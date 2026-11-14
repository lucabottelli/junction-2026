"""INT8 post-training quantization (NNCF) of an OpenVINO IR / ONNX model from a folder of calibration images, plus
the accuracy guard: anomaly-score drift FP vs INT8 on the same images (and decision flips against the model's
calibrated thresholds.json, if there is one). INT8 is what the NPU runs best and usually halves CPU/iGPU latency;
the guard says whether the cheaper model still gives the same answers.

nncf is imported lazily. It is already in the Anomalib venv (anomalib[openvino]); in other venvs `pip install nncf`
(check it does not downgrade numpy there). The guard alone (score_drift / --guard-only) needs only openvino.

    ~/anomalib/.venv/bin/python -m bench.quantize models/front/red/weights/openvino/model.xml \
        --calib training_images/front/inhand/good                       # -> .../model_int8.xml + model_int8.quant.json
    python -m bench.quantize model.xml --calib DIR --ignore-types MatMul   # PatchCore: keep the memory-bank distance FP
    python -m bench.quantize model.xml --guard-only other_model.xml --calib DIR   # drift between any two models
Images are fed exactly as bench_model feeds them (RGB, resized to the input, 0..1, NCHW): the same contract as
Anomalib's OpenVINOInferencer, so the scores compare with the sorter's thresholds.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

from .bench_model import _name, list_images, make_feeds, parse_shape, static_shapes


def _nncf():
    try:
        import nncf
        return nncf
    except ImportError:
        raise SystemExit("nncf is not installed in this venv. Run this with the Anomalib venv (~/anomalib/.venv/bin/python -m "
                         "bench.quantize ...), which has it, or `pip install nncf` here (check it keeps your numpy/openvino).")


def score_of(cm, outputs):
    """Anomaly score from one inference: output 'pred_score' if present, else a 1-element output, else max of output 0."""
    by = {}
    for i, port in enumerate(cm.outputs):
        names = port.get_names() or {f"out{i}"}
        by.update({n: outputs[port] for n in names}); by.setdefault(i, outputs[port])
    for k, v in by.items():
        if isinstance(k, str) and "pred_score" in k: return float(np.asarray(v).reshape(-1)[0])
    for v in by.values():
        if np.asarray(v).size == 1: return float(np.asarray(v).reshape(-1)[0])
    return float(np.asarray(by[0]).max())


def find_thresholds(model_path):
    """scorer.py keeps thresholds.json at the model root (models/<cam>/<brick>/), a few levels above model.xml."""
    p = Path(model_path).resolve().parent
    for _ in range(5):
        f = p / "thresholds.json"
        if f.exists():
            try: return json.loads(f.read_text()), str(f)
            except Exception: return None, None
        p = p.parent
    return None, None


def image_files(src, limit=300):
    """A folder of images, or a .txt file with one image path per line."""
    if src and Path(src).is_file() and Path(src).suffix == ".txt":
        fs = [l.strip() for l in Path(src).read_text().splitlines() if l.strip() and Path(l.strip()).exists()]
        return fs[:: max(1, len(fs) // limit)][:limit]
    return list_images(src, limit)


def _rank(a): return np.argsort(np.argsort(a)).astype(float)


def _corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return None
    return round(float(np.corrcoef(a, b)[0, 1]), 5)


def scores(model_path, images, device="CPU", shape=None, hw=256, scale="unit", bgr=False):
    import openvino as ov
    core = ov.Core(); m = core.read_model(str(model_path))
    shapes, _ = static_shapes(m, shape, hw)
    if any(p.get_partial_shape().is_dynamic for p in m.inputs):
        m.reshape({m.inputs[i]: ov.PartialShape(s) for i, s in shapes.items()})
    cm = core.compile_model(m, device)
    feeds, _ = make_feeds(m, shapes, images, k=len(images), scale=scale, bgr=bgr)
    return np.array([score_of(cm, cm(f)) for f in feeds])


def score_drift(ref_model, new_model, images, device="CPU", thresholds=None, **kw):
    """Scores of both models on the same images -> drift statistics (+ decision flips with t_low/t_high)."""
    if not images: raise RuntimeError("no images for the accuracy guard")
    a = scores(ref_model, images, device, **kw); b = scores(new_model, images, device, **kw)
    d = b - a; rng = float(a.max() - a.min()) or 1.0
    out = {"n": int(len(a)), "device": device, "ref_mean": round(float(a.mean()), 5), "new_mean": round(float(b.mean()), 5),
           "mean_abs": round(float(np.abs(d).mean()), 6), "max_abs": round(float(np.abs(d).max()), 6), "bias": round(float(d.mean()), 6),
           "max_abs_rel_range": round(float(np.abs(d).max() / rng), 5), "pearson": _corr(a, b), "spearman": _corr(_rank(a), _rank(b)),
           "per_image": [{"image": Path(f).name, "ref": round(float(x), 5), "new": round(float(y), 5)} for f, x, y in zip(images, a, b)][:200]}
    if thresholds and "t_low" in thresholds and "t_high" in thresholds:
        tl, th = float(thresholds["t_low"]), float(thresholds["t_high"])
        dec = lambda s: np.where(s >= th, "defect", np.where(s <= tl, "clean", "uncertain"))
        da, db = dec(a), dec(b)
        out.update(t_low=tl, t_high=th, decision_flips=int((da != db).sum()),
                   flips=[{"image": Path(f).name, "ref": x, "new": y} for f, x, y in zip(images, da, db) if x != y][:50])
    return out


def quantize(src, calib, dst=None, subset=300, preset="performance", ignore_types=(), shape=None, hw=256, scale="unit",
             bgr=False, guard_device="CPU", guard_images=None, log=print):
    nncf = _nncf()
    import openvino as ov
    src = Path(src); dst = Path(dst) if dst else src.with_name(src.stem + "_int8.xml")
    core = ov.Core(); model = core.read_model(str(src))
    imgs = image_files(calib, subset)
    if not imgs: raise SystemExit(f"no calibration images in {calib}")
    shapes, _ = static_shapes(model, shape, hw)
    feeds, data = make_feeds(model, shapes, imgs, k=len(imgs), scale=scale, bgr=bgr)
    named = [{_name(model.inputs[i], i): v for i, v in f.items()} for f in feeds]
    log(f"quantizing {src} with {len(named)} calibration images ({data}), preset {preset}"
        + (f", ignoring op types {list(ignore_types)}" if ignore_types else ""))
    kw = {"preset": nncf.QuantizationPreset.MIXED if preset == "mixed" else nncf.QuantizationPreset.PERFORMANCE,
          "subset_size": min(subset, len(named))}
    if ignore_types: kw["ignored_scope"] = nncf.IgnoredScope(types=list(ignore_types))
    t0 = time.time(); q = nncf.quantize(model, nncf.Dataset(named), **kw); qs = time.time() - t0
    ov.save_model(q, str(dst))
    log(f"saved {dst} ({qs:.1f} s)")
    thr, thr_file = find_thresholds(src)
    rep = {"src": str(src), "dst": str(dst), "calib_dir": str(calib), "n_calib": len(named), "preset": preset,
           "ignore_types": list(ignore_types), "quantize_s": round(qs, 2), "nncf": getattr(nncf, "__version__", None),
           "openvino": ov.get_version(), "thresholds_file": thr_file,
           "guard_images": str(guard_images or calib),
           "guard": score_drift(src, dst, image_files(guard_images, 200) if guard_images else imgs, guard_device, thr,
                                shape=shape, hw=hw, scale=scale, bgr=bgr)}
    dst.with_suffix(".quant.json").write_text(json.dumps(rep, indent=1))
    return rep


def guard_text(g):
    t = (f"accuracy guard on {g['n']} images ({g['device']}): mean |dscore| {g['mean_abs']:.4g}, max {g['max_abs']:.4g} "
         f"({100 * g['max_abs_rel_range']:.1f}% of the FP score range), bias {g['bias']:+.4g}, pearson {g['pearson']}, spearman {g['spearman']}")
    if "decision_flips" in g: t += f"\n  decisions vs thresholds ({g['t_low']}, {g['t_high']}): {g['decision_flips']} flip(s)"
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("model")
    ap.add_argument("--calib", required=True, help="folder (or .txt list) of calibration images: good TRAINING crops, a few hundred")
    ap.add_argument("--guard-images", help="folder / .txt list for the accuracy guard (default: the calibration images); "
                                           "best: held-out good + defect crops")
    ap.add_argument("--out", help="output .xml (default <model>_int8.xml)")
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--preset", default="performance", choices=["performance", "mixed"])
    ap.add_argument("--ignore-types", default="", help="op types to keep in FP, e.g. MatMul (PatchCore memory bank)")
    ap.add_argument("--shape"); ap.add_argument("--hw", type=int, default=256)
    ap.add_argument("--scale", default="unit", choices=["unit", "raw", "imagenet"]); ap.add_argument("--bgr", action="store_true")
    ap.add_argument("--device", default="CPU", help="device for the accuracy guard")
    ap.add_argument("--guard-only", metavar="OTHER_MODEL", help="no quantization: compare MODEL with OTHER_MODEL")
    ap.add_argument("--max-drift", type=float, help="exit 3 if max |dscore| exceeds this, or any decision flips")
    a = ap.parse_args()
    kw = dict(shape=parse_shape(a.shape), hw=a.hw, scale=a.scale, bgr=a.bgr)
    if a.guard_only:
        thr, _ = find_thresholds(a.model)
        g = score_drift(a.model, a.guard_only, image_files(a.guard_images or a.calib, a.subset), a.device, thr, **kw)
    else:
        g = quantize(a.model, a.calib, a.out, a.subset, a.preset, [t for t in a.ignore_types.split(",") if t], guard_device=a.device,
                     guard_images=a.guard_images, **kw)["guard"]
    print(guard_text(g))
    if a.max_drift is not None and (g["max_abs"] > a.max_drift or g.get("decision_flips")):
        print(f"GUARD FAILED (max drift > {a.max_drift} or decision flips): do not deploy this model"); sys.exit(3)


if __name__ == "__main__":
    main()
