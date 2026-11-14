"""Train + export one anomaly model per brick and camera for the sorter. Run it with the ANOMALIB venv:

    ~/anomalib/.venv/bin/python ~/junction26/anomaly_capture/train_anomaly.py --brick red          # PatchCore, auto device
    ~/anomalib/.venv/bin/python .../train_anomaly.py --brick red --model efficientad --xpu       # fast first-stage model
    ~/anomalib/.venv/bin/python .../train_anomaly.py --brick red --cam wrist                     # top photo on the tape spot
    ~/anomalib/.venv/bin/python .../train_anomaly.py --brick red --image-size 384 --tag s384     # a variant to --compare
    python3 .../train_anomaly.py --brick red --dry-run                                           # only print the split
One command for train main + fast -> export -> eval -> thresholds: anomaly_pipeline.sh (same folder).

Input : training_images/<cam>/inhand/good/*_<brick>_*_crop.png (and defect/ for the internal test metrics), i.e.
        exactly the crops prep.inspect_crop() / prep.release_crop() make, the same functions the sorter uses live.
        cam front = brick held in front of the camera; cam wrist = released brick on the tape spot from above.
Split : HELD OUT BY EPISODE. An episode = one handling of the brick = (run, cycle) of meta.jsonl (all its views and
        cameras share the grip/pose); without a meta line the capture group of the file name is used. ~20 % of the
        good episodes (--holdout-frac, deterministic per --seed) are never trained on: the page's "calibrate" and
        eval_anomaly.py score them (a model scores its own training images too low -> too tight thresholds).
        Defect crops are never trained on (decision 13: one-class models, defects only evaluate/calibrate).
Output: models/<cam>/<brick>[_<model> unless patchcore][_<tag>]/   (or --out DIR)
          weights/openvino/model.xml (+.bin)  OpenVINO IR, FP16 weights by default; scorer.py picks it up
          holdout.json   held-out good crop names (what sorter.calibrate() reads)
          split.json     train / holdout / defect file names with their episode ids (what eval_anomaly.py reads)
          model_card.json  data counts, split, params, device, versions, train time, Anomalib's internal test metrics
          work/          Anomalib's work dir (checkpoint); --clean-work deletes it after the export
        then: eval_anomaly.py (metrics + thresholds.json) or press "calibrate" on the page.
Resolution policy (decision 11): one model per camera/stage, trained at the scale it runs at (--image-size; the
crop is resized to it inside the exported graph, so the run-time path resizes identically). Do not mix cameras.
Checked against the Anomalib 2.7.1.dev0 source (Folder, Patchcore/Padim/EfficientAd/Dinomaly, Engine.export with
CompressionType.FP16). Station has XPU; --device auto falls back to CPU when torch.xpu is not available.
"""
import argparse, hashlib, json, shutil, sys, tempfile, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("patchcore", "efficientad", "padim", "dinomaly")


# ---------------------------------------------------------------- data + split (no anomalib needed: tests use these)
def load_meta(data):
    """raw-photo stem -> meta.jsonl record (the crop is <stem>_crop.png)."""
    m = {}
    try:
        for line in open(Path(data) / "meta.jsonl"):
            line = line.strip()
            if not line: continue
            try: j = json.loads(line)
            except Exception: continue
            m[Path(j["file"]).stem] = j
    except FileNotFoundError: pass
    return m


def crop_stem(f):
    s = Path(f).stem
    return s[:-len("_crop")] if s.endswith("_crop") else s


def brick_of(f):
    p = Path(f).stem.split("_")
    return p[3] if len(p) > 4 else None


def episode_of(f, meta):
    """Episode id of a crop: 'run/cycle' from meta.jsonl, else the capture group (date_time_ms of the file name).
    Same grouping as sorter.calibrate() (which keys on (run, cycle))."""
    j = meta.get(crop_stem(f))
    if j and j.get("run") is not None and j.get("cycle") is not None: return f"{j['run']}/{j['cycle']}"
    return "group:" + "_".join(Path(f).stem.split("_")[:3])


def list_crops(data, cam, label, brick="", views=None):
    """Same selection as the original trainer / sorter.calibrate(): *_crop.png, brick = 4th '_' field."""
    fs = sorted((Path(data) / cam / "inhand" / label).glob("*_crop.png"))
    fs = [f for f in fs if not brick or brick_of(f) == brick]
    if views: fs = [f for f in fs if crop_stem(f).split("_", 4)[-1] in views]
    return fs


def split_by_episode(files, meta, frac=0.2, seed=0):
    """-> (train, holdout, episodes {name: ep}). Whole episodes go to one side; deterministic for (episode, seed)."""
    eps = {f.name: episode_of(f, meta) for f in files}
    uniq = sorted(set(eps.values()))
    if frac <= 0 or len(uniq) < 2: return list(files), [], eps
    rank = sorted(uniq, key=lambda e: hashlib.sha1(f"{seed}:{e}".encode()).hexdigest())
    n_hold = min(len(uniq) - 1, max(1, round(frac * len(uniq))))
    hold = set(rank[:n_hold])
    return [f for f in files if eps[f.name] not in hold], [f for f in files if eps[f.name] in hold], eps


def thin_by_episode(files, eps, n):
    """At most n files, spread over as many episodes as possible (for quick CPU runs)."""
    if not n or len(files) <= n: return files
    by = {}
    for f in files: by.setdefault(eps[f.name], []).append(f)
    out, i = [], 0
    while len(out) < n:
        for e in sorted(by):
            if i < len(by[e]) and len(out) < n: out.append(by[e][i])
        i += 1
    return sorted(out)


def out_dir(models, cam, name, model, tag=""):
    """models/<cam>/<name>[_<model> unless patchcore][_<tag>]: <name> = main model, <name>_efficientad = cascade."""
    return Path(models) / cam / (name + ("" if model == "patchcore" else f"_{model}") + (f"_{tag}" if tag else ""))


# ---------------------------------------------------------------- anomalib
def pick_device(want):
    import torch
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if want == "xpu" and not xpu: print("WARNING: --device xpu but torch.xpu is not available: training on CPU")
    return "xpu" if want in ("xpu", "auto") and xpu else "cpu"


def post_processor(kind):
    """Score scale of the exported model. 'anomalib': min-max normalised to 0..1 around the val threshold (0.5) and
    CLAMPED, so the hardest goods and the defects can all read 1.000 (ties -> sorter.thresholds_from can't separate).
    'unclamped' (default): the same scale (0.5 = Anomalib's threshold, sorter defaults 0.4/0.6 still sensible) without
    the clamp, so the ranking survives. 'raw': the model's raw distance (thresholds.json is then mandatory)."""
    if kind == "anomalib": return True
    from anomalib.post_processing import PostProcessor
    if kind == "raw": return PostProcessor(enable_normalization=False)
    import torch

    class UnclampedPostProcessor(PostProcessor):
        @staticmethod
        def _normalize(preds, norm_min, norm_max, threshold):
            if preds is None: return preds
            threshold = torch.where(torch.isnan(threshold), (norm_max + norm_min) / 2, threshold)
            rng = norm_max - norm_min
            rng = torch.where(rng == 0, torch.ones_like(rng), rng)
            has = ~torch.isnan(norm_min) & ~torch.isnan(norm_max)
            return torch.where(has, (preds - threshold) / rng + 0.5, preds)
    if not callable(getattr(PostProcessor, "_normalize", None)):
        print("WARNING: this Anomalib has no PostProcessor._normalize: using its default (clamped) scores"); return True
    return UnclampedPostProcessor()


def build_model(a, M):
    size = (a.image_size, a.image_size)
    if a.model == "patchcore":
        cls = M.Patchcore
        kw = dict(backbone=a.backbone or "wide_resnet50_2", layers=a.layers or ["layer2", "layer3"],
                  coreset_sampling_ratio=a.coreset, num_neighbors=a.neighbors, pre_trained=not a.no_pretrained)
    elif a.model == "padim":
        cls = M.Padim
        kw = dict(backbone=a.backbone or "resnet18", layers=a.layers or ["layer1", "layer2", "layer3"], pre_trained=not a.no_pretrained)
    elif a.model == "efficientad":
        cls = M.EfficientAd
        kw = dict(model_size=a.model_size, imagenet_dir=str(Path(a.imagenet_dir).expanduser()))
    else:
        cls = M.Dinomaly; kw = {}
    model = cls(pre_processor=cls.configure_pre_processor(image_size=size), post_processor=post_processor(a.score_norm),
                visualizer=False, **kw)
    return model, {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in kw.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="train + export one anomaly model (see the module docstring)")
    ap.add_argument("--brick", default="", help="brick name in the file names ('' = all bricks -> models/<cam>/all)")
    ap.add_argument("--cam", default="front")
    ap.add_argument("--model", default="patchcore", choices=MODELS)
    ap.add_argument("--xpu", action="store_true", help="train on the Intel GPU (XPU) (= --device xpu)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "xpu"], help="auto = XPU if available, else CPU")
    ap.add_argument("--no-holdout", action="store_true", help="train on every good crop (thresholds will be optimistic)")
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="share of good EPISODES held out (default 0.2)")
    ap.add_argument("--seed", type=int, default=0, help="split + training seed")
    ap.add_argument("--image-size", type=int, default=256, help="model input side in px (decision 11: 256, then 384/512)")
    ap.add_argument("--backbone", default="", help="patchcore: wide_resnet50_2 (default), resnet18, ...; padim: resnet18")
    ap.add_argument("--layers", default="", help="comma list, e.g. layer2,layer3")
    ap.add_argument("--coreset", type=float, default=0.1, help="patchcore coreset_sampling_ratio")
    ap.add_argument("--neighbors", type=int, default=9, help="patchcore num_neighbors")
    ap.add_argument("--no-pretrained", action="store_true", help="random backbone (only when ImageNet weights can't be downloaded)")
    ap.add_argument("--epochs", type=int, default=20, help="efficientad / dinomaly epochs")
    ap.add_argument("--model-size", default="small", choices=["small", "medium"], help="efficientad S / M")
    ap.add_argument("--imagenet-dir", default="~/anomalib/datasets/imagenette", help="efficientad: imagenette (downloaded once)")
    ap.add_argument("--compress", default="fp16", choices=["fp16", "none", "int8", "int8_ptq"], help="OpenVINO export weights")
    ap.add_argument("--score-norm", default="unclamped", choices=["unclamped", "anomalib", "raw"],
                    help="score scale of the export (see post_processor()); unclamped = 0..1-ish without ties")
    ap.add_argument("--views", default="", help="only these views (comma list), e.g. wp2-1,wp2-2")
    ap.add_argument("--max-train", type=int, default=0, help="use at most N good crops for training (quick CPU runs)")
    ap.add_argument("--tag", default="", help="variant suffix of the output dir, e.g. s384")
    ap.add_argument("--out", default="", help="explicit output dir (overrides models/<cam>/...)")
    ap.add_argument("--data", default=str(ROOT / "training_images"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--clean-work", action="store_true", help="delete Anomalib's work dir (checkpoints) after the export")
    ap.add_argument("--dry-run", action="store_true", help="print the split and exit (no Anomalib needed)")
    a = ap.parse_args(argv)
    if a.xpu: a.device = "xpu"
    a.layers = [s for s in a.layers.split(",") if s]
    views = [v for v in a.views.split(",") if v] or None
    data = Path(a.data); name = a.brick or "all"
    out = Path(a.out) if a.out else out_dir(a.models, a.cam, name, a.model, a.tag)

    meta = load_meta(data)
    good, bad = list_crops(data, a.cam, "good", a.brick, views), list_crops(data, a.cam, "defect", a.brick, views)
    src = data / a.cam / "inhand"
    if len(good) < 10: raise SystemExit(f"only {len(good)} good crops for '{name}' in {src}/good: collect more first")
    train, holdout, eps = split_by_episode(good, meta, 0 if a.no_holdout else a.holdout_frac, a.seed)
    train = thin_by_episode(train, eps, a.max_train)
    deps = {f.name: episode_of(f, meta) for f in bad}
    no_meta = sum(1 for e in list(eps.values()) + list(deps.values()) if e.startswith("group:"))
    assert not {eps[f.name] for f in train} & {eps[f.name] for f in holdout}, "episode leakage between train and holdout"
    split = {"by": "episode = (run, cycle) from meta.jsonl; capture group if no meta line", "holdout_frac": a.holdout_frac,
             "seed": a.seed, "no_meta_files": no_meta,
             "train": [f.name for f in train], "holdout": [f.name for f in holdout], "defect": [f.name for f in bad],
             "episodes": {**eps, **deps}}
    counts = {"good_total": len(good), "train": len(train), "holdout": len(holdout), "defect": len(bad),
              "episodes_train": len({eps[f.name] for f in train}), "episodes_holdout": len({eps[f.name] for f in holdout}),
              "defect_episodes": len(set(deps.values()))}
    print(f"{name}/{a.cam}: {counts['train']} good crops to train on ({counts['episodes_train']} episodes), "
          f"{counts['holdout']} good held out ({counts['episodes_holdout']} episodes), {counts['defect']} defect crops "
          f"({counts['defect_episodes']} episodes)" + (f"  [{no_meta} files without a meta.jsonl line]" if no_meta else ""))
    if a.dry_run:
        print(json.dumps({"out": str(out), **counts}, indent=1)); return {"out": out, "split": split, "counts": counts}
    if len(train) < 5: raise SystemExit("fewer than 5 training crops after the split")

    import torch
    from anomalib import __version__ as anomalib_version
    from anomalib import models as M
    from anomalib.data import Folder
    from anomalib.deploy import CompressionType, ExportType
    from anomalib.engine import Engine
    from lightning.pytorch import seed_everything
    seed_everything(a.seed, workers=True)

    tmp = Path(tempfile.mkdtemp(prefix=f"anom_{name}_"))
    for lab, fs in (("train_good", train), ("holdout_good", holdout), ("defect", bad)):
        (tmp / lab).mkdir()
        for f in fs: (tmp / lab / f.name).symlink_to(f.resolve())
    # test = held-out good + defect; validation = the same set (only sets Anomalib's monotonic min-max normalisation of
    # the exported score to 0..1; ranking metrics and eval_anomaly.py's thresholds don't depend on it)
    dm = Folder(name=name, root=tmp, normal_dir="train_good", normal_test_dir="holdout_good" if holdout else None,
                abnormal_dir="defect" if bad else None, train_batch_size=1 if a.model == "efficientad" else 8,
                eval_batch_size=8, num_workers=0, val_split_mode="same_as_test", seed=a.seed)
    model, params = build_model(a, M)
    device = pick_device(a.device)
    kw = {"default_root_dir": str(out / "work"), "logger": False, "devices": 1}
    if a.model in ("efficientad", "dinomaly"): kw["max_epochs"] = a.epochs
    if device == "xpu":
        from anomalib.engine.accelerator.xpu import XPUAccelerator
        from anomalib.engine.strategy.xpu_single import SingleXPUStrategy
        kw.update(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())
    else:
        kw["accelerator"] = "cpu"
    engine = Engine(**kw)
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    t_train = time.time() - t0
    test_metrics = {}
    if bad and holdout:
        try:
            r = engine.test(model=model, datamodule=dm)
            test_metrics = {k: float(v) for k, v in (r[0] if r else {}).items()}
        except Exception as e:
            test_metrics = {"error": f"{type(e).__name__}: {e}"}
    if (out / "weights").exists(): shutil.rmtree(out / "weights")
    comp = {"fp16": CompressionType.FP16, "none": None, "int8": CompressionType.INT8,
            "int8_ptq": CompressionType.INT8_PTQ}[a.compress]
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=out, compression_type=comp,
                  datamodule=dm if a.compress == "int8_ptq" else None)
    xml = sorted(out.rglob("model.xml"))
    (out / "holdout.json").write_text(json.dumps([f.name for f in holdout]))
    (out / "split.json").write_text(json.dumps(split, indent=1))
    card = {"created": datetime.now().isoformat(timespec="seconds"), "brick": name, "cam": a.cam, "model": a.model,
            "tag": a.tag, "synthetic_data": (data / "synth_info.json").exists(), "data_dir": str(data),
            "views": views, "image_size": a.image_size, "params": params, "compress": a.compress, "score_norm": a.score_norm,
            "device": device, "seed": a.seed, "data": counts,
            "split": {k: split[k] for k in ("by", "holdout_frac", "seed", "no_meta_files")},
            "train_s": round(t_train, 1), "export": [str(p.relative_to(out)) for p in xml],
            "anomalib": anomalib_version, "torch": torch.__version__,
            "anomalib_test_metrics": test_metrics,
            "anomalib_test_note": "Anomalib's own metrics on held-out good + all defect crops of this brick (photo level; "
                                  "includes defect photos where the cross faces away). eval_anomaly.py is the reference.",
            "command": " ".join(sys.argv)}
    (out / "model_card.json").write_text(json.dumps(card, indent=1))
    if a.clean_work: shutil.rmtree(out / "work", ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("exported:", *xml)
    print(f"next: eval_anomaly.py --model-dir {out}   (or on /sort choose judge = model, brick = {name}, press 'calibrate')")
    return {"out": out, "card": card}


if __name__ == "__main__":
    main()
