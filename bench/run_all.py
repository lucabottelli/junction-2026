"""The one command for the station: environment report -> idle baseline -> (optional INT8 quantization) -> every
model on every device -> sorter cycle statistics -> bench/results/REPORT.md + section9_rows.md + charts.

Models found automatically:
  anomaly  ROOT/models/<cam>/<name>/**/model.xml          (train_anomaly.py layout; *_int8.xml siblings too)
           input images = the model's held-out + defect crops from split.json (training_images/<cam>/inhand/...),
           else training_images/<cam>/inhand/good; role "fast" for <name> containing _efficientad, else "main"
  policy   ~/.local/share/physicalai/models/<id>/exports/openvino/*.xml (else exports/onnx/*.onnx)    role "policy"
           (also ROOT/models/**/exports/*/ with a manifest.json), random inputs of the export's shapes

    python -m bench                       # full run (~15-25 min with a few models)
    python -m bench --quick               # LATENCY only, shorter windows (~5 min)
    python -m bench --models a.xml b.xml  # just these
    python -m bench --report-only         # rebuild the report from bench/results + runs/
"""
import argparse, importlib.util, json, os, subprocess, sys
from pathlib import Path

from . import BENCH, RESULTS, ROOT


def home():
    """The real user's home also under sudo (Studio's models live there)."""
    u = os.environ.get("SUDO_USER")
    return Path(os.path.expanduser(f"~{u}")) if u else Path.home()


def anomaly_models(root=ROOT):
    out = []
    mdir = Path(root) / "models"
    if not mdir.is_dir(): return out
    for cam in sorted(p for p in mdir.iterdir() if p.is_dir()):
        for d in sorted(p for p in cam.iterdir() if p.is_dir()):
            xmls = sorted(x for x in d.rglob("*.xml") if "work" not in x.relative_to(d).parts[:1] and x.with_suffix(".bin").exists())
            if not xmls: continue
            imgs = model_images(root, cam.name, d)
            for x in xmls:
                tag = f"{cam.name}/{d.name}" + ("" if x.stem == "model" else f"-{x.stem.replace('model_', '')}")
                out.append({"path": str(x), "tag": tag, "role": "fast" if "efficientad" in d.name else "main", "images": imgs or None,
                            "kind": "anomaly", "model_dir": str(d), "cam": cam.name})
    return out


def model_images(root, cam, model_dir, limit=64):
    base = Path(root) / "training_images" / cam / "inhand"
    try:
        sp = json.loads((Path(model_dir) / "split.json").read_text())
        fs = [base / "good" / n for n in sp.get("holdout", [])] + [base / "defect" / n for n in sp.get("defect", [])]
        fs = [str(f) for f in fs if f.exists()]
        if fs: return fs[:: max(1, len(fs) // limit)][:limit]
    except Exception: pass
    from .bench_model import list_images
    return list_images(base / "good", limit) or None


def policy_models(roots=None, root=ROOT):
    roots = roots or [home() / ".local/share/physicalai/models", Path(root) / "models"]
    out, seen = [], set()
    for r in map(Path, roots):
        if not r.is_dir(): continue
        for mf in sorted(set(r.glob("*/exports/*/manifest.json")) | set(r.glob("**/exports/*/manifest.json"))):
            exp = mf.parent.parent
            if exp in seen: continue
            pick = None
            for be, pat in (("openvino", "*.xml"), ("onnx", "*.onnx")):
                fs = sorted((exp / be).glob(pat)) if (exp / be).is_dir() else []
                if fs: pick = fs[0]; break
            if not pick: continue
            seen.add(exp)
            name = ""
            for f in sorted(exp.glob("*/manifest.json")):
                try: name = name or (json.loads(f.read_text()).get("policy") or {}).get("name") or ""
                except Exception: pass
            mid = exp.parent.name
            out.append({"path": str(pick), "tag": f"policy/{name or 'policy'}-{mid[:8]}", "role": "policy", "kind": "policy", "images": None})
    return out


def has_nncf(): return importlib.util.find_spec("nncf") is not None


def quantize_all(models, root=ROOT, log=print):
    """INT8 for every anomaly model without one yet: in this venv if it has nncf, else with the Anomalib venv."""
    py = None
    if not has_nncf():
        ap = Path(os.environ.get("ANOMALIB_PY", home() / "anomalib/.venv/bin/python"))
        if ap.exists(): py = str(ap)
        else:
            log("INT8: nncf not installed here and no Anomalib venv found: skipping quantization (see bench/README.md)"); return []
    new = []
    for m in [m for m in models if m["kind"] == "anomaly" and Path(m["path"]).stem == "model"]:
        dst = Path(m["path"]).with_name("model_int8.xml")
        if dst.exists() or not m.get("images"): continue
        cal, guard = calib_lists(m, dst, root)
        cmd = [py or sys.executable, "-m", "bench.quantize", m["path"], "--calib", cal, "--out", str(dst)] + (["--guard-images", guard] if guard else [])
        log("INT8: " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(BENCH.parent), text=True, capture_output=True)
        out = [l for l in r.stdout.strip().splitlines() if l.startswith(("accuracy guard", "  decisions"))]
        log("\n".join("   " + l.strip() for l in out) if r.returncode == 0 else f"   FAILED: {(r.stderr or r.stdout).strip()[-400:]}")
        if r.returncode == 0 and dst.exists():
            new.append({**m, "path": str(dst), "tag": m["tag"] + "-int8"})
    return new


def calib_lists(m, dst, root=ROOT):
    """Calibration = the model's TRAINING crops (split.json "train"), guard = its held-out + defect crops (m["images"]).
    Written as .txt lists next to the INT8 model. Without split.json: the good folder for both."""
    base = Path(root) / "training_images" / m.get("cam", "front") / "inhand"
    try: train = [str(base / "good" / n) for n in json.loads((Path(m["model_dir"]) / "split.json").read_text()).get("train", [])]
    except Exception: train = []
    train = [f for f in train if Path(f).exists()]
    if not train: return str(base / "good"), None
    cal = Path(dst).with_name("calib_train.txt"); cal.write_text("\n".join(train) + "\n")
    guard = None
    if isinstance(m.get("images"), list):
        guard = Path(dst).with_name("guard_images.txt"); guard.write_text("\n".join(m["images"]) + "\n"); guard = str(guard)
    return str(cal), guard


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--models", nargs="*", help="explicit model files instead of the discovery")
    ap.add_argument("--images", help="images for --models")
    ap.add_argument("--devices", default="CPU,GPU,NPU,AUTO")
    ap.add_argument("--hints", default="LATENCY,THROUGHPUT")
    ap.add_argument("--precisions", default="default,f32")
    ap.add_argument("--quick", action="store_true", help="LATENCY only, n=100, 3 s windows, 10 s idle")
    ap.add_argument("--quantize", choices=["auto", "yes", "no"], default="auto", help="auto = if nncf or the Anomalib venv is there")
    ap.add_argument("--no-policies", action="store_true"); ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--min-seconds", type=float, default=5.0)
    ap.add_argument("--async-seconds", type=float, default=5.0); ap.add_argument("--idle-seconds", type=float, default=15.0)
    ap.add_argument("--cooldown", type=float, default=20.0); ap.add_argument("--shape"); ap.add_argument("--hw", type=int, default=256)
    ap.add_argument("--runs", nargs="*", help="sorter runs (default ROOT/runs)"); ap.add_argument("--truth")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--root", default=str(ROOT), help="project root with models/, training_images/, runs/ (default: this repo)")
    ap.add_argument("--studio-models", default=str(home() / ".local/share/physicalai/models"), help="Studio's models dir")
    a = ap.parse_args(argv)
    if a.quick:                                  # shorter defaults; values given explicitly on the command line win
        for k, v in {"hints": "LATENCY", "n": 100, "min_seconds": 3.0, "async_seconds": 3.0, "idle_seconds": 10.0, "cooldown": 10.0}.items():
            if getattr(a, k) == ap.get_default(k): setattr(a, k, v)
    if importlib.util.find_spec("openvino") is None:
        print(f"openvino is not importable with {sys.executable}: run with Studio's venv "
              "(PY=~/physical-ai-studio/application/backend/.venv/bin/python bench/run_all.sh) or the Anomalib venv"); return 2
    from . import devices, report
    from .bench_model import bench_models, parse_shape
    if not a.report_only:
        env = devices.report()
        print(devices.text(env), "\n")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        if a.models:
            models = [{"path": m, "tag": None, "images": a.images, "kind": "given"} for m in a.models]
        else:
            models = ([] if a.no_anomaly else anomaly_models(a.root)) + \
                     ([] if a.no_policies else policy_models([a.studio_models, Path(a.root) / "models"]))
            if a.quantize == "yes" or (a.quantize == "auto" and models):
                models += quantize_all(models, a.root)
        if not models:
            print(f"no models found (looked in {Path(a.root) / 'models'} and {a.studio_models}): "
                  "train/export first, or pass --models"); return 1
        print(f"{len(models)} model(s): " + ", ".join(m.get("tag") or Path(m["path"]).name for m in models))
        from .bench_model import Session
        from .sensors import Sensors, headline, measure_idle
        sen = Sensors()
        idle = None
        if a.idle_seconds > 0:
            print(f"\nidle baseline {a.idle_seconds:.0f} s: hands off, close Studio's camera previews / browser tabs ...")
            idle = measure_idle(sen, a.idle_seconds); print("  idle: " + headline(idle))
        sess = Session(a.out, {**vars(a), "argv": sys.argv}, sen, idle, env)
        bench_models(models, a.devices.split(","), a.hints.split(","), a.precisions.split(","), a.n, a.min_seconds, 20, a.async_seconds,
                     0, a.cooldown, 0.25, parse_shape(a.shape), a.hw, session=sess, sensors=sen, args=vars(a))
        from .cycle_stats import collect
        try: sess.data["cycle_stats"] = collect(a.runs or [Path(a.root) / "runs"], a.truth); sess.save()
        except Exception as e: print("cycle stats:", e)
        print("\nresults:", sess.path)
    print()
    report.build(a.out, "all", a.runs or [Path(a.root) / "runs"], a.truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
