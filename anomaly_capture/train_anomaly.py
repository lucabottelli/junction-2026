"""Train + export one anomaly model per brick for the sorter. Run it with the ANOMALIB venv:

    ~/anomalib/.venv/bin/python ~/junction26/anomaly_capture/train_anomaly.py --brick red          # PatchCore, CPU/auto
    ~/anomalib/.venv/bin/python .../train_anomaly.py --brick red --model efficientad --xpu       # fast first-stage model
    ~/anomalib/.venv/bin/python .../train_anomaly.py --brick red --cam wrist                     # top photo on the tape spot

Input : training_images/<cam>/inhand/good/*_<brick>_*_crop.png (and defect/ for the test metrics), i.e. exactly
        the crops prep.inspect_crop() / prep.release_crop() make, the same functions the sorter uses live.
        cam front = brick held in front of the camera; cam wrist = released brick on the tape spot from above.
Output: models/<cam>/<brick>/weights/openvino/model.xml  (+ the work dir with metrics). scorer.py picks it up;
        then press "calibrate" on the page to set the thresholds through the live scoring path.
Written against Anomalib 2.x (Folder / Patchcore / Engine.export); if your 2.7 build names differ, the error
message will say which.
"""
import argparse, shutil, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brick", default="", help="brick name in the file names ('' = all bricks -> models/<cam>/all)")
    ap.add_argument("--cam", default="front")
    ap.add_argument("--model", default="patchcore", choices=["patchcore", "efficientad", "padim"])
    ap.add_argument("--xpu", action="store_true", help="train on the Intel GPU (XPU)")
    ap.add_argument("--no-holdout", action="store_true", help="train on every good crop (thresholds will be optimistic)")
    args = ap.parse_args()

    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib import models as M

    src = ROOT / "training_images" / args.cam / "inhand"
    name = args.brick or "all"

    def pick(lab):
        fs = sorted((src / lab).glob("*_crop.png"))
        return [f for f in fs if not args.brick or (len(f.stem.split("_")) > 4 and f.stem.split("_")[3] == args.brick)]

    good, bad = pick("good"), pick("defect")
    if len(good) < 10: raise SystemExit(f"only {len(good)} good crops for '{name}' in {src}/good: collect more first")
    # every 5th good crop is NOT trained on: the page's "calibrate" uses these to set the thresholds honestly
    # (a model scores the images it was trained on too low, which would make the thresholds too tight)
    holdout = good[4::5] if not args.no_holdout else []
    good = [f for f in good if f not in set(holdout)]
    print(f"{name}: {len(good)} good crops to train on, {len(holdout)} good held out for calibration, {len(bad)} defect crops")
    tmp = Path(tempfile.mkdtemp(prefix=f"anom_{name}_"))
    for lab, fs in (("good", good), ("defect", bad)):
        (tmp / lab).mkdir()
        for f in fs: (tmp / lab / f.name).symlink_to(f)

    dm = Folder(name=name, root=tmp, normal_dir="good", abnormal_dir="defect" if bad else None,
                train_batch_size=8, eval_batch_size=8, num_workers=0)
    model = {"patchcore": lambda: M.Patchcore(), "efficientad": lambda: M.EfficientAd(), "padim": lambda: M.Padim()}[args.model]()
    out = ROOT / "models" / args.cam / (name if args.model == "patchcore" else f"{name}_{args.model}")
    kw = {"default_root_dir": str(out / "work")}
    if args.model == "efficientad": kw["max_epochs"] = 20
    if args.xpu:
        from anomalib.engine.strategy.xpu_single import SingleXPUStrategy
        from anomalib.engine.accelerator.xpu import XPUAccelerator
        kw.update(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())
    engine = Engine(**kw)
    engine.fit(model=model, datamodule=dm)
    if bad: engine.test(model=model, datamodule=dm)
    if (out / "weights").exists(): shutil.rmtree(out / "weights")
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=out)
    import json
    (out / "holdout.json").write_text(json.dumps([f.name for f in holdout]))
    print("exported:", *sorted(out.rglob("model.xml")))
    print("next: on /arm section 8 choose judge = model, brick =", name, "and press 'calibrate'")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
