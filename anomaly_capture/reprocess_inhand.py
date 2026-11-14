"""Re-make the depth-masked images and anomaly crops of existing in-hand captures with the CURRENT mask rules.

Why: photos collected with an older prep.inhand_mask (e.g. before the empty-scene depth background existed) may have
masked the table instead of the held brick. The raw photo and its aligned depth are saved with every capture
(<stem>.png + <stem>_depth.png), so the masked view and the training crop can be made again at any time.

    python anomaly_capture/reprocess_inhand.py                 # dry run: what would change (nothing is written)
    python anomaly_capture/reprocess_inhand.py --apply         # do it
    python anomaly_capture/reprocess_inhand.py --apply --since 20261114_19   # only captures from that time on

What it does, per in-hand capture of the camera (meta.jsonl records with stage "inhand" and a _depth.png):
  * runs prep.inhand_mask / prep.inspect_crop (the functions the collector and the sorter use) with the station's
    current config from cam_settings.json: front detection area, held-object distance limit, auto-tune distance and
    the depth background (used only if it matches the depth config the photo was taken with: rs_depth.background_status);
  * writes a new <stem>_masked.jpg and <stem>_crop.png when the mask is accepted. When it is refused, NO crop is left
    (a crop of the wrong object must not reach training: train_anomaly.py reads every *_crop.png);
  * never touches <stem>.png or <stem>_depth.png. The previous _masked.jpg / _crop.png and meta.jsonl are MOVED to
    <data>/_reprocess_backup/<time>/ (same relative paths), outside the folders the trainer reads, so nothing is lost;
  * adds "reprocessed" (time, result, reason, config, background) to each record and updates "depth" /
    "depth_mask_why" to the new result.
Run it while nothing is collecting (meta.jsonl is rewritten).
"""
import argparse, json, shutil, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep, rs_depth  # noqa: E402

DEFAULT_FRONT_ROI = [840, 240, 1380, 780]          # = capture.DEFAULT_ROI["front"]


def load_station(settings_path, use_bg=True):
    """cam_settings.json -> (inhand config dict, background desc or None, background array or None)."""
    s = json.loads(Path(settings_path).read_text()) if Path(settings_path).exists() else {}
    raw = s.get("front_depth") if isinstance(s.get("front_depth"), dict) else {}
    cfg = rs_depth.resolve_config(raw)
    roi = (s.get("roi") or {}).get("front") or DEFAULT_FRONT_ROI
    inh = {"roi": list(roi), "expect_z_mm": cfg.get("expect_z_mm"), "max_obj_mm": cfg.get("max_obj_mm")}
    desc = raw.get("background") if use_bg and isinstance(raw.get("background"), dict) else None
    bg = None
    if desc:
        f = Path(settings_path).parent / (desc.get("file") or "front_depth_bg.png")
        bg = cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f.exists() else None
        if bg is None or bg.dtype != np.uint16: desc, bg = None, None
    return inh, desc, bg


def reprocess(data, settings, cam="front", apply=False, since="", use_bg=True, log=print):
    data = Path(data)
    mp = data / "meta.jsonl"
    if not mp.exists(): raise SystemExit(f"no {mp}")
    lines = mp.read_text().splitlines()
    inh, desc, bg = load_station(settings, use_bg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = data / "_reprocess_backup" / stamp
    stats, reasons, out_lines = Counter(), Counter(), []
    for line in lines:
        if not line.strip(): out_lines.append(line); continue
        try: rec = json.loads(line)
        except ValueError: out_lines.append(line); continue
        f = rec.get("file", "")
        if rec.get("cam") != cam or rec.get("stage") != "inhand" or not f.endswith(".png") or (since and str(rec.get("time", "")) < since):
            out_lines.append(line); continue
        raw = data / f; stem = raw.with_suffix("")
        dep_p = Path(str(stem) + "_depth.png")
        if not raw.exists() or not dep_p.exists():
            stats["skipped (no raw or depth file)"] += 1; out_lines.append(line); continue
        img = cv2.imread(str(raw)); dep = cv2.imread(str(dep_p), cv2.IMREAD_UNCHANGED)
        if img is None or dep is None or dep.dtype != np.uint16:
            stats["skipped (unreadable)"] += 1; out_lines.append(line); continue
        dcfg = rec.get("depth_cfg") or {}
        fx = float(dcfg.get("fx_color") or 1380.0)
        bg_ok, bg_why = rs_depth.background_status(desc, dcfg) if desc else (False, "no depth background")
        prep.configure_inhand(fx=fx, **inh)
        prep.set_background(bg if bg_ok else None, desc if bg_ok else None)
        mask, minfo = prep.inhand_mask(dep)
        mask = prep.complete_mask(img, mask)
        crop, cinfo = prep.inspect_crop(img, dep)
        ok = mask is not None and cinfo.get("masked")
        stats["masked" if ok else "refused"] += 1
        if not ok: reasons[str(cinfo.get("why") or minfo)[:90]] += 1
        old = {"depth": rec.get("depth"), "masked_before": Path(str(stem) + "_crop.png").exists()}
        if ok and not old["masked_before"]: stats["newly masked"] += 1
        if not ok and old["masked_before"]: stats["crop removed (mask now refused)"] += 1
        if apply:
            for suf in ("_masked.jpg", "_crop.png"):
                p = Path(str(stem) + suf)
                if p.exists():
                    dst = backup / p.relative_to(data); dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dst))
            if ok:
                cv2.imwrite(str(stem) + "_masked.jpg", prep.apply_mask(img, mask), [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(str(stem) + "_crop.png", crop)
            rec["depth"] = minfo if isinstance(minfo, dict) else None
            rec["depth_mask_why"] = None if ok else str(cinfo.get("why") or minfo)
            rec["inhand_cfg"] = cinfo.get("inhand_cfg")
            rec["reprocessed"] = {"time": stamp, "masked": bool(ok), "why": rec["depth_mask_why"],
                                  "z_mm": minfo.get("z_mm") if isinstance(minfo, dict) else None,
                                  "background": desc if bg_ok else None, "background_why": None if bg_ok else bg_why,
                                  "before": old, "backup": str(backup.relative_to(data))}
            out_lines.append(json.dumps(rec))
        else:
            out_lines.append(line)
    if apply and (stats["masked"] or stats["refused"]):
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mp, backup / "meta.jsonl")
        tmp = mp.with_name("meta.jsonl.tmp"); tmp.write_text("".join(l + "\n" for l in out_lines)); tmp.replace(mp)
    prep.set_background(None)
    log(("APPLIED" if apply else "DRY RUN (use --apply to write)") + f": {dict(stats)}")
    for r, n in reasons.most_common(8): log(f"  refused x{n}: {r}")
    if apply and (stats["masked"] or stats["refused"]): log(f"  previous crops/masks + meta.jsonl backed up in {backup}")
    if not desc: log("  note: no usable depth background in the settings: masks rely on the plane/size rules only")
    return {"stats": dict(stats), "reasons": dict(reasons), "backup": str(backup) if apply else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(HERE.parent / "training_images"))
    ap.add_argument("--settings", default=str(HERE / "cam_settings.json"))
    ap.add_argument("--cam", default="front")
    ap.add_argument("--since", default="", help="only captures whose time stamp is >= this (e.g. 20261114_19)")
    ap.add_argument("--no-bg", action="store_true", help="ignore the stored depth background")
    ap.add_argument("--apply", action="store_true", help="write the results (default: dry run)")
    a = ap.parse_args()
    reprocess(a.data, a.settings, a.cam, a.apply, a.since, not a.no_bg)


if __name__ == "__main__":
    main()
