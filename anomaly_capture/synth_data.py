"""SYNTHETIC training_images/ tree in exactly the station's on-disk format, to test the anomaly pipeline without the station.

    .venv/bin/python anomaly_capture/synth_data.py --out /tmp/synth/training_images            # 3 bricks, front + wrist
    .venv/bin/python anomaly_capture/synth_data.py --out X --bricks red --good 20 --defect 8 --no-raw --seed 1

Why: train_anomaly.py / eval_anomaly.py / sorter.calibrate() need real crops that only exist on the station. This
renders Duplo-like bricks (colour, studs, ridge roof, lighting / pose / grip jitter, sensor noise, the gripper fingers)
and, on defect bricks, pen-drawn crosses of controlled size, so the whole chain can be run end-to-end tonight and
"smallest detectable cross" trends estimated before real defect data exists. NOT a substitute for real numbers.

Format (= arm.capture() as called by collect.phase2(), see SO101_SETUP_STATUS decisions 12, 16, 20):
  <out>/<cam>/inhand/<label>/<group>_<brick>_<view>.png          full-res frame (1920x1080)        [--no-raw: skipped]
                                       ..._<view>_depth.png        aligned depth, uint16 mm (front)  [--no-raw: skipped]
                                       ..._<view>_masked.jpg       prep.apply_mask(frame, mask)      [--no-raw: skipped]
                                       ..._<view>_crop.png         THE model input
  <out>/meta.jsonl   one line per photo, the same keys arm.capture writes (file, stage, cam, group, label, brick, view,
                     cycle, run, routine, joints, motion, frames_averaged, time, depth, cam_settings)
  front = brick held at WP2 (crop made by the REAL prep.inspect_crop() from a synthetic frame + depth map),
  wrist = released brick on the tape spot seen from above (view tape_top, stage release, fixed crop).
  One episode = one (run, cycle) = one handling of the brick: every view of it shares the grip / pose.
Side files (evaluation only, ignored by every loader):
  <out>/synth_truth.jsonl  per crop: episode, face shown, cross face, visible, cross_mm, thickness, ink, contrast,
                           visible_frac (share of the cross not hidden by a finger / off the face), hard-case tags
  <out>/synth_info.json    seed, assumed mm/px per camera, counts; its presence marks the data as SYNTHETIC
Deterministic for a given --seed. Assumed scales (no station measurement yet): front 0.25 mm/px (D415 at ~35 cm),
wrist 0.20 mm/px (close-up from above).
"""
import argparse, json, math, sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep  # noqa: E402  (the real crop function: synthetic frames go through the same code as the station's)

W, H = 1920, 1080
STUD_PITCH, STUD_D, STUD_H, BRICK_H = 16.0, 9.6, 4.5, 19.2                # Duplo mm
BRICKS = {"pink": {"bgr": (125, 135, 232), "studs": (2, 2), "roof": False},   # salmon 2x2 (replaced the red roof 2026-11-14)
          "red": {"bgr": (38, 40, 190), "studs": (2, 4), "roof": True},      # 2x4 ridge roof (tent prism, no studs)
          "green": {"bgr": (55, 140, 40), "studs": (2, 3), "roof": False},
          "blue": {"bgr": (165, 85, 25), "studs": (2, 2), "roof": False}}
INKS = {"black": (28, 28, 30), "blue": (150, 70, 35), "red": (45, 40, 165)}
MM_PER_PX = {"front": 0.25, "wrist": 0.20}
FRONT_VIEWS = {"wp2-1": "side_a", "wp2-2": "side_b", "wp2-3": "bottom"}     # pose -> face it shows (sort_settings faces)
RELEASE_ROI = (0.36, 0.28, 0.66, 0.72)                                        # assumed prep.release_roi("wrist")
LADDER = (1.5, 2.0, 3.0, 5.0, 8.0, 10.0)
CAM_SETTINGS = {"front": {"exposure": 313, "gain": 96.0, "white_balance": 4750}, "wrist": {}}
AVG = {"front": 16, "wrist": 8}
SS = 2                                                                        # supersampling of the local canvas


def face_dims(brick, face):
    """(width, height) in mm of a face; side faces are the long sides, bottom/top the footprint."""
    a, b = BRICKS[brick]["studs"]
    long_, short = b * STUD_PITCH - 0.2, a * STUD_PITCH - 0.2
    return (long_, BRICK_H) if face.startswith("side") else (long_, short)


def shade(img, rng, strength):
    """Multiply by a smooth directional lighting gradient + vignette (per-photo jitter)."""
    h, w = img.shape[:2]
    a = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = ((xx / w - .5) * np.cos(a) + (yy / h - .5) * np.sin(a))
    r2 = ((xx / w - .5) ** 2 + (yy / h - .5) ** 2)
    return img * (1 + strength * g - 0.25 * strength * r2)[..., None]


def draw_cross(ink_mask, cx, cy, L, t, ang, rng):
    """Two slightly wobbly pen strokes (anti-aliased) into a float mask (0..1)."""
    for k in (0, 1):
        a = ang + k * np.pi / 2 + rng.normal(0, 0.06)
        dx, dy = np.cos(a) * L / 2, np.sin(a) * L / 2
        n = 6; pts = []
        for i in range(n + 1):
            f = -1 + 2 * i / n
            w = rng.normal(0, max(0.3, t * 0.25))
            pts.append((cx + f * dx - np.sin(a) * w, cy + f * dy + np.cos(a) * w))
        m = np.zeros_like(ink_mask, np.uint8)
        cv2.polylines(m, [np.round(np.array(pts) * 16).astype(np.int32)], False, 255, max(1, int(round(t))), cv2.LINE_AA, 4)
        ink_mask[:] = np.maximum(ink_mask, m.astype(np.float32) / 255)


def render_face(brick, face, k, rng, defect, light):
    """Local canvas of one face (+ studs on top for a side view) at k px/mm.
    -> rgb float, object mask, ink-visible mask, ink-total px (cross drawn without clipping)."""
    spec = BRICKS[brick]; fw, fh = face_dims(brick, face)
    top = 0 if (spec["roof"] or not face.startswith("side")) else STUD_H
    m = 2.0
    cw, ch = int((fw + 2 * m) * k), int((fh + top + 2 * m) * k)
    col = np.array(spec["bgr"], np.float32) * rng.uniform(0.95, 1.05, 3)
    rgb = np.zeros((ch, cw, 3), np.float32); obj = np.zeros((ch, cw), np.uint8)
    x0, y0, x1, y1 = int(m * k), int((m + top) * k), int((m + fw) * k), int((m + top + fh) * k)
    cv2.rectangle(obj, (x0, y0), (x1, y1), 255, -1)
    rgb[obj > 0] = col
    if face.startswith("side") and spec["roof"]:            # ridge roof side: the slope catches more light near the top
        band = np.linspace(1.18, 0.92, y1 - y0, dtype=np.float32)[:, None, None]
        rgb[y0:y1, x0:x1] *= band
    if face.startswith("side") and not spec["roof"]:        # studs seen from the side = bumps on the top edge
        n = spec["studs"][1]
        for i in range(n):
            sx = x0 + int(((i + 0.5) * STUD_PITCH - STUD_D / 2) * k)
            cv2.rectangle(obj, (sx, int(m * k)), (sx + int(STUD_D * k), y0), 255, -1)
            rgb[int(m * k):y0, sx:sx + int(STUD_D * k)] = col * 1.08
    if face == "bottom":                                    # open bottom: dark cavity + tubes
        w_ = int(1.6 * k)
        cv2.rectangle(rgb, (x0 + w_, y0 + w_), (x1 - w_, y1 - w_), tuple(float(c) for c in col * 0.45), -1)
        for i in range(max(1, spec["studs"][1] - 1)):
            cx = x0 + int((i + 1) * STUD_PITCH * k)
            cv2.circle(rgb, (cx, (y0 + y1) // 2), int(3.3 * k), tuple(float(c) for c in col * 0.85), -1, cv2.LINE_AA)
    if face == "top":
        if spec["roof"]:                                    # ridge line along the middle
            yc = (y0 + y1) // 2
            rgb[y0:yc, x0:x1] *= 1.12; rgb[yc:y1, x0:x1] *= 0.9
            cv2.line(rgb, (x0, yc), (x1, yc), tuple(float(c) for c in col * 0.7), max(1, int(0.4 * k)), cv2.LINE_AA)
        else:
            for i in range(spec["studs"][1]):
                for j in range(spec["studs"][0]):
                    c = (x0 + int((i + 0.5) * STUD_PITCH * k), y0 + int((j + 0.5) * STUD_PITCH * k))
                    cv2.circle(rgb, (c[0] + int(0.6 * k), c[1] + int(0.6 * k)), int(STUD_D / 2 * k), tuple(float(v) for v in col * 0.6), -1, cv2.LINE_AA)
                    cv2.circle(rgb, c, int(STUD_D / 2 * k), tuple(float(v) for v in col * 1.12), -1, cv2.LINE_AA)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), tuple(float(c) for c in col * 0.7), max(1, int(0.3 * k)), cv2.LINE_AA)
    ink_vis = np.zeros((ch, cw), np.float32); total = 0.0; info = None
    if defect:
        L, t = defect["cross_mm"] * k, max(1.0, defect["thickness_mm"] * k)
        # position on the face in face fractions (edge cases sit at the border and get clipped)
        cx, cy = x0 + defect["pos"][0] * (x1 - x0), y0 + defect["pos"][1] * (y1 - y0)
        big = np.zeros((ch * 3, cw * 3), np.float32)
        draw_cross(big, cx + cw, cy + ch, L, t, defect["angle"], np.random.default_rng(defect["stroke_seed"]))
        total = float(big.sum()); ink = big[ch:2 * ch, cw:2 * cw] * (obj > 0)   # clipped to the face
        ink_vis = ink
        inkc = np.array(INKS[defect["ink"]], np.float32)
        a = (ink * defect["alpha"])[..., None]
        rgb = rgb * (1 - a) + inkc * a
        info = {"cx": cx, "cy": cy}
    rgb = shade(rgb, rng, light) * (obj > 0)[..., None]
    return rgb, obj, ink_vis, total, (x0, y0, x1, y1), info


def place(frame, depth, rgb, obj, extra_masks, cx, cy, ang, z_mm):
    """Rotate the local canvas (rendered at SS x) by ang and paste it centred at (cx, cy) of the full frame."""
    ch, cw = obj.shape
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0 / SS)
    M[0, 2] += cx - cw / 2; M[1, 2] += cy - ch / 2
    warp = lambda a, fl=cv2.INTER_LINEAR: cv2.warpAffine(a, M, (W, H), flags=fl, borderValue=0)
    o = warp(obj.astype(np.float32) / 255)
    frame[:] = frame * (1 - o[..., None]) + warp(rgb) * (o[..., None] > 0)  # rgb already premultiplied by the mask
    if depth is not None: depth[o > 0.5] = np.uint16(z_mm)
    return o, [warp(m) for m in extra_masks]


def background(rng, kind):
    small = rng.uniform(0, 1, (9, 16, 3)).astype(np.float32)
    if kind == "front":       # the room behind the held brick (depth mask greys it out anyway)
        bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC) * 45 + np.array([75, 85, 95], np.float32)
    else:                     # orange tape on the table seen from above
        bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC) * 18 + np.array([45, 125, 215], np.float32)
    return bg


def finish(frame, rng, roi, sigma):
    """Sensor noise (after frame averaging) + slight lens blur, only noisy inside roi (keeps the raw PNGs small)."""
    x0, y0, x1, y1 = roi
    sub = cv2.GaussianBlur(frame[y0:y1, x0:x1], (0, 0), 0.6)
    sub += rng.normal(0, sigma, sub.shape).astype(np.float32)
    frame[y0:y1, x0:x1] = sub
    return np.clip(frame, 0, 255).astype(np.uint8)


def front_photo(brick, face, ep, rng, defect):
    k = SS / MM_PER_PX["front"] * ep["scale"]
    rgb, obj, ink, total, (fx0, fy0, fx1, fy1), _ = render_face(brick, face, k, rng, defect if defect and defect["face"] == face else None, 0.35)
    ch, cw = obj.shape
    # gripper: two dark fingers clamping the ends (they overlap the face by ep["finger_mm"]), reaching up out of view.
    # No closed jaw outline: prep.fill_holes() would fill a background hole enclosed by jaws + brick (check on station)
    fo = int(ep["finger_mm"] * k); fwid = int(8 * k)
    pad = int(fwid + 2 * k); top_h = int(18 * k)
    grow = lambda a: cv2.copyMakeBorder(a, top_h, int(2 * k), pad, pad, cv2.BORDER_CONSTANT, value=0)
    rgb, obj, ink = grow(rgb), grow(obj), grow(ink); fing = np.zeros_like(obj)
    fy = fy0 + top_h + int(ep["grip_dy"] * k)
    for xa, xb in ((fx0 + pad - fwid + fo, fx0 + pad + fo), (fx1 + pad - fo, fx1 + pad + fwid - fo)):
        fing_ = np.zeros_like(obj); cv2.rectangle(fing_, (xa, fy - int(6 * k)), (xb, fy + int(0.8 * (fy1 - fy0))), 255, -1)
        cv2.rectangle(fing_, (xa, 0), (xb, fy), 255, -1)
        fing = np.maximum(fing, fing_)
    fcol = np.array([58, 60, 62], np.float32) * rng.uniform(0.95, 1.05)
    frgb = shade(np.ones((*fing.shape, 3), np.float32) * fcol, rng, 0.3)
    rgb = np.where((fing > 0)[..., None], frgb, rgb); ink = ink * (fing == 0)
    obj = np.maximum(obj, fing)
    frame = background(rng, "front"); depth = np.zeros((H, W), np.uint16)
    depth[::7, ::5] = 1600                                  # sparse far background
    cx, cy = 1178 + ep["dx"], 540 + ep["dy"]
    o, (inkw,) = place(frame, depth, rgb, obj, [ink], cx, cy, ep["angle"] + rng.normal(0, 0.6), ep["z_mm"])
    frame = finish(frame, rng, (700, 150, 1660, 950), rng.uniform(1.4, 2.6))
    crop, info = prep.inspect_crop(frame, depth)
    vis = float(inkw.sum()) * SS * SS / total if total else 0.0
    return frame, depth, crop, info, min(1.0, vis)


def wrist_photo(brick, ep, rng, defect):
    k = SS / MM_PER_PX["wrist"] * ep["scale"]
    rgb, obj, ink, total, _, _ = render_face(brick, "top", k, rng, defect if defect and defect["face"] == "top" else None, 0.3)
    frame = background(rng, "wrist")
    # soft shadow of the brick on the tape
    sh = cv2.GaussianBlur(obj.astype(np.float32) / 255, (0, 0), 6 * SS)
    M0 = np.float32([[1, 0, 6 * SS], [0, 1, 8 * SS]]); sh = cv2.warpAffine(sh, M0, (obj.shape[1], obj.shape[0]))
    cx, cy = int(W * 0.51) + ep["dx"], int(H * 0.5) + ep["dy"]
    o_s, _ = place(frame.copy(), None, np.zeros_like(rgb), (sh * 255).astype(np.uint8), [], cx, cy, ep["yaw"], 0)
    frame *= (1 - 0.35 * o_s)[..., None]
    o, (inkw,) = place(frame, None, rgb, obj, [ink], cx, cy, ep["yaw"], 0)
    frame = finish(frame, rng, (int(W * RELEASE_ROI[0]), int(H * RELEASE_ROI[1]), int(W * RELEASE_ROI[2]), int(H * RELEASE_ROI[3])),
                   rng.uniform(1.8, 3.2))
    x0, y0, x1, y1 = int(RELEASE_ROI[0] * W), int(RELEASE_ROI[1] * H), int(RELEASE_ROI[2] * W), int(RELEASE_ROI[3] * H)
    crop = frame[y0:y1, x0:x1].copy()
    vis = float(inkw.sum()) * SS * SS / total if total else 0.0
    return frame, crop, {"masked": True, "kind": "release", "bbox": [x0, y0, x1, y1]}, min(1.0, vis)


def make_defect(rng, faces):
    """One cross per defect brick: face, size, thickness, ink, contrast, position; ~15 % each hard-case tag."""
    hard = [t for t, p in (("tiny", .12), ("low_contrast", .15), ("edge", .15), ("hidden", .12)) if rng.random() < p]
    size = float(rng.uniform(0.8, 1.5)) if "tiny" in hard else float(rng.choice(LADDER)) * float(rng.uniform(0.9, 1.1))
    ink = str(rng.choice(list(INKS)))
    alpha = float(rng.uniform(0.25, 0.4)) if "low_contrast" in hard else float(rng.uniform(0.7, 0.95))
    pos = [float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.3, 0.7))]
    if "edge" in hard: pos[int(rng.integers(2))] = float(rng.choice([0.02, 0.98]))
    if "hidden" in hard: pos[0] = float(rng.choice([0.01, 0.99])); pos[1] = 0.5
    return {"face": str(rng.choice(faces)), "cross_mm": round(size, 2), "thickness_mm": round(float(rng.uniform(0.35, 0.8)), 2),
            "ink": ink, "alpha": round(alpha, 2), "pos": [round(p, 3) for p in pos], "angle": float(rng.uniform(0, np.pi)),
            "stroke_seed": int(rng.integers(1 << 30)), "hard": hard}


def generate(out, bricks=("pink", "green", "blue"), n_good=30, n_defect=10, cams=("front", "wrist"),
             views=("wp2-1", "wp2-2"), seed=0, raw=True, session_size=25):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = datetime(2026, 11, 15, 9, 0, 0) + timedelta(minutes=int(rng.integers(0, 600)))
    meta = open(out / "meta.jsonl", "a"); truth = open(out / "synth_truth.jsonl", "a")
    faces_seen = [FRONT_VIEWS[v] for v in views if "front" in cams] + (["top"] if "wrist" in cams else [])
    counts = {}
    for brick in bricks:
        for label, n in (("good", n_good), ("defect", n_defect)):
            run = None
            for e in range(n):
                if e % session_size == 0:                   # a new collection session (collect.py: <time>_<brick>)
                    t += timedelta(minutes=7); run = t.strftime("%Y%m%d_%H%M%S") + f"_{brick}"; cyc0 = e
                cycle = e - cyc0 + 1
                ep = {"scale": float(rng.uniform(0.96, 1.04)), "angle": float(rng.normal(0, 2.5)), "dx": int(rng.integers(-25, 26)),
                      "dy": int(rng.integers(-20, 21)), "grip_dy": float(rng.uniform(-3, 3)), "finger_mm": float(rng.uniform(0.5, 3.0)),
                      "z_mm": int(rng.integers(340, 380)), "yaw": float(rng.normal(0, 5))}
                # a cross can also sit on a face this camera set never shows (an end face): "not visible" bricks
                dfx = make_defect(rng, faces_seen + ["end"]) if label == "defect" else None
                if dfx and rng.random() > 0.1 and dfx["face"] == "end": dfx["face"] = str(rng.choice(faces_seen))
                t += timedelta(seconds=int(rng.integers(14, 24)))
                shots = [("front", v) for v in views if "front" in cams] + ([("wrist", "tape_top")] if "wrist" in cams else [])
                for cam, view in shots:
                    t += timedelta(milliseconds=int(rng.integers(800, 3000)))
                    group = t.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    d = out / cam / "inhand" / label; d.mkdir(parents=True, exist_ok=True)
                    stem = f"{group}_{brick}_{view}"
                    joints = {j: int(2048 + rng.integers(-600, 600)) for j in
                              ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
                    if cam == "front":
                        face = FRONT_VIEWS[view]
                        frame, depth, crop, cinfo, vis = front_photo(brick, face, ep, rng, dfx)
                        dinfo = {k: cinfo[k] for k in ("z_mm", "area_px") if k in cinfo}
                        if raw:
                            cv2.imwrite(str(d / f"{stem}.png"), frame); cv2.imwrite(str(d / f"{stem}_depth.png"), depth)
                            mask, _ = prep.inhand_mask(depth)
                            cv2.imwrite(str(d / f"{stem}_masked.jpg"), prep.apply_mask(frame, mask), [cv2.IMWRITE_JPEG_QUALITY, 90])
                        stage = "inhand"
                    else:
                        face = "top"
                        frame, crop, cinfo, vis = wrist_photo(brick, ep, rng, dfx); dinfo = None; stage = "release"
                        if raw: cv2.imwrite(str(d / f"{stem}.png"), frame)
                    if cinfo.get("masked"): cv2.imwrite(str(d / f"{stem}_crop.png"), crop)
                    meta.write(json.dumps({"file": f"{cam}/inhand/{label}/{stem}.png", "stage": stage, "cam": cam, "group": group,
                                           "label": label, "brick": brick, "view": view, "cycle": cycle, "run": run,
                                           "routine": "wp2", "joints": joints, "motion": round(float(rng.uniform(0.1, 0.6)), 2),
                                           "frames_averaged": AVG[cam], "time": group, "depth": dinfo,
                                           "cam_settings": CAM_SETTINGS[cam]}) + "\n")
                    visible = bool(dfx and dfx["face"] == face and vis > 0.02)
                    truth.write(json.dumps({"file": f"{cam}/inhand/{label}/{stem}_crop.png", "episode": f"{run}/{cycle}",
                                            "label": label, "brick": brick, "cam": cam, "view": view, "face": face,
                                            "crop_px": list(crop.shape[:2]), "mm_per_px": MM_PER_PX[cam],
                                            **({"cross_face": dfx["face"], "visible": visible, "visible_frac": round(vis, 3),
                                                "cross_mm": dfx["cross_mm"], "cross_px": round(dfx["cross_mm"] / MM_PER_PX[cam], 1),
                                                "thickness_mm": dfx["thickness_mm"], "ink": dfx["ink"], "alpha": dfx["alpha"],
                                                "hard": dfx["hard"]} if dfx else {})}) + "\n")
                    counts[f"{cam}/{label}"] = counts.get(f"{cam}/{label}", 0) + 1
    meta.close(); truth.close()
    info = {"synthetic": True, "generator": "anomaly_capture/synth_data.py", "seed": seed, "bricks": list(bricks),
            "good_episodes": n_good, "defect_episodes": n_defect, "cams": list(cams), "views": list(views),
            "mm_per_px": MM_PER_PX, "release_roi": RELEASE_ROI, "raw_frames": raw, "photos": counts,
            "note": "SYNTHETIC rendered bricks - use only to test the pipeline and for size TRENDS, never as real results"}
    (out / "synth_info.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the training_images/ dir to create (appends to meta.jsonl if it exists)")
    ap.add_argument("--bricks", default="pink,green,blue")
    ap.add_argument("--good", type=int, default=30, help="good episodes per brick")
    ap.add_argument("--defect", type=int, default=10, help="defect episodes (bricks with one cross) per brick")
    ap.add_argument("--cams", default="front,wrist")
    ap.add_argument("--views", default="wp2-1,wp2-2", help=f"front WP2 poses, of {','.join(FRONT_VIEWS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-raw", action="store_true", help="only crops + meta (skip full frames, depth, masked: much smaller)")
    a = ap.parse_args()
    info = generate(a.out, a.bricks.split(","), a.good, a.defect, a.cams.split(","), a.views.split(","), a.seed, not a.no_raw)
    print(json.dumps(info["photos"]), "->", a.out, "(SYNTHETIC)")


if __name__ == "__main__":
    main()
