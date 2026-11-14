"""Image preprocessing shared by data collection, dataset export and the live robot runtime.

RULE: training images and live images must go through the SAME function here, otherwise the models see
something different at run time than they learned.

Background movement (people walking past the fixed cameras) is handled in two ways:
  * ACT pick policy: uses only the TOP camera cropped to the table workspace (people stand outside the crop)
    and the WRIST camera (looks down at the plate). The FRONT camera mostly sees the room -> not a policy input.
  * Anomaly images of the held brick: the front RealSense depth keeps only the nearest object cluster
    (gripper + brick) and paints everything behind it flat grey (`inhand_mask`).
"""
import json
from pathlib import Path

import cv2
import numpy as np

INSPECT_SETTINGS = Path(__file__).resolve().parent / "inspect_settings.json"   # release-view crop per camera
ACT_W, ACT_H = 640, 480     # matches the Studio environment's camera shape (480x640x3)
GREY = 128


def fit_43(box, W=1920, H=1080):
    """Grow a box (x0,y0,x1,y1) to 4:3 around its centre, kept inside the image."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w * 3 > h * 4: h = w * 3 / 4
    else: w = h * 4 / 3
    w, h = min(w, W), min(h, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(round(min(max(cx - w / 2, 0), W - w))); y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return [x0, y0, x0 + int(round(w)), y0 + int(round(h))]


def default_act_roi(cam, W=1920, H=1080):
    return fit_43([0, 0, W, H], W, H)          # centre 4:3 crop


def act_frame(img, roi):
    """Full-res BGR frame -> 640x480 BGR policy input (crop to the 4:3 `roi`, then downscale)."""
    x0, y0, x1, y1 = roi
    return cv2.resize(img[y0:y1, x0:x1], (ACT_W, ACT_H), interpolation=cv2.INTER_AREA)


ACT_JPEG_Q = 93             # collect.py stores every recorded policy frame as JPEG at this quality


def act_jpeg(img, roi):
    """What collect.py writes per recorded frame: the policy crop as JPEG bytes."""
    return cv2.imencode(".jpg", act_frame(img, roi), [cv2.IMWRITE_JPEG_QUALITY, ACT_JPEG_Q])[1].tobytes()


def act_policy_image(img, roi, jpeg=True):
    """THE live ACT image input: exactly what export_act.py gives Studio for training, from a full-res BGR frame.
    Training path: act_jpeg() -> episodes/.../<cam>/NNNNNN.jpg -> cv2.imread -> BGR2RGB (export_act.py).
    jpeg=True repeats the JPEG round trip so even the compression artefacts match. Returns 480x640x3 uint8 RGB."""
    if jpeg:
        bgr = cv2.imdecode(np.frombuffer(act_jpeg(img, roi), np.uint8), cv2.IMREAD_COLOR)
    else:
        bgr = act_frame(img, roi)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# In-hand mask configuration, shared by EVERY caller (arm.capture training crops, sorter live photos, collect's live
# view), so training and run time always use the same rule. capture.py fills it from the station (configure_inhand):
INHAND = {"roi": None,          # search area, full-res x0,y0,x1,y1 (= the front detection area); None = image centre
          "fx": 1380.0,         # colour focal length in px at 1920 wide (the aligned depth lives in colour pixels)
          "expect_z_mm": None,  # distance of the held cluster measured by /depth/autotune at WP2 (None = no prior)
          "min_mm": 50,         # nearer than this is noise: the D415 can't measure it even with a disparity shift
          "max_obj_mm": 400}    # a HELD object is nearer than this; the room/table/robot column behind it is farther
_CFG = object()                 # "take this parameter from INHAND"
TUNING = False                  # capture.py sets this during /depth/autotune: photos taken meanwhile are not masked


def configure_inhand(**kw):
    """Set INHAND keys (unknown keys are ignored). Returns the current config."""
    INHAND.update({k: v for k, v in kw.items() if k in INHAND})
    return dict(INHAND)


def fill_holes(mask, depth=None, z_mm=None, band_mm=70, px_per_mm=None, max_fill_mm=8.0, stats=None):
    """Fill the holes of a mask (regions enclosed by it that don't reach the image border).

    Without depth: every hole is filled (the old behaviour: the pen cross / depth holes inside the brick stay IN).
    With depth (same shape as mask, mm) and the cluster distance z_mm, holes are filled only when they can't be
    background: a hole is left OPEN when
      * most of it has valid depth farther than z_mm + band_mm (background seen through the gap between the jaws
        and the brick), or
      * it has no usable depth and is wider than max_fill_mm (inscribed circle, needs px_per_mm): a pen stroke or a
        specular spot is thin, a gap enclosed by jaws + brick (background beyond the far limit = no depth) is not.
    `stats` (dict, optional) receives {"open_seen_px", "open_missing_px", "filled_px"}.
    """
    if depth is None or z_mm is None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
        return out
    out = mask.copy()
    inv = (mask == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()))
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    r_max = (max_fill_mm / 2.0) * px_per_mm if px_per_mm else np.inf
    seen = missing = filled = 0
    for i in range(1, n):
        if i in border: continue
        x, y, w, h, a = st[i]
        sl = (slice(y, y + h), slice(x, x + w)); hole = lab[sl] == i
        dv = depth[sl][hole].astype(np.int32)
        if (dv > z_mm + band_mm).mean() >= 0.5: seen += int(a); continue          # see-through: background
        if dt[sl][hole].max() > r_max:                                            # wide hole without depth
            missing += int(a)
            if stats is not None: stats["largest_missing_px"] = max(stats.get("largest_missing_px", 0), int(a))
            continue
        out[sl][hole] = 255; filled += int(a)
    if stats is not None: stats.update({"open_seen_px": seen, "open_missing_px": missing, "filled_px": filled})
    return out


def _search_region(shape, roi, scale):
    """Bool mask (small image) of where the held object must be: the ROI, or the centre of the image."""
    h, w = shape
    reg = np.zeros((h, w), bool)
    if roi:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in roi)
    else:                                    # no ROI configured: central 60 % x 70 % of the frame
        x0, y0, x1, y1 = int(w * 0.2), int(h * 0.15), int(w * 0.8), int(h * 0.85)
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1)); y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    reg[y0:y1, x0:x1] = True
    return reg


def inhand_mask(depth_mm, band_mm=70, max_mm=900, min_px=4000, min_mm=_CFG, roi=_CFG, expect_z_mm=_CFG, fx=_CFG,
                min_obj_mm=30, min_density=0.4, max_missing=0.25, z_tol_mm=None, max_obj_mm=_CFG,
                max_hole_mm=25, max_obj_size_mm=300, max_outside=0.75):
    """Mask of the held object (gripper + brick) from aligned depth (uint16 mm, 0 = no data).

    Returns (mask uint8 0/255 at the depth's resolution, info dict) or (None, reason str). It refuses (None) rather
    than return a wrong object: the sorter treats "no depth mask" as uncertain (never good), collection saves no crop.

      1. Depth peaks between min_mm and max_mm, counted only inside the search area (`roi`, else the image centre),
         nearest first. A substantial peak has >= 25 % of the largest 20 mm bin. With `expect_z_mm` (the auto-tune's
         measurement at WP2) only peaks within z_tol_mm (default max(60, 40 %)) of it are considered.
      2. Pixels within +-band_mm of the peak, closed, connected components; the one with most pixels in the search
         area is the candidate. It must be
           * substantial: >= min_px pixels AND at least the projected size of a min_obj_mm square at that distance
             ((fx * min_obj_mm / z)^2: a held brick + jaws is bigger than 30 x 30 mm), at least half of that inside
             the search area, and its centre inside the search area;
           * solid: >= min_density of it has real depth in the band (closed-up sparse speckle is not an object).
           * near: its distance z <= max_obj_mm (a held object is close to the camera; a wall/table/robot column
             behind it is not, even when it is the nearest thing with depth because the brick is inside Min-Z);
           * not the background: at most the projected size of a max_obj_size_mm square ((fx*300/z)^2), and at most
             max_outside of its outline outside the search area.
         A candidate that fails moves on to the next peak (a small near speckle blob is skipped); none left -> None.
      3. Holes are filled (pen cross, small depth holes stay IN), except background seen through a gap (jaws + brick)
         and wide holes without depth (fill_holes). -> None if any single no-depth hole is bigger than a
         max_hole_mm square at that distance (a brick-sized object without depth sits in front of the cluster: the
         brick is inside the camera's minimum range), or all of them together exceed max_missing of the outline.
    Parameters left at their default come from INHAND (configure_inhand); pass None to switch roi/expect_z_mm off.
    """
    if depth_mm is None: return None, "no depth"
    d = depth_mm
    H, W = d.shape[:2]
    s = 4
    if W // s < 4 or H // s < 4: return None, f"depth image too small ({W}x{H})"
    min_mm = INHAND["min_mm"] if min_mm is _CFG else (min_mm or 0)
    zmax = INHAND.get("max_obj_mm") if max_obj_mm is _CFG else max_obj_mm
    roi = INHAND["roi"] if roi is _CFG else roi
    expect = INHAND["expect_z_mm"] if expect_z_mm is _CFG else expect_z_mm
    fxi = float(INHAND["fx"] if fx is _CFG else fx) * W / 1920.0          # focal length in pixels of THIS image
    small = cv2.resize(d, (W // s, H // s), interpolation=cv2.INTER_NEAREST)
    reg = _search_region(small.shape, roi, W / 1920.0 / s)
    si = small.astype(np.int32)
    valid = (small > min_mm) & (small < max_mm)
    v = small[valid & reg]
    min_small = max(1, min_px // (s * s))
    if v.size < min_small:
        return None, f"no depth within {min_mm}-{max_mm} mm in the search area ({v.size * s * s} px)"
    edges = np.arange(min_mm, max_mm + 20, 20)
    hist, edges = np.histogram(v, bins=edges)
    thr = max(1, min_small // 4, int(hist.max() * 0.25))
    tol = z_tol_mm if z_tol_mm is not None else (max(60.0, 0.4 * expect) if expect else None)
    reasons, tried = [], []
    for b in np.nonzero(hist >= thr)[0]:
        sel = v[(v >= edges[b]) & (v < edges[b] + 40)]
        if sel.size == 0: continue
        z = float(np.median(sel))
        if any(abs(z - t) <= band_mm for t in tried): continue                 # same cluster as one already tried
        if expect and abs(z - expect) > tol:
            reasons.append(f"cluster at {z:.0f} mm, expected {expect:.0f} +- {tol:.0f} mm (re-run the depth auto-tune?)")
            continue
        tried.append(z)
        if len(tried) > 4: break
        if zmax and z > zmax:
            reasons.append(f"nearest object at {z:.0f} mm is farther than a held one can be ({zmax:.0f} mm): background? "
                           "brick inside the depth camera's minimum range?")
            break                                                               # everything after is farther still
        band = valid & (np.abs(si - z) <= band_mm)
        m = cv2.morphologyEx(band.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, lab, st, cent = cv2.connectedComponentsWithStats(m)
        if n <= 1: reasons.append(f"no cluster at {z:.0f} mm"); continue
        inreg = np.bincount(lab[reg], minlength=n); inreg[0] = 0
        k = int(np.argmax(inreg))
        if inreg[k] == 0: reasons.append(f"no cluster at {z:.0f} mm in the search area"); continue
        need = int(max(min_px, (fxi * min_obj_mm / max(z, 1.0)) ** 2))
        cx, cy = cent[k]
        if inreg[k] * s * s < need / 2 or not reg[min(int(cy), reg.shape[0] - 1), min(int(cx), reg.shape[1] - 1)]:
            reasons.append(f"cluster at {z:.0f} mm is not in the search area"); continue
        comp = lab == k
        dens = float(band[comp].mean())
        if dens < min_density:
            reasons.append(f"cluster at {z:.0f} mm too sparse ({dens:.0%} has depth)"); continue
        hs = {}
        mk = fill_holes(comp.astype(np.uint8) * 255, small, z, band_mm, fxi / max(z, 1.0) / s, stats=hs)
        tot = int(cv2.countNonZero(mk)) + hs["open_missing_px"] + hs["open_seen_px"]     # outline area (small px)
        if tot * s * s < need:
            reasons.append(f"cluster at {z:.0f} mm too small ({tot * s * s} px < {need} px)"); continue
        big = (fxi * max_obj_size_mm / max(z, 1.0)) ** 2
        out_frac = 1.0 - float(((mk > 0) & reg).sum()) / max(1, int((mk > 0).sum()))
        if tot * s * s > big or out_frac > max_outside:
            reasons.append(f"cluster at {z:.0f} mm is background-sized ({tot * s * s} px, {out_frac:.0%} outside the search area)")
            continue
        hole_lim = (fxi * max_hole_mm / max(z, 1.0)) ** 2
        if hs.get("largest_missing_px", 0) * s * s > hole_lim:
            return None, (f"object at {z:.0f} mm has a {hs['largest_missing_px'] * s * s} px hole without depth (more than "
                          f"{max_hole_mm}x{max_hole_mm} mm): something in front of it is inside the depth camera's minimum "
                          "range (run the depth auto-tune)")
        if hs["open_missing_px"] > max_missing * tot:
            return None, (f"object at {z:.0f} mm: {hs['open_missing_px'] / tot:.0%} of it has no depth "
                          "(too close for the depth window? run the depth auto-tune)")
        mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
        return mk, {"z_mm": round(z), "area_px": int(cv2.countNonZero(mk)), "band_mm": band_mm, "density": round(dens, 2),
                    "search": "roi" if roi else "centre", "holes_open_px": (hs["open_seen_px"] + hs["open_missing_px"]) * s * s}
    if not reasons: reasons.append("no substantial depth peak in the search area")
    return None, "; ".join(reasons[:3])


def apply_mask(img, mask):
    """Everything outside the mask -> flat grey (so background changes can't look like anomalies)."""
    out = np.full_like(img, GREY)
    out[mask > 0] = img[mask > 0]
    return out


def inspect_crop(img, depth_mm=None, pad=0.08):
    """THE anomaly-model input for a held brick (training crops AND live inspection use this, see RULE above).

    Depth-masked (background -> flat grey), then a square crop around the nearest object cluster (gripper + brick),
    padded with grey where the square leaves the image. Without usable depth the full masked/unmasked frame is
    returned and info["masked"] is False (the caller should treat that photo with suspicion).
    Returns (crop BGR, info dict).
    """
    cfg = {"inhand_cfg": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in INHAND.items()}}
    if TUNING: return img, {"masked": False, "why": "depth auto-tune running", **cfg}
    if depth_mm is None: return img, {"masked": False, "why": "no depth", **cfg}
    m, info = inhand_mask(depth_mm)
    if m is None: return img, {"masked": False, "why": info, **cfg}
    info = {**info, **cfg}
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = apply_mask(img, m)
    b = mask_bbox(m, pad)
    if b is None: return out, {"masked": False, "why": "empty mask"}
    x0, y0, x1, y1 = b
    side = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0, sy0 = cx - side // 2, cy - side // 2
    H, W = img.shape[:2]
    crop = np.full((side, side, 3), GREY, np.uint8)
    ax0, ay0, ax1, ay1 = max(0, sx0), max(0, sy0), min(W, sx0 + side), min(H, sy0 + side)
    crop[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0] = out[ay0:ay1, ax0:ax1]
    return crop, {"masked": True, "bbox": [int(sx0), int(sy0), int(sx0 + side), int(sy0 + side)], **info}


def release_roi(cam):
    """Crop (x0,y0,x1,y1 as fractions of the frame) for photos of the RELEASED brick on the tape spot, per camera.
    Stored in inspect_settings.json so collection, training crops and the live sorter always agree."""
    try: r = json.loads(INSPECT_SETTINGS.read_text())["release_roi"][cam]
    except Exception: r = [0.0, 0.0, 1.0, 1.0]
    return [float(min(1, max(0, v))) for v in r]


def set_release_roi(cam, roi):
    try: d = json.loads(INSPECT_SETTINGS.read_text())
    except Exception: d = {}
    x0, y0, x1, y1 = [float(min(1, max(0, v))) for v in roi]
    if x1 - x0 < 0.05 or y1 - y0 < 0.05: raise ValueError("crop too small")
    d.setdefault("release_roi", {})[cam] = [x0, y0, x1, y1]
    INSPECT_SETTINGS.write_text(json.dumps(d, indent=1))


def release_crop(img, cam):
    """THE anomaly-model input for the brick lying released on the tape spot (seen from above, e.g. wrist camera).
    The brick is always at the same place there, so a fixed crop is enough (no depth needed)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = release_roi(cam)
    b = [int(x0 * W), int(y0 * H), max(int(x0 * W) + 8, int(x1 * W)), max(int(y0 * H) + 8, int(y1 * H))]
    return img[b[1]:b[3], b[0]:b[2]].copy(), {"masked": True, "kind": "release", "bbox": b}


def mask_bbox(mask, pad=0.08):
    ys, xs = np.nonzero(mask)
    if xs.size == 0: return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    p = int(max(x1 - x0, y1 - y0) * pad)
    H, W = mask.shape
    return [max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)]
