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


def fill_holes(mask):
    """Fill everything enclosed by the outer outline (the pen cross / depth holes inside the brick stay IN)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def inhand_mask(depth_mm, band_mm=70, max_mm=900, min_px=4000):
    """Mask of the nearest object cluster (gripper + held brick) from aligned depth (uint16 mm, 0 = no data).

    Finds the nearest substantial depth peak (closer than `max_mm`), keeps pixels within +-band_mm of it, takes
    the largest connected component, fills its holes and pads it slightly. Returns (mask uint8 0/255, info) or
    (None, reason) if depth is missing/unusable.
    """
    if depth_mm is None: return None, "no depth"
    d = depth_mm
    small = cv2.resize(d, (d.shape[1] // 4, d.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
    v = small[(small > 150) & (small < max_mm)]
    if v.size < min_px // 16: return None, "nothing within %d mm" % max_mm
    hist, edges = np.histogram(v, bins=np.arange(150, max_mm + 20, 20))
    thr = max(min_px // 16 // 4, int(hist.max() * 0.25))
    peak = int(np.argmax(hist >= thr))                       # NEAREST bin with a substantial amount of pixels
    z = float(np.median(v[(v >= edges[peak]) & (v < edges[peak] + 40)])) if v.size else 0
    m = ((small > 0) & (np.abs(small.astype(np.int32) - z) <= band_mm)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None, "no cluster"
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8) * 255
    m = fill_holes(m)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.resize(m, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m, {"z_mm": round(z), "area_px": int(cv2.countNonZero(m))}


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
    if depth_mm is None: return img, {"masked": False, "why": "no depth"}
    m, info = inhand_mask(depth_mm)
    if m is None: return img, {"masked": False, "why": info}
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
