"""Image preprocessing shared by data collection, dataset export and the live robot runtime.

RULE: training images and live images must go through the SAME function here, otherwise the models see
something different at run time than they learned.

Background movement (people walking past the fixed cameras) is handled in two ways:
  * ACT pick policy: uses only the TOP camera cropped to the table workspace (people stand outside the crop)
    and the WRIST camera (looks down at the plate). The FRONT camera mostly sees the room -> not a policy input.
  * Anomaly images of the held brick: the front RealSense depth keeps only the nearest object cluster
    (gripper + brick) and paints everything behind it flat grey (`inhand_mask`).
"""
import cv2
import numpy as np

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


def mask_bbox(mask, pad=0.08):
    ys, xs = np.nonzero(mask)
    if xs.size == 0: return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    p = int(max(x1 - x0, y1 - y0) * pad)
    H, W = mask.shape
    return [max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)]
