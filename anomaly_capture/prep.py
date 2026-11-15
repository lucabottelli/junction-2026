"""Image preprocessing shared by data collection, dataset export and the live robot runtime.

RULE: training images and live images must go through the SAME function here, otherwise the models see
something different at run time than they learned.

Background movement (people walking past the fixed cameras) is handled in two ways:
  * ACT pick policy: uses only the TOP camera cropped to the table workspace (people stand outside the crop)
    and the WRIST camera (looks down at the plate). The FRONT camera mostly sees the room -> not a policy input.
  * Anomaly images of the held brick: the front RealSense depth keeps only the nearest object cluster
    (gripper + brick) and paints everything behind it flat grey (`inhand_mask`).
"""
import json, warnings
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
          "max_obj_mm": 400,    # a HELD object is nearer than this; the room/table/robot column behind it is farther
          "brick": None,        # brick colour name: set -> the mask comes from the brick's COLOUR (brick_mask), not depth
          "bg": None}           # description of the empty-scene depth background in use (file, taken, config) or None
_CFG = object()                 # "take this parameter from INHAND"
TUNING = False                  # capture.py sets this during /depth/autotune: photos taken meanwhile are not masked
_BG = {"depth": None}           # the empty-scene depth background itself (uint16 mm, aligned to colour), see set_background
BG_MARGIN_MM = 15               # a pixel is foreground when it is nearer than the background by this + 3 % of its depth


def configure_inhand(**kw):
    """Set INHAND keys (unknown keys are ignored). Returns the current config."""
    INHAND.update({k: v for k, v in kw.items() if k in INHAND})
    return dict(INHAND)


def set_background(depth_mm, desc=None):
    """Use this empty-scene depth (uint16 mm, same alignment as the live depth; None = no background) for every
    in-hand mask. desc (JSON-safe dict: file, taken, resolution, shifts) is recorded with each photo as provenance."""
    _BG["depth"] = None if depth_mm is None else np.asarray(depth_mm, np.uint16)
    INHAND["bg"] = (dict(desc) if desc else {"source": "in memory"}) if depth_mm is not None else None


MASK_SCALE = 4                  # inhand_mask works on depth downscaled by this (nearest neighbour)


def small_depth(depth_mm):
    """The depth image inhand_mask actually looks at: every MASK_SCALE-th pixel (nearest neighbour). A background
    stored at this size lines up pixel for pixel with it."""
    H, W = depth_mm.shape[:2]
    return cv2.resize(depth_mm, (W // MASK_SCALE, H // MASK_SCALE), interpolation=cv2.INTER_NEAREST)


def median_depth(frames, min_valid=0.5):
    """Per-pixel median of the VALID values of several depth frames (0 = no data); pixels valid in fewer than
    min_valid of the frames -> 0. Used for the empty-scene background."""
    st = np.stack([np.asarray(f, np.float32) for f in frames])
    st[st == 0] = np.nan
    ok = np.isfinite(st).mean(0) >= min_valid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)            # all-NaN pixels: masked out by `ok` anyway
        med = np.nanmedian(st, 0)
    return np.where(ok, np.nan_to_num(med), 0).round().astype(np.uint16)


def merge_backgrounds(layers):
    """Background depths taken under several depth windows (disparity shifts) -> one: each pixel from the first
    layer where it has depth (they measure the same surface; a near window sees what a far one can't and back)."""
    out = np.zeros_like(layers[0])
    for L in layers:
        out = np.where(out == 0, L, out)
    return out


def foreground(depth_mm, bg_mm, margin_mm=BG_MARGIN_MM):
    """Bool: pixels with depth that are clearly NEARER than the empty-scene background (or where the background
    had no depth). Both arrays in mm, same shape."""
    d = depth_mm.astype(np.int32); b = bg_mm.astype(np.int32)
    return (d > 0) & ((b == 0) | (d < b - (margin_mm + 0.03 * b)))


def plane_fit(depth_small, comp, fx_small, cx, cy, max_pts=4000):
    """Fit a plane to the 3D points of a cluster (small-image pixels `comp`). -> (inlier fraction, in-plane spread
    of the inliers: two standard deviations in mm, residual tolerance mm, tilt of the plane from facing the camera in
    degrees). A table top is one large plane seen at a steep angle; gripper + brick are several surfaces at
    different depths, or a small face."""
    ys, xs = np.nonzero(comp)
    z = depth_small[ys, xs].astype(np.float64)
    ok = z > 0; ys, xs, z = ys[ok], xs[ok], z[ok]
    if z.size < 30: return 0.0, (0.0, 0.0), 0.0, 0.0
    if z.size > max_pts:
        idx = np.random.default_rng(0).choice(z.size, max_pts, replace=False); ys, xs, z = ys[idx], xs[idx], z[idx]
    P = np.stack([(xs - cx) * z / fx_small, (ys - cy) * z / fx_small, z], 1)
    tol = max(4.0, 0.02 * float(np.median(z)))
    inl = np.ones(len(P), bool)
    for _ in range(3):                                        # least squares, then refit on the inliers
        c = P[inl].mean(0); _, _, vt = np.linalg.svd(P[inl] - c, full_matrices=False)
        res = np.abs((P - c) @ vt[2])
        new = res <= tol
        if new.sum() < 30: break
        inl = new
    c = P[inl].mean(0); _, sv, vt = np.linalg.svd(P[inl] - c, full_matrices=False)
    spread = (float(sv[0] / np.sqrt(inl.sum())), float(sv[1] / np.sqrt(inl.sum())))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(vt[2][2])))))
    return float(inl.mean()), spread, tol, tilt


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
                min_obj_mm=18, min_density=0.4, max_missing=0.25, z_tol_mm=None, max_obj_mm=_CFG,
                max_hole_mm=25, max_obj_size_mm=300, max_outside=0.75, bg=_CFG, bg_margin_mm=BG_MARGIN_MM,
                plane_inliers=0.85, plane_spread_mm=25, plane_tilt_deg=35):
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
    0. With an empty-scene depth background (`bg`, default: set_background()), only pixels clearly nearer than
       the background count (foreground()): the table, plates and the parked arm can never be the cluster.
       Without one, a cluster that is one big plane seen at a steep angle (>= plane_inliers of it within a few mm
       of a plane spanning >= plane_spread_mm standard deviation in both directions, tilted >= plane_tilt_deg from
       facing the camera) is refused: that is the table top in front of the camera, not a held brick.
    Parameters left at their default come from INHAND (configure_inhand); pass None to switch roi/expect_z_mm/bg off.
    """
    if depth_mm is None: return None, "no depth"
    d = depth_mm
    H, W = d.shape[:2]
    s = MASK_SCALE
    if W // s < 4 or H // s < 4: return None, f"depth image too small ({W}x{H})"
    min_mm = INHAND["min_mm"] if min_mm is _CFG else (min_mm or 0)
    zmax = INHAND.get("max_obj_mm") if max_obj_mm is _CFG else max_obj_mm
    roi = INHAND["roi"] if roi is _CFG else roi
    expect = INHAND["expect_z_mm"] if expect_z_mm is _CFG else expect_z_mm
    fxi = float(INHAND["fx"] if fx is _CFG else fx) * W / 1920.0          # focal length in pixels of THIS image
    small = small_depth(d)
    reg = _search_region(small.shape, roi, W / 1920.0 / s)
    si = small.astype(np.int32)
    valid = (small > min_mm) & (small < max_mm)
    bgd = _BG["depth"] if bg is _CFG else bg
    use_bg, bg_note = False, None
    if bgd is not None:                      # stored at the small size (small_depth) or at the depth's own size
        bsmall = bgd if bgd.shape[:2] == small.shape[:2] else small_depth(bgd) if bgd.shape[:2] == d.shape[:2] else None
        if bsmall is None: bg_note = f"background {bgd.shape[1]}x{bgd.shape[0]} doesn't fit depth {W}x{H}: ignored"
        else: valid &= foreground(small, bsmall, bg_margin_mm); use_bg = True
    v = small[valid & reg]
    min_small = max(1, min_px // (s * s))
    if v.size < min_small:
        return None, (f"no depth within {min_mm}-{max_mm} mm in the search area ({v.size * s * s} px)"
                      + (" in front of the depth background" if use_bg else "") + (f"; {bg_note}" if bg_note else ""))
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
        if not use_bg:
            frac, spread, ptol, tilt = plane_fit(small, comp, fxi / s, (W / s) / 2.0, (H / s) / 2.0)
            if frac >= plane_inliers and min(spread) >= plane_spread_mm and tilt >= plane_tilt_deg:
                reasons.append(f"cluster at {z:.0f} mm is a flat surface ({frac:.0%} within {ptol:.0f} mm of a plane "
                               f"tilted {tilt:.0f} deg, {2 * spread[0]:.0f} x {2 * spread[1]:.0f} mm): the table? "
                               "store a depth background")
                continue
        mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
        return mk, {"z_mm": round(z), "area_px": int(cv2.countNonZero(mk)), "band_mm": band_mm, "density": round(dens, 2),
                    "search": "roi" if roi else "centre", "holes_open_px": (hs["open_seen_px"] + hs["open_missing_px"]) * s * s,
                    "bg": use_bg, **({"bg_note": bg_note} if bg_note else {})}
    if not reasons: reasons.append("no substantial depth peak in the search area" + (" in front of the background" if use_bg else ""))
    if bg_note: reasons.append(bg_note)
    return None, "; ".join(reasons[:3])


def complete_mask(img, mask, s=4, sat_min=185, val_min=80, hue_tol=9, grow_px=120, min_peak=0.15):
    """Fill what depth misses on the held object, by colour. At ~10 cm glossy Duplo returns depth on only ~half of
    its surface, so the depth mask has holes and missing edges. The saturated hues found INSIDE the depth mask (the
    brick, the jaw pad) are grown into same-hue pixels within grow_px of it; only components touching the depth mask
    are kept, then holes are filled (a pen cross stays inside). Table / tape / plates are excluded by saturation, hue
    or distance. Shared by training captures and live inspection. -> mask (same size), unchanged if no colour peak."""
    if mask is None or img is None: return mask
    H, W = mask.shape[:2]
    sm = cv2.resize(img, (W // s, H // s), interpolation=cv2.INTER_AREA)
    seed = cv2.resize(mask, (W // s, H // s), interpolation=cv2.INTER_NEAREST) > 0
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV); h = hsv[..., 0].astype(np.int16)
    sat = (hsv[..., 1] >= sat_min) & (hsv[..., 2] >= val_min)
    hs = h[seed & sat]
    if hs.size < 200: return mask
    raw = np.bincount(hs, minlength=180).astype(float); tot = raw.sum()
    sm_h = sum(np.roll(raw, k) for k in range(-3, 4))           # circular smoothing, for finding peaks
    col = np.zeros_like(seed)
    for i in range(3):                                          # up to 3 dominant hues (brick, jaw pad, ...)
        h0 = int(np.argmax(sm_h))
        idx = [(h0 + k) % 180 for k in range(-hue_tol, hue_tol + 1)]
        mass = raw[idx].sum()
        if mass <= 0 or (i and mass < min_peak * tot): break
        d = np.abs(h - h0); d = np.minimum(d, 180 - d)
        col |= sat & (d <= hue_tol)
        for k in range(-12, 13): sm_h[(h0 + k) % 180] = 0; raw[(h0 + k) % 180] = 0
    near = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (grow_px // s) + 1,) * 2)) > 0
    u = (seed | (col & near)).astype(np.uint8)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab = cv2.connectedComponents(u)
    keep = np.isin(lab, np.unique(lab[seed & (u > 0)])) & (lab > 0)
    out = fill_holes(keep.astype(np.uint8) * 255)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)


BRICK_HUES = {"pink": (168, 1), "red": (168, 1), "blue": (95, 115), "green": (22, 88)}   # OpenCV hue 0-179, range may wrap
# the lime-green 2x4 is hue 23-30 and paler (S 120-200) than pink/blue; the yellow plate is hue 18-20
BRICK_SAT = {"green": (150, 100)}
BRICK_HULL = {"green"}                      # box-shaped brick whose hue the jaws don't share: the mask is the blob's convex hull           # per brick: (sat_min, sat_weak) instead of the defaults


def brick_mask(img, brick, s=4, sat_min=190, val_min=70, min_frac=0.004, sat_weak=140):
    """Mask of the held brick by its COLOUR (saturated pixels in the brick's hue range): the largest such blob, weighted
    toward the image centre, closed and hole-filled (a pen cross stays inside). Robust at 10 cm where depth is patchy.
    For pink the red jaws have the same hue (175-179) and are included (in every photo alike); the orange wrist-camera mount (hue 1-7) is not. -> (mask, info) / (None, why)."""
    if brick not in BRICK_HUES: return None, f"no colour range for brick '{brick}'"
    H, W = img.shape[:2]; sm = cv2.resize(img, (W // s, H // s), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV); h = hsv[..., 0]; lo, hi = BRICK_HUES[brick]
    sat_min, sat_weak = BRICK_SAT.get(brick, (sat_min, sat_weak))
    inh = ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
    m = (inh & (hsv[..., 1] >= sat_min) & (hsv[..., 2] >= val_min)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None, f"no {brick} blob in the picture"
    hh, ww = m.shape; best, bs = 0, 0.0
    for k in range(1, n):
        d = np.hypot((cen[k][0] - ww / 2) / ww, (cen[k][1] - hh / 2) / hh)
        sc = st[k, cv2.CC_STAT_AREA] * (1.0 - min(0.8, d * 1.2))            # big and near the centre wins
        if sc > bs: best, bs = k, sc
    area = int(st[best, cv2.CC_STAT_AREA])
    if area < min_frac * hh * ww: return None, f"{brick} blob too small ({area * s * s} px)"
    seed = lab == best
    # a brightly lit brick face is paler (S ~ 170 < sat_min): grow the blob into connected same-hue pixels >= sat_weak
    weak = (inh & (hsv[..., 1] >= sat_weak) & (hsv[..., 2] >= val_min)).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, lab2 = cv2.connectedComponents(weak | seed.astype(np.uint8))
    out = fill_holes(np.isin(lab2, np.unique(lab2[seed])).astype(np.uint8) * 255)
    if brick in BRICK_HULL:                   # glossy hollow underside: grey reflections split the blob -> its convex hull
        cs, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            hull = cv2.fillConvexPoly(np.zeros_like(out), cv2.convexHull(np.vstack(cs)), 255)
            # inside the hull everything is brick (shadows and grey reflections included) except a red jaw in front of
            # it and the bright orange plate / table in a corner; then the outline is smoothed (no bites, no spikes)
            el = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
            jl, jh = BRICK_HUES["red"]; jaw = ((h >= jl) | (h <= jh)) & (hsv[..., 1] >= 150)
            jaw = cv2.morphologyEx(jaw.astype(np.uint8), cv2.MORPH_CLOSE, el(9))            # the ribbed jaw as one piece
            plate = ((h >= 3) & (h <= 19) & (hsv[..., 1] >= 190) & (hsv[..., 2] >= 110)).astype(np.uint8)
            bad = cv2.morphologyEx(jaw | plate, cv2.MORPH_OPEN, el(5))
            out = np.where((hull > 0) & ((out > 0) | (bad == 0)), 255, 0).astype(np.uint8)
            out = cv2.morphologyEx(cv2.morphologyEx(out, cv2.MORPH_OPEN, el(11)), cv2.MORPH_CLOSE, el(11))
            n3, lab3, st3, _ = cv2.connectedComponentsWithStats(out)
            if n3 > 2: out = (lab3 == 1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
            out = fill_holes(out)
            big = cv2.GaussianBlur(cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 5)
            big = cv2.erode((big >= 128).astype(np.uint8) * 255, el(7))                    # smooth edge, 3 px inside
            return big, {"z_mm": None, "area_px": int((big > 0).sum()), "search": "colour", "brick": brick}
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST), {"z_mm": None, "area_px": area * s * s, "search": "colour", "brick": brick}


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
    if INHAND.get("brick"):                      # colour mask of the named brick (robust at 10 cm, no depth needed)
        m, info = brick_mask(img, INHAND["brick"])
        if m is None: return img, {"masked": False, "why": info, **cfg}
        info = {**info, **cfg}
    else:
        if depth_mm is None: return img, {"masked": False, "why": "no depth", **cfg}
        m, info = inhand_mask(depth_mm)
        if m is None: return img, {"masked": False, "why": info, **cfg}
        info = {**info, **cfg}
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = complete_mask(img, m)                # colour fills what depth missed on the glossy brick
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
