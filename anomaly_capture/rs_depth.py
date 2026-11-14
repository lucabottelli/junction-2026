"""Close-range depth on the front RealSense D415: disparity-shift math, advanced-mode setup, auto-tune scoring.

Why: at the WP2 photo poses the held brick is ~8-15 cm from the front camera. A D4xx only measures depth where the
stereo disparity d (pixels) lies inside its search window [shift, shift + 126], and z = fx * B / d
(fx = focal length of the depth stream in pixels, B = stereo baseline in mm). So
    near (Min-Z) = fx * B / (shift + 126)        far = fx * B / shift   (no far limit when shift = 0)
fx scales with the depth resolution: at 1280x720 (fx ~ 900-950, B ~ 55 mm) Min-Z is ~40-45 cm, at 424x240
(fx ~ 300-310) it is ~13-15 cm. A disparity shift (advanced mode, `STDepthTableControl.disparityShift`) moves the
window closer at the price of a far limit; everything beyond `far` simply has no depth (= also a free background cut).

Everything here is pure or takes the `pyrealsense2` module as an argument (`rs`), so it is unit-tested without a camera
(tests/test_depth_mask.py). capture.py (RSCam, /depth/* endpoints) and mock_hw.py (FakeCam) use it.
"""
import math
import time

import numpy as np

DISP_RANGE = 126          # disparity search range of the D4xx stereo matcher (pixels)
SHIFT_MAX = 511           # largest disparityShift we send (the RealSense Viewer's advanced-mode slider range is 0..511)
MODES = [(1280, 720), (848, 480), (640, 480), (640, 360), (480, 270), (424, 240)]   # D415 depth modes we accept
DEFAULT = {"width": 424, "height": 240, "fps": 30, "shift": None, "near_mm": None, "advanced": True,
           "expect_z_mm": None, "max_obj_mm": 400}
NEAR_TARGETS = (None, 140, 120, 100, 85, 70, 60, 50)   # auto-tune sweep: shift 0 + the shifts that put Min-Z here (mm)


# ------------------------------------------------------------------ pure math
def depth_window(fx, baseline_mm, shift):
    """(near_mm, far_mm) measurable with this disparity shift. far_mm is math.inf for shift <= 0."""
    if fx <= 0 or baseline_mm <= 0: raise ValueError("fx and baseline must be positive")
    shift = max(0, int(shift))
    fb = float(fx) * float(baseline_mm)
    return fb / (shift + DISP_RANGE), (fb / shift if shift > 0 else math.inf)


def shift_for_near(fx, baseline_mm, near_mm):
    """Smallest disparity shift whose Min-Z is at or below `near_mm`. -> (shift, clamped).
    clamped=True: the request was beyond what the hardware window allows (shift capped at SHIFT_MAX, or
    near_mm farther than the unshifted Min-Z, where shift 0 is already enough)."""
    if fx <= 0 or baseline_mm <= 0: raise ValueError("fx and baseline must be positive")
    if near_mm is None: return 0, False
    if near_mm <= 0: return SHIFT_MAX, True     # "as close as possible"
    s = math.ceil(float(fx) * float(baseline_mm) / float(near_mm) - DISP_RANGE - 1e-9)
    if s < 0: return 0, False                 # shift 0 already measures this close
    if s > SHIFT_MAX: return SHIFT_MAX, True
    return int(s), False


def window_info(fx, baseline_mm, shift):
    """JSON-friendly description of the window (far_mm None = no far limit)."""
    near, far = depth_window(fx, baseline_mm, shift)
    return {"shift": int(max(0, shift)), "near_mm": round(near), "far_mm": None if math.isinf(far) else round(far)}


def resolve_config(saved):
    """cam_settings.json "front_depth" -> complete, validated config. Never raises: every bad or unknown value falls
    back to its DEFAULT and is listed in c["config_warning"] (unsupported resolution -> 424x240)."""
    c, bad = dict(DEFAULT), []
    if saved is not None and not isinstance(saved, dict): bad.append(f"front_depth must be an object, got {saved!r}")
    elif saved: c.update({k: v for k, v in saved.items() if k in DEFAULT})

    def num(k, kind, lo, hi, none_ok):
        v = c.get(k)
        if v is None and none_ok: return None
        try:
            if isinstance(v, bool): raise ValueError
            x = kind(v)
            if not (math.isfinite(x) and lo <= x <= hi): raise ValueError
            return x
        except (TypeError, ValueError, OverflowError):
            bad.append(f"{k}={v!r}"); return DEFAULT[k]
    c["width"] = num("width", int, 1, 10000, False); c["height"] = num("height", int, 1, 10000, False)
    c["fps"] = num("fps", int, 1, 300, False)
    if c["fps"] not in (6, 15, 30, 60, 90): bad.append(f"fps={c['fps']}"); c["fps"] = DEFAULT["fps"]
    if (c["width"], c["height"]) not in MODES:
        bad.append(f"resolution {c['width']}x{c['height']} not supported"); c["width"], c["height"] = DEFAULT["width"], DEFAULT["height"]
    sh = num("shift", float, -1e9, 1e9, True)
    c["shift"] = None if sh is None else int(min(SHIFT_MAX, max(0, int(sh))))
    c["near_mm"] = num("near_mm", float, 1e-6, 1e5, True)
    c["expect_z_mm"] = num("expect_z_mm", float, 1e-6, 1e5, True)
    c["max_obj_mm"] = num("max_obj_mm", float, 1e-6, 1e5, False)
    if not isinstance(c["advanced"], bool): bad.append(f"advanced={c['advanced']!r}"); c["advanced"] = DEFAULT["advanced"]
    c["config_warning"] = ("bad front_depth settings ignored: " + ", ".join(bad)) if bad else None
    return c


def initial_shift(cfg, fx, baseline_mm):
    """Shift to apply at start-up: an explicit "shift" wins, else the one for "near_mm", else 0. -> (shift, note)."""
    if cfg.get("shift") is not None: return int(cfg["shift"]), "from settings"
    if cfg.get("near_mm") and not (fx and baseline_mm): return 0, "near_mm ignored: depth intrinsics unknown"
    if cfg.get("near_mm"):
        s, clamped = shift_for_near(fx, baseline_mm, float(cfg["near_mm"]))
        return s, (f"for near_mm={cfg['near_mm']}" + (" (clamped to the hardware limit)" if clamped else ""))
    return 0, "default"


def candidate_shifts(fx, baseline_mm, nears=NEAR_TARGETS):
    """Shifts to try in the auto-tune: shift 0 and the shifts that move Min-Z to each near target. Sorted, unique."""
    out = {0}
    if not (fx and baseline_mm): return sorted(out)
    for n in nears:
        if n: out.add(shift_for_near(fx, baseline_mm, n)[0])
    return sorted(out)


# ------------------------------------------------------------------ measuring a depth frame (auto-tune, /depth/status)
def roi_valid_fraction(depth, roi):
    """Fraction of pixels with depth inside roi (full-res x0,y0,x1,y1; scaled if depth is smaller than 1920 wide)."""
    if depth is None: return 0.0
    s = depth.shape[1] / 1920.0
    x0, y0, x1, y1 = (int(round(v * s)) for v in roi)
    sub = depth[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return float((sub > 0).mean()) if sub.size else 0.0


def measure(depth, roi, mask_fn):
    """Stats of one aligned depth frame (uint16 mm): valid fraction in the ROI, the in-hand cluster from `mask_fn`
    (prep.inhand_mask without the expected-z prior), and how many pixels of it really carry depth (`object_px`,
    the auto-tune score: hole-filled pixels don't count)."""
    out = {"valid_roi": round(roi_valid_fraction(depth, roi), 3), "masked": False, "z_mm": None, "object_px": 0}
    if depth is None: out["why"] = "no depth"; return out
    m, info = mask_fn(depth)
    if m is None: out["why"] = info; return out
    z = info["z_mm"]; band = info.get("band_mm", 70)
    real = (m > 0) & (depth > 0) & (np.abs(depth.astype(np.int32) - z) <= band)
    out.update({"masked": True, "z_mm": z, "object_px": int(real.sum()), "area_px": info.get("area_px")})
    return out


def pick_shift(results, plateau=0.95, near_band=70):
    """Best shift from auto-tune results [{shift, masked, object_px, z_mm, ...}].
    Only results whose cluster is the NEAREST one seen in the sweep (within near_band mm) compete: at small shifts a
    brick inside Min-Z is invisible and the background behind it can be the cluster, with far more pixels.
    Score = object_px (pixels of the held cluster that really have depth). Every shift within `plateau` of the best
    score is equally good; we take the MIDDLE of that plateau, which leaves margin on both sides of the window
    (pose jitter towards the camera -> near limit, away -> far limit). None if no shift gave a mask."""
    ok = [r for r in results if r.get("masked") and r.get("object_px", 0) > 0 and r.get("z_mm") is not None]
    if not ok: return None
    zmin = min(r["z_mm"] for r in ok)                  # the HELD object is the nearest cluster any shift found; a
    ok = [r for r in ok if r["z_mm"] <= zmin + near_band]   # bigger, farther cluster (wall, table) never wins
    best = max(r["object_px"] for r in ok)
    good = sorted((r for r in ok if r["object_px"] >= plateau * best), key=lambda r: r["shift"])
    return good[(len(good) - 1) // 2]


# ------------------------------------------------------------------ device (pyrealsense2 passed in as `rs`)
def find_device(rs, serial):
    for d in rs.context().query_devices():
        try:
            if d.get_info(rs.camera_info.serial_number) == serial: return d
        except Exception: continue
    return None


def enable_advanced(rs, serial, timeout=20.0, poll=0.5, sleep=time.sleep, now=time.monotonic, log=print):
    """Make sure the D4xx with `serial` is in advanced mode, BEFORE its pipeline starts.
    toggle_advanced_mode(True) resets the device: it disconnects and re-enumerates (several seconds), so the old
    handle is useless afterwards; we poll rs.context() for the serial until it is back in advanced mode or `timeout`.
    The flag is stored on the device (survives restarts), so this reset happens once per camera, not per start.
    -> (ok, message). Never raises."""
    try:
        dev = find_device(rs, serial)
        if dev is None: return False, f"device {serial} not found"
        if rs.rs400_advanced_mode(dev).is_enabled(): return True, "advanced mode already on"
        log(f"depth {serial}: enabling advanced mode, the camera resets (takes a few seconds) ...")
        rs.rs400_advanced_mode(dev).toggle_advanced_mode(True)
    except Exception as e:
        return False, f"advanced mode not available ({e})"
    del dev
    sleep(min(3.0, timeout))                       # it first disconnects: don't find the old, dying instance
    t0 = now()
    while now() - t0 < timeout:
        try:
            d = find_device(rs, serial)
            if d is not None and rs.rs400_advanced_mode(d).is_enabled(): return True, "advanced mode enabled (camera was reset)"
        except Exception: pass                     # half-enumerated device: try again
        sleep(poll)
    return False, f"camera did not come back in advanced mode within {timeout:.0f} s"


def set_shift(rs, dev, shift):
    """Write the disparity shift into the depth table (works while streaming). -> the value read back."""
    adv = rs.rs400_advanced_mode(dev)
    t = adv.get_depth_table()
    t.disparityShift = int(min(SHIFT_MAX, max(0, int(shift))))
    adv.set_depth_table(t)
    return int(adv.get_depth_table().disparityShift)
