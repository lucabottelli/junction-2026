"""Anomaly dataset capture tool (Junction 2026, SO-101 brick sorting) - multi-camera.

Browser UI at http://127.0.0.1:7870 — live preview of all cameras (top, front, wrist), auto brick
detection per camera, live full-res zoom, and one key press saves lossless full-res crops from every
enabled camera at the same moment (linked by a shared "group" id in meta.jsonl).

Run (Studio's Environments preview must be CLOSED so the cameras are free):
    ~/physical-ai-studio/application/backend/.venv/bin/python ~/junction26/anomaly_capture/capture.py

Keys: B = empty background (all enabled cams; refused if a brick is in the area, Shift+B forces), G = good, D = defect (any size; measured later from pixels), U = undo last save.
Drag on a camera's preview to redraw its detection area (saved per camera). Tick/untick cameras to include.

Mock station (no hardware, separate data root; see docs/DEV.md and mock_hw.py):
    JUNCTION_MOCK=1 [JUNCTION_PORT=7871] [JUNCTION_DATA_ROOT=...] python anomaly_capture/capture.py   (or --mock)
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse

MOCK = os.environ.get("JUNCTION_MOCK") == "1" or "--mock" in sys.argv   # hardware-free mock station (mock_hw.py)
if MOCK:
    os.environ["JUNCTION_MOCK"] = "1"     # arm.py picks the fake servo bus from this
    import mock_hw
else:
    import pyrealsense2 as rs
import prep, rs_depth      # noqa: E402  (after mock_hw: in mock mode it redirects prep's settings file first)
PORT = int(os.environ.get("JUNCTION_PORT", "7870"))

CAMS = ["top", "front", "wrist"]
RS_SERIAL = {"top": "243622063627", "front": "243622061066"}
DEPTH_CAMS = {"front"}   # aligned depth (mm) kept alongside colour for background masking
# Depth stream config per depth camera: cam_settings.json "<cam>_depth" (see rs_depth.DEFAULT): resolution (Min-Z scales
# with it: 424x240 measures from ~14 cm, 1280x720 only from ~42 cm), disparity "shift" (or "near_mm" to compute it),
# "advanced" (allow advanced mode, needed for the shift) and "expect_z_mm" (held-brick distance found by /depth/autotune)
WRIST_DEV = 16
ROOT = Path(__file__).resolve().parent.parent  # ~/junction26
DATA = ROOT / "training_images"
SETTINGS = ROOT / "anomaly_capture" / "cam_settings.json"
LABELS = {"good": "good", "defect": "defect"}
TARGET = {"good": 150, "defect": 30}   # suggested counts per camera
PAD = 0.25            # crop padding around brick bbox (fraction of brick size)
AREA = {"top": (1500, 60000), "front": (1500, 250000), "wrist": (1500, 1500000)}   # blob pixel-area limits
SIDE = {"top": (30, 200), "front": (30, 600), "wrist": (40, 1900)}                 # bbox side limits (full-res)
# detection area (full-res x0, y0, x1, y1); drag in the UI to change (saved in cam_settings.json "roi")
DEFAULT_ROI = {"top": [712, 278, 1132, 638], "front": [840, 240, 1380, 780], "wrist": [300, 150, 1620, 950]}
AVG = {"top": 16, "front": 16, "wrist": 8}  # frames averaged per saved image (~sqrt(N) less sensor noise)
MOTION_MAX = 4.0     # mean abs change (0-255) inside the brick box during averaging; above = "moved"
PREVIEW_W = 640
VIEW_W = 1280        # width of the clean (no boxes) camera view of the user page
# thin-structure filter size at 1/4 scale: about half the brick's smallest side on that camera
# (top brick ~20 px, front ~50 px, wrist ~30+ px at 1/4 scale)
OPEN_K = {"top": 9, "front": 15, "wrist": 13}
STALE_FRAC = 0.04    # >4% of the detection area differs from the background outside the brick -> warn "press B again"
if MOCK:   # never touch the station's data/settings: everything goes to the mock data root
    ROOT = mock_hw.install(ROOT); DATA = ROOT / "training_images"; SETTINGS = ROOT / "settings" / "cam_settings.json"


def load_settings():
    return json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}


def store_settings(key, value):
    s = load_settings(); s[key] = value; SETTINGS.write_text(json.dumps(s, indent=2))


# ---------------------------------------------------------------- cameras
class RSCam:
    """D415 colour stream at 1080p with exposure + white balance locked (calibrated once, then reused)."""

    def __init__(self, name):
        self.name = name
        self.depth, self.depth_t, self.align, self.n, self.depth_n = None, 0.0, None, 0, 0
        self.depth_scale, self.dinfo, self.dev = 0.001, None, None
        self.pipe = None
        prof = self._start_depth_safe() if name in DEPTH_CAMS else None   # depth greys out everything behind the brick
        if prof is None:
            self.pipe = rs.pipeline()                 # fresh: an advanced-mode toggle may have reset the device
            cfg = rs.config(); cfg.enable_device(RS_SERIAL[name])
            cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            prof = self.pipe.start(cfg)
        self.sen = prof.get_device().first_color_sensor()
        self.settings = self._lock()

    def _start_depth_safe(self):
        """_start_depth that never raises: any unexpected error -> colour only (as before), reason in self.dinfo."""
        try:
            return self._start_depth()
        except Exception as e:
            print(f"!! {self.name}: depth setup failed ({e!r}), colour only")
            try:
                if self.pipe is not None: self.pipe.stop()
            except Exception: pass
            self.pipe, self.align, self.dev = None, None, None
            self.dinfo = {"enabled": False, "width": None, "height": None, "shift": 0, "advanced": False, "near_mm": None,
                          "far_mm": None, "fx_depth": None, "baseline_mm": None, "fx_color": None, "expect_z_mm": None,
                          "max_obj_mm": None, "warning": f"depth setup failed ({e}): colour only"}
            return None

    def _start_depth(self):
        """Colour 1080p + close-range depth (resolution and disparity shift from cam_settings.json "<cam>_depth").
        Never raises: advanced mode / shift problems -> shift 0 with a warning in self.dinfo; no depth stream at all
        -> None (the caller starts colour only, as before)."""
        serial = RS_SERIAL[self.name]
        c = rs_depth.resolve_config(load_settings().get(f"{self.name}_depth"))
        st = {"enabled": False, "width": c["width"], "height": c["height"], "fps": c["fps"], "shift": 0, "advanced": False,
              "near_mm": None, "far_mm": None, "fx_depth": None, "baseline_mm": None, "fx_color": None,
              "expect_z_mm": c.get("expect_z_mm"), "max_obj_mm": c.get("max_obj_mm"), "warning": None}
        warn, adv_ok = ([c["config_warning"]] if c.get("config_warning") else []), False
        if c["advanced"]:      # BEFORE the pipeline starts: enabling it resets the camera (handled in enable_advanced)
            adv_ok, msg = rs_depth.enable_advanced(rs, serial)
            if not adv_ok: warn.append(f"advanced mode unavailable ({msg}): disparity shift stays 0")
        try:
            self.pipe = rs.pipeline()                 # created AFTER a possible advanced-mode reset
            cfg = rs.config(); cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, c["width"], c["height"], rs.format.z16, c["fps"])
            prof = self.pipe.start(cfg)
        except Exception as e:
            print(f"!! {self.name}: depth stream unavailable ({e}), colour only")
            self.pipe = None; self.dinfo = {**st, "warning": f"depth stream unavailable ({e}): colour only"}
            return None
        self.align = rs.align(rs.stream.color); st["enabled"] = True
        dev = prof.get_device(); dsen = dev.first_depth_sensor()
        try: self.depth_scale = float(dsen.get_depth_scale())
        except Exception: pass
        try:
            st["fx_depth"] = round(prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().fx, 2)
            st["fx_color"] = round(prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics().fx, 2)
            st["baseline_mm"] = round(abs(float(dsen.get_option(rs.option.stereo_baseline))), 3)
        except Exception as e:
            warn.append(f"intrinsics/baseline unreadable ({e})")
        shift, note = (rs_depth.initial_shift(c, st["fx_depth"], st["baseline_mm"])
                       if st["fx_depth"] and st["baseline_mm"] else (0, "default"))
        if adv_ok:
            self.dev = dev
            try:
                shift = rs_depth.set_shift(rs, dev, shift); st["advanced"] = True
            except Exception as e:
                warn.append(f"could not set disparity shift {shift} ({e}): using 0")
                try: shift = rs_depth.set_shift(rs, dev, 0)
                except Exception: pass
                shift = 0
        else:
            if shift: warn.append(f"wanted disparity shift {shift} ({note}) but advanced mode is off: using 0")
            shift = 0
            try:                                      # the device may still be in advanced mode with a stale shift
                if rs.rs400_advanced_mode(dev).is_enabled(): rs_depth.set_shift(rs, dev, 0)
            except Exception: pass
        st["shift"] = shift
        if st["fx_depth"] and st["baseline_mm"]: st.update(rs_depth.window_info(st["fx_depth"], st["baseline_mm"], shift))
        st["warning"] = "; ".join(warn) or None
        self.dinfo = st
        if warn: print(f"!! {self.name} depth WARNING: {st['warning']}")
        print(f"{self.name} depth: {c['width']}x{c['height']}, disparity shift {shift} ({note}) -> measures "
              f"{st['near_mm']}-{st['far_mm'] or 'inf'} mm")
        return prof

    def depth_info(self):
        """Depth config + window (None for a colour-only camera). Shown in /status and /depth/status, saved in meta."""
        return dict(self.dinfo) if self.dinfo else None

    def set_shift(self, shift):
        """Change the disparity shift live (no restart). -> the shift now in the depth table."""
        d = self.dinfo or {}
        if not (d.get("advanced") and self.dev is not None):
            raise RuntimeError("advanced mode is off on this camera: the disparity shift can't be changed")
        s = rs_depth.set_shift(rs, self.dev, shift)
        d["shift"] = s
        if d.get("fx_depth") and d.get("baseline_mm"): d.update(rs_depth.window_info(d["fx_depth"], d["baseline_mm"], s))
        return s

    def _grab(self, n=1):
        for _ in range(n):
            f = self.pipe.wait_for_frames()
        return np.asanyarray(f.get_color_frame().get_data()).copy()

    def _lock(self):
        O = rs.option
        saved = load_settings()
        if self.name in saved:
            s = saved[self.name]
        else:   # match auto-exposure brightness/colour, then freeze
            self.sen.set_option(O.enable_auto_exposure, 1); self.sen.set_option(O.enable_auto_white_balance, 1)
            ref = self._grab(90).reshape(-1, 3).mean(0)
            self.sen.set_option(O.enable_auto_exposure, 0); self.sen.set_option(O.enable_auto_white_balance, 0)
            best = None
            for wb in range(3000, 6501, 250):
                self.sen.set_option(O.white_balance, wb); m = self._grab(8).reshape(-1, 3).mean(0)
                err = abs(np.log((m[0] + 1) / (m[2] + 1)) - np.log((ref[0] + 1) / (ref[2] + 1)))
                best = min(best or (9, wb), (err, wb))
            self.sen.set_option(O.white_balance, best[1])
            for gain in (32.0, 48.0, 64.0, 80.0, 96.0, 128.0):   # lowest gain that reaches auto brightness
                self.sen.set_option(O.gain, gain); lo, hi = 1.0, 330.0
                for _ in range(10):
                    mid = (lo + hi) / 2; self.sen.set_option(O.exposure, mid)
                    if self._grab(6).mean() < ref.mean(): lo = mid
                    else: hi = mid
                if hi < 320: break
            s = {"exposure": round((lo + hi) / 2), "gain": gain, "white_balance": best[1],
                 "auto_ref_mean_bgr": [round(float(v), 1) for v in ref]}
            store_settings(self.name, s)
        self.sen.set_option(O.enable_auto_exposure, 0); self.sen.set_option(O.enable_auto_white_balance, 0)
        self.sen.set_option(O.gain, s["gain"]); self.sen.set_option(O.white_balance, s["white_balance"])
        self.sen.set_option(O.exposure, s["exposure"])
        self._grab(15)
        print(f"{self.name} camera locked:", s)
        return s

    def read(self):
        f = self.pipe.wait_for_frames(2000)          # raises after 2 s instead of blocking forever
        self.n += 1
        # a stalled COLOUR stream still yields framesets (depth keeps arriving, the syncer repeats the last colour frame):
        # treat a colour frame number that doesn't advance for 1 s as "no frames", so the reconnect logic takes over
        try: cnum = f.get_color_frame().get_frame_number()
        except Exception: cnum = None
        now = time.time()
        if cnum is not None and cnum == getattr(self, "_cnum", None):
            if now - self._cnum_t > 1.0: raise RuntimeError(f"colour frames frozen (frame {cnum} for {now - self._cnum_t:.1f} s)")
        else:
            self._cnum, self._cnum_t = cnum, now
        if self.align is not None and self.n % 3 == 0:   # depth aligned to colour pixels, ~10 Hz (CPU)
            try:
                fa = self.align.process(f); d = fa.get_depth_frame()
                if d:
                    dep = np.asanyarray(d.get_data())
                    if abs(self.depth_scale - 0.001) > 1e-6:     # keep millimetres whatever the depth units are
                        dep = np.clip(dep * (self.depth_scale * 1000.0), 0, 65535).astype(np.uint16)
                    self.depth = dep.copy(); self.depth_t = time.time(); self.depth_n += 1
                return np.asanyarray(fa.get_color_frame().get_data()).copy()
            except Exception as e:
                print(f"{self.name}: depth align failed: {e}")
        return np.asanyarray(f.get_color_frame().get_data()).copy()


class WristCam:
    """USB wrist camera at 1080p MJPG; auto exposure/WB settle then freeze (best effort via V4L2)."""

    def __init__(self):
        self.cap = cv2.VideoCapture(WRIST_DEV, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(60): self.cap.read()
        exp = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # V4L2: 1 = manual
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.settings = {"exposure": exp, "wb_temp": self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}
        print("wrist camera locked (best effort):", self.settings)

    def read(self):
        ok, f = self.cap.read()
        return f if ok else None


def open_cam(name):
    if MOCK:
        c = mock_hw.open_cam(name)
        if c is not None and name in DEPTH_CAMS:
            try: c.configure_depth(rs_depth.resolve_config(load_settings().get(f"{name}_depth")))
            except Exception as e: print(f"!! {name}: mock depth config failed ({e}), defaults"); c.configure_depth(rs_depth.resolve_config(None))
        return c
    try:
        return RSCam(name) if name in RS_SERIAL else WristCam()
    except Exception as e:      # a missing camera must not take the whole tool down
        print(f"!! {name} camera failed to open: {e}")
        return None


# ---------------------------------------------------------------- detection
def detect(img, bg, cam, roi):
    """Return brick bboxes (x, y, w, h) at full resolution, fully inside the area, closest to its centre first."""
    s = 4
    small = cv2.resize(img, (img.shape[1] // s, img.shape[0] // s), interpolation=cv2.INTER_AREA)
    if bg is not None:   # difference from the empty-scene reference (works for any brick colour)
        bsmall = cv2.resize(bg, small.shape[1::-1], interpolation=cv2.INTER_AREA)
        d = cv2.absdiff(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), cv2.cvtColor(bsmall, cv2.COLOR_BGR2LAB))
        m = (d.astype(np.int32).sum(2) > 40).astype(np.uint8) * 255
    else:                # fallback: pixels whose colour differs from the table (= median colour of the area)
        x0, y0, x1, y1 = (v // s for v in roi)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int32)
        table = np.median(lab[y0:y1, x0:x1].reshape(-1, 3), 0)
        m = (np.abs(lab - table).sum(2) > 45).astype(np.uint8) * 255
    x0, y0, x1, y1 = (v // s for v in roi)
    m[:y0] = 0; m[y1:] = 0; m[:, :x0] = 0; m[:, x1:] = 0   # only look inside the area (no merging with outside blobs)
    # remove thin structures (edges of plates/objects nudged since the background was taken, cables) but keep
    # solid blobs: a brick is >= ~16 px wide at this 1/4 scale, a shifted rim is only a few px thick
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K[cam],) * 2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    lo, hi = AREA[cam]; rx0, ry0, rx1, ry1 = roi
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i] * np.array([s, s, s, s, s * s]))
        touches = x <= rx0 + s or y <= ry0 + s or x + w >= rx1 - s or y + h >= ry1 - s   # cut off by the area edge
        if (not touches and lo <= a <= hi and SIDE[cam][0] <= min(w, h) and max(w, h) <= SIDE[cam][1]
                and 0.3 < w / max(h, 1) < 3.3):
            boxes.append((x, y, w, h, (x + w / 2 - (rx0 + rx1) / 2) ** 2 + (y + h / 2 - (ry0 + ry1) / 2) ** 2))
    return [b[:4] for b in sorted(boxes, key=lambda b: b[4])]   # closest to the area centre first


def stale_fraction(img, bg, roi, boxes):
    """Fraction of the detection area (outside detected bricks) that differs from the saved background.
    High = something on the table moved (plate, tape, camera) -> the background should be retaken."""
    if bg is None: return 0.0
    s = 4; small = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
    bs = cv2.resize(bg, (480, 270), interpolation=cv2.INTER_AREA)
    m = cv2.absdiff(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), cv2.cvtColor(bs, cv2.COLOR_BGR2LAB)).astype(np.int32).sum(2) > 40
    x0, y0, x1, y1 = (v // s for v in roi)
    for (x, y, w, h) in boxes:
        m[max(0, y // s - 3):(y + h) // s + 3, max(0, x // s - 3):(x + w) // s + 3] = False
    return float(m[y0:y1, x0:x1].mean())


def crop_box(img, box):
    x, y, w, h = box
    side = min(int(max(w, h) * (1 + 2 * PAD)), img.shape[0]); cx, cy = x + w // 2, y + h // 2
    x0 = min(max(0, cx - side // 2), img.shape[1] - side); y0 = min(max(0, cy - side // 2), img.shape[0] - side)
    return img[y0:y0 + side, x0:x0 + side]


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def jpg(img, q=80):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


BLANK = jpg(np.full((300, 300, 3), 40, np.uint8))
OFFLINE = jpg(cv2.putText(np.full((360, 640, 3), 30, np.uint8), "camera offline", (180, 190),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (90, 90, 255), 2))


# ---------------------------------------------------------------- state
class State:
    def __init__(self):
        with ThreadPoolExecutor(3) as ex:   # open (and if needed calibrate) all cameras in parallel
            self.cams = dict(zip(CAMS, ex.map(open_cam, CAMS)))
        self.live = [k for k in CAMS if self.cams[k] is not None]
        self.enabled = {k: k in self.live for k in CAMS}
        self.frames = {k: None for k in CAMS}
        self.hist = {k: [] for k in CAMS}
        self.frame_t = {k: 0.0 for k in CAMS}; self.reconnects = {k: 0 for k in CAMS}
        self.bg = {k: (cv2.imread(str(DATA / k / "background.png")) if (DATA / k / "background.png").exists() else None)
                   for k in CAMS}
        saved = load_settings().get("roi", {})
        self.roi = {k: saved.get(k, DEFAULT_ROI[k]) for k in CAMS}
        self.lock = threading.Lock()
        self.saved = []           # undo stack: list of file lists (one entry per key press)
        self.recent = []          # (caption, png bytes), newest first
        self.boxes = {k: [] for k in CAMS}
        self.stale = {k: 0.0 for k in CAMS}
        self.preview = {k: None for k in CAMS}; self.zoomj = {k: None for k in CAMS}; self.clean = {k: None for k in CAMS}
        self.msg, self.ok = "ready", True
        for k in self.live:
            threading.Thread(target=self._loop, args=(k,), daemon=True).start()
            threading.Thread(target=self._render, args=(k,), daemon=True).start()

    def _loop(self, k):
        """Read frames forever. A stall/exception never kills the thread: after 3 s without a frame the
        camera is closed and re-opened (same locked settings from cam_settings.json)."""
        last_ok = time.time()
        while True:
            try:
                f = self.cams[k].read() if self.cams[k] is not None else None
            except Exception as e:
                print(f"{k}: read error: {e}"); f = None
            if f is not None:
                last_ok = time.time()
                with self.lock:
                    self.frames[k] = f; self.frame_t[k] = last_ok
                    self.hist[k] = (self.hist[k] + [f])[-AVG[k]:]
                continue
            time.sleep(0.05)
            if time.time() - last_ok > 3:
                print(f"{k}: no frames for 3 s, reconnecting camera"); self.reconnects[k] += 1
                try:
                    c = self.cams[k]
                    if hasattr(c, "pipe"): c.pipe.stop()
                    elif hasattr(c, "cap"): c.cap.release()
                except Exception: pass
                time.sleep(1); self.cams[k] = open_cam(k); last_ok = time.time()

    def _render(self, k):
        """Detection ~8x/s per camera; preview + zoom are served from this cache."""
        while True:
            f = self.frame(k)
            if f is None: time.sleep(0.05); continue
            roi = self.roi[k]; boxes = detect(f, self.bg[k], k, roi); self.boxes[k] = boxes
            self.stale[k] = stale_fraction(f, self.bg[k], roi, boxes)
            if boxes:
                z = crop_box(f, boxes[0]); zs = 300 / z.shape[0]
                self.zoomj[k] = jpg(cv2.resize(z, None, fx=zs, fy=zs,
                                               interpolation=cv2.INTER_NEAREST if zs > 1 else cv2.INTER_AREA), 92)
            else:
                self.zoomj[k] = None
            # the same frame WITHOUT the boxes, larger: the user page's camera views (/view/<cam>)
            self.clean[k] = jpg(cv2.resize(f, (VIEW_W, VIEW_W * f.shape[0] // f.shape[1]), interpolation=cv2.INTER_AREA), 80)
            vis = f.copy()
            cv2.rectangle(vis, tuple(roi[:2]), tuple(roi[2:]), (255, 200, 0), 4)
            for i, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0) if i == 0 else (0, 200, 255), 5)
            if not self.enabled[k]: vis = (vis * 0.4).astype(np.uint8)
            self.preview[k] = jpg(cv2.resize(vis, (PREVIEW_W, PREVIEW_W * vis.shape[0] // vis.shape[1]),
                                             interpolation=cv2.INTER_AREA), 75)
            time.sleep(0.12)

    def frame(self, k):
        with self.lock:
            f = self.frames[k]
            return None if f is None else f.copy()

    def averaged(self, cams, n=None):
        """Average AVG fresh frames on all given cams in the same time window -> {cam: (mean, first, last)}.
        n = frames to average instead of AVG[cam] (flow.py: a shorter stop at the photo pose)."""
        with self.lock:
            for k in cams: self.hist[k] = []
        t0 = time.time(); out = {}
        while len(out) < len(cams) and time.time() - t0 < 5:
            with self.lock:
                for k in cams:
                    if k not in out and len(self.hist[k]) >= (n or AVG[k]):
                        st = np.stack(self.hist[k][:n or AVG[k]]).astype(np.float32)
                        out[k] = (st.mean(0).round().astype(np.uint8), st[0], st[-1])
            time.sleep(0.02)
        return out

    def counts(self):
        return {cam: {lab: (len(list((DATA / cam / d).glob("*.png"))) if (DATA / cam / d).exists() else 0)
                      for lab, d in LABELS.items()} for cam in CAMS}

    def active(self):
        return [k for k in self.live if self.enabled[k]]

    def say(self, msg, ok=True):
        self.msg, self.ok = msg, ok


S: State = None
app = FastAPI()


def mjpeg(getter, period):
    def gen():
        while True:
            b = getter()
            if b: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n"
            time.sleep(period)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/view/{cam}")
def view(cam: str):
    """The live camera picture without the detection boxes (user page). Falls back to the boxed preview."""
    c = getattr(S, "clean", {}).get(cam) or S.preview.get(cam)
    return Response(c or OFFLINE, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/snap/{cam}")
def snap(cam: str):
    return Response(S.preview.get(cam) or OFFLINE, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/snapzoom/{cam}")
def snapzoom(cam: str):
    return Response(S.zoomj.get(cam) or BLANK, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/stream/{cam}")
def stream(cam: str): return mjpeg(lambda: S.preview.get(cam) or OFFLINE, 0.12)


@app.get("/zoom/{cam}")
def zoom(cam: str): return mjpeg(lambda: S.zoomj.get(cam) or BLANK, 0.2)


@app.get("/recent/{i}")
def recent(i: int):
    if i >= len(S.recent): return Response(status_code=204)
    return Response(S.recent[i][1], media_type="image/png")


@app.get("/status")
def status():
    return {"cams": CAMS, "live": S.live, "enabled": S.enabled, "msg": S.msg, "ok": S.ok, "counts": S.counts(),
            "target": TARGET, "has_bg": {k: v is not None for k, v in S.bg.items()}, "roi": S.roi,
            "n_boxes": {k: len(v) for k, v in S.boxes.items()},
            "stale": {k: round(v, 3) for k, v in S.stale.items()},
            "age": {k: (round(time.time() - t, 1) if t else None) for k, t in S.frame_t.items()},
            "reconnects": S.reconnects, "stale_max": STALE_FRAC, "recent": [r[0] for r in S.recent],
            "settings": {k: (_with_depth(c) if c else None) for k, c in S.cams.items()},
}


def _with_depth(c):
    di = c.depth_info() if hasattr(c, "depth_info") else None
    return {**c.settings, "depth": di} if di else c.settings


@app.post("/enable/{cam}")
def enable(cam: str, on: bool):
    if cam in S.live: S.enabled[cam] = on
    S.say(f"{cam} {'included in' if on else 'excluded from'} captures"); return status()


@app.post("/roi/{cam}")
def set_roi(cam: str, x0: int, y0: int, x1: int, y1: int):
    if cam == "front" and TUNE["running"]: S.say("NOT SET: depth auto-tune running", False); return status()
    x0, x1 = sorted((max(0, x0), min(1920, x1))); y0, y1 = sorted((max(0, y0), min(1080, y1)))
    if x1 - x0 < 60 or y1 - y0 < 60: S.say("NOT SET: detection area too small", False); return status()
    S.roi[cam] = [x0, y0, x1, y1]; store_settings("roi", S.roi); sync_inhand()
    S.say(f"{cam}: detection area set to {S.roi[cam]}"); return status()


@app.post("/roi/{cam}/reset")
def reset_roi(cam: str):
    if cam == "front" and TUNE["running"]: S.say("NOT RESET: depth auto-tune running", False); return status()
    S.roi[cam] = list(DEFAULT_ROI[cam]); store_settings("roi", S.roi); sync_inhand()
    S.say(f"{cam}: detection area reset"); return status()


@app.post("/background")
def background(force: bool = False):
    cams = S.active()
    if not cams: S.say("NOT SET: no camera enabled", False); return status()
    shots = S.averaged(cams)
    if not force:   # safety: the background must be EMPTY, otherwise the brick becomes "table" and is never detected
        busy = [k for k, (f, _, _) in shots.items() if detect(f, None, k, S.roi[k])]
        if busy:
            S.say(f"NOT SET: something is still in the detection area on {', '.join(busy)}. Remove the brick "
                  "(and hands) first. If the area really is empty, press Shift+B to force.", False)
            return status()
    for k, (f, _, _) in shots.items():
        d = DATA / k; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "background.png"), f); S.bg[k] = f
    S.say(f"empty background saved for: {', '.join(cams)}"); return status()


@app.post("/background/clear")
def clear_background():
    for k in CAMS:
        (DATA / k / "background.png").unlink(missing_ok=True); S.bg[k] = None
    S.say("backgrounds cleared: using colour detection until you press B with the area empty"); return status()


# ---------------------------------------------------------------- front depth: close range, auto-tune
TUNE = {"running": False, "last": None}
TUNE_LOCK = threading.Lock()
JOBS = {}          # set in __main__: {"arm": ARM, "collector": ..., "sorter": ...} (read-only use: their busy flags)


def _front():
    c = S.cams.get("front") if S else None
    return c if c is not None and hasattr(c, "depth_info") and c.depth_info() else None


def _busy():
    """Names of jobs that move the arm or take photos right now (an auto-tune or depth reset must not overlap)."""
    out = []
    for name, key, flag in (("an arm routine", "arm", "running"), ("collection", "collector", "active"), ("the sorter", "sorter", "active")):
        o = JOBS.get(key)
        if o is not None and getattr(o, flag, False): out.append(name)
    return out


def _front_depth_cfg():
    v = load_settings().get("front_depth")
    return dict(v) if isinstance(v, dict) else {}


BG_CACHE = {"key": None, "depth": None, "usable": False, "warning": None}


def _bg_path(desc):
    return SETTINGS.parent / (desc.get("file") or "front_depth_bg.png")


def sync_inhand():
    """Give prep.inhand_mask the station's search area (front detection area), colour focal length, the held-object
    distance limit, the held-brick distance found by the auto-tune and the empty-scene depth background (only if it
    matches the running depth config, rs_depth.background_status). Every in-hand mask (training crops, live sorting,
    the live view) uses these."""
    if S is None: return
    c = _front(); di = c.depth_info() if c else {}
    raw = _front_depth_cfg(); cfg = rs_depth.resolve_config(raw)
    prep.configure_inhand(roi=list(S.roi["front"]), fx=float(di.get("fx_color") or 1380.0), expect_z_mm=cfg.get("expect_z_mm"),
                          max_obj_mm=cfg.get("max_obj_mm"))
    desc = raw.get("background") if isinstance(raw.get("background"), dict) else None
    usable, warn = rs_depth.background_status(desc, di)
    bg = None
    if usable:
        f = _bg_path(desc)
        try:
            key = (str(f), f.stat().st_mtime)
            if BG_CACHE["key"] != key:
                arr = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if arr is None or arr.dtype != np.uint16: raise ValueError("not a 16-bit depth PNG")
                BG_CACHE.update(key=key, depth=arr)
            bg = BG_CACHE["depth"]
        except Exception as e:
            usable, warn = False, f"depth background file {f.name} unreadable ({e}): store it again"
    if warn and usable is False and desc: print(f"!! front depth background: {warn}")
    BG_CACHE.update(usable=usable, warning=warn)
    prep.set_background(bg, {**desc, "file": _bg_path(desc).name} if usable else None)


def _bg_state():
    raw = _front_depth_cfg(); desc = raw.get("background") if isinstance(raw.get("background"), dict) else None
    out = {"exists": bool(desc) and _bg_path(desc).exists(), "usable": BG_CACHE["usable"], "warning": BG_CACHE["warning"]}
    if desc:
        out.update({k: desc.get(k) for k in ("taken", "width", "height", "shifts", "frames", "valid_frac", "file")})
        try: out["age_s"] = round(time.time() - datetime.fromisoformat(desc["taken"]).timestamp())
        except Exception: out["age_s"] = None
    return out


def _claim():
    """Exclusive use of the front depth window (auto-tune, background): -> (ok, release fn or error message).
    Refused while an arm routine, collection or the sorter runs; holds the arm's busy flag (so none of them can
    start) and prep.TUNING (photos taken anyway are not masked) until release()."""
    if not TUNE_LOCK.acquire(blocking=False): return False, "the depth auto-tune or background capture is already running"
    busy = _busy()
    if busy: TUNE_LOCK.release(); return False, f"busy: {', '.join(busy)} is running. Stop it first."
    arm = JOBS.get("arm"); claimed = arm is not None
    if claimed: arm.running = True
    TUNE["running"] = True; prep.TUNING = True

    def release():
        TUNE["running"] = False; prep.TUNING = False
        if claimed: arm.running = False
        TUNE_LOCK.release()
    return True, release


def _probe_mask(d):
    return prep.inhand_mask(d, expect_z_mm=None)      # the auto-tune must not be biased by the previous result


def _fresh_depths(c, n, timeout=4.0):
    """n depth frames produced after now (the first one after a table change is skipped: it may be in flight)."""
    time.sleep(0.3); n0 = c.depth_n; last = n0; out = []; t0 = time.time()
    while len(out) < n and time.time() - t0 < timeout:
        if c.depth_n != last:
            last = c.depth_n
            if last > n0 + 1 and c.depth is not None: out.append(c.depth)
        time.sleep(0.02)
    return out


def _has_intrinsics(di):
    return bool(di.get("fx_depth")) and bool(di.get("baseline_mm"))


@app.get("/depth/status")
def depth_status():
    """Front depth: resolution, disparity shift, measurable window, live valid fraction in the detection area and the
    held cluster's distance; mask_ok/mask_why = what an in-hand photo taken now would get."""
    c = _front()
    if c is None: return {"available": False, "msg": "front depth camera not available"}
    dep = c.depth if c.depth_t and time.time() - c.depth_t < 1.0 else None
    st = rs_depth.measure(dep, S.roi["front"], _probe_mask)
    m, why = prep.inhand_mask(dep) if dep is not None else (None, "no fresh depth")
    return {"available": True, **c.depth_info(), "roi": S.roi["front"],
            "age_s": round(time.time() - c.depth_t, 2) if c.depth_t else None,
            "valid_roi": st["valid_roi"], "z_mm": st["z_mm"], "object_px": st["object_px"],
            "mask_ok": m is not None, "mask_why": None if m is not None else why,
            "inhand": dict(prep.INHAND), "tuning": TUNE["running"], "last_tune": TUNE["last"], "busy": _busy(),
            "background": _bg_state()}


@app.post("/depth/autotune")
def depth_autotune(shifts: str = "", frames: int = 3, no_bg: bool = False):
    """Hold a brick at a WP2 photo pose, then call this: sweeps disparity shifts live, measures the held cluster
    (pixels with real depth, its distance) and the valid fraction in the front detection area per shift, keeps the
    middle of the best plateau of the NEAREST cluster (rs_depth.pick_shift), stores shift + cluster distance in
    cam_settings.json ("front_depth") and applies it. shifts="0,40,80" overrides the sweep.
    Exclusive: refused while an arm routine, collection or the sorter runs; while it runs it holds the arm's busy flag
    (so none of them can start) and prep.TUNING (photos taken anyway are not masked)."""
    c = _front()
    if c is None: return {"ok": False, "msg": "front depth camera not available"}
    di = c.depth_info()
    if not di.get("enabled"):
        return {"ok": False, "msg": f"front depth stream is off: {di.get('warning') or 'see the restart log'}", "depth": di}
    if not di.get("advanced"):
        return {"ok": False, "msg": "advanced mode is off on the front camera (see warning): can't change the disparity "
                                    "shift. Check the log of the last restart.", "depth": di}
    if not _has_intrinsics(di):
        return {"ok": False, "msg": "depth intrinsics/baseline unknown (see warning): can't compute the depth window", "depth": di}
    fx, B = di["fx_depth"], di["baseline_mm"]
    try: cands = sorted({int(min(rs_depth.SHIFT_MAX, max(0, int(x)))) for x in shifts.split(",") if x.strip()})
    except ValueError: return {"ok": False, "msg": "shifts must be integers, e.g. 0,40,80"}
    cands = cands or rs_depth.candidate_shifts(fx, B)
    if not BG_CACHE["usable"] and not no_bg:      # without it the table in front of the camera can win the sweep
        return {"ok": False, "msg": "store a depth background first (empty scene, arm parked away from WP2): "
                                    + (BG_CACHE["warning"] or "none usable"), "background": _bg_state()}
    ok, release = _claim()
    if not ok: return {"ok": False, "msg": release + (" Hold the brick at WP2 with nothing running, then tune." if "busy" in release else "")}
    try:
        orig, roi, results = di["shift"], list(S.roi["front"]), []
        try:
            for sh in cands:
                got = c.set_shift(sh)
                ms = [rs_depth.measure(d, roi, _probe_mask) for d in _fresh_depths(c, max(1, min(10, frames)))]
                if not ms: results.append({"shift": got, **rs_depth.window_info(fx, B, got), "masked": False, "why": "no depth frames"}); continue
                r = sorted(ms, key=lambda x: x["object_px"])[len(ms) // 2]          # median frame by score
                results.append({**r, **rs_depth.window_info(fx, B, got)})
            best = rs_depth.pick_shift(results)
            if best is None:
                c.set_shift(orig); _fresh_depths(c, 1)
                out = {"ok": False, "msg": "no disparity shift gave a usable in-hand mask: hold a brick at a WP2 photo pose "
                                           f"(in the front detection area) and try again. Shift back to {orig}.",
                       "results": results, "depth": c.depth_info()}
            else:
                c.set_shift(best["shift"]); _fresh_depths(c, 1)          # status/photos right after see the new window
                cfg = _front_depth_cfg()
                cfg.update({"width": di["width"], "height": di["height"], "shift": best["shift"], "expect_z_mm": best["z_mm"],
                            "tuned_at": datetime.now().isoformat(timespec="seconds"),
                            "tuned_near_mm": best["near_mm"], "tuned_far_mm": best["far_mm"]})
                store_settings("front_depth", cfg)
                c.dinfo["expect_z_mm"] = best["z_mm"]; sync_inhand()
                out = {"ok": True, "msg": f"depth tuned: shift {best['shift']} measures {best['near_mm']}-{best['far_mm'] or 'inf'} mm, "
                                          f"held cluster at {best['z_mm']} mm, {best['valid_roi']:.0%} of the detection area has depth",
                       "best": best, "results": results, "depth": c.depth_info()}
        except Exception as e:
            try: c.set_shift(orig)
            except Exception: pass
            out = {"ok": False, "msg": f"auto-tune failed: {e}", "results": results}
    finally:
        release()
        sync_inhand()                                 # the shift may have changed: re-check the background
    TUNE["last"] = {k: out.get(k) for k in ("ok", "msg", "best")}
    S.say(out["msg"], out["ok"])
    return out


@app.post("/depth/background")
def depth_background(frames: int = 20):
    """Store the EMPTY-SCENE front depth (park the arm away from WP2, nothing held, nobody leaning in): per-pixel
    median of `frames` aligned depth frames. With advanced mode it is taken at every auto-tune shift and merged, so it
    stays valid after a tune; without, only at the current shift. Saved as a 16-bit PNG next to cam_settings.json,
    described in cam_settings "front_depth"/"background". In-hand masks then only keep what is nearer than it."""
    c = _front()
    if c is None: return {"ok": False, "msg": "front depth camera not available"}
    di = c.depth_info()
    if not di.get("enabled"): return {"ok": False, "msg": f"front depth stream is off: {di.get('warning') or 'see the restart log'}"}
    n = max(5, min(60, int(frames)))
    sweep = (rs_depth.candidate_shifts(di["fx_depth"], di["baseline_mm"]) if di.get("advanced") and _has_intrinsics(di)
             else [int(di.get("shift") or 0)])
    ok, release = _claim()
    if not ok: return {"ok": False, "msg": release + (" Park the arm away from WP2 first." if "busy" in release else "")}
    orig = int(di.get("shift") or 0); layers = []
    try:
        for sh in sweep:
            if di.get("advanced"): c.set_shift(sh)
            fr = _fresh_depths(c, n, timeout=n / 5.0 + 3.0)
            if len(fr) >= max(3, n // 2): layers.append(prep.median_depth([prep.small_depth(x) for x in fr]))
        if not layers: out = {"ok": False, "msg": "no depth frames arrived: is the front camera running?"}
        else:
            bg = prep.merge_backgrounds(layers)
            desc = {"file": "front_depth_bg.png", "taken": datetime.now().isoformat(timespec="seconds"),
                    "width": di["width"], "height": di["height"], "shifts": sorted(set(sweep)), "frames": n,
                    "size": [int(bg.shape[1]), int(bg.shape[0])], "scale": prep.MASK_SCALE,
                    "fx_color": di.get("fx_color"), "valid_frac": round(float((bg > 0).mean()), 3)}
            f = _bg_path(desc); tmp = f.with_name(f.stem + ".tmp.png")
            if not cv2.imwrite(str(tmp), bg): raise RuntimeError(f"could not write {tmp}")
            tmp.replace(f)
            cfg = _front_depth_cfg(); cfg["background"] = desc; store_settings("front_depth", cfg)
            note = ("It covers every auto-tune shift, so it stays valid after a tune." if len(desc["shifts"]) > 1 else
                    "Taken at the current shift only (no advanced mode): store it again after changing the shift.")
            out = {"ok": True, "msg": f"depth background stored ({desc['valid_frac']:.0%} of the frame has depth). {note} "
                                      "Store it again if the camera, table or plates move.", "background": desc}
    except Exception as e:
        out = {"ok": False, "msg": f"background capture failed: {e}"}
    finally:
        try:
            if di.get("advanced") and c.depth_info().get("shift") != orig: c.set_shift(orig); _fresh_depths(c, 1)
        except Exception: pass
        release()
        sync_inhand()
    S.say(out["msg"], out["ok"])
    return {**out, "status": _bg_state()}


@app.post("/depth/background/clear")
def depth_background_clear():
    if TUNE["running"]: return {"ok": False, "msg": "depth auto-tune running"}
    cfg = _front_depth_cfg(); cfg.pop("background", None); store_settings("front_depth", cfg)
    sync_inhand(); S.say("depth background cleared")
    return {"ok": True, "status": _bg_state()}


@app.post("/depth/reset")
def depth_reset():
    """Forget the auto-tune (shift from "near_mm" or 0 again, no expected distance). For a re-taught WP2.
    Refused while the auto-tune, an arm routine, collection or the sorter runs (it changes the live depth window)."""
    if TUNE["running"]: return {**depth_status(), "ok": False, "msg": "depth auto-tune running"}
    busy = _busy()
    if busy: return {**depth_status(), "ok": False, "msg": f"busy: {', '.join(busy)} is running"}
    c = _front()
    cfg = _front_depth_cfg()
    for k in ("shift", "expect_z_mm", "tuned_at", "tuned_near_mm", "tuned_far_mm"): cfg.pop(k, None)
    store_settings("front_depth", cfg)
    if c is not None:
        di = c.depth_info(); c.dinfo["expect_z_mm"] = None
        if di.get("advanced"):
            c.set_shift(rs_depth.initial_shift(rs_depth.resolve_config(cfg), di.get("fx_depth"), di.get("baseline_mm"))[0])
            _fresh_depths(c, 1)
    sync_inhand(); S.say("front depth tuning reset")
    return {**depth_status(), "ok": True}


@app.post("/save/{label}")
def save(label: str, brick: str = ""):
    """ALWAYS saves. Every enabled camera: full raw frame + crop of the most central detected brick
    (or, if no brick is found, the whole detection area). Nothing is ever refused; problems are only noted."""
    if label not in LABELS: S.say("bad label", False); return status()
    cams = [k for k in S.live if S.enabled[k]] or list(S.live)
    shots = S.averaged(cams)
    for k in cams:                     # a camera that delivered no fresh frames: use its latest frame
        if k not in shots:
            f = S.frame(k)
            if f is not None: shots[k] = (f, f.astype(np.float32), f.astype(np.float32))
    if not shots: S.say("NOT SAVED: no camera is delivering frames", False); return status()
    with S.lock: S.shot_no = getattr(S, "shot_no", 0) + 1
    group = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    files, n, notes = [], 0, []
    for k, (f, first, last) in shots.items():
        boxes = detect(f, S.bg[k], k, S.roi[k])
        if boxes:
            b = boxes[0]; c = crop_box(f, b); found = True
        else:                          # no brick found: save the whole detection area so the shot is never lost
            x0, y0, x1, y1 = S.roi[k]; b = (x0, y0, x1 - x0, y1 - y0); c = f[y0:y1, x0:x1]; found = False
            notes.append(f"{k}: no brick found, saved whole area")
        x, y, w, h = b
        motion = float(np.abs(cv2.blur(first[y:y + h, x:x + w], (5, 5)) - cv2.blur(last[y:y + h, x:x + w], (5, 5))).mean())
        d = DATA / k / LABELS[label]; rawd = DATA / k / "raw" / LABELS[label]
        d.mkdir(parents=True, exist_ok=True); rawd.mkdir(parents=True, exist_ok=True)
        raw = rawd / f"{group}.png"; cv2.imwrite(str(raw), f); files.append(raw)
        p = d / f"{group}{'' if found else '_area'}.png"; cv2.imwrite(str(p), c); files.append(p); n += 1
        sc = 160 / c.shape[0]
        big = cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST if sc > 1 else cv2.INTER_AREA)
        S.recent = ([(f"{k} · {LABELS[label]}", cv2.imencode(".png", big)[1].tobytes())] + S.recent)[:9]
        with open(DATA / "meta.jsonl", "a") as m:
            m.write(json.dumps({"file": str(p.relative_to(DATA)), "raw": str(raw.relative_to(DATA)), "cam": k,
                                "group": group, "label": LABELS[label], "brick": brick, "bbox": list(b),
                                "brick_found": found, "n_bricks": len(boxes), "motion": round(motion, 2),
                                "crop_px": list(c.shape[:2]), "sharpness": round(sharpness(c), 1),
                                "frames_averaged": AVG[k], "roi": S.roi[k], "time": group,
                                "cam_settings": S.cams[k].settings}) + "\n")
    S.saved.append((files, n))
    S.say(f"✓ SAVED shot #{S.shot_no} ({LABELS[label]}) from {', '.join(shots)}" + (f" · brick {brick}" if brick else "")
          + (f"  ·  note: {'; '.join(notes)}" if notes else ""), True)
    return status()


# ---------------------------------------------------------------- auto capture
def brick_pose(img, bg, box):
    """(cx, cy, angle_deg, symmetry_deg) of the brick inside `box`, from its outline (minAreaRect).
    Robust to noise/shadows/lighting because it only looks at the brick's geometry."""
    x, y, w, h = box; m = 12
    x0, y0 = max(0, x - m), max(0, y - m); x1, y1 = min(img.shape[1], x + w + m), min(img.shape[0], y + h + m)
    c = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.int32)
    if bg is not None:
        ref = cv2.cvtColor(bg[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.int32)
    else:                      # table colour = median of the crop border
        border = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
        ref = np.median(border, 0)
    mk = (np.abs(c - ref).sum(-1) > 40).astype(np.uint8)
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts: return (x + w / 2, y + h / 2, 0.0, 90.0)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    sym = 90.0 if max(rw, rh) / max(1.0, min(rw, rh)) < 1.25 else 180.0   # square brick looks the same every 90 deg
    return (cx + x0, cy + y0, ang % sym, sym)


def pose_change(a, b):
    """(pixel shift, rotation in degrees) between two poses, respecting the brick's rotational symmetry."""
    d = abs(a[2] - b[2]) % a[3]
    return float(np.hypot(a[0] - b[0], a[1] - b[1])), float(min(d, a[3] - d))


class Auto:
    """Guided capture. Each tick (~7/s) it measures the brick's pose on the reference camera (overhead).
    A shot is taken when (1) the brick count is right, (2) the pose has been stable for STILL_FOR seconds
    (hands out, nothing moving) and (3) the pose differs from the previous shot by >= MOVE_PX or MOVE_DEG.
    SPACE forces the next shot as soon as the brick is still (escape hatch when the pose check disagrees)."""
    STILL_FOR = 0.8      # s
    STILL_PX, STILL_DEG = 3.0, 3.0      # jitter allowed while "still"
    MOVE_PX, MOVE_DEG = 12.0, 10.0      # needed between shots (top cam: 12 px ~ 6 mm)

    def __init__(self):
        self.running, self.label, self.brick, self.target, self.n = False, "good", "", 0, 0
        self.phase, self.shot_id, self.force, self.hint = "idle", 0, False, ""
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, label, target, brick):
        self.label, self.target, self.brick, self.n = label, target, brick, 0
        self.last, self.hist, self.move_since, self.force, self.hint = None, [], None, False, ""
        self.still_since = None
        self.running = True; self.phase = "place"

    def stop(self, why="stopped"):
        self.running = False; self.phase = why; self.force = False

    def _tick(self):
        cams = S.active()
        if not cams: self.stop("no camera enabled"); return
        ref = "top" if "top" in cams else cams[0]
        f = S.frame(ref)
        if f is None: return
        boxes = detect(f, S.bg[ref], ref, S.roi[ref])
        ok_count = len(boxes) >= 1 if self.label == "good" else len(boxes) == 1
        if not ok_count:
            self.hist = []; self.move_since = None; self.still_since = None
            self.phase = "place" if not boxes else "one"; self.hint = ""; return
        pose = brick_pose(f, S.bg[ref], boxes[0])
        now = time.time()
        self.hist = [(t, p) for t, p in self.hist if now - t <= self.STILL_FOR] + [(now, pose)]
        jitter = [pose_change(pose, p) for _, p in self.hist]
        if any(dp > self.STILL_PX or da > self.STILL_DEG for dp, da in jitter):
            self.hist = [(now, pose)]; self.still_since = now; self.phase = "moving"; self.move_since = None; return
        self.still_since = self.still_since or now
        if now - self.still_since < self.STILL_FOR:
            self.phase = "hold"; return
        if self.last is not None and not self.force:
            dp, da = pose_change(pose, self.last)
            if dp < self.MOVE_PX and da < self.MOVE_DEG:
                self.move_since = self.move_since or now
                self.phase = "move"
                self.hint = (f"moved {dp:.0f} px / rotated {da:.0f}\u00b0 since last shot; need {self.MOVE_PX:.0f} px "
                             f"or {self.MOVE_DEG:.0f}\u00b0. Press SPACE to shoot anyway") if now - self.move_since > 4 else ""
                return
        self.phase = "capturing"; self.hint = ""
        save(self.label, self.brick)
        if S.ok and S.msg.startswith("saved"):
            self.n += 1; self.shot_id += 1; self.last = pose; self.force = False
            self.hist = []; self.move_since = None; self.still_since = None; self.phase = "move"
            if self.n >= self.target: self.stop("done")
        else:
            self.hist = []; self.still_since = None; self.phase = "retry"; self.hint = S.msg

    def _loop(self):
        while True:
            time.sleep(0.15)
            if not self.running or S is None: continue
            try:
                self._tick()
            except Exception as e:      # never let one bad frame kill the auto loop
                self.phase, self.hint = "retry", f"internal error: {e}"
                print("auto tick error:", repr(e))


A: Auto = None


# removed: ("/auto/start")
def auto_start(label: str = "good", target: int = 60, brick: str = ""):
    if label not in LABELS: S.say("bad label", False); return status()
    A.start(label, max(1, min(500, target)), brick)
    S.say(f"auto capture started: {target} x {LABELS[label]}" + (f" (brick {brick})" if brick else "")); return status()


# removed: ("/auto/shoot")
def auto_shoot():
    if A.running: A.force = True; S.say("next shot forced: hold still")
    else: S.say("auto capture is not running", False)
    return status()


# removed: ("/auto/stop")
def auto_stop():
    A.stop(); S.say(f"auto capture stopped after {A.n} shot(s)"); return status()


@app.post("/undo")
def undo():
    if not S.saved: S.say("nothing to undo", False); return status()
    files, n = S.saved.pop()
    for p in files: p.unlink(missing_ok=True)
    names = {str(p.relative_to(DATA)) for p in files}
    mp = DATA / "meta.jsonl"
    if mp.exists():
        keep = [l for l in mp.read_text().splitlines() if l.strip() and json.loads(l)["file"] not in names]
        mp.write_text("".join(l + "\n" for l in keep))
    S.recent = S.recent[n:]
    S.say(f"undid {n} crop(s)"); return status()


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Brick Capture</title><style>
:root{--bg:#121417;--panel:#1b1f24;--line:#2c323a;--text:#e8eaed;--muted:#9aa3ad;--good:#2fa36b;--bad:#d2582a;--ok:#5fd38d;--err:#ff6b6b}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:12px 16px}
button{font:600 15px system-ui;padding:9px 13px;border-radius:7px;border:0;cursor:pointer;color:#fff;background:#3d4854}
button.g{background:var(--good)}button.d{background:var(--bad)}button:active{transform:translateY(1px)}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;position:sticky;top:0;background:var(--bg);padding:4px 0;z-index:2}
.autobar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:6px}.autobar input{width:80px;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:8px;font-size:15px}
#banner{display:none;font:800 34px system-ui;text-align:center;padding:18px;border-radius:10px;margin-top:8px;color:#111}
.bar input{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:8px;width:150px;font-size:15px}
#msg{font-size:17px;font-weight:600;padding:8px 12px;border-radius:7px;margin:8px 0;background:var(--panel)}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
.card.off{opacity:.55}.head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px}
.head b{font-size:16px;text-transform:uppercase;letter-spacing:.05em}.tag{font-size:12px;color:var(--muted)}
.tag.warn{color:#ffb44d}label{font-size:14px;cursor:pointer}
.wrap{position:relative;user-select:none}.pv{width:100%;display:block;border-radius:6px;cursor:crosshair}
.drag{position:absolute;border:2px dashed #ffd23f;pointer-events:none;display:none}
.zrow{display:flex;gap:10px;align-items:flex-start;margin-top:8px}.zoom{width:150px;height:150px;border-radius:6px;image-rendering:pixelated;background:#222}
.zrow .tag a{color:#8ab4f8}
.bottom{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:12px;margin-top:12px}@media(max-width:1000px){.bottom{grid-template-columns:1fr}}
h3{margin:2px 0 8px;font-size:13px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
.pb{height:5px;background:#2a3038;border-radius:3px}.pb>i{display:block;height:5px;border-radius:3px;background:var(--good)}
#recent{display:flex;flex-wrap:wrap;gap:6px}#recent figure{margin:0;font-size:11px;color:var(--muted);text-align:center}
#recent img{width:100px;height:100px;image-rendering:pixelated;border-radius:4px;display:block}
.flash{animation:fl .4s}@keyframes fl{from{background:#2f6}to{}}
</style></head><body>
<div class=bar>
<button onclick="p('/background')">[B] empty background</button>
<button class=g onclick="sv('good')" style="font-size:20px;padding:14px 26px">[G] CAPTURE GOOD</button><button class=d onclick="sv('defect')" style="font-size:20px;padding:14px 26px">[D] CAPTURE DEFECT</button><button onclick="p('/undo')">[U] undo</button>
<a href="/arm" style="color:#8ab4f8;font-weight:600;margin-left:8px">🤖 arm auto-capture →</a>
<input id=brick placeholder="brick id (optional)" title="e.g. blue-3: keeps all shots of one defect brick on the same side of the train/test split">
</div>

<div id=msg>loading…</div>
<div class=cams id=cams></div>
<div class=bottom>
 <div class=card><h3>progress (crops per camera)</h3><table id=cnt></table></div>
 <div class=card><h3>last saved</h3><div id=recent></div></div>
</div>
<script>
const $=id=>document.getElementById(id);let busy=false,built=false;
let q=Promise.resolve(),pending=0;   // every click is queued and executed in order: no click is ever dropped
function p(u){pending++;busy=true;$('msg').textContent='capturing… hold still ('+pending+' queued)';$('msg').style.color='var(--muted)';
 q=q.then(()=>fetch(u,{method:'POST'}).then(r=>r.json()).then(show).catch(e=>{$('msg').textContent='ERROR: '+e;$('msg').style.color='var(--err)'}))
  .finally(()=>{pending--;busy=pending>0})}
function as(l){p('/auto/start?label='+l+'&target='+($('an').value||60)+'&brick='+encodeURIComponent($('brick').value.trim()))}
let lastShot=-1,ac=null;function beep(f,d){try{ac=ac||new AudioContext();const o=ac.createOscillator(),g=ac.createGain();o.frequency.value=f;o.connect(g);g.connect(ac.destination);g.gain.value=.15;o.start();o.stop(ac.currentTime+d)}catch(e){}}
const PH={place:['PLACE THE BRICK in the yellow box','#ffd23f'],one:['ONLY ONE BRICK for defect shots','#ff8a3d'],
 moving:['HANDS OUT, keep it still…','#ffd23f'],hold:['HOLD STILL…','#ffd23f'],capturing:['CAPTURING, hold still','#ffd23f'],
 move:['✓ SHOT TAKEN: now MOVE (≥1 cm) or ROTATE (≥15°) the brick','#5fd38d'],retry:['NOT SAVED: see message, adjust','#ff6b6b'],
 done:['✓ DONE: all shots taken','#5fd38d']};
function banner(a){const b=$('banner');const ph=PH[a.phase];
 if(a.shot_id!==lastShot){if(lastShot>=0&&a.shot_id>lastShot)beep(a.phase=='done'?1320:880,a.phase=='done'?.4:.12);lastShot=a.shot_id}
 if(!a.running&&a.phase!='done'){b.style.display='none';return}
 b.style.display='block';b.style.background=ph?ph[1]:'#888';b.innerHTML=(ph?ph[0]:a.phase)+'   ('+a.n+' / '+a.target+' '+a.label+')'+(a.hint?'<div style="font-size:16px;font-weight:600;margin-top:6px">'+a.hint.replace(/</g,'&lt;')+'</div>':'')}
function sv(l){p('/save/'+l+'?brick='+encodeURIComponent($('brick').value.trim()))}
function build(s){$('cams').innerHTML=s.cams.map(c=>`<div class=card id=card-${c}>
 <div class=head><b>${c}</b><span class=tag id=tag-${c}></span>
 <label><input type=checkbox id=en-${c} onchange="p('/enable/${c}?on='+this.checked)"> capture</label></div>
 <div class=wrap><img class=pv id=pv-${c} src="/snap/${c}"><div class=drag id=dr-${c}></div></div>
 <div class=zrow><img class=zoom id=zm-${c} src="/snapzoom/${c}"><div class=tag>Live full-res zoom of the green brick.<br>
 Yellow = detection area: drag on the image to redraw it (<a href="#" onclick="p('/roi/${c}/reset');return false">reset</a>).</div></div></div>`).join('');
 s.cams.forEach(dragger);s.cams.forEach(c=>{live('pv-'+c,'/snap/'+c,120);live('zm-'+c,'/snapzoom/'+c,250)});built=true}
function live(id,url,ms){const im=$(id);const next=()=>setTimeout(()=>{im.src=url+'?t='+Date.now()},ms);im.onload=next;im.onerror=()=>setTimeout(next,1000)}
function show(s){if(!built)build(s);const m=$('msg');m.textContent=s.msg;m.style.color=s.ok?'var(--ok)':'var(--err)';
 if(s.ok&&s.msg.startsWith('✓')){m.classList.remove('flash');void m.offsetWidth;m.classList.add('flash')}
 for(const c of s.cams){const live=s.live.includes(c),t=$('tag-'+c);
  $('en-'+c).checked=s.enabled[c];$('en-'+c).disabled=!live;$('card-'+c).classList.toggle('off',!s.enabled[c]);
  const st=s.has_bg[c]&&s.stale[c]>s.stale_max;
  const fr=s.age[c]!=null&&s.age[c]>2;t.textContent=fr?'FROZEN: no new frame for '+s.age[c]+' s (auto-reconnecting)':!live?'offline':(!s.has_bg[c]?'NO BACKGROUND':st?'BACKGROUND OUTDATED: clear the area, press B':'background set')+' · '+s.n_boxes[c]+' brick(s)';
  t.classList.toggle('warn',fr||!live||!s.has_bg[c]||st||s.n_boxes[c]!=1)}
 let h='<tr><th></th>'+s.cams.map(c=>'<th>'+c+'</th>').join('')+'</tr>';
 for(const k in s.target){h+='<tr><td>'+k+'</td>';for(const c of s.cams){const n=s.counts[c][k],t=s.target[k];
  h+='<td>'+n+' / '+t+'<div class=pb><i style="width:'+Math.min(100,100*n/t)+'%"></i></div></td>'}h+='</tr>'}
 $('cnt').innerHTML=h;
 $('recent').innerHTML=s.recent.map((l,i)=>'<figure><img src="/recent/'+i+'?'+Date.now()+'">'+l+'</figure>').join('')}
const K={b:'/background',u:'/undo'},L={g:'good',d:'defect'};
document.onkeydown=e=>{if(e.target.tagName=='INPUT'&&e.target.type!='checkbox')return;const k=e.key.toLowerCase();if(e.key=='B'){p('/background?force=true');return}if(K[k])p(K[k]);else if(L[k])sv(L[k])};
function dragger(c){let st=null;const pv=$('pv-'+c),dr=$('dr-'+c);
 const pos=e=>{const r=pv.getBoundingClientRect();return[Math.max(0,Math.min(r.width,e.clientX-r.left)),Math.max(0,Math.min(r.height,e.clientY-r.top)),r]};
 pv.addEventListener('mousedown',e=>{e.preventDefault();st=pos(e)});
 window.addEventListener('mousemove',e=>{if(!st)return;const[x,y]=pos(e);Object.assign(dr.style,{display:'block',left:Math.min(x,st[0])+'px',top:Math.min(y,st[1])+'px',width:Math.abs(x-st[0])+'px',height:Math.abs(y-st[1])+'px'})});
 window.addEventListener('mouseup',e=>{if(!st)return;const[x,y,r]=pos(e),sx=1920/r.width,sy=1080/r.height;dr.style.display='none';
  const a=st;st=null;if(Math.abs(x-a[0])<8||Math.abs(y-a[1])<8)return;
  p(`/roi/${c}?x0=${Math.round(Math.min(x,a[0])*sx)}&y0=${Math.round(Math.min(y,a[1])*sy)}&x1=${Math.round(Math.max(x,a[0])*sx)}&y1=${Math.round(Math.max(y,a[1])*sy)}`)})}
fetch('/status').then(r=>r.json()).then(show);
setInterval(()=>{if(!busy)fetch('/status').then(r=>r.json()).then(show)},400);
</script></body></html>"""


@app.get("/manual")
def index():
    return HTMLResponse(PAGE)          # old manual-capture page (kept, not linked); the main UI is / (ui.py)


@app.get("/")
def root():
    import ui                          # the user-facing page (web/operator.html); the dev interface is /dev and the page paths
    return ui.operator()


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    S = State()
    sync_inhand()
    A = None   # auto-capture removed: capture only happens on a click/key press

    def say_recent(caption, img):
        sc = 160 / img.shape[0]
        S.recent = ([(caption, cv2.imencode(".png", cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))[1].tobytes())]
                    + S.recent)[:9]
    try:
        import arm
        router, ARM, ACTX = arm.make_router(S, DATA, AVG, say_recent)
        app.include_router(router)
        import collect, ui
        app.include_router(ui.router)
        crouter = collect.make_router(S, ARM, ACTX, DATA, ROOT)
        app.include_router(crouter)
        import sorter
        srouter = sorter.make_router(S, ARM, ACTX, crouter.collector, DATA, ROOT)
        app.include_router(srouter)
        JOBS.update(arm=ARM, collector=crouter.collector, sorter=srouter.sorter)    # depth auto-tune exclusivity
        import feed                                                                 # hand-feed anomaly capture (/anomaly)
        frouter = feed.make_router(S, ARM, ACTX, crouter.collector, DATA, JOBS)
        app.include_router(frouter)
        import autocollect                                                          # automatic pick -> WP2 photos -> put back
        acrouter = autocollect.make_router(S, ARM, ACTX, crouter.collector, srouter.sorter, frouter.feeder, DATA)
        app.include_router(acrouter)
        import flow                                                                 # production sort flow (/flow/*)
        app.include_router(flow.make_router(S, ARM, ACTX, crouter.collector, srouter.sorter, frouter.feeder, acrouter.autocollect, DATA))
        if MOCK: mock_hw.attach(app, S, ARM, crouter.collector, srouter.sorter)
        try:
            import telemetry                                                        # live power / temperature / device load (/telemetry)
            app.include_router(telemetry.make_router())
        except Exception as e:
            print("telemetry not loaded:", e)
        print(f"control UI: http://127.0.0.1:{PORT}/  (steps: /connect /waypoints /crop /record /episodes /train /anomaly /sort)")
    except Exception as e:
        print("arm module not loaded:", e)
    print("live cameras:", S.live)
    print(f"open http://127.0.0.1:{PORT}" + ("   (MOCK STATION, no hardware)" if MOCK else ""))
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", timeout_graceful_shutdown=1)  # live streams never end on their own
