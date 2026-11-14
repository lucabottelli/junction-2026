"""Anomaly dataset capture tool (Junction 2026, SO-101 brick sorting) - multi-camera.

Browser UI at http://127.0.0.1:7870 — live preview of all cameras (top, front, wrist), auto brick
detection per camera, live full-res zoom, and one key press saves lossless full-res crops from every
enabled camera at the same moment (linked by a shared "group" id in meta.jsonl).

Run (Studio's Environments preview must be CLOSED so the cameras are free):
    ~/physical-ai-studio/application/backend/.venv/bin/python ~/junction26/anomaly_capture/capture.py

Keys: B = empty background (all enabled cams; refused if a brick is in the area, Shift+B forces), G = good, D = defect (any size; measured later from pixels), U = undo last save.
Drag on a camera's preview to redraw its detection area (saved per camera). Tick/untick cameras to include.
"""
import json, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse

CAMS = ["top", "front", "wrist"]
RS_SERIAL = {"top": "243622063627", "front": "243622061066"}
DEPTH_CAMS = {"front"}   # aligned depth (mm) kept alongside colour for background masking
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
# thin-structure filter size at 1/4 scale: about half the brick's smallest side on that camera
# (top brick ~20 px, front ~50 px, wrist ~30+ px at 1/4 scale)
OPEN_K = {"top": 9, "front": 15, "wrist": 13}
STALE_FRAC = 0.04    # >4% of the detection area differs from the background outside the brick -> warn "press B again"


def load_settings():
    return json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}


def store_settings(key, value):
    s = load_settings(); s[key] = value; SETTINGS.write_text(json.dumps(s, indent=2))


# ---------------------------------------------------------------- cameras
class RSCam:
    """D415 colour stream at 1080p with exposure + white balance locked (calibrated once, then reused)."""

    def __init__(self, name):
        self.name = name
        self.depth, self.depth_t, self.align, self.n = None, 0.0, None, 0
        self.pipe = rs.pipeline()
        prof = None
        if name in DEPTH_CAMS:      # depth lets us grey out everything behind the held brick (people walking past)
            try:
                cfg = rs.config(); cfg.enable_device(RS_SERIAL[name])
                cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
                cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                prof = self.pipe.start(cfg); self.align = rs.align(rs.stream.color)
                self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
            except Exception as e:
                print(f"{name}: depth stream unavailable ({e}), colour only"); prof = None; self.pipe = rs.pipeline()
        if prof is None:
            cfg = rs.config(); cfg.enable_device(RS_SERIAL[name])
            cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            prof = self.pipe.start(cfg)
        self.sen = prof.get_device().first_color_sensor()
        self.settings = self._lock()

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
        if self.align is not None and self.n % 3 == 0:   # depth aligned to colour pixels, ~10 Hz (CPU)
            try:
                fa = self.align.process(f); d = fa.get_depth_frame()
                if d:
                    self.depth = np.asanyarray(d.get_data()).copy(); self.depth_t = time.time()
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
        self.preview = {k: None for k in CAMS}; self.zoomj = {k: None for k in CAMS}
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

    def averaged(self, cams):
        """Average AVG fresh frames on all given cams in the same time window -> {cam: (mean, first, last)}."""
        with self.lock:
            for k in cams: self.hist[k] = []
        t0 = time.time(); out = {}
        while len(out) < len(cams) and time.time() - t0 < 5:
            with self.lock:
                for k in cams:
                    if k not in out and len(self.hist[k]) >= AVG[k]:
                        st = np.stack(self.hist[k]).astype(np.float32)
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
            "settings": {k: (c.settings if c else None) for k, c in S.cams.items()},
}


@app.post("/enable/{cam}")
def enable(cam: str, on: bool):
    if cam in S.live: S.enabled[cam] = on
    S.say(f"{cam} {'included in' if on else 'excluded from'} captures"); return status()


@app.post("/roi/{cam}")
def set_roi(cam: str, x0: int, y0: int, x1: int, y1: int):
    x0, x1 = sorted((max(0, x0), min(1920, x1))); y0, y1 = sorted((max(0, y0), min(1080, y1)))
    if x1 - x0 < 60 or y1 - y0 < 60: S.say("NOT SET: detection area too small", False); return status()
    S.roi[cam] = [x0, y0, x1, y1]; store_settings("roi", S.roi)
    S.say(f"{cam}: detection area set to {S.roi[cam]}"); return status()


@app.post("/roi/{cam}/reset")
def reset_roi(cam: str):
    S.roi[cam] = list(DEFAULT_ROI[cam]); store_settings("roi", S.roi)
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
    return HTMLResponse(PAGE)          # old manual-capture page (kept, not linked); the main page is /arm


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/arm")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    S = State()
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
        app.include_router(collect.make_router(S, ARM, ACTX, DATA, ROOT))
        print("pick+inspect data collection: http://127.0.0.1:7870/collect")
        print("arm auto-capture: http://127.0.0.1:7870/arm")
    except Exception as e:
        print("arm module not loaded:", e)
    print("live cameras:", S.live)
    print("open http://127.0.0.1:7870")
    uvicorn.run(app, host="127.0.0.1", port=7870, log_level="warning", timeout_graceful_shutdown=1)  # live streams never end on their own
