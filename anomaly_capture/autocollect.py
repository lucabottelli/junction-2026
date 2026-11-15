"""Automatic clean-photo collection for the anomaly models (no hands): the arm feeds itself.

One cycle:  look at the plate (top camera) -> vision pick (vpick.py) of the colour that still needs the most photos
            -> WP1 -> WP2, a photo at every taught WP2 pose (same capture path, mask, per-brick lift and settle wait
            as the hand-feed page, label "good") -> back over WP1 -> set the brick down at a random free spot of the
            start plate with a random turn -> home.
Repeats until every colour on the plate has `target` good crops (training_images/front/inhand/good/*_<brick>_*_crop.png)
or `max_cycles` is reached. Stops by itself after 3 failed picks in a row, 3 handlings in a row without a usable
mask, or when a colour that still needs photos is no longer seen on the plate.

It runs as a sorter job (Sorter.active), so the top-bar STOP / Space / Esc and /sort/stop stop it like any other
routine and nothing else can start meanwhile. Log: runs/autocollect_<time>/cycles.jsonl.
record=1 (default): every pick is also saved as a training episode (sorter.VRecorder), from the start pose until the
arm is at WP1; the WP2 photos and the put-back are not in it. Episodes: episodes/<time>_vpick/.

    POST /autocollect/start?target=100&max_cycles=200&record=1      (per_block=40&since=20261115_0330: instead count
                                                 handlings PICKED BY THE ARM since that time, 40 per block)
       GET /autocollect/status     POST /autocollect/stop
"""
import json, threading, time
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter

import policy as pol
import prep
from arm import Stopped

BRICKS = ("pink", "green", "blue")
MIN_AREA = 40_000        # px: a colour mask smaller than this at a WP2 pose is not a held brick (pink: see feed.MIN_AREA)
AUTO_LIFT = {"wp2-1": 22.0, "wp2-3": 12.0}    # mm, see AutoCollect.photos
# green: its feed_lift.json entry (+28 mm at wp2-1) was tuned for the hand-fed end grip; picked off the plate, +50 mm
# in total pushed it out of the TOP of the frame at wp2-1 (2026-11-15, 4 of 4 handlings)
AUTO_LIFT_BRICK = {"green": {"wp2-1": -6.0, "wp2-3": 0.0}}    # wp2-3: +12 cut a stud-up green block at the top twice
# Inspection only: a 4th photo with the fingers pointing straight down, so the front camera (which sits higher) looks
# onto the block's TOP face, the one between the jaws that wp2-1..3 only show edge-on (2026-11-15: tape on the stud
# face of a pink block was missed twice). Model elevation -51 deg = straight down on the real arm (the taught pose
# this starts from, 2026-11-14 21:59, kept in routines.json as wp2-3's "replaced"); its grasp point is raised by up_mm so
# the hanging block is in the frame (at the taught height the frame's bottom edge cut it).
TOP_VIEW = {"name": "wp2-top", "elev_deg": -41.0, "up_mm": 40.0}    # from the hand-taught straight-down pose (wp2-3 "replaced");
                                                                    # -51 deg is only reachable up to +20 mm
# Top-camera check before the pick (2026-11-15): the face that points UP while the block lies on the plate ends up
# against the gripper and is never shown to the front camera. Limit per brick = largest patch on / next to the brick
# that is neither its colour, nor the plate's orange, nor another brick, as a share of the brick's area. Pink only so
# far: 73 clean pink views <= 0.066, two taped ones 0.13 and 0.16. For blue and green the same measure reads 0.07-0.4
# on CLEAN blocks (their shaded faces change hue), so it is off until tuned.
TOP_FOREIGN_MAX = {"pink": 0.10}
MAX_FAILS = 3            # failed picks in a row / handlings without a usable mask in a row -> stop and ask


def _hdist(H, h0):
    d = np.abs(H - h0); return np.minimum(d, 180 - d)


def top_foreign(img, brick, others, plate, grow=1.25, up=4):
    """Tape seen by the TOP camera on a brick lying on the plate. img = raw top frame; brick / others = entries of
    vpick.detect (box = 4 corners in full-frame px, size = (long, short) px); plate = (cx, cy, r) or None.
    -> largest foreign patch as a share of the brick's area, or None if the brick's colour can't be read."""
    box = np.asarray(brick["box"], np.float32); c = box.mean(0); pts = c + (box - c) * grow
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2; x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    x0, y0 = max(0, x0), max(0, y0)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    P = lambda q: ((np.asarray(q, np.float32) - [x0, y0]) * up).astype(np.int32)
    m = np.zeros(crop.shape[:2], np.uint8); cv2.fillConvexPoly(m, P(pts), 255)
    for o in others:                                      # a neighbouring brick is not tape
        ob = np.asarray(o["box"], np.float32); oc = ob.mean(0); cv2.fillConvexPoly(m, P(oc + (ob - oc) * 1.12), 0)
    if plate is not None:                                 # stay on the plate (the table is not saturated)
        pm = np.zeros_like(m); cv2.circle(pm, (int((plate[0] - x0) * up), int((plate[1] - y0) * up)), int(plate[2] * 0.93 * up), 255, -1)
        m &= pm
    inner = np.zeros_like(m); cv2.fillConvexPoly(inner, P(c + (box - c) * 0.7), 255)
    m = m > 0; hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(int); H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (inner > 0) & m & (S >= 110) & (V >= 60)
    if sat.sum() < 40: return None
    hd = int(np.median(H[sat]))                           # the brick's own hue; the plate's orange is hue ~10
    f = m & (((S < 70) & (V > 75) & (V < 235)) | ((S >= 70) & (V > 55) & (_hdist(H, hd) > 18) & (_hdist(H, 10) > 10)))
    el = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    f = cv2.morphologyEx(cv2.morphologyEx(f.astype(np.uint8), cv2.MORPH_OPEN, el(2 * up + 1)), cv2.MORPH_CLOSE, el(4 * up + 1))
    n, _, st, _ = cv2.connectedComponentsWithStats(f)
    return float(st[1:, 4].max()) / (float(brick["size"][0] * brick["size"][1]) * up * up) if n > 1 else 0.0


class AutoCollect:
    def __init__(self, S, A, ctx, C, T, F, DATA):
        self.S, self.A, self.ctx, self.C, self.T, self.F, self.DATA = S, A, ctx, C, T, F, DATA
        self.running = False; self.target = 100; self.cycles = []; self.session = None; self.start_counts = {}; self.rec_session = None; self.per_block = 0; self.since = ""; self.last_inspect = None

    def counts(self):
        d = self.DATA / "front" / "inhand" / "good"
        return {b: len(list(d.glob(f"*_{b}_*_crop.png"))) for b in BRICKS}

    def picked(self, since=""):
        """Handlings PICKED BY THE ARM per brick (meta.jsonl runs ending in _auto, capture time >= since) that gave at
        least one good crop. This is what the sorter will see, unlike the hand-fed photos."""
        h = {b: set() for b in BRICKS}; d = self.DATA
        try:
            for line in open(d / "meta.jsonl"):
                j = json.loads(line)
                if (j.get("stage") == "inhand" and j.get("label") == "good" and str(j.get("run", "")).endswith("_auto")
                        and j.get("brick") in h and j.get("time", "") >= since
                        and (d / j["file"]).with_name((d / j["file"]).stem + "_crop.png").exists()):
                    h[j["brick"]].add((j["run"], j.get("cycle")))
        except Exception: pass
        return {b: len(v) for b, v in h.items()}

    def progress(self):
        """What the run counts toward its goal: picked handlings (per_block mode) or all good crops."""
        return self.picked(self.since) if self.per_block else self.counts()

    def busy(self):
        T, C, A, F = self.T, self.C, self.A, self.F
        return bool(T.active or C.active or A.running or F.active or prep.TUNING)

    def edge_cut(self, file, top=40, bottom=1076):
        """'top' / 'bottom' if the mask of this capture reaches the frame edge there (1080 rows), else None."""
        try:
            import cv2, numpy as np
            p = self.DATA / file; m = cv2.imread(str(p.with_name(p.stem + "_masked.jpg")))
            ys = np.where((np.abs(m.astype(int) - np.array(prep.GREY)).sum(2) > 30).any(1))[0]
            s = m.shape[0] / 1080.0
            return "top" if ys.min() <= top * s else "bottom" if ys.max() >= bottom * s else None
        except Exception:
            return None

    def top_pose(self, wp21):
        """Joint ticks of the top-face view (TOP_VIEW), solved once from the hand-taught straight-down pose. Raises if
        out of reach."""
        key = tuple(sorted(wp21.items()))
        if getattr(self, "_top", (None,))[0] != key:
            import vpick
            if self.F.kin is None: self.F.kin = vpick.Kin()
            kin = self.F.kin; p = kin.fk(kin.rad(wp21))[0].copy(); p[2] += TOP_VIEW["up_mm"] / 1000.0
            t, err = kin.pose_at(wp21, p, TOP_VIEW["elev_deg"])
            if err > 0.004: raise RuntimeError(f"top-view pose out of reach ({err * 1000:.0f} mm off)")
            for n, (lo, hi) in self.A.fl.items():
                if n in t and n != "gripper" and not lo + 20 <= t[n] <= hi - 20: raise RuntimeError(f"top-view pose: {n} at its limit")
            self._top = (key, t)
        return dict(self._top[1])

    def photos(self, brick, g_lock, wp1, idx, run_id, label="good"):
        """WP1 -> WP2, one masked photo per WP2 pose, back up over WP2 and to WP1. -> list of photo records."""
        T, C, F = self.T, self.C, self.F
        import feed
        # a block picked off the plate hangs lower in the jaws than a hand-fed one (2026-11-15: the frame's bottom edge
        # cut it at wp2-1 and wp2-3), so these poses go up by AUTO_LIFT on top of the brick's own feed_lift.json entry
        taught = C.wps("wp2"); old = F.lift
        try:
            F.lift = {**old, brick: {w["name"]: float((old.get(brick) or {}).get(w["name"]) or 0) + AUTO_LIFT_BRICK.get(brick, AUTO_LIFT).get(w["name"], 0)
                                     for w in taught}}
            try: wp2 = F.lifted(taught, brick)
            except Exception as e:                       # lifted pose out of reach: the hand-feed poses
                F.lift = old; wp2 = F.lifted(taught, brick); T.say(f"auto lift not reachable ({e}): using the feed poses", False)
        finally:
            F.lift = old
        T.state = "to_wp2"; T.say(f"cycle {idx}: {brick} to WP2 (clamp locked)")
        C.to_wp2(wp1, wp2, g_lock)
        out = []
        for i, w in enumerate(wp2):
            T.check()
            if i: C.goto(w["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)
            T.state = "photo"; T.say(f"cycle {idx}: {brick} photo {i + 1}/{len(wp2)} at {w['name']}")
            F.settle(); T.check()
            prep.configure_inhand(brick=brick)
            nf = self.ctx.capture(["front"], label, brick, w["name"], idx, run_id)
            rec = next((m for m in reversed(F._last_meta(3)) if m.get("cam") == "front"), {}) if nf else {}
            dep = rec.get("depth") if isinstance(rec.get("depth"), dict) else {}
            area = dep.get("area_px")
            ok = bool(nf) and bool(dep) and (area or 0) >= max(MIN_AREA, feed.MIN_AREA.get(brick, 0))
            why = rec.get("depth_mask_why") if nf else "no fresh front frame"
            if ok:                                       # a block cut by the frame's top or bottom edge is no training photo
                cut = self.edge_cut(rec.get("file"))
                if cut: ok, why = False, f"block cut by the {cut} of the frame"
            elif why is None: why = "mask too small: no block in the jaws"
            out.append({"view": w["name"], "file": rec.get("file"), "mask_ok": bool(dep), "area_px": area, "ok": ok, "why": why})
            F._grab_last(rec)
            if not ok and rec.get("file") and label == "good":   # nothing (or not this brick) in the jaws: not a training photo
                stem = (self.DATA / rec["file"]).with_suffix("")
                for f in stem.parent.glob(stem.name + "*"):
                    try: f.unlink()
                    except OSError: pass
                out[-1]["deleted"] = True
        if label != "good":                              # inspection: also look at the top face
            try:
                seed = next((w["replaced"]["pos"] for w in taught if isinstance(w.get("replaced"), dict) and w["name"] == "wp2-3"), None)
                if seed is None: raise RuntimeError("no hand-taught straight-down pose (wp2-3 'replaced') in the routine")
                top = self.top_pose(seed)
                C.goto(wp2[0]["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)
                C.goto(top, C.cfg["speed"] * 0.4, gripper=g_lock)
                T.state = "photo"; T.say(f"cycle {idx}: {brick} photo of the top face ({TOP_VIEW['name']})")
                F.settle(); T.check(); prep.configure_inhand(brick=brick)
                nf = self.ctx.capture(["front"], label, brick, TOP_VIEW["name"], idx, run_id)
                rec = next((m for m in reversed(F._last_meta(3)) if m.get("cam") == "front"), {}) if nf else {}
                dep = rec.get("depth") if isinstance(rec.get("depth"), dict) else {}
                out.append({"view": TOP_VIEW["name"], "file": rec.get("file"), "mask_ok": bool(dep), "area_px": dep.get("area_px"),
                            "ok": bool(nf) and bool(dep), "extra": True, "why": rec.get("depth_mask_why") if nf else "no fresh front frame"})
                F._grab_last(rec)
            except Stopped: raise
            except Exception as e:
                out.append({"view": TOP_VIEW["name"], "file": None, "mask_ok": False, "ok": False, "extra": True, "why": f"top view not taken: {e}"})
        T.state = "returning"
        if len(wp2) > 1 or label != "good": C.goto(wp2[0]["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)      # leave the way it came in
        C.up_wp2(wp1, wp2, g_lock)
        C.goto(wp1, gripper=g_lock)
        return out

    def run(self, target, max_cycles, record=True, per_block=0, since=""):
        T, C, A = self.T, self.C, self.A
        self.running = True; self.cycles = []; self.per_block = per_block; self.since = since
        if per_block: target = per_block
        self.target = target; unit = "handlings picked by the arm" if per_block else "good crops"
        T.active = True; A.running = True; C.active = True
        for e in (T.stop_flag, A.stop, C.stop_flag): e.clear()
        T.session = self.session = datetime.now().strftime("autocollect_%Y%m%d_%H%M%S")
        d = T.RUNS / T.session; d.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_auto"
        self.start_counts = self.counts()
        (d / "settings.json").write_text(json.dumps({"target": target, "max_cycles": max_cycles, "run": run_id,
                                                     "counts_at_start": self.start_counts}, indent=1))
        T.trials = []; T._vdir = d; pick_fails = mask_fails = 0; n = 0; vrec = None; miss = {b: 0 for b in BRICKS}
        try:
            vp, kin, cal = T.get_vision()
            if record:                                   # every pick = one training episode, start pose -> WP1
                from sorter import VRecorder
                vrec = VRecorder(T, max_cycles, d); self.rec_session = vrec.session
            home = C.wps("home")[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]; g_open = T.g_open()
            while n < max_cycles:
                T.check()
                cnt = self.progress(); need = [b for b in BRICKS if cnt[b] < target]
                if not need:
                    T.state = "done"; T.say(f"✓ AUTO-COLLECT DONE: every block has {target}+ {unit} ({cnt})"); return
                T.state = "look"; C.goto(home, gripper=g_open)
                _, bricks, _ = vp.VisionPick(A, C, self.S, kin, cal, check=T.check, say=T.say).look()
                seen = {b["colour"] for b in bricks}
                cand = [b for b in need if b in seen]
                if not cand:
                    T.state = "help"
                    T.say(f"auto-collect stopped: {', '.join(need)} still need photos but the top camera doesn't see "
                          f"{'them' if len(need) > 1 else 'it'} on the plate (sees: {', '.join(sorted(seen)) or 'nothing'}). "
                          f"Put the block(s) on the plate and start again. Counts: {cnt}", False); return
                # a block that was missed twice in a row is probably tipped over or on the plate rim: leave it for a
                # person and go on with the other colours instead of hammering it
                skip = [b for b in cand if miss[b] >= 2]; cand = [b for b in cand if miss[b] < 2]
                if not cand:
                    T.state = "help"
                    T.say(f"auto-collect stopped: {', '.join(skip)} missed twice in a row each (tipped over or on the plate "
                          f"rim?). Stand the block(s) upright near the middle of the plate and start again. {unit}: {cnt}", False); return
                colour = min(cand, key=lambda b: cnt[b])
                n += 1; t0 = time.time(); rec = {"cycle": n, "colour": colour, "counts": cnt}
                T._vopt = {"target": colour, "retries": 1}
                if vrec:                                     # one attempt = one episode (a retry would restart the buffer)
                    vrec.begin(n); T._vopt = {"target": colour, "retries": 0, "on_event": vrec.on_event, "tuner": vrec.tuner}
                try:
                    g_lock = T.pick_vision()
                    if vrec: vrec.end(True, g_lock)          # the cut at WP1: photos and put-back are not in the episode
                except pol.PickFailed as e:
                    if vrec: vrec.end(False)
                    pick_fails += 1; miss[colour] += 1; rec.update(ok=False, error=str(e), t_s=round(time.time() - t0, 1)); self._log(d, rec)
                    if not e.retry or pick_fails >= 2 * MAX_FAILS:
                        T.state = "help"
                        T.say(f"auto-collect stopped after {pick_fails} failed pick(s) in a row: {e}", False); return
                    C.goto(dict(A.cmd), 0.4, gripper=g_open); C.rise(wp1); C.goto(home, gripper=g_open); continue
                except BaseException:
                    if vrec: vrec.end(False, discard=True)
                    raise
                pick_fails = 0; miss[colour] = 0
                brick = ((T.act.attempts or [{}])[-1]).get("brick") or colour
                ph = self.photos(brick, g_lock, wp1, n, run_id)
                good = sum(p["ok"] for p in ph)
                mask_fails = 0 if good else mask_fails + 1
                placed = None
                for k in range(3):                                   # a random spot can be out of reach: draw another
                    try: placed = T.place_and_look(n, g_lock, home, g_open, d); break
                    except pol.PickFailed as e:
                        if k == 2: raise RuntimeError(f"no reachable spot to set the {brick} block down: {e}")
                        C.goto(wp1, gripper=g_lock)
                if vrec: vrec.verify(bool(placed))
                rec.update(ok=True, brick=brick, photos=ph, good_photos=good, placed_seen=placed,
                           t_s=round(time.time() - t0, 1)); self._log(d, rec)
                T.say(f"cycle {n}: {brick} · {good}/{len(ph)} photos masked · {unit} now {self.progress()}", good == len(ph))
                if mask_fails >= MAX_FAILS:
                    T.state = "help"
                    T.say(f"auto-collect stopped: {mask_fails} handlings in a row without a usable mask (last: "
                          f"{ph[-1].get('why') or 'mask too small'}). Check the front camera view and the block.", False); return
                time.sleep(0.3)
            T.state = "done"; T.say(f"auto-collect: {max_cycles} cycles done · {unit} {self.progress()}")
        except Stopped:
            T.state = "stopped"; T.say("STOPPED: arm holds its position", False)
        except Exception as e:
            T.state = "error"; T.say(f"ERROR in auto-collect: {e} (arm holds its position)", False)
        finally:
            try: C.hold_here()
            except Exception: pass
            T.active = False; A.running = False; C.active = False; self.running = False

    def inspect(self, colour):
        """One live test of the anomaly model: pick the `colour` block, photograph it at the WP2 poses (saved under
        front/inhand/test, NOT in the training folders), score every crop with that block's model, put the block back.
        Verdict: any photo >= t_high -> ANOMALOUS; all <= t_low -> CLEAN; otherwise UNCERTAIN. -> self.last_inspect."""
        T, C, A = self.T, self.C, self.A
        self.running = True; self.last_inspect = {"colour": colour, "running": True}
        T.active = True; A.running = True; C.active = True
        for e in (T.stop_flag, A.stop, C.stop_flag): e.clear()
        T.session = datetime.now().strftime("inspect_%Y%m%d_%H%M%S"); d = T.RUNS / T.session; d.mkdir(parents=True, exist_ok=True)
        T._vdir = d; res = {"colour": colour, "session": T.session}
        try:
            model, root = T.scorer.find("front", colour); th = T.scorer.thresholds(root) or {}
            t_low, t_high = th.get("t_low"), th.get("t_high")
            if t_low is None or t_high is None: raise RuntimeError(f"no thresholds.json for the {colour} model")
            home = C.wps("home")[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]; g_open = T.g_open()
            # 1) top camera, block still on the plate: its upward face (never shown to the front camera later)
            top = {"view": "top camera", "extra": True, "ok": False, "foreign": None}
            try:
                vp, kin, cal = T.get_vision()
                T.state = "look"; C.goto(home, gripper=g_open)
                frame, bricks, plate = vp.VisionPick(A, C, self.S, kin, cal, check=T.check, say=T.say).look()
                mine = [b for b in bricks if b["colour"] == colour]
                if mine:
                    fs = top_foreign(frame, mine[0], [b for b in bricks if b is not mine[0]], plate)
                    cv2.imwrite(str(d / "top_raw.jpg"), frame)
                    top.update(ok=fs is not None, foreign=None if fs is None else round(fs, 3), limit=TOP_FOREIGN_MAX.get(colour))
            except Stopped: raise
            except Exception as e: top["why"] = f"top-camera check failed: {e}"
            T._vopt = {"target": colour, "retries": 3}        # the user's rule: a failed pick is tried again
            g_lock = T.pick_vision()
            brick = ((T.act.attempts or [{}])[-1]).get("brick") or colour
            ph = self.photos(brick, g_lock, wp1, 1, T.session, label="test")
            T.state = "judge"; T.say(f"scoring the {brick} photos")
            reasons = []
            lim = TOP_FOREIGN_MAX.get(brick)
            if top["foreign"] is not None and lim is not None and top["foreign"] >= lim:
                top["verdict"] = "anomalous"
                reasons.append(f"top camera: a patch that is not {brick} covers {top['foreign']:.0%} of the block's upward face on the plate")
            else: top["verdict"] = "clean" if (top["foreign"] is not None and lim is not None) else "not judged"
            for p in ph:
                why = str(p.get("why") or "")
                if p.get("extra"):                           # top-face view: no model for it, only the colour check counts
                    fs = prep.foreign_share(cv2.imread(str(self.DATA / p["file"])), brick) if p.get("file") and p["ok"] else None
                    p["foreign"] = None if fs is None else round(fs[0], 3)
                    if fs is not None and fs[0] >= prep.FOREIGN_MAX:
                        p["verdict"] = "anomalous"; reasons.append(f"{p['view']}: a patch that is not {brick} covers {fs[0]:.0%} of the block's top")
                    else: p["verdict"] = "clean" if fs is not None else "not seen"
                    continue
                if "cut by" in why or not p.get("file"):
                    p["verdict"] = "uncertain"; reasons.append(f"{p['view']}: {why or 'no photo'}"); continue
                if not p["ok"]:
                    # the pick was verified (the block left the plate) but its colour can't be seen at this pose:
                    # something covers it (tape over the whole face) - or the jaws hide it. Either way not clean.
                    p["verdict"] = "anomalous"; reasons.append(f"{p['view']}: the {brick} block is held but its colour is hidden (covered?)"); continue
                f = self.DATA / p["file"]; crop = f.with_name(f.stem + "_crop.png")
                fs = prep.foreign_share(cv2.imread(str(f)), brick)
                p["foreign"] = None if fs is None else round(fs[0], 3)
                r = T.scorer.score(model, crop, T.cfg["device"]); p["score"] = round(float(r["score"]), 3)
                if p["foreign"] is not None and p["foreign"] >= prep.FOREIGN_MAX:
                    p["verdict"] = "anomalous"; reasons.append(f"{p['view']}: a patch that is not {brick} covers {p['foreign']:.0%} of the block")
                elif p["score"] >= t_high:
                    # the user's rule (2026-11-15, after a clean pink block scored 0.89): never call a block anomalous
                    # when nothing can be SEEN on it. A high model score alone is only a doubt.
                    p["verdict"] = "uncertain"; reasons.append(f"{p['view']}: model score {p['score']:.2f} >= {t_high} but no visible patch")
                elif p["score"] > t_low:
                    p["verdict"] = "uncertain"; reasons.append(f"{p['view']}: model score {p['score']:.2f} between {t_low} and {t_high}")
                else: p["verdict"] = "clean"
            sc = [p["score"] for p in ph if "score" in p]; ph = [top] + ph; vs = [p["verdict"] for p in ph]
            verdict = "ANOMALOUS" if "anomalous" in vs else "UNCERTAIN" if "uncertain" in vs else "CLEAN"
            res["reasons"] = reasons
            res.update(brick=brick, verdict=verdict, t_low=t_low, t_high=t_high, max_score=max(sc) if sc else None, photos=ph)
            self.last_inspect = {**res, "running": True}
            for k in range(3):
                try: res["placed_seen"] = T.place_and_look(1, g_lock, home, g_open, d); break
                except pol.PickFailed as e:
                    if k == 2: raise RuntimeError(f"no reachable spot to set the block down: {e}")
                    C.goto(wp1, gripper=g_lock)
            T.state = "done"; T.say(f"INSPECT {brick}: {verdict}" + (f" · {'; '.join(reasons)}" if reasons else f" (highest score {max(sc):.2f}, clean <= {t_low})"),
                                    verdict == "CLEAN")
        except Stopped:
            res["error"] = "stopped"; T.state = "stopped"; T.say("STOPPED: arm holds its position", False)
        except Exception as e:
            res["error"] = str(e); T.state = "error"; T.say(f"ERROR in inspect: {e} (arm holds its position)", False)
        finally:
            try: C.hold_here()
            except Exception: pass
            (d / "inspect.json").write_text(json.dumps(res, indent=1))
            self.last_inspect = {**res, "running": False}
            T.active = False; A.running = False; C.active = False; self.running = False

    def _log(self, d, rec):
        self.cycles.append({k: v for k, v in rec.items() if k != "photos"})
        with open(d / "cycles.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")

    def status(self):
        T = self.T; c = self.counts()
        return {"running": self.running, "state": T.state if self.running else "idle", "msg": T.msg, "ok": T.ok,
                "target": self.target, "per_block": self.per_block, "since": self.since, "picked": self.picked(self.since), "counts": c, "added": {b: c[b] - self.start_counts.get(b, c[b]) for b in BRICKS},
                "session": self.session, "episodes": self.rec_session, "cycles": self.cycles[-10:], "n_cycles": len(self.cycles)}


def make_router(S, A, ctx, C, T, F, DATA):
    AC = AutoCollect(S, A, ctx, C, T, F, DATA)
    r = APIRouter()

    @r.get("/autocollect/status")
    def status(): return AC.status()

    @r.post("/autocollect/start")
    def start(target: int = 100, max_cycles: int = 200, record: bool = True, per_block: int = 0, since: str = ""):
        def bad(m): return {**AC.status(), "started": False, "msg": m, "ok": False}
        if AC.busy(): return bad("busy: STOP the running job first")
        if A.f is None: return bad("connect the arm first")
        try:
            for k in ("home", "wp1", "wp2"): C.wps(k)
        except Exception as e: return bad(str(e))
        miss = [c for c in ("top", "front") if c not in S.live]
        if miss: return bad(f"camera not live: {', '.join(miss)}")
        threading.Thread(target=AC.run, args=(max(1, min(1000, target)), max(1, min(1000, max_cycles)), bool(record), max(0, min(500, per_block)), since), daemon=True).start()
        time.sleep(0.1); return {**AC.status(), "started": True}

    @r.post("/autocollect/inspect")
    def inspect(colour: str = "blue"):
        """Pick that block, photograph it, score it with its anomaly model, put it back. Result: GET /autocollect/inspect."""
        if colour not in BRICKS: return {"started": False, "msg": f"colour must be one of {BRICKS}"}
        if AC.busy(): return {"started": False, "msg": "busy: STOP the running job first"}
        if A.f is None: return {"started": False, "msg": "connect the arm first"}
        threading.Thread(target=AC.inspect, args=(colour,), daemon=True).start()
        time.sleep(0.1); return {"started": True}

    @r.get("/autocollect/inspect")
    def inspect_result(): return AC.last_inspect or {}

    @r.post("/autocollect/stop")
    def stop():
        T.stop_flag.set(); A.stop.set(); C.stop_flag.set()
        return AC.status()

    r.autocollect = AC
    return r
