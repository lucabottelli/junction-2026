"""Hand-feed anomaly capture (step 7, /anomaly): one block at a time, no picking.

    READY   the arm waits at the rest pose with the clamp open; you put a block between the fingers
    CLOSE   "Close clamp": closes until the fingers stall on the block, squeezes a bit past it, checks it is held,
            goes to WP2 (through 'Above WP2' if that is taught), photographs at EVERY taught WP2 pose (front camera),
            back to the rest pose, opens the clamp -> READY for the next block

Every photo goes through ctx.capture (arm.py), exactly like collection: averaged full-res frame + aligned depth +
depth-masked image + the anomaly-model crop (prep.inspect_crop) + a meta.jsonl line. One "Close clamp" = one
handling = one episode for the trainer's held-out split (run = this session, cycle = the handling number).
Photos whose depth mask failed keep their raw + depth files (reprocess_inhand.py can redo them) but get no crop.

STOP (Space / Esc / the top bar) freezes the arm wherever it is, with the block still clamped.
"""
import json, threading, time
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import APIRouter
from fastapi.responses import Response

import prep
from arm import Stopped
from collect import REGRIP_SQUEEZE

BRICKS = ("pink", "green", "blue")
LABELS = ("good", "defect")
RESTS = ("home", "wp1")          # where the arm waits for the next block
OPEN_FRAC = 0.8                  # clamp opening at the rest pose (fraction of the gripper range), like the sorter's hand-feed
SETTLE_S = 0.4                   # wait after arriving at a WP2 pose before the photo (the block stops swinging)
STILL_S, SETTLE_MAX_S = 0.5, 4.0   # the arm counts as still after this long without joint movement; give up after
THIN_MIN = 8                     # ticks above the empty clamp's stop that count as 'something is held'
FLOOR_MM = 15.0                  # tilt poses keep the arm at least this far above the table (kinematic model)
LIFT = Path(__file__).resolve().parent / "feed_lift.json"   # {brick: {wp2 pose name: mm higher}}: see Feeder.lifted
CFG = Path(__file__).resolve().parent / "feed_cfg.json"     # the selected brick / label / rest pose, kept over a restart
MIN_AREA = {"pink": 120_000}     # px: a colour mask smaller than this is not that brick. Only pink needs it: the red jaws
                                 # pass the pink colour test (69k-107k px alone; with a pink block 170k-380k), so a block
                                 # of another colour under 'pink' still gets a mask (2026-11-15 01:39: a green one did)
HIGH_MM = 60.0                   # WP2 without a taught 'Above WP2' is allowed if the arm stays this far above the table


class Feeder:
    def __init__(self, S, A, ctx, C, DATA, jobs):
        self.S, self.A, self.ctx, self.C, self.DATA, self.jobs = S, A, ctx, C, DATA, jobs
        self.cfg = {"brick": "pink", "label": "good", "rest": "home"}
        try:                        # 2026-11-15: a restart reset green to pink and a green block was saved as pink
            j = json.loads(CFG.read_text())
            if j.get("brick") in BRICKS: self.cfg["brick"] = j["brick"]
            if j.get("label") in LABELS: self.cfg["label"] = j["label"]
            if j.get("rest") in RESTS: self.cfg["rest"] = j["rest"]
        except Exception: pass
        self.state, self.msg, self.ok = "idle", "Press Get ready: the arm goes to the rest pose and opens the clamp", True
        self.run = datetime.now().strftime("%Y%m%d_%H%M%S") + "_feed"
        self.n = 0                  # handlings done this session
        self.photos = []            # last handling: [{view, file, mask_ok, why, z_mm}]
        self.last_img = None        # jpg bytes of the last photo (masked if the mask worked)
        self.active = False         # a motion thread is running
        self.lock = threading.Lock()
        self.kin = None             # vpick.Kin, loaded when the tilt poses are made
        self.shut = None            # gripper ticks where the EMPTY clamp stops (measured at the first Get ready)
        self.grip = {}              # last Close clamp: stall / lock / shut / floor ticks
        self.thin_ok = False        # the user confirmed a thin grip (see close); until the next Get ready
        try: self.lift = json.loads(LIFT.read_text())
        except Exception: self.lift = {}
        self._lifted = {}           # cache: (pose ticks, mm) -> lifted pose ticks

    def say(self, m, ok=True): self.msg, self.ok = m, ok

    # ---------------------------------------------------------------- helpers
    def busy(self):
        out = []
        if self.A.running and not self.active: out.append("an arm routine")
        for name, key in (("collection", "collector"), ("the sorter", "sorter")):
            o = self.jobs.get(key)
            if o is not None and getattr(o, "active", False): out.append(name)
        if prep.TUNING: out.append("the depth auto-tune")
        return out

    def g_open(self):
        gmin, gmax = self.A.fl["gripper"]
        return gmin + OPEN_FRAC * (gmax - gmin)

    def settle(self):
        """Wait until the arm really stands still before a photo: A.goto returns within 3.5 deg of the pose while
        the joints still creep into place, and the photo averages 16 frames (2026-11-14: a ghosted double image at
        the pose after the biggest move). Still = no body joint moved more than 2 ticks for STILL_S, then SETTLE_S
        more for the block to stop swinging. Gives up after SETTLE_MAX_S (a joint that keeps hunting)."""
        from arm import BODY
        t0 = time.time(); last = dict(self.A.pos); still_since = t0
        while time.time() - t0 < SETTLE_MAX_S:
            time.sleep(0.05); self.C.check(); now = dict(self.A.pos)
            if max(abs(now[n] - last[n]) for n in BODY) > 2: still_since = time.time(); last = now
            elif time.time() - still_since >= STILL_S: break
        time.sleep(SETTLE_S)

    def _shut_io(self, value=...):
        """The empty-clamp stop is kept next to the clamp range (arm.GRIP_FLOOR), so a restart doesn't re-measure it
        (a re-measure with a block left in the clamp would be wrong). value=None forgets it; no argument reads it."""
        import arm as armmod
        try: j = json.loads(armmod.GRIP_FLOOR.read_text())
        except Exception: j = {}
        if value is ...:
            ok = j.get("shut_floor") == round(self.floor()) if self.A.f else False
            return float(j["shut"]) if ok and j.get("shut") is not None else None
        j["shut"] = value; j["shut_floor"] = round(self.floor())
        try: armmod.GRIP_FLOOR.write_text(json.dumps(j, indent=1))
        except Exception: pass

    def floor(self):
        """Lowest gripper command: the calibrated closed end, or past it if the arm allows (arm.set_grip_floor)."""
        return float(getattr(self.A, "g_floor", None) or self.A.fl["gripper"][0])

    def held(self):
        """Is something between the fingers? They stand THIN_MIN ticks or more above where the EMPTY clamp stops
        (measured at the first Get ready) and are still being squeezed. Replaces the collector's rule (50 ticks
        above the calibrated end) here: a 2x2 block reads only 1919-1932 against an end of 1875, so a block held
        across a thin side was taken for nothing. Until the empty clamp is measured, the collector's rule is used."""
        A = self.A; g, c = A.pos["gripper"], A.cmd["gripper"]
        if self.shut is None: return self.C.holding()
        return g - self.shut >= THIN_MIN and c < g - 6

    def lifted(self, wp2, brick):
        """The WP2 photo poses for this brick: the taught ones, each moved straight up by the brick's entry in
        feed_lift.json (same angle, like /feed/nudge). 2026-11-15: the green 2x3 is longer than the 2x2 the poses
        were taught with and hangs lower, so the frame's bottom edge cut it in 19/24 photos at wp2-1 and 8/24 at
        wp2-3. Pink and blue keep the taught poses. Raises if a lifted pose is out of reach."""
        off = self.lift.get(brick) or {}
        if not any(off.get(w["name"]) for w in wp2): return wp2
        import vpick
        from arm import BODY, JOINTS
        if self.kin is None: self.kin = vpick.Kin()
        kin, out = self.kin, []
        for w in wp2:
            up = float(off.get(w["name"]) or 0)
            if not up: out.append(w); continue
            key = (tuple(sorted(w["pos"].items())), up)
            if key not in self._lifted:
                q = kin.rad(w["pos"]); p = kin.fk(q)[0].copy(); p[2] += up / 1000.0
                t, err = kin.pose_at(w["pos"], p, kin.elevation(q))
                lim = self.A.fl or kin.cal["follower_limits"]
                if err >= 0.001 or not all(lim[j][0] + 20 <= t[j] <= lim[j][1] - 20 for j in BODY):
                    raise RuntimeError(f"{w['name']} can't go {up:.0f} mm higher for {brick} (out of reach, error "
                                       f"{err * 1000:.0f} mm): lower it with /feed/lift")
                self._lifted[key] = {**w["pos"], **{j: int(round(t[j])) for j in BODY}}
            out.append({**w, "pos": self._lifted[key]})
        return out

    def wp2_clearance(self, wp1, wp2):
        """Lowest height (mm above the table, kinematic model) of any arm link on the way rest -> WP1-height pose
        over WP2 -> WP2 poses -> back, moving joint-to-joint as the capture does. None if the model can't load."""
        try:
            import vpick
            from arm import JOINTS
            if self.kin is None: self.kin = vpick.Kin()
            above = self.C.above_wp2(wp1, wp2[0]["pos"])[0]
            seq = [self.rest_pos(), above] + [w["pos"] for w in wp2] + [wp2[0]["pos"], self.C.above_wp2(wp1, wp2[-1]["pos"])[0]]
            low = 9.0
            for a, b in zip(seq, seq[1:]):
                for k in range(13):
                    low = min(low, self.kin.lowest({j: a[j] + (b[j] - a[j]) * k / 12 for j in JOINTS}))
            return low * 1000
        except Exception:
            return None

    def rest_pos(self):
        return self.C.wps(self.cfg["rest"])[0]["pos"]

    def counts(self):
        """Raw photos and model crops per brick and label in training_images/front/inhand/."""
        out = {}
        for lab in LABELS:
            d = self.DATA / "front" / "inhand" / lab
            if not d.is_dir(): continue
            for p in d.glob("*.png"):
                parts = p.stem.split("_")
                if len(parts) < 4: continue
                b = parts[3]
                o = out.setdefault(b, {"good": 0, "defect": 0, "good_crops": 0, "defect_crops": 0})
                if p.stem.endswith("_crop"): o[lab + "_crops"] += 1
                elif not p.stem.endswith("_depth"): o[lab] += 1
        return out

    def _last_meta(self, n):
        """The last n meta.jsonl records (the photos ctx.capture just wrote)."""
        try:
            lines = (self.DATA / "meta.jsonl").read_text().strip().splitlines()[-n:]
            return [json.loads(x) for x in lines]
        except Exception:
            return []

    def _start(self, fn, name):
        with self.lock:
            if self.active: return False, "the arm is still moving"
            b = self.busy()
            if b: return False, f"busy: {', '.join(b)} is running. Stop it first."
            if not self.A.f: return False, "the arm is not connected (Connect on the Waypoints page)"
            self.active = True; self.A.running = True; self.A.stop.clear(); self.C.stop_flag.clear()
        threading.Thread(target=self._run, args=(fn, name), daemon=True).start()
        return True, ""

    def _run(self, fn, name):
        try:
            self.C.hold_here()
            fn()
        except Stopped:
            self.state = "stopped"; self.say("STOPPED: the arm holds where it is (a block in the clamp stays clamped). "
                                             "Get ready to go back to the rest pose.", False)
        except Exception as e:
            self.state = "error"; self.say(f"{name} failed: {e}", False)
        finally:
            try: self.C.hold_here()
            except Exception: pass
            self.active = False; self.A.running = False

    # ---------------------------------------------------------------- the two actions
    def ready(self):
        def go():
            self.state = "moving"; self.say(f"to the rest pose ({self.cfg['rest']}), opening the clamp")
            self.C.goto(self.rest_pos(), gripper=self.g_open())
            self.thin_ok = False
            if self.shut is None: self.shut = self._shut_io()
            if self.shut is None:                    # once per clamp range: where do the EMPTY fingers stop?
                self.say("measuring the empty clamp (keep it empty): closing once, then opening")
                self.C.goto(dict(self.A.cmd), 0.25, gripper=self.floor()); time.sleep(0.4)
                self.shut = float(self.A.pos["gripper"]); self._shut_io(self.shut)
                self.C.goto(dict(self.A.cmd), 0.5, gripper=self.g_open())
            self.state = "ready"
            self.say(f"READY: put a {self.cfg['brick']} block ({self.cfg['label']}) between the open fingers, any "
                     "orientation, then press Close clamp")
        return self._start(go, "get ready")

    def close(self):
        if self.state not in ("ready", "thin"):
            return False, "not ready: press Get ready first (the arm must wait at the rest pose, clamp open)"
        confirm = self.state == "thin"      # second press on a thin grip: the user says the block IS in the clamp
        C, A = self.C, self.A
        try: wp2 = self.lifted(C.wps("wp2"), self.cfg["brick"])
        except Exception as e: return False, str(e)
        wp1 = C.wps("wp1")[0]["pos"]
        if not (self.ctx.R["routines"].get("wp2_above") or {}).get("waypoints"):
            # No taught "Above WP2": fine when WP2 is held high (2026-11-14: front camera raised, block ~17 cm up).
            # The collector then comes in over WP2 at WP1's arm height. Only a LOW WP2 needs the taught straight drop.
            low = self.wp2_clearance(wp1, wp2)
            if low is None or low < HIGH_MM:
                return False, ("teach 'Above WP2' (wp2_above) first: the arm comes down into WP2 from there"
                               + ("" if low is None else f" (without it the arm passes {low:.0f} mm above the table)"))
        if "front" not in self.S.live: return False, "the front camera is not live"
        brick, label, idx = self.cfg["brick"], self.cfg["label"], self.n + 1

        def go():
            floor = self.floor()
            self.state = "closing"; self.say("closing the clamp on the block: keep holding it until the gripper has it")
            C.goto(dict(A.cmd), 0.25, gripper=floor); time.sleep(0.3)     # stops when the fingers stall on the block
            stall = float(A.pos["gripper"])
            g_lock = max(floor, stall - REGRIP_SQUEEZE)                   # squeeze a little past the block's width
            C.goto(dict(A.cmd), 0.5, gripper=g_lock); time.sleep(0.3)
            self.grip = {"stall": round(stall), "lock": round(g_lock), "shut": None if self.shut is None else round(self.shut),
                         "floor": round(floor)}
            # The fingers are flexible: around a block held across its thin side (a 2x2 on edge) they wrap and close
            # almost as far as empty (2026-11-14: 1853 with the block, 1849 empty), so the reading can't tell the two
            # apart. Don't open and drop the block: stay shut and let the user say it is there (once per Get ready).
            thin = not self.held()
            self.grip["thin"] = thin
            if thin and confirm: self.thin_ok = True
            if thin and not self.thin_ok:
                self.state = "thin"
                self.say(f"the fingers closed almost fully (stopped at {stall:.0f}; the empty clamp stops at "
                         f"{'?' if self.shut is None else round(self.shut)}). Block held on its thin side? Press Close "
                         "clamp again: the photos go on, and the next thin grips don't ask. Clamp empty? Press Open "
                         "clamp and put the block in again.", False)
                return
            self.n = idx                                                  # a new handling (episode id), even if STOPPED later
            self.state = "to_wp2"; self.say("to WP2 (clamp locked)")
            C.to_wp2(wp1, wp2, g_lock)
            photos = []; self.photos = []
            for i, w in enumerate(wp2):
                C.check()
                if i: C.goto(w["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)       # slow between photo poses
                self.state = "photo"; self.say(f"photo {i + 1}/{len(wp2)} at {w['name']}")
                self.settle(); C.check()
                prep.configure_inhand(brick=brick)                        # mask by this brick's colour
                nf = self.ctx.capture(["front"], label, brick, w["name"], idx, self.run)
                if not nf: raise RuntimeError(f"the front camera gave no fresh frame at {w['name']} (frozen or "
                                              "reconnecting): nothing saved. STOP, wait for the feed, Get ready again")
                rec = next((m for m in reversed(self._last_meta(3)) if m.get("cam") == "front"), {})
                dep = rec.get("depth") or {}
                ph = {"view": w["name"], "file": rec.get("file"), "files": nf, "mask_ok": bool(dep),
                      "why": rec.get("depth_mask_why"), "z_mm": dep.get("z_mm") if isinstance(dep, dict) else None,
                      "area_px": dep.get("area_px") if isinstance(dep, dict) else None}
                photos.append(ph); self.photos = list(photos)
                self._grab_last(rec)
            held = thin or self.held()                # a thin grip can't be told from empty: see the photos
            self.state = "returning"; self.say("back to the rest pose")
            if len(wp2) > 1: C.goto(wp2[0]["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)   # leave the way it came in
            C.up_wp2(wp1, wp2, g_lock)
            C.goto(self.rest_pos(), gripper=g_lock)
            C.goto(dict(A.cmd), 0.4, gripper=self.g_open()); time.sleep(0.2)
            bad = [p["view"] for p in photos if not p["mask_ok"]]
            note = (f" · NO depth mask at {', '.join(bad)} (kept raw + depth; no crop)" if bad else " · all masked")
            if not held: note += " · the block seemed to slip during the photos: check it"
            if thin: note += " · thin grip: check the block is in the photos"
            small = [p["view"] for p in photos if p["mask_ok"] and p.get("area_px") is not None
                     and p["area_px"] < MIN_AREA.get(brick, 0)]
            if small:
                note = (f" · THE MASK DOESN'T LOOK LIKE A {brick.upper()} BLOCK at {', '.join(small)} (too small): "
                        f"is the brick selector right? These photos are saved as {brick}") + note.replace(" · all masked", "")
            self.state = "ready"
            self.say(f"handling {idx} done: {len(photos)} photo{'s' * (len(photos) != 1)}{note}. "
                     f"Take the block out, put the next one in, Close clamp", ok=not bad and held and not small)
        return self._start(go, "close clamp")

    def release(self):
        """Open the clamp where the arm is (to take a block out after a STOP or an error)."""
        def go():
            self.say("opening the clamp"); self.C.goto(dict(self.A.cmd), 0.4, gripper=self.g_open())
            if self.state == "thin":
                self.state = "ready"; self.say("clamp open: put the block between the fingers, then Close clamp"); return
            self.state = "stopped" if self.state in ("stopped", "error") else self.state
            self.say("clamp open. Get ready to go back to the rest pose.")
        return self._start(go, "open clamp")

    def _grab_last(self, rec):
        try:
            f = rec.get("file")
            if not f: return
            p = self.DATA / f
            m = p.with_name(p.stem + "_masked.jpg")
            img = cv2.imread(str(m if m.exists() else p))
            if img is None: return
            img = cv2.resize(img, (640, 640 * img.shape[0] // img.shape[1]), interpolation=cv2.INTER_AREA)
            self.last_img = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        except Exception:
            pass

    def status(self):
        try: poses = [w["name"] for w in self.C.wps("wp2")]
        except Exception: poses = []
        rest_ok = bool((self.ctx.R["routines"].get(self.cfg["rest"]) or {}).get("waypoints"))
        return {"state": self.state, "msg": self.msg, "ok": self.ok, "active": self.active, "cfg": self.cfg,
                "run": self.run, "n": self.n, "photos": self.photos, "has_img": self.last_img is not None, "grip": self.grip,
                "wp2_poses": poses, "rest_taught": rest_ok, "connected": bool(self.A.f), "busy": self.busy(),
                "lift": self.lift, "counts": self.counts(), "bricks": BRICKS, "labels": LABELS, "rests": RESTS}


def make_router(S, A, ctx, C, DATA, jobs):
    F = Feeder(S, A, ctx, C, DATA, jobs)
    r = APIRouter()

    def res(ok_msg):
        ok, m = ok_msg
        if not ok: F.say(m, False)
        return F.status()

    @r.get("/feed/status")
    def st(): return F.status()

    @r.post("/feed/cfg")
    def cfg(brick: str = "", label: str = "", rest: str = ""):
        if F.active: F.say("can't change the settings while the arm moves", False); return F.status()
        if brick:
            b = brick.strip().lower()
            if b in BRICKS: F.cfg["brick"] = b
        if label in LABELS: F.cfg["label"] = label
        if rest in RESTS: F.cfg["rest"] = rest
        try: CFG.write_text(json.dumps(F.cfg, indent=1))
        except Exception: pass
        if F.state == "ready" and (brick or label):
            F.say(f"READY: put a {F.cfg['brick']} block ({F.cfg['label']}) in, then Close clamp")
        return F.status()

    @r.post("/feed/ready")
    def ready(): return res(F.ready())

    @r.post("/feed/close")
    def close(): return res(F.close())

    @r.post("/feed/release")
    def release(): return res(F.release())

    @r.post("/feed/tilts")
    def tilts(angles: str = "", deg: float = 0.0, on: bool = True):
        """Extra WP2 photo poses WITHOUT re-clamping and WITHOUT rolling the wrist: copies of the first WP2 pose with
        the wrist FLEXED by each angle in `angles` (signed degrees, comma separated; default "-25,-40").
        Negative = the gripper tips nose-up: the face below the studs turns toward the front camera (about -25 is
        head-on for the taught WP2), further shows it from underneath. Positive tips the gripper down and shows
        the stud face head-on, hiding the lower face (2026-11-14: not wanted as a view).
        Shoulder lift and elbow are re-solved (vpick.Kin.pivot) so the grasp point stays where it is: the block
        turns in place and keeps its position in the picture. Stored as real WP2 waypoints (wp2-flex-25, ...), in
        the order given, so the hand-feed capture AND the sorter use the same views. A tilt that a joint can't
        reach, or that brings the arm closer than FLOOR_MM to the table, is reduced in 5 deg steps.
        on=false removes them (and the older wp2-roll* poses)."""
        import arm as armmod
        if F.active or A.running: F.say("can't change WP2 while the arm moves", False); return F.status()
        R = ctx.R["routines"]; w = (R.get(C.cfg["wp2"]) or {}).get("waypoints") or []
        if not w: F.say("teach the first WP2 pose first", False); return F.status()
        keep = [x for x in w if not (x["name"].startswith("wp2-flex") or x["name"].startswith("wp2-roll"))]
        notes = []
        if on:
            try:
                want = [float(x) for x in angles.split(",") if x.strip()] or ([-abs(deg), abs(deg)] if deg else [-25.0, -40.0])
            except ValueError:
                F.say("angles must be numbers like -25,-40", False); return F.status()
            try:
                import vpick
                if F.kin is None: F.kin = vpick.Kin()
            except Exception as e:
                F.say(f"the kinematic model did not load ({e}): no poses added", False); return F.status()
            base = keep[0]["pos"]; lim = A.fl or F.kin.cal["follower_limits"]; names = set()
            for a in want[:4]:
                sgn = -1 if a < 0 else 1; d = max(5.0, min(60.0, abs(a))); pose = None; why = ""
                while d >= 5:
                    t, err = F.kin.pivot(base, sgn * d)
                    low = F.kin.lowest(t) * 1000
                    if err >= 0.001 or not all(lim[j][0] + 20 <= t[j] <= lim[j][1] - 20 for j in armmod.BODY): why = "the joints"
                    elif low < FLOOR_MM: why = f"the table (arm {low:.0f} mm above it)"
                    else: pose = {j: int(round(t[j])) for j in armmod.JOINTS}; break
                    d -= 5
                s_ = "+" if sgn > 0 else "-"; name = f"wp2-flex{s_}{int(d)}"
                if pose is None: notes.append(f"flex {a:+.0f} is out of reach"); continue
                if d != abs(a): notes.append(f"flex {a:+.0f} limited to {s_}{int(d)} deg by {why}")
                if name in names: continue
                names.add(name); keep.append({"name": name, "pos": pose, "capture": True})
        R[C.cfg["wp2"]]["waypoints"] = keep; armmod.store_routines(R)
        F.say(f"WP2 poses now: {', '.join(x['name'] for x in keep)}" + (" · " + "; ".join(notes) if notes else "")
              + (". The wrist roll is not touched; the block should stay in place in the picture. Watch the first "
                 "handling with a hand on STOP." if on else ""), ok=not notes)
        return F.status()

    @r.post("/feed/clamp")
    def clamp(extra: int = 120):
        """Let the clamp close up to `extra` ticks past the gripper's calibrated closed end (0 = back to normal), so
        a block held across a thin side is squeezed properly. Stored in the gripper servo; the clamp must be empty
        (it goes limp for a moment). The empty-clamp position is measured again at the next Get ready."""
        if F.active or A.running or F.busy(): F.say("can't change the clamp while the arm is busy", False); return F.status()
        try:
            got = A.set_grip_floor(extra); F.shut = None; F._shut_io(None)
            F.say(f"the clamp can now close to {got} (calibrated end {A.fl['gripper'][0]}). Press Get ready with the clamp EMPTY: "
                  "it measures where the empty fingers stop.")
        except Exception as e:
            F.say(f"clamp range not changed: {e}", False)
        return F.status()

    @r.post("/feed/nudge")
    def nudge(pose: int = 1, up_mm: float = 0.0, right_mm: float = 0.0, toward_mm: float = 0.0):
        """Move WP2 photo pose number `pose` (1 = first) by a few mm WITHOUT changing its angle: up, to the arm's
        right (seen from the robot base), toward the far end of the fingers' reach (away from the base). The block
        keeps the same orientation in the picture and only shifts. The previous pose is kept in "replaced"."""
        import arm as armmod, numpy as np
        if F.active or A.running: F.say("can't change WP2 while the arm moves", False); return F.status()
        R = ctx.R["routines"]; w = (R.get(C.cfg["wp2"]) or {}).get("waypoints") or []
        i = int(pose) - 1
        if not 0 <= i < len(w): F.say(f"there is no WP2 pose {pose}", False); return F.status()
        try:
            import vpick
            if F.kin is None: F.kin = vpick.Kin()
        except Exception as e:
            F.say(f"the kinematic model did not load ({e}): pose not changed", False); return F.status()
        kin = F.kin; old = w[i]["pos"]; q = kin.rad(old); p = kin.fk(q)[0]
        out = p[:2] / max(1e-9, np.linalg.norm(p[:2])); right = np.array([out[1], -out[0]])
        d = np.r_[out * toward_mm + right * right_mm, up_mm] / 1000.0
        if np.linalg.norm(d) > 0.04: F.say("nudge at most 40 mm at a time", False); return F.status()
        t, err = kin.pose_at(old, p + d, kin.elevation(q))
        lim = A.fl or kin.cal["follower_limits"]; low = kin.lowest(t) * 1000
        if err >= 0.001 or not all(lim[j][0] + 20 <= t[j] <= lim[j][1] - 20 for j in armmod.BODY) or low < FLOOR_MM:
            F.say(f"that shift is out of reach for WP2 pose {pose} (error {err * 1000:.0f} mm, {low:.0f} mm above the table)", False)
            return F.status()
        w[i] = {**w[i], "pos": {j: int(round(t[j])) for j in armmod.JOINTS},
                "replaced": w[i].get("replaced") or {"name": w[i]["name"], "pos": old}}
        armmod.store_routines(R)
        F.say(f"WP2 pose {pose} ({w[i]['name']}) moved {up_mm:+.0f} mm up, {right_mm:+.0f} mm right, {toward_mm:+.0f} mm out; "
              "same angle. Watch the next handling with a hand on STOP.")
        return F.status()

    @r.post("/feed/lift")
    def lift(brick: str, pose: int = 1, up_mm: float = 0.0):
        """Photograph `brick` with WP2 pose number `pose` held up_mm higher than taught (0 = as taught). Only the
        hand-feed capture of that brick changes; the taught poses and the other bricks stay as they are."""
        if F.active or A.running: F.say("can't change WP2 while the arm moves", False); return F.status()
        try: w = C.wps("wp2")
        except Exception: w = []
        i = int(pose) - 1
        if brick not in BRICKS or not 0 <= i < len(w) or not -40 <= up_mm <= 40:
            F.say(f"lift: brick one of {BRICKS}, pose 1..{len(w)}, up_mm within ±40", False); return F.status()
        old = json.loads(json.dumps(F.lift)); F.lift.setdefault(brick, {})[w[i]["name"]] = float(up_mm)
        try: F.lifted(w, brick)
        except Exception as e: F.lift = old; F.say(str(e), False); return F.status()
        LIFT.write_text(json.dumps(F.lift, indent=1))
        F.say(f"{brick}: {w[i]['name']} is photographed {up_mm:+.0f} mm higher than taught. Watch the next handling "
              "with a hand on STOP.")
        return F.status()

    @r.post("/feed/third")
    def third(frac: float = 0.5):
        """Replace (or add) the THIRD WP2 photo pose with one computed from the first two: the grasp point is put
        `frac` of the way from pose 1 to pose 2 and the fingers at the angle `frac` of the way between theirs
        (0.5 = halfway), wrist roll averaged. The block therefore stays where it is in photos 1 and 2 and is seen
        from a third angle in between. The pose it replaces is kept inside the waypoint ("replaced") for undo."""
        import arm as armmod
        if F.active or A.running: F.say("can't change WP2 while the arm moves", False); return F.status()
        R = ctx.R["routines"]; w = (R.get(C.cfg["wp2"]) or {}).get("waypoints") or []
        if len(w) < 2: F.say("teach the first two WP2 poses first", False); return F.status()
        try:
            import vpick
            if F.kin is None: F.kin = vpick.Kin()
        except Exception as e:
            F.say(f"the kinematic model did not load ({e}): pose not changed", False); return F.status()
        kin = F.kin; frac = max(0.1, min(0.9, float(frac))); a, b = w[0]["pos"], w[1]["pos"]
        pa, pb = kin.fk(kin.rad(a))[0], kin.fk(kin.rad(b))[0]
        ea, eb = kin.elevation(kin.rad(a)), kin.elevation(kin.rad(b))
        seed = {j: a[j] + (b[j] - a[j]) * frac for j in armmod.JOINTS}
        t, err = kin.pose_at(seed, pa + (pb - pa) * frac, ea + (eb - ea) * frac)
        lim = A.fl or kin.cal["follower_limits"]; low = kin.lowest(t) * 1000
        if err >= 0.001 or not all(lim[j][0] + 20 <= t[j] <= lim[j][1] - 20 for j in armmod.BODY) or low < FLOOR_MM:
            F.say(f"no reachable pose between WP2 poses 1 and 2 (error {err * 1000:.0f} mm, {low:.0f} mm above the table)", False)
            return F.status()
        pose = {j: int(round(t[j])) for j in armmod.JOINTS}
        new = {"name": "wp2-3", "pos": pose, "capture": True, "computed": f"between {w[0]['name']} and {w[1]['name']} at {frac:.2f}"}
        if len(w) >= 3:
            old = w[2]; new["replaced"] = old.get("replaced") or {"name": old["name"], "pos": old["pos"]}
            w[2] = new
        else: w.append(new)
        armmod.store_routines(R)
        F.say(f"WP2 pose 3 is now computed: fingers {ea + (eb - ea) * frac - ea:+.0f} deg from pose 1 "
              f"(pose 2 is {eb - ea:+.0f} deg), block held between where poses 1 and 2 hold it. "
              "Watch the first handling with a hand on STOP.")
        return F.status()

    @r.get("/feed/last.jpg")
    def last():
        if F.last_img is None: return Response(status_code=204)
        return Response(F.last_img, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    r.feeder = F
    return r
