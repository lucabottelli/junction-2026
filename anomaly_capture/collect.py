"""Data collection for phase 1 (learned pick) and phase 2 (front-camera images).

Page: http://127.0.0.1:7870/collect (served by capture.py, shares its cameras and the arm from arm.py)

Task phases (only 1 and 2 exist yet; sorting etc. comes later, see SO101_SETUP_STATUS.md):
  PHASE 1  pick a brick lying anywhere/any way on the orange plate and bring it to WAYPOINT 1 (above the
           orange plate). This is what ACT learns. An episode = human teleoperated pick (until the gripper
           stalls on the brick) + the robot's own automatic rise and carry to WP1, recorded as ONE episode,
           so the carry part of every demonstration is clean and identical.
  PHASE 2  WP1 -> WAYPOINT 2 in front of the front camera with the gripper command LOCKED (clamp force
           unchanged, only the other joints move), then averaged full-res front images + aligned depth +
           depth-masked image are saved (anomaly training data, label good/defect).
  RESET    (collection only) back via WP1, drop at a random taught `drops` pose with a random wrist turn,
           back to the start pose (`home`) -> next episode.

Per episode: START POSE -> MATCH (move the leader to the robot's pose, mirroring starts when every joint is
within 8 deg) -> PICK (rec) -> CARRY to WP1 (rec) -> WP2 capture -> RESET.
Keys: G = grasped (take over now), X = discard pick, K = start mirroring anyway, N = next after a discard,
Space/Esc = STOP (arm freezes, torque on).
Raw episodes go to ~/junction26/episodes/<session>/<episode>/ ; export_act.py cleans them and writes the
Studio/LeRobot dataset for ACT.
"""
import json, random, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse, Response

import prep
from arm import BODY, JOINTS, TICK_PER_DEG, Stopped

FPS = 30
REC_CAMS = ["top", "wrist", "front"]     # stored per frame; ACT uses top + wrist (see prep.py)
MATCH_DEG = 8.0          # leader must be within this of the follower (every body joint) to start mirroring
HOLD_GAP = 40            # ticks: gripper commanded this much more closed than it actually is ...
HOLD_STILL = 4           # ... and not moving more than this over 0.25 s ...
HOLD_MIN_OPEN = 50       # ... and not fully shut (ticks above the closed end) = holding something
HOLD_TIME = 0.3          # s the above must hold
TAIL_S = 0.4             # keep recording this long after the grasp (the policy must learn to close firmly)
MAX_PICK_S = 90


class Collector:
    def __init__(self, S, A, ctx, DATA, ROOT):
        self.S, self.A, self.ctx, self.DATA = S, A, ctx, DATA
        self.EP = ROOT / "episodes"; self.EP.mkdir(exist_ok=True)
        self.cfg_path = Path(__file__).resolve().parent / "collect_settings.json"
        self.cfg = {"home": "home", "wp1": "wp1", "wp2": "wp2", "drops": "drops", "phase2": False, "regrasp": False, "speed": 0.5, "roll_jitter": 45,
                    "act_roi": {k: prep.default_act_roi(k) for k in REC_CAMS}}
        try: self.cfg.update(json.loads(self.cfg_path.read_text()))
        except Exception: pass
        self.state, self.msg, self.ok = "idle", "set up the 3 routines on /arm, then START", True
        self.info = {}
        self.active = False; self.stop_flag = threading.Event()
        self.flags = {"grasped": False, "discard": False, "skip": False, "next": threading.Event()}
        self.session = None; self.n_done = 0; self.target = 0; self.rec = None

    def save_cfg(self): self.cfg_path.write_text(json.dumps(self.cfg, indent=1))

    def say(self, m, ok=True): self.msg, self.ok = m, ok

    def wps(self, name):
        r = self.ctx.R["routines"].get(self.cfg[name])
        if not r or not r["waypoints"]: raise RuntimeError(f"routine '{self.cfg[name]}' ({name}) is missing or empty - teach it on /arm")
        return r["waypoints"]

    def check(self):
        if self.stop_flag.is_set() or self.A.stop.is_set(): raise Stopped()

    # ------------------------------------------------------------------ recording (30 fps)
    def _record(self, buf, stop_evt):
        S, A, roi = self.S, self.A, self.cfg["act_roi"]
        t0 = time.monotonic(); i = 0
        while not stop_evt.is_set():
            now = time.monotonic()
            fr, ages = {}, {}
            for k in REC_CAMS:
                f = S.frames.get(k)          # reference to the latest frame (frames are replaced, never mutated)
                if f is None: continue
                fr[k] = cv2.imencode(".jpg", prep.act_frame(f, roi[k]), [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tobytes()
                ages[k] = time.time() - S.frame_t[k]
            buf["t"].append(now - t0); buf["frames"].append(fr); buf["age"].append(ages)
            buf["state"].append([A.pos[n] for n in JOINTS]); buf["action"].append([A.goal[n] for n in JOINTS])
            buf["leader"].append([A.lpos.get(n, 0) for n in JOINTS])
            i += 1
            time.sleep(max(0.0, t0 + i / FPS - time.monotonic()))

    def _write(self, d, buf, meta):
        d.mkdir(parents=True, exist_ok=True)
        for k in REC_CAMS: (d / k).mkdir(exist_ok=True)
        for i, fr in enumerate(buf["frames"]):
            for k, jb in fr.items(): (d / k / f"{i:06d}.jpg").write_bytes(jb)
        ages = np.array([[a.get(k, 99.0) for k in REC_CAMS] for a in buf["age"]], np.float32)
        np.savez_compressed(d / "data.npz", t=np.array(buf["t"]), state=np.array(buf["state"], np.float32),
                            action=np.array(buf["action"], np.float32), leader=np.array(buf["leader"], np.float32),
                            cam_age=ages, cams=np.array(REC_CAMS))
        meta["max_cam_age_s"] = round(float(ages.max()), 3) if ages.size else None
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    # ------------------------------------------------------------------ motion helpers
    def goto(self, pos, speed=None, **override):
        tgt = {k: float(v) for k, v in pos.items()}; tgt.update(override)
        for n in JOINTS: tgt[n] = min(self.A.fl[n][1], max(self.A.fl[n][0], tgt[n]))
        self.A.goto(tgt, speed or self.cfg["speed"])

    def hold_here(self):
        """Stop following the leader; keep the commanded gripper (= keep squeezing the brick)."""
        with self.A.io:
            self.A.mode = "hold"; self.A.goal = dict(self.A.cmd)

    def above_wp2(self, wp1, wp2pos):
        """Pose straight above WP2: the taught `wp2_above` slot if it exists (recommended: ~5 cm straight up),
        otherwise approximated as WP2's pan/roll at WP1's height (lift/elbow/wrist-flex of WP1)."""
        r = self.ctx.R["routines"].get("wp2_above")
        if r and r["waypoints"]: return r["waypoints"][0]["pos"], True
        return {**wp2pos, "shoulder_lift": wp1["shoulder_lift"], "elbow_flex": wp1["elbow_flex"],
                "wrist_flex": wp1["wrist_flex"]}, False

    def slot(self, name):
        """Waypoints of an OPTIONAL routine slot ([] if not taught)."""
        r = self.ctx.R["routines"].get(name)
        return r["waypoints"] if r else []

    def place_on_tape(self, g_lock, g_open):
        """Gently set the held brick down on the tape spot and let go; ends above the spot, gripper open."""
        above = self.slot("tape_above")[0]["pos"]; place = self.slot("tape_place")[0]["pos"]; A = self.A
        self.goto(above, gripper=g_lock)
        self.goto(place, self.cfg["speed"] * 0.3, gripper=g_lock)          # slow, straight down = gentle
        self.goto(dict(A.cmd), 0.3, gripper=g_open); time.sleep(0.2)       # open slowly
        self.goto(above, self.cfg["speed"] * 0.5, gripper=g_open)

    def regrasp_90(self, g_lock, g_open):
        """Set down on the tape spot, turn the wrist 90 deg, grip again with the SAME clamp command, lift.
        The faces that were under the fingers are now visible. Returns True if the brick is held again."""
        A = self.A
        above = self.slot("tape_above")[0]["pos"]; place = self.slot("tape_place")[0]["pos"]
        self.place_on_tape(g_lock, g_open)
        lo, hi = A.fl["wrist_roll"]; r90 = 90 * TICK_PER_DEG
        roll = place["wrist_roll"] + r90 if place["wrist_roll"] + r90 <= hi - 20 else place["wrist_roll"] - r90
        self.check()
        self.goto(above, self.cfg["speed"] * 0.5, gripper=g_open, wrist_roll=roll)    # turn while up
        self.goto(place, self.cfg["speed"] * 0.3, gripper=g_open, wrist_roll=roll)    # straight down, open
        self.goto(dict(A.cmd), 0.5, gripper=g_lock); time.sleep(0.2)                  # same clamp command
        held = self.holding()
        self.goto(above, self.cfg["speed"] * 0.5, gripper=g_lock, wrist_roll=roll)    # straight up
        return held and self.holding()

    def up_wp2(self, wp1, wp2, g_lock):
        """Straight UP out of WP2 to the pose above it."""
        above, _ = self.above_wp2(wp1, wp2[-1]["pos"])
        self.goto(above, self.cfg["speed"] * 0.5, gripper=g_lock)

    def to_wp2(self, wp1, wp2, g_lock):
        """WP1 -> above WP2 (at height) -> straight DOWN into WP2 as the very last motion, slower."""
        above, taught = self.above_wp2(wp1, wp2[0]["pos"])
        if not taught: self.say("phase 2: 'above WP2' not taught - using an estimate (teach it for a straight drop)", False)
        self.goto(above, gripper=g_lock)
        self.check()
        self.goto(wp2[0]["pos"], self.cfg["speed"] * 0.5, gripper=g_lock)

    def from_wp2(self, wp1, wp2, g_lock):
        """Reverse: straight UP out of WP2 first, then over to WP1 (never sweeps across what's below WP2)."""
        above, _ = self.above_wp2(wp1, wp2[-1]["pos"])
        self.goto(above, self.cfg["speed"] * 0.5, gripper=g_lock)
        self.goto(wp1, gripper=g_lock)

    def test_wp2(self, kind="wp2"):
        """Dry runs with whatever is in the gripper (nothing saved, <=30 % speed).
        kind=wp2: WP1 -> above WP2 -> down into WP2 -> up -> WP1.  kind=regrasp: tape spot set-down + 90 deg re-grip."""
        A = self.A
        self.active = True; A.running = True; self.stop_flag.clear(); A.stop.clear(); self.state = "capture"
        try:
            if A.mode != "hold": A.set_mode("hold", force=True)
            wp1 = self.wps("wp1")[0]["pos"]; wp2 = self.wps("wp2"); g = A.cmd["gripper"]
            spd = self.cfg["speed"]; self.cfg["speed"] = min(spd, 0.3)
            try:
                if kind == "regrasp":
                    if not (self.slot("tape_above") and self.slot("tape_place")): raise RuntimeError("teach the tape spot (above + on) first")
                    gmin, gmax = A.fl["gripper"]
                    g_open = max(self.wps("home")[0]["pos"]["gripper"], gmin + 0.35 * (gmax - gmin))
                    self.say("TEST: set down on the tape spot, re-grip turned 90°")
                    ok = self.regrasp_90(g, g_open)
                    if not ok: raise RuntimeError("re-grip test: brick NOT held after re-gripping")
                else:
                    self.say("TEST: to WP1"); self.goto(wp1, gripper=g)
                    self.say("TEST: above WP2, then straight down"); self.to_wp2(wp1, wp2, g)
                    for w in wp2[1:]: self.goto(w["pos"], gripper=g)
                    self.say("TEST: straight up, back to WP1"); self.from_wp2(wp1, wp2, g)
            finally:
                self.cfg["speed"] = spd
            self.state = "done"; self.say(f"✓ test of the {'tape re-grip' if kind == 'regrasp' else 'WP1 → WP2 path'} done (nothing saved)")
        except Stopped:
            self.state = "stopped"; self.say("STOPPED: arm holds its position", False)
        except Exception as e:
            self.state = "error"; self.say(f"ERROR: {e}", False)
        finally:
            try: self.hold_here()
            except Exception: pass
            self.active = False; A.running = False

    def holding(self):
        A = self.A; g, c = A.pos["gripper"], A.cmd["gripper"]
        return g - A.fl["gripper"][0] > HOLD_MIN_OPEN and c < g - HOLD_GAP / 2

    # ------------------------------------------------------------------ one episode
    def rise(self, wp1):
        """Go straight up over the current spot first (WP1's arm height/reach, current pan + roll), so the carry
        never sweeps low across the plate. Gripper stays at its commanded (squeezing) value."""
        p = dict(self.A.cmd)
        self.goto(p, self.cfg["speed"] * 0.6, shoulder_lift=wp1["shoulder_lift"], elbow_flex=wp1["elbow_flex"],
                  wrist_flex=wp1["wrist_flex"])

    def episode(self, idx, brick, label):
        A = self.A
        home = self.wps("home")[0]["pos"]; wp1 = self.wps("wp1")[0]["pos"]
        wp2 = self.wps("wp2") if self.cfg.get("phase2") else []
        drops = self.slot("drops")
        gmin, gmax = A.fl["gripper"]
        g_open = max(home["gripper"], gmin + 0.35 * (gmax - gmin))
        ep_id = f"ep{idx:04d}_{datetime.now().strftime('%H%M%S')}"
        meta = {"episode": ep_id, "session": self.session, "brick": brick, "label": label, "fps": FPS,
                "cams": REC_CAMS, "act_roi": self.cfg["act_roi"], "joints": JOINTS,
                "follower_limits": A.fl, "leader_limits": A.ll, "task": f"pick up the {brick} brick",
                "phase1": "human pick + automatic carry to WP1 (recorded for ACT)",
                "start": datetime.now().isoformat(timespec="seconds")}

        # 0 HOME (start pose)
        self.state = "home"; self.say(f"episode {idx}: robot going to the start pose")
        if A.mode != "hold": A.set_mode("hold", force=True)
        self.goto(home, self.cfg["speed"], gripper=g_open)

        # 0b MATCH leader
        A.ensure_leader(); self.state = "match"; self.flags["skip"] = False; ok_since = None
        while True:
            self.check()
            lt = A.leader_target()
            gaps = {n: abs(lt[n] - A.cmd[n]) / TICK_PER_DEG for n in BODY}
            self.info = {"gaps": {k: round(v, 1) for k, v in gaps.items()}}
            worst = max(gaps, key=gaps.get)
            if gaps[worst] <= MATCH_DEG: ok_since = ok_since or time.time()
            else: ok_since = None
            self.say(f"move the LEADER to the robot's pose: biggest gap {worst} {gaps[worst]:.0f}° (needs ≤{MATCH_DEG:.0f}°)")
            if (ok_since and time.time() - ok_since > 0.4) or self.flags["skip"]: break
            time.sleep(0.05)

        # 1 PHASE 1a: human pick (recorded)
        A.set_mode("mirror", force=True)
        buf = {k: [] for k in ("t", "frames", "age", "state", "action", "leader")}
        rec_stop = threading.Event(); rec = threading.Thread(target=self._record, args=(buf, rec_stop), daemon=True)
        self.flags["grasped"] = self.flags["discard"] = False
        self.state = "pick"; t0 = time.time(); attempts, was_open = 0, True
        hold_since, gq, reason = None, [], None
        rec.start()
        try:
            while True:
                self.check()
                now = time.time()
                g, c = A.pos["gripper"], A.goal["gripper"]
                gn = (c - gmin) / max(1, gmax - gmin) * 100
                if was_open and gn < 25: attempts += 1; was_open = False
                elif gn > 40: was_open = True
                gq = [(t, v) for t, v in gq if now - t < 0.25] + [(now, g)]
                still = max(v for _, v in gq) - min(v for _, v in gq) <= HOLD_STILL and now - gq[0][0] > 0.2
                if c < g - HOLD_GAP and still and g - gmin > HOLD_MIN_OPEN:
                    hold_since = hold_since or now
                else: hold_since = None
                self.info = {"attempts": attempts, "t": round(now - t0, 1), "frames": len(buf["t"])}
                self.say(f"● REC  pick the brick  ·  {now - t0:.1f} s  ·  grasp attempts {attempts}")
                if hold_since and now - hold_since >= HOLD_TIME: reason = "auto"; break
                if self.flags["grasped"]: reason = "manual"; break
                if self.flags["discard"]: reason = "discard"; break
                if now - t0 > MAX_PICK_S: reason = "timeout"; break
                time.sleep(0.02)
            handoff = len(buf["t"])
            pick_s = round(time.time() - t0, 2)
            if reason in ("discard", "timeout"):
                rec_stop.set(); rec.join(1)
            else:
                # PHASE 1b: automatic carry to WP1, STILL RECORDED (clean robot-made demo of the carry)
                time.sleep(TAIL_S)
                self.hold_here(); g_lock = A.cmd["gripper"]          # clamp command fixed from here on
                grasp_pose = dict(A.cmd)                             # where the brick was: put it back here later
                self.state = "carry"; self.say("● REC  robot carries the brick to waypoint 1: hands off")
                self.rise(wp1)
                self.goto(wp1, gripper=g_lock)
                time.sleep(0.3)
                rec_stop.set(); rec.join(1)
        finally:
            rec_stop.set()
        self.hold_here()
        meta.update({"pick_s": pick_s, "handoff_frame": handoff, "handoff": reason, "grasp_attempts": attempts,
                     "n_frames": len(buf["t"]), "wp1_frame": len(buf["t"]) - 1})

        if reason in ("discard", "timeout"):
            meta["outcome"] = reason
            threading.Thread(target=self._write, args=(self.EP / self.session / ep_id, buf, meta), daemon=True).start()
            self.state = "reset"; self.flags["next"].clear()
            self.say(f"episode {reason}ed: put the brick back on the orange plate (by hand or with the leader), then NEXT", False)
            A.set_mode("mirror", force=True)          # let the human fix the scene with the leader
            while not self.flags["next"].wait(0.1): self.check()
            A.set_mode("hold", force=True)
            return False

        if not self.holding():
            meta["outcome"] = "missed"
            threading.Thread(target=self._write, args=(self.EP / self.session / ep_id, buf, meta), daemon=True).start()
            self.goto(home, gripper=g_open)
            self.say("brick NOT in the gripper at waypoint 1: episode marked 'missed' (not used for training)", False)
            time.sleep(1.5); return False
        meta["outcome"] = "success"; meta["gripper_lock"] = round(g_lock, 1)
        threading.Thread(target=self._write, args=(self.EP / self.session / ep_id, buf, meta), daemon=True).start()

        # 2 PHASE 2: WP1 -> above WP2 -> down -> photo -> up -> tape spot: set down, re-grip turned 90 deg
        #   -> above WP2 -> down -> photo of the rotated brick -> up.  Clamp command identical throughout.
        n_img, on_tape = 0, False
        if self.cfg.get("phase2"):
            n_img, on_tape = self.phase2(wp1, wp2, g_lock, g_open, label, brick, idx)
        self.info = {"images": n_img}
        self.reset(home, wp1, drops, grasp_pose, g_lock, g_open, on_tape)
        return True

    def phase2(self, wp1, wp2, g_lock, g_open, label, brick, idx):
        """OPTIONAL: WP1 -> above WP2 -> down -> photo -> up [-> tape spot re-grip 90 deg -> photo 2]."""
        self.state = "capture"; n_img = 0
        cams = [c for c in self.cfg.get("inspect_cams", ["front"]) if c in self.S.live]
        tape = self.cfg.get("regrasp") and self.slot("tape_above") and self.slot("tape_place")

        def shoot(suffix, note):
            n = 0
            for i, w in enumerate(wp2):
                self.check(); self.info = {"step": f"WP2{suffix} {i + 1}/{len(wp2)} {w['name']}"}
                self.say(f"phase 2: front camera photo {note} ({w['name']}), clamp unchanged")
                if i: self.goto(w["pos"], gripper=g_lock)
                time.sleep(0.25); self.check()
                n += self.ctx.capture(cams, label, brick, w["name"] + suffix, idx, self.session)
            return n

        self.say("phase 2: to ABOVE waypoint 2, then straight down (clamp unchanged)")
        self.to_wp2(wp1, wp2, g_lock)
        n_img += shoot("", "(as picked)")
        self.up_wp2(wp1, wp2, g_lock)
        on_tape = False
        if tape:
            self.state = "regrasp"; self.say("phase 2: set the brick down on the tape spot, re-grip it turned 90°")
            if self.regrasp_90(g_lock, g_open):
                self.state = "capture"
                self.to_wp2(wp1, wp2, g_lock)
                n_img += shoot("_r90", "(turned 90°)")
                self.up_wp2(wp1, wp2, g_lock)
            else:
                on_tape = True
                self.say("re-grip FAILED: the brick is left on the tape spot (put it back on the orange plate)", False)
        return n_img, on_tape

    def reset(self, home, wp1, drops, grasp_pose, g_lock, g_open, on_tape):
        """Collection only: put the brick back on the orange plate so the next pick can start.
        Reset spots taught -> drop at a random one. Otherwise -> put it back exactly where it was picked
        (that path is known to be clear), turned by a random wrist angle for variety."""
        A = self.A; self.state = "reset_auto"
        if on_tape:
            self.goto(home, gripper=g_open); time.sleep(1.0); return
        jit = random.uniform(-1, 1) * self.cfg["roll_jitter"] * TICK_PER_DEG
        if drops:
            self.say("resetting: robot drops the brick at a random reset spot on the orange plate")
            self.goto(wp1, gripper=g_lock)
            d = random.choice(drops)["pos"]
            self.goto(d, gripper=g_lock, wrist_roll=d["wrist_roll"] + jit)
            self.goto(dict(A.cmd), 1.0, gripper=g_open); time.sleep(0.2)      # release
            self.rise(wp1)
        else:
            self.say("resetting: robot puts the brick back where it picked it, turned a bit")
            self.goto(wp1, gripper=g_lock)
            above = {**grasp_pose, "shoulder_lift": wp1["shoulder_lift"], "elbow_flex": wp1["elbow_flex"],
                     "wrist_flex": wp1["wrist_flex"]}
            roll = grasp_pose["wrist_roll"] + jit
            self.goto(above, gripper=g_lock, wrist_roll=roll)
            self.goto(grasp_pose, self.cfg["speed"] * 0.4, gripper=g_lock, wrist_roll=roll)   # gently down
            self.goto(dict(A.cmd), 0.4, gripper=g_open); time.sleep(0.2)                       # let go
            self.goto(above, self.cfg["speed"] * 0.6, gripper=g_open, wrist_roll=roll)          # straight up
        self.goto(home, gripper=g_open)

    # ------------------------------------------------------------------ session
    def run(self, n, brick, label):
        self.active = True; self.A.running = True; self.stop_flag.clear(); self.A.stop.clear()
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{brick}"; self.n_done = 0; self.target = n
        idx = 0
        try:
            while self.n_done < n:
                idx += 1
                if self.episode(idx, brick, label): self.n_done += 1
            self.state = "done"; self.say(f"✓ DONE: {self.n_done} successful episodes in {self.session}")
        except Stopped:
            self.state = "stopped"; self.say(f"STOPPED ({self.n_done} successful episodes saved): arm holds its position", False)
        except Exception as e:
            self.state = "error"; self.say(f"ERROR: {e} (arm holds its position)", False)
        finally:
            try: self.hold_here()
            except Exception: pass
            self.active = False; self.A.running = False


def make_router(S, A, ctx, DATA, ROOT):
    C = Collector(S, A, ctx, DATA, ROOT)
    r = APIRouter()
    export = {"proc": None, "log": ""}

    def status():
        sessions = []
        for sd in sorted(C.EP.glob("*"), reverse=True)[:12]:
            eps = []
            for md in sorted(sd.glob("*/meta.json")):
                try: m = json.loads(md.read_text())
                except Exception: continue
                eps.append({k: m.get(k) for k in ("episode", "outcome", "pick_s", "grasp_attempts", "handoff", "keep")})
            sessions.append({"session": sd.name, "episodes": eps})
        if export["proc"] is not None and export["proc"].poll() is not None:
            export["log"] += f"\n[exit code {export['proc'].returncode}]"; export["proc"] = None
        return {"state": C.state, "msg": C.msg, "ok": C.ok, "info": C.info, "active": C.active, "done": C.n_done,
                "target": C.target, "session": C.session, "cfg": C.cfg, "connected": A.f is not None,
                "routines": list(ctx.R["routines"]), "sessions": sessions, "live": S.live,
                "export_running": export["proc"] is not None, "export_log": export["log"][-4000:]}

    @r.get("/collect")
    def page(): return RedirectResponse("/arm#collect")      # everything lives on /arm now (ui.py)

    @r.get("/collect/counts")
    def counts():
        """Front-camera anomaly images per brick and label (originals only, not _depth/_masked)."""
        out = {}
        for lab in ("good", "defect"):
            for f in (DATA / "front" / "inhand" / lab).glob("*.png"):
                if f.stem.endswith("_depth"): continue
                parts = f.stem.split("_")
                b = parts[3] if len(parts) > 4 else "?"
                out.setdefault(b, {"good": 0, "defect": 0})[lab] += 1
        return out

    @r.get("/collect/status")
    def st(): return status()

    @r.post("/collect/start")
    def start(n: int = 10, brick: str = "", label: str = "good"):
        if C.active: C.say("already running", False); return status()
        if A.f is None: C.say("connect the arm on /arm first", False); return status()
        if not brick.strip(): C.say("type the brick name first", False); return status()
        try:
            for k in ("home", "wp1"): C.wps(k)
            if C.cfg.get("phase2"): C.wps("wp2")
            if C.cfg.get("phase2") and C.cfg.get("regrasp") and not (C.slot("tape_above") and C.slot("tape_place")):
                raise RuntimeError("re-grip is on: teach the tape spot (above + on) first, or switch the re-grip off")
        except Exception as e:
            C.say(str(e), False); return status()
        threading.Thread(target=C.run, args=(max(1, n), brick.strip(), label), daemon=True).start()
        time.sleep(0.1); return status()

    @r.post("/collect/test_wp2")
    def test_wp2(kind: str = "wp2"):
        if C.active or A.running: C.say("busy: stop first", False); return status()
        if A.f is None: C.say("connect the arm first", False); return status()
        threading.Thread(target=C.test_wp2, args=(kind,), daemon=True).start(); time.sleep(0.1); return status()

    @r.post("/collect/stop")
    def stop():
        C.stop_flag.set(); A.stop.set(); C.flags["next"].set()
        time.sleep(0.06)
        if A.f:
            with A.io: A.mode = "hold"; A.goal = dict(A.cmd)
        C.say("STOPPED: arm holds its position", False); return status()

    @r.post("/collect/flag/{flag}")
    def flag(flag: str):
        if flag in ("grasped", "discard", "skip"): C.flags[flag] = True
        elif flag == "next": C.flags["next"].set()
        return status()

    @r.post("/collect/cfg")
    def cfg(home: str = "", wp1: str = "", wp2: str = "", drops: str = "", speed: float = 0,
            roll_jitter: float = -1, inspect_cams: str = "", phase2: str = "", regrasp: str = ""):
        if phase2: C.cfg["phase2"] = phase2 == "true"
        if regrasp: C.cfg["regrasp"] = regrasp == "true"
        if home: C.cfg["home"] = home
        if wp1: C.cfg["wp1"] = wp1
        if wp2: C.cfg["wp2"] = wp2
        if drops: C.cfg["drops"] = drops
        if speed: C.cfg["speed"] = min(1.0, max(0.1, speed))
        if roll_jitter >= 0: C.cfg["roll_jitter"] = min(90, roll_jitter)
        if inspect_cams: C.cfg["inspect_cams"] = [c for c in inspect_cams.split(",") if c]
        C.save_cfg(); return status()

    @r.post("/collect/roi/{cam}")
    def roi(cam: str, x0: int, y0: int, x1: int, y1: int):
        if cam in REC_CAMS and x1 - x0 > 100 and y1 - y0 > 75:
            C.cfg["act_roi"][cam] = prep.fit_43([x0, y0, x1, y1]); C.save_cfg()
            C.say(f"policy view of {cam} set to {C.cfg['act_roi'][cam]} (4:3)")
        return status()

    @r.post("/collect/roi/{cam}/reset")
    def roi_reset(cam: str):
        C.cfg["act_roi"][cam] = prep.default_act_roi(cam); C.save_cfg(); return status()

    @r.post("/collect/keep")
    def keep(session: str, episode: str, on: bool):
        mp = C.EP / session / episode / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text()); m["keep"] = on; mp.write_text(json.dumps(m, indent=1))
        return status()

    @r.get("/collect/view/{cam}")
    def view(cam: str):
        """Exactly what the policy sees (top/wrist) - or, for front, the depth-masked in-hand view."""
        f = S.frames.get(cam)
        if f is None: return Response(status_code=204)
        if cam == "front" and getattr(S.cams.get("front"), "depth", None) is not None:
            m, _ = prep.inhand_mask(S.cams["front"].depth)
            img = prep.apply_mask(f, m) if m is not None else f
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        else:
            img = prep.act_frame(f, C.cfg["act_roi"][cam])
        return Response(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @r.post("/collect/export")
    def run_export(speedup: float = 1.5, max_attempts: int = 1, name: str = "", dry: bool = False, bricks: str = ""):
        if export["proc"] is not None: return status()
        args = [sys.executable, "-u", str(Path(__file__).resolve().parent / "export_act.py"),
                "--speedup", str(speedup), "--max-attempts", str(max_attempts)]
        if name: args += ["--name", name]
        if bricks: args += ["--bricks"] + [b for b in bricks.split(",") if b]
        if dry: args += ["--dry-run"]
        export["proc"] = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        export["log"] = " ".join(args[2:]) + "\n"

        def pump(p=export["proc"]):
            for line in p.stdout: export["log"] += line
        threading.Thread(target=pump, daemon=True).start()
        return status()

    return r


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Pick + Inspect Collection</title><style>
:root{--bg:#121417;--panel:#1b1f24;--line:#2c323a;--text:#e8eaed;--muted:#9aa3ad;--good:#2fa36b;--bad:#d2582a;--ok:#5fd38d;--err:#ff6b6b}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:12px 16px}
button{font:600 14px system-ui;padding:8px 12px;border-radius:7px;border:0;cursor:pointer;color:#fff;background:#3d4854}
button.g{background:var(--good)}button.d{background:var(--bad)}button.big{font-size:18px;padding:12px 20px}
#stop{background:#c62828;font:800 22px system-ui;padding:14px 30px}
input,select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px;font-size:14px}
#banner{font:800 26px system-ui;padding:16px;border-radius:10px;margin:10px 0;text-align:center;color:#111;background:#888}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}@media(max-width:1000px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
h3{margin:0 0 10px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px}.views img{width:100%;border-radius:6px;display:block}
.views figure{margin:0;font-size:12px;color:var(--muted)}.wrap{position:relative}.drag{position:absolute;border:2px dashed #ffd23f;display:none;pointer-events:none}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left}
pre{white-space:pre-wrap;font-size:12px;max-height:260px;overflow:auto;background:#0d0f12;padding:8px;border-radius:6px}
.small{font-size:12px;color:var(--muted)}ol{margin:4px 0 0 18px;padding:0;font-size:14px;line-height:1.5}
.bar{height:10px;background:#2a3038;border-radius:5px;flex:1;min-width:80px}.bar i{display:block;height:10px;border-radius:5px}
</style></head><body>
<div class=row><button id=stop onclick="p('/collect/stop')">■ STOP (Space)</button>
<span class=small>brick</span><input id=brick placeholder="e.g. red" style="width:100px">
<select id=label title="label for the inspection images (the pick episodes are useful either way)"><option value=good>good brick</option><option value=defect>defect brick</option></select>
<span class=small>episodes</span><input id=n type=number value=20 min=1 style="width:70px">
<button class="g big" onclick="p('/collect/start?n='+$('n').value+'&brick='+encodeURIComponent($('brick').value.trim())+'&label='+$('label').value)">▶ START</button>
<a href="/arm" style="color:#8ab4f8;margin-left:auto">arm / teach →</a> <a href="/" style="color:#8ab4f8">manual capture →</a></div>
<div id=banner>…</div>
<div class=row id=keys><button class=g onclick="p('/collect/flag/grasped')">[G] grasped: take over</button>
<button class=d onclick="p('/collect/flag/discard')">[X] discard this pick</button>
<button onclick="p('/collect/flag/skip')">[K] start mirroring anyway</button><button onclick="p('/collect/flag/next')">[N] NEXT</button>
<span id=gaps class=small></span></div>
<div class=grid><div>
 <div class=card><h3>what the models see (live)</h3><div class=views>
  <figure><div class=wrap><img id=v-top><div class=drag id=dr-top></div></div>top · policy input (drag on the full view below to change)</figure>
  <figure><img id=v-wrist>wrist · policy input</figure>
  <figure><img id=v-front>front · depth-masked (anomaly view)</figure></div>
  <h3 style="margin-top:12px">top camera full view: drag to set the policy crop (table only, no people)</h3>
  <div class=wrap><img id=full-top style="width:100%;border-radius:6px;cursor:crosshair"><div class=drag id=dr-full></div></div></div>
</div><div>
 <div class=card><h3>setup</h3>
  <div class=row>start <select id=c-home></select> WP1 <select id=c-wp1></select> WP2 <select id=c-wp2></select> drops <select id=c-drops></select></div>
  <div class=row>robot speed <input id=c-speed type=range min=10 max=100><span id=spv class=small></span> · drop turn ±<input id=c-jit type=number min=0 max=90 style="width:60px">°</div>
  <ol class=small>
   <li>On <a href="/arm" style="color:#8ab4f8">/arm</a> teach 4 routines (1 pose each, drops: 3-5): <b>home</b> = start pose, gripper open, NOT blocking the top camera's view of the plate · <b>wp1</b> = above the orange plate · <b>wp2</b> = brick in front of the front camera (every pose in it is photographed) · <b>drops</b> = ~3 cm above different spots of the orange plate.</li>
   <li>Put a brick on the orange plate. START. Match the leader, pick the brick. When the gripper closes on it the robot carries it to WP1 (phase 1, recorded for ACT), then to WP2 with the clamp unchanged and photographs it (phase 2), then drops it somewhere new.</li>
   <li>Fumbled? X discards it. Every pick is kept raw; only clean ones are exported.</li>
  </ol></div>
 <div class=card><h3>episodes</h3><div id=eps style="max-height:330px;overflow:auto"></div></div>
 <div class=card><h3>export cleaned dataset to Studio (for ACT)</h3>
  <div class=row>speed-up <input id=x-sp type=number value=1.5 step=0.1 min=1 max=3 style="width:65px">× · max grasp attempts <input id=x-at type=number value=1 min=1 max=5 style="width:55px">
  · name <input id=x-name placeholder="pick-bricks" style="width:130px"></div>
  <div class=row><button onclick="ex(true)">preview (dry run)</button><button class=g onclick="ex(false)">EXPORT to Studio</button></div>
  <pre id=xlog></pre></div>
</div></div>
<script>
const $=id=>document.getElementById(id);let q=Promise.resolve();
function p(u){q=q.then(()=>fetch(u,{method:'POST'}).then(r=>r.json()).then(show))}
function ex(dry){p(`/collect/export?speedup=${$('x-sp').value}&max_attempts=${$('x-at').value}&name=${encodeURIComponent($('x-name').value)}&dry=${dry}`)}
function live(id,url,ms){const im=$(id);const next=()=>setTimeout(()=>{im.src=url+'?t='+Date.now()},ms);im.onload=next;im.onerror=()=>setTimeout(next,1000);next()}
live('v-top','/collect/view/top',200);live('v-wrist','/collect/view/wrist',200);live('v-front','/collect/view/front',400);live('full-top','/snap/top',300);
const COL={idle:'#888',home:'#ffd23f',match:'#ffb44d',pick:'#ff5252',carry:'#ff5252',capture:'#5fd38d',reset_auto:'#8ab4f8',reset:'#ff8a3d',done:'#5fd38d',stopped:'#ff6b6b',error:'#ff6b6b'};
const TXT={home:'ROBOT GOING HOME',match:'MATCH THE LEADER TO THE ROBOT',pick:'PHASE 1 ● REC: PICK THE BRICK',carry:'PHASE 1 ● REC: ROBOT → WAYPOINT 1 (hands off)',capture:'PHASE 2: WAYPOINT 2, FRONT CAMERA (hands off)',reset_auto:'RESET: ROBOT DROPS THE BRICK',reset:'RESET THE SCENE, THEN NEXT',done:'DONE',stopped:'STOPPED',error:'ERROR',idle:'READY'};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let cfgShown=false;
function show(s){const b=$('banner');b.style.background=COL[s.state]||'#888';
 b.innerHTML=(TXT[s.state]||s.state)+(s.active?`  (${s.done}/${s.target})`:'')+`<div style="font-size:16px;font-weight:600;margin-top:6px">${esc(s.msg)}</div>`;
 $('gaps').innerHTML=s.info&&s.info.gaps?Object.entries(s.info.gaps).map(([k,v])=>`${k} <b style="color:${v<=8?'#5fd38d':'#ffb44d'}">${v}°</b>`).join(' · '):'';
 if(!cfgShown){cfgShown=true;for(const k of ['home','wp1','wp2','drops']){$('c-'+k).innerHTML=s.routines.map(r=>`<option ${r==s.cfg[k]?'selected':''}>${esc(r)}</option>`).join('')||'<option>(teach on /arm)</option>';
   $('c-'+k).onchange=()=>p(`/collect/cfg?${k}=`+encodeURIComponent($('c-'+k).value))}
  $('c-speed').value=Math.round(s.cfg.speed*100);$('spv').textContent=$('c-speed').value+'%';$('c-speed').onchange=()=>{$('spv').textContent=$('c-speed').value+'%';p('/collect/cfg?speed='+$('c-speed').value/100)};
  $('c-jit').value=s.cfg.roll_jitter;$('c-jit').onchange=()=>p('/collect/cfg?roll_jitter='+$('c-jit').value)}
 $('eps').innerHTML=s.sessions.map(se=>`<div class=small style="margin-top:6px"><b>${esc(se.session)}</b></div><table>`+se.episodes.map(e=>{const bad=e.outcome!='success';
  return `<tr><td>${esc(e.episode)}</td><td style="color:${bad?'var(--err)':'var(--ok)'}">${esc(e.outcome)}</td><td>${e.pick_s??''} s</td><td>${e.grasp_attempts??''} att.</td><td>${esc(e.handoff)}</td>
  <td>${bad?'':`<label><input type=checkbox ${e.keep===false?'':'checked'} onchange="p('/collect/keep?session=${encodeURIComponent(se.session)}&episode=${encodeURIComponent(e.episode)}&on='+this.checked)"> use</label>`}</td></tr>`}).join('')+'</table>').join('')||'<span class=small>none yet</span>';
 const xl=$('xlog');if(xl.textContent!=s.export_log){xl.textContent=s.export_log;xl.scrollTop=1e9}}
document.onkeydown=e=>{if(e.key==' '||e.key=='Escape'){e.preventDefault();p('/collect/stop');return}
 if(e.target.tagName=='INPUT'&&e.target.type!='checkbox'&&e.target.type!='range')return;
 const m={g:'grasped',x:'discard',k:'skip',n:'next'}[e.key.toLowerCase()];if(m)p('/collect/flag/'+m)};
(function(){let st=null;const pv=$('full-top'),dr=$('dr-full');
 const pos=e=>{const r=pv.getBoundingClientRect();return[Math.max(0,Math.min(r.width,e.clientX-r.left)),Math.max(0,Math.min(r.height,e.clientY-r.top)),r]};
 pv.addEventListener('mousedown',e=>{e.preventDefault();st=pos(e)});
 window.addEventListener('mousemove',e=>{if(!st)return;const[x,y]=pos(e);Object.assign(dr.style,{display:'block',left:Math.min(x,st[0])+'px',top:Math.min(y,st[1])+'px',width:Math.abs(x-st[0])+'px',height:Math.abs(y-st[1])+'px'})});
 window.addEventListener('mouseup',e=>{if(!st)return;const[x,y,r]=pos(e),sx=1920/r.width,sy=1080/r.height;dr.style.display='none';const a=st;st=null;
  if(Math.abs(x-a[0])<10||Math.abs(y-a[1])<10)return;
  p(`/collect/roi/top?x0=${Math.round(Math.min(x,a[0])*sx)}&y0=${Math.round(Math.min(y,a[1])*sy)}&x1=${Math.round(Math.max(x,a[0])*sx)}&y1=${Math.round(Math.max(y,a[1])*sy)}`)})})();
setInterval(()=>fetch('/collect/status').then(r=>r.json()).then(show),300);fetch('/collect/status').then(r=>r.json()).then(show);
</script></body></html>"""
