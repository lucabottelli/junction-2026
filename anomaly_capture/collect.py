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
from coverage import DEFAULTS as COV_DEFAULTS

FPS = 30
REC_CAMS = ["top", "wrist", "front"]     # stored per frame; ACT uses top + wrist (see prep.py)
MATCH_DEG = 8.0          # leader must be within this of the follower (every body joint) to start mirroring
HOLD_GAP = 20            # ticks: gripper commanded this much more closed than it actually is ... (real SO-101 squeezing a
                         # Duplo brick: 28-37, the leader's closed end limits it; 40 never triggered -> everything needed G)
HOLD_STILL = 4           # ... and not moving more than this over 0.25 s ...
HOLD_MIN_OPEN = 15       # ... and not fully shut (ticks above the closed end) = holding something
HOLD_TIME = 0.3          # s the above must hold
TAIL_S = 0.4             # keep recording this long after the grasp (the policy must learn to close firmly)
MAX_PICK_S = 90
PUT_PAN_DEG = 5.0        # put-back: shift the block sideways by up to this much shoulder pan (new start position each episode)
PUT_MIN_TURN = 0.3       # put-back: turn at least this fraction of roll_jitter, so every episode starts clearly different
REGRIP_OPEN = 0.9        # gripper opening (fraction of its range) when coming down on the released brick
REGRIP_SQUEEZE = 80      # ticks the re-grip closes past the width where the fingers stalled on the brick
RELEASE_TICKS = 200      # set-down: at table height the gripper opens only this far past the brick width (~18 deg of the
                         # moving jaw), so the jaw lets go without sweeping or tipping the brick; it opens fully once lifted
LIFT_FIRST = 0.3         # lifting off the tape spot: the first 30 % of the way up runs slowly (nothing is dragged along)
TURN_TOL_DEG = COV_DEFAULTS["margin_deg"]   # a taught tape_regrip must turn exactly 90 deg up to this (|roll err| + |pan
                         # change|): coverage.py credits every re-grip as exactly 90 deg with only this margin, more would
                         # credit directions that were really under the fingers
STEP_TIMEOUT_S = 120     # step-by-step re-grip test: no NEXT for this long -> the test stops, the arm holds (loaded, squeezing)
TAPE_SLOTS = ("tape_above", "tape_place", "tape_regrip", "tape_regrip_above")
DIAG_CAMS = ("wrist", "top")     # frames saved at every sub-step of the re-grip test
REGRIP_STEPS = {         # sub-steps of the tape set-down + re-grip (failure reasons and the step-by-step test name them)
    "above_tape": "above the tape spot, holding the brick",
    "lowered": "lowered: brick on the tape, still held",
    "released": "released: gripper opened a little at table height",
    "lifted": "lifted: straight up, brick left on the tape, gripper open",
    "turned_above": "turned 90° above the brick, gripper wide open",
    "above_open": "above the brick, gripper wide open (not turned)",
    "lowered_open": "lowered open: the jaws should straddle the brick",
    "closed": "closed until the fingers stalled, then squeezed",
    "lifted_held": "lifted straight up with the brick",
    "reopened": "not held: reopened at table height and lifted",
}
REGRIP_PLAN = ("above_tape", "lowered", "released", "lifted", "turned_above", "lowered_open", "closed", "lifted_held")


class StepTimeout(Stopped):
    """Step-by-step re-grip test: no NEXT in time. Handled exactly like STOP (the arm holds where it is)."""


class GraspDetector:
    """The automatic hand-off rule of episode() (and sorter.pick_teleop), as a reusable object for policy.py:
    the gripper is commanded HOLD_GAP ticks more closed than it actually is, has not moved more than HOLD_STILL
    over 0.25 s and is not fully shut (HOLD_MIN_OPEN above the closed end), continuously for HOLD_TIME.
    update(now, actual gripper, commanded gripper target) -> True once the fingers have stalled on an object."""
    def __init__(self, gmin):
        self.gmin, self.gq, self.since = gmin, [], None

    def update(self, now, g, c):
        self.gq = [(t, v) for t, v in self.gq if now - t < 0.25] + [(now, g)]
        still = max(v for _, v in self.gq) - min(v for _, v in self.gq) <= HOLD_STILL and now - self.gq[0][0] > 0.2
        self.since = (self.since or now) if (c < g - HOLD_GAP and still and g - self.gmin > HOLD_MIN_OPEN) else None
        return bool(self.since and now - self.since >= HOLD_TIME)


LIFT_J = ("shoulder_lift", "elbow_flex", "wrist_flex")   # the joints that raise / lower the gripper
LIFT_FRAC = 0.2          # hand-off when the closed gripper has been raised this fraction of the way from the grasp pose to WP1 (~3 cm)
STALL_FALLBACK_S = 1.5   # ... or when the fingers have stalled on the brick and the arm is held still this long (no lift)


class HandoffDetector:
    """When the robot takes over the pick (collection episode() and the ACT run-time in policy.py use the SAME rule):
      * LIFT  (main): the gripper is closed on something (commanded closed, fingers not fully shut) and the arm has been
               raised `lift_frac` of the way from the pose where it closed toward WP1 (measured on lift/elbow/wrist flex,
               projected on the grasp -> WP1 direction, so it needs no calibration) for 0.1 s;
      * STALL (fallback): GraspDetector's stall rule, held for `stall_s` (you closed firmly and wait without lifting).
    update(now, pos, goal) -> True once; then .why is "lift" or "auto". .lift = progress 0..1 for the UI."""
    def __init__(self, glim, wp1, lift_frac=LIFT_FRAC, stall_s=STALL_FALLBACK_S):
        self.gmin, self.gmax = glim; self.wp1, self.frac, self.stall_s = wp1, lift_frac, stall_s
        self.g0 = None; self.stall = GraspDetector(self.gmin); self.lift = 0.0; self.why = None; self.lift_since = None

    def update(self, now, pos, goal):
        g, c = pos["gripper"], goal["gripper"]
        gn = (c - self.gmin) / max(1, self.gmax - self.gmin) * 100
        closed_on = gn < 25 and g - self.gmin > HOLD_MIN_OPEN
        if gn > 40: self.g0 = None                                     # opened again: a new grasp attempt
        elif closed_on and self.g0 is None: self.g0 = {n: pos[n] for n in LIFT_J}
        self.lift = 0.0
        if self.g0 is not None and closed_on:
            d = {n: self.wp1[n] - self.g0[n] for n in LIFT_J}; dd = sum(v * v for v in d.values())
            if dd > (10 * TICK_PER_DEG) ** 2:                            # WP1 clearly away from the grasp pose
                self.lift = max(0.0, sum((pos[n] - self.g0[n]) * d[n] for n in LIFT_J) / dd)
        self.lift_since = (self.lift_since or now) if self.lift >= self.frac else None
        if self.lift_since and now - self.lift_since >= 0.1: self.why = "lift"; return True
        self.stall.update(now, g, c)
        if self.stall.since and now - self.stall.since >= self.stall_s: self.why = "auto"; return True
        return False


class Collector:
    def __init__(self, S, A, ctx, DATA, ROOT):
        self.S, self.A, self.ctx, self.DATA = S, A, ctx, DATA
        self.EP = ROOT / "episodes"; self.EP.mkdir(exist_ok=True)
        self.cfg_path = Path(__file__).resolve().parent / "collect_settings.json"
        self.fast = None        # fast mode: (target speed as a share of VMAX for a move at cfg speed, ramp seconds); sorter.apply_motion
        self.cfg = {"home": "home", "wp1": "wp1", "wp2": "wp2", "drops": "drops", "phase2": False, "regrasp": False, "speed": 0.5, "roll_jitter": 90, "lift_frac": LIFT_FRAC,
                    "act_roi": {k: prep.default_act_roi(k) for k in REC_CAMS},
                    # tape re-grip: opening past the brick width at release, optional contact detection from the following
                    # error of the lift joints when the open gripper comes down (deg; 0 = off), step-mode timeout
                    "release_ticks": RELEASE_TICKS, "regrip_contact_deg": 0.0, "step_timeout_s": STEP_TIMEOUT_S}
        try: self.cfg.update(json.loads(self.cfg_path.read_text()))
        except Exception: pass
        # where recorded episodes go: "episodes_dir" in collect_settings.json (default ~/junction26/episodes); read at start.
        # 2026-11-14 22:40: new pick dataset -> ~/junction26/tDataset2 (the user's request; old episodes/ stays as it is)
        if self.cfg.get("episodes_dir"):
            self.EP = Path(str(self.cfg["episodes_dir"])).expanduser(); self.EP.mkdir(parents=True, exist_ok=True)
        self.state, self.msg, self.ok = "idle", "ready: name the block and press Start", True
        self.info = {}
        self.active = False; self.stop_flag = threading.Event()
        self.flags = {"grasped": False, "discard": False, "skip": False, "next": threading.Event()}
        self.session = None; self.n_done = 0; self.target = 0; self.rec = None
        self.fails = 0; self.attempt = 0   # failed episodes in this run (paused for N, run continues) · episode number being attempted
        self.g_relock = None      # clamp command after the last tape re-grip (grip_on_tape)
        self.RUNS = ROOT / "runs"  # re-grip test diagnostics: runs/regrip_test_<time>/
        self.sub = None           # last finished sub-step of the tape set-down / re-grip (failure reasons name it)
        self.diag = None          # the last re-grip test: {dir, step_mode, active, waiting, i, rows, result, ...}
        self.regrip_info = None   # what the last regrip() did (the sorter logs it)
        self._tgt = None          # target of the last goto() (diagnostics: target vs actual)
        self._tape_snap = None    # tape slots frozen for one cycle / test / episode (freeze_tape): one turn direction

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
                fr[k] = prep.act_jpeg(f, roi[k])     # same bytes as before; policy.py decodes the same way
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
        for n in JOINTS:        # the gripper may be commanded past its calibrated end if the arm allows it (arm.set_grip_floor)
            lo = (getattr(self.A, "g_floor", None) or self.A.fl[n][0]) if n == "gripper" else self.A.fl[n][0]
            tgt[n] = min(self.A.fl[n][1], max(min(lo, self.A.fl[n][0]), tgt[n]))
        self._tgt = tgt
        # fast mode (sorter cfg motion_fast): long moves cruise at the target speed; slower moves keep their ratio to it
        self.A.fast = (self.fast[0] / self.cfg["speed"], self.fast[1]) if self.fast else None
        try: return self.A.goto(tgt, speed or self.cfg["speed"])
        finally: self.A.fast = None

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

    # ------------------------------------------------------------------ tape spot: set down + 90 deg re-grip
    def tslot(self, name):
        """A tape slot: the frozen copy while a cycle / test / episode runs (freeze_tape), else the live one."""
        if self._tape_snap is not None: return self._tape_snap.get(name) or []
        return self.slot(name)

    def freeze_tape(self):
        """Copy the tape slots once per inspect cycle / re-grip test / collection episode and validate them BEFORE the
        brick is set down (raises RuntimeError: nothing has moved yet). Every re-grip of the cycle then uses the same
        poses, so the turn direction can't change mid-cycle (coverage.py assumes it doesn't)."""
        self._tape_snap = None
        snap = {n: json.loads(json.dumps(self.slot(n))) for n in TAPE_SLOTS}
        self._tape_snap = snap
        try:
            self.regrip_poses(False); self.regrip_poses(True)
        except Exception:
            self._tape_snap = None; raise

    def thaw_tape(self): self._tape_snap = None

    def regrip_check(self):
        """None if the tape spot + re-grip poses are usable as taught now, else why not (for the START checks)."""
        if self._tape_snap is not None: return None
        try:
            self.regrip_poses(False); self.regrip_poses(True); return None
        except (RuntimeError, IndexError, KeyError) as e:
            return str(e) if not isinstance(e, IndexError) else "teach the tape spot (above + on) first"

    def tape_poses(self):
        """(above, place) of the tape spot as taught (raises IndexError if not taught)."""
        return self.tslot("tape_above")[0]["pos"], self.tslot("tape_place")[0]["pos"]

    def regrip_turn(self):
        """Change of the taught tape_regrip vs tape_place in degrees: {"roll", "pan"} (None if either is not taught).
        With the gripper pointing down, shoulder_pan turns the gripper about (nearly) the same vertical axis as the
        wrist roll, so the effective turn is roll +- pan (the sign depends on the servo directions)."""
        rg, pl = self.tslot("tape_regrip"), self.tslot("tape_place")
        if not (rg and pl): return None
        d = lambda n: (rg[0]["pos"][n] - pl[0]["pos"][n]) / TICK_PER_DEG
        return {"roll": d("wrist_roll"), "pan": d("shoulder_pan")}

    def regrip_turn_deg(self):
        t = self.regrip_turn(); return None if t is None else t["roll"]

    def regrip_problem(self):
        """Why the taught tape_regrip can't be used (None = fine or not taught). Fail-safe: coverage.py credits every
        re-grip as EXACTLY 90 deg (either direction) with a margin of only margin_deg, so the worst-case real turn,
        | |roll| - 90 | + |pan|, must stay within it; otherwise directions that were really under the fingers would be
        credited as seen and a defect there could reach the good plate."""
        t = self.regrip_turn()
        if t is None: return None
        er, ep = abs(abs(t["roll"]) - 90), abs(t["pan"])
        if er + ep <= TURN_TOL_DEG: return None
        parts = []
        if er >= 0.05: parts.append(f"wrist roll {t['roll']:+.1f}° from tape_place ({er:.1f}° off 90°)")
        if ep >= 0.05: parts.append(f"shoulder pan {t['pan']:+.1f}° from tape_place (pan also turns the gripper)")
        return (f"the taught re-grip pose (tape_regrip) is not a clean 90° turn: {'; '.join(parts)}; together "
                f"{er + ep:.1f}°, allowed {TURN_TOL_DEG:g}° (the coverage margin). Re-teach it (roll exactly ±90°, pan as "
                f"in tape_place), or delete it to use the computed turn")

    def regrip_status(self):
        t = self.regrip_turn()
        return {"taught": bool(self.tslot("tape_regrip")), "above": bool(self.tslot("tape_regrip_above")),
                "turn_deg": None if t is None else round(t["roll"], 1), "pan_deg": None if t is None else round(t["pan"], 1),
                "problem": self.regrip_problem()}

    def regrip_poses(self, turned=True):
        """(approach from above, grip pose, how) for gripping the brick that lies released on the tape spot.
          turned=False            -> tape_above / tape_place as taught ("as placed": pick it up as it lies)
          turned, tape_regrip     -> the taught pose ("taught"): jaws straddling the released brick, wrist turned ~90 deg.
                                     Approach: tape_regrip_above if taught, else tape_regrip at tape_above's height
                                     (lift / elbow / wrist flex of tape_above), always with tape_regrip's wrist roll,
                                     so the last motion is straight down.
          turned, not taught      -> tape_above / tape_place with the wrist rolled +90 deg, -90 deg if +90 would pass the
                                     roll limit ("computed"). The turn is about the wrist-roll axis, which is NOT where
                                     the brick is: the SO-101 has one fixed jaw and the brick was pressed against it.
        The turn direction is the same for every re-grip of a cycle (coverage.py tracks both directions, but assumes
        that each re-grip turns the same way)."""
        above, place = self.tape_poses()
        if not turned: return dict(above), dict(place), "as placed"
        rg = self.tslot("tape_regrip")
        if rg:
            prob = self.regrip_problem()
            if prob: raise RuntimeError(prob)
            down = {k: float(v) for k, v in rg[0]["pos"].items()}
            ra = self.tslot("tape_regrip_above")
            up = {k: float(v) for k, v in ra[0]["pos"].items()} if ra else {**down, **{n: float(above[n]) for n in LIFT_J}}
            up["wrist_roll"] = down["wrist_roll"]
            return up, down, "taught"
        lo, hi = self.A.fl["wrist_roll"]; r90 = 90 * TICK_PER_DEG; roll = place["wrist_roll"]
        if roll + r90 <= hi - 20: roll = roll + r90
        elif roll - r90 >= lo + 20: roll = roll - r90
        else: raise RuntimeError("tape_place's wrist roll leaves no room for a 90° turn either way: re-teach it nearer the middle")
        return {**above, "wrist_roll": roll}, {**place, "wrist_roll": roll}, "computed"

    def lift(self, to, g, slow=1.0):
        """Up off the tape spot from where the arm is: the first LIFT_FIRST of the way slowly, then the rest."""
        frm = dict(self.A.cmd)
        mid = {n: frm[n] + LIFT_FIRST * (float(to[n]) - frm[n]) for n in JOINTS}
        self.goto(mid, self.cfg["speed"] * 0.3 * slow, gripper=g)
        self.goto(to, self.cfg["speed"] * 0.5 * slow, gripper=g)

    def place_on_tape(self, g_lock, g_open):
        """Gently set the held brick down on the tape spot and let go; ends above the spot, gripper open (g_open).
        At table height the gripper opens only `release_ticks` past the brick width (the moving jaw lets go without
        sweeping or tipping the brick), the first part of the lift is slow, and it opens to g_open once up."""
        A = self.A; above, place = self.tape_poses(); spd = self.cfg["speed"]
        self.goto(above, gripper=g_lock); self.substep("above_tape")
        self.goto(place, spd * 0.3, gripper=g_lock); self.substep("lowered")        # slow, straight down = gentle
        gmax = A.fl["gripper"][1]
        g_rel = min(gmax, max(A.pos["gripper"], A.cmd["gripper"]) + float(self.cfg.get("release_ticks", RELEASE_TICKS)))
        self.goto(dict(A.cmd), 0.3, gripper=g_rel); time.sleep(0.2); self.substep("released")
        self.lift(above, g_rel)
        self.goto(dict(A.cmd), 0.5, gripper=max(g_rel, g_open)); self.substep("lifted")

    def grip_on_tape(self, g_open, turned=True, slow=1.0, open_low=False):
        """ONE attempt to grip the brick lying released on the tape spot (the arm is above it), then lift.
        turned=True: wrist turned 90 deg (the taught tape_regrip pose if there is one, see regrip_poses), so the faces
        that were under the fingers become visible.
        Works for any brick width: opens wide, comes straight down, closes until the fingers STALL on the brick, then
        squeezes a fixed amount past that width (a long brick turned 90 deg is much wider than in the first grip).
        A miss lets go at table height and lifts straight up (a half-gripped brick is never dragged up).
        open_low=True (retry after "slipped while lifting": holding() may have been a false negative): the gripper is
        NOT opened at height; it goes down closed to just above the brick (where the slow part of the lift starts),
        opens there, then continues as usual.
        Returns (held, clamp command, why): why is None when held, else '<sub-step>: what went wrong'."""
        A = self.A; spd = self.cfg["speed"]
        up, down, how = self.regrip_poses(turned)
        gmin, gmax = A.fl["gripper"]
        g_wide = max(g_open, gmin + REGRIP_OPEN * (gmax - gmin))
        self.check()
        if open_low:
            g_now = A.cmd["gripper"]; near = {n: down[n] + LIFT_FIRST * (up[n] - down[n]) for n in JOINTS}
            self.goto(up, spd * 0.5 * slow, gripper=g_now)                       # turn while up, gripper unchanged
            self.goto(near, spd * 0.3 * slow, gripper=g_now)                     # down to just above the brick
            self.goto(dict(A.cmd), 0.4, gripper=g_wide)                          # open only there
        else:
            self.goto(up, spd * 0.5 * slow, gripper=g_wide)                      # open wide + turn while up
        self.substep("turned_above" if turned else "above_open", pose=how)
        try:
            self.goto(down, spd * 0.3 * slow, gripper=g_wide)                    # straight down, open
        except RuntimeError as e:
            if "BLOCKED" not in str(e): raise
            self.goto(up, spd * 0.3, gripper=g_wide); self.substep("reopened")
            return False, None, f"lowered open: {e} (a jaw on the brick?)"
        lerr = max(abs(A.pos[n] - self._tgt[n]) for n in LIFT_J) / TICK_PER_DEG
        self.substep("lowered_open", pose=how, lift_err_deg=round(lerr, 1))
        cd = float(self.cfg.get("regrip_contact_deg") or 0)
        if cd > 0 and lerr > cd:                                                 # optional, off by default
            self.lift(up, g_wide, slow); self.substep("reopened")
            return False, None, (f"lowered open: the arm stopped {lerr:.0f}° short of the grip pose "
                                 f"(contact detection > {cd:g}°: a jaw is probably on the brick)")
        self.goto(dict(A.cmd), 0.4 * slow, gripper=gmin)                         # close until it stalls on the brick
        stall = A.pos["gripper"]
        g_new = max(gmin, stall - REGRIP_SQUEEZE)
        self.goto(dict(A.cmd), 0.5, gripper=g_new); time.sleep(0.2)
        held = self.holding(); self.substep("closed", stall=stall)
        if not held:
            why = ("closed: the fingers closed on nothing (the brick is not between the jaws)" if stall - gmin <= HOLD_MIN_OPEN
                   else f"closed: the fingers stopped {100 * (stall - gmin) / max(1, gmax - gmin):.0f} % open but do not hold the brick")
            self.goto(dict(A.cmd), 0.4, gripper=g_wide)                          # let go at table height ...
            self.lift(up, g_wide, slow); self.substep("reopened")                # ... then straight up
            return False, None, why
        self.lift(up, g_new, slow)                                               # straight up
        if not self.holding():
            self.substep("lifted_held")
            return False, None, "lifted: the brick slipped out of the fingers while lifting"
        self.substep("lifted_held")
        return True, g_new, None

    def regrip(self, g_open, turned=True, attempts=2):
        """grip_on_tape with a retry that is not a blind repeat: a failed attempt has already let go at table height and
        lifted straight up (a tipped brick settles, a brick the closing jaw pushed now lies against the fixed jaw's
        line), and the retry comes down and closes slower. The turn direction stays the same (coverage.py).
        Returns (held, clamp command or None, [why per failed attempt]); self.regrip_info records it for the logs."""
        self.regrip_info = None
        how = self.regrip_poses(turned)[2]                     # a badly taught pose raises here, before any motion
        whys, g = [], None
        for a in range(attempts):
            held, g, why = self.grip_on_tape(g_open, turned, slow=1.0 if a == 0 else 0.6,
                                             open_low=bool(whys) and whys[-1].startswith("lifted:"))
            if held: break
            whys.append(why)
            if a + 1 < attempts: self.say(f"re-grip attempt {a + 1} failed ({why}): reopened and lifted, trying once more, slower")
        else:
            held = False
        self.regrip_info = {"turned": turned, "pose": how, "attempts": len(whys) + (1 if held else 0), "held": held, "failed": whys}
        return held, (g if held else None), whys

    @staticmethod
    def why_text(whys):
        return "; ".join(f"attempt {i + 1}: {w}" for i, w in enumerate(whys))

    def regrasp_90(self, g_lock, g_open, on_released=None):
        """Set down on the tape spot, [on_released(): e.g. photograph the released brick from above], turn the wrist
        90 deg, grip again (one retry), lift. Returns True if the brick is held again; the new clamp command is
        self.g_relock, the reasons of failed attempts self.regrip_info["failed"]."""
        self.place_on_tape(g_lock, g_open)
        if on_released: on_released()
        held, self.g_relock, _ = self.regrip(g_open, turned=True)
        return held

    # ------------------------------------------------------------------ re-grip diagnostics (Test re-grip)
    def start_diag(self, step_mode):
        base = self.RUNS / datetime.now().strftime("regrip_test_%Y%m%d_%H%M%S"); d, n = base, 1
        while d.exists(): n += 1; d = base.with_name(f"{base.name}_{n}")
        d.mkdir(parents=True)
        A = self.A; held = self.holding(); self.regrip_info = None
        self.diag = {"dir": d, "step_mode": bool(step_mode), "active": True, "waiting": False, "i": 0, "rows": [],
                     "t0": time.time(), "result": None, "width0": A.pos.get("gripper") if held else None, "started_holding": held}
        self.info = {"regrip": self.diag_view()}
        return d

    def end_diag(self, result, ok):
        d = self.diag
        if not d: return
        try: held = bool(self.holding())
        except Exception: held = None
        d.update({"active": False, "waiting": False, "result": result, "ok": ok, "held_end": held})
        try:
            (d["dir"] / "summary.json").write_text(json.dumps({
                "result": result, "ok": ok, "holding_at_end": held, "step_mode": d["step_mode"], "started_holding": d["started_holding"],
                "first_grip_width_ticks": d["width0"], "regrip_pose": self.regrip_status(), "regrip": self.regrip_info,
                "steps": [r["step"] for r in d["rows"]], "limits": self.A.fl, "settings": {k: self.cfg.get(k) for k in (
                    "speed", "release_ticks", "regrip_contact_deg", "step_timeout_s")},
                "time": datetime.now().isoformat(timespec="seconds")}, indent=1))
        except Exception as e:
            print("re-grip test: summary not written:", e)
        self.info = {"regrip": self.diag_view()}

    def diag_view(self):
        d = self.diag
        if not d: return None
        return {"dir": f"{d['dir'].parent.name}/{d['dir'].name}", "step_mode": d["step_mode"], "active": d["active"],
                "waiting": d["waiting"], "rows": d["rows"], "plan": list(REGRIP_PLAN), "labels": REGRIP_STEPS,
                "result": d["result"], "ok": d.get("ok"),
                "holding": d.get("held_end") if not d["active"] else (d["rows"][-1]["holding"] if d["rows"] else None)}

    def substep(self, name, **extra):
        """A sub-step of the tape set-down / re-grip is done. Always: remember it. In the re-grip test also: log target vs
        actual joints (a lift / elbow error at tape_place = pressing on the table or stopped short), the gripper (stall
        position, opening, vs the first grip), holding(), and save camera frames to runs/regrip_test_<time>/; in step
        mode wait for NEXT (N) before going on. STOP or no NEXT for step_timeout_s raises (the arm holds)."""
        self.sub = name
        d = self.diag
        if not d or not d["active"]: return
        A = self.A; tgt = dict(self._tgt or A.cmd); pos = dict(A.pos)
        gmin, gmax = A.fl["gripper"]
        pct = lambda t: round(100 * (t - gmin) / max(1, gmax - gmin), 1)
        err = {n: round((pos[n] - tgt[n]) / TICK_PER_DEG, 1) for n in BODY}
        d["i"] += 1; i = d["i"]
        grip = {"cmd": round(float(A.cmd["gripper"]), 1), "pos": int(pos["gripper"]), "open_pct": pct(pos["gripper"]),
                "cmd_pct": pct(A.cmd["gripper"]), "jaw_deg_from_closed": round((pos["gripper"] - gmin) / TICK_PER_DEG, 1)}
        if "stall" in extra:
            st = extra.pop("stall"); grip.update({"stall": int(st), "stall_pct": pct(st)})
            if d["width0"] is not None: grip["stall_minus_first_grip_ticks"] = int(round(st - d["width0"]))
        rec = {"i": i, "step": name, "label": REGRIP_STEPS.get(name, name), "t_s": round(time.time() - d["t0"], 2),
               "target": {n: round(float(tgt[n]), 1) for n in JOINTS}, "actual": {n: int(pos[n]) for n in JOINTS},
               "err_deg": err, "max_err_deg": max(abs(v) for v in err.values()), "gripper": grip,
               "holding": bool(self.holding()), **extra, "frames": []}
        for cam in DIAG_CAMS:
            f = (getattr(self.S, "frames", None) or {}).get(cam)
            if f is None: continue
            fn = f"{i:02d}_{name}_{cam}.jpg"
            try:
                cv2.imwrite(str(d["dir"] / fn), f, [cv2.IMWRITE_JPEG_QUALITY, 85]); rec["frames"].append(fn)
            except Exception as e:
                rec.setdefault("frame_errors", []).append(f"{cam}: {e}")
        try:
            with open(d["dir"] / "steps.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
        except Exception as e:                           # a full disk must not end the test mid-sequence
            print("re-grip test: steps.jsonl not written:", e)
        d["rows"].append({"i": i, "step": name, "label": rec["label"], "max_err_deg": rec["max_err_deg"],
                          "lift_err_deg": {n: err[n] for n in LIFT_J}, "grip_pct": grip["open_pct"],
                          "stall_pct": grip.get("stall_pct"), "stall_vs_first": grip.get("stall_minus_first_grip_ticks"),
                          "holding": rec["holding"], "frames": rec["frames"]})
        self.info = {"regrip": self.diag_view()}
        if not d["step_mode"]: return
        self.flags["next"].clear(); d["waiting"] = True; self.info = {"regrip": self.diag_view()}
        le = ", ".join(f"{n.split('_')[0]} {err[n]:+.1f}°" for n in LIFT_J)
        self.say(f"STEP {i}: {rec['label']} · error {le} · gripper {grip['open_pct']:.0f} % open · "
                 f"{'holding' if rec['holding'] else 'not holding'} · N = next step, STOP ends the test")
        lim = float(self.cfg.get("step_timeout_s") or STEP_TIMEOUT_S); t0 = time.monotonic()
        try:
            while not self.flags["next"].wait(0.1):
                self.check()
                if time.monotonic() - t0 > lim: raise StepTimeout(f"no NEXT for {lim:g} s")
            self.check()
        finally:
            d["waiting"] = False; self.info = {"regrip": self.diag_view()}

    def up_wp2(self, wp1, wp2, g_lock):
        """Straight UP out of WP2 to the pose above it."""
        above, _ = self.above_wp2(wp1, wp2[-1]["pos"])
        self.goto(above, self.cfg["speed"] * 0.5, gripper=g_lock)

    def to_wp2(self, wp1, wp2, g_lock):
        """WP1 -> above WP2 (at height) -> straight DOWN into WP2 as the very last motion, slower."""
        above, taught = self.above_wp2(wp1, wp2[0]["pos"])
        # "Above WP2" is optional (a WP2 held high needs no straight drop): without it the arm comes in over WP2 at
        # WP1's arm height, which is what above_wp2() returns then. No warning.
        self.goto(above, gripper=g_lock)
        self.check()
        self.goto(wp2[0]["pos"], self.cfg["speed"] * 0.5, gripper=g_lock)

    def from_wp2(self, wp1, wp2, g_lock):
        """Reverse: straight UP out of WP2 first, then over to WP1 (never sweeps across what's below WP2)."""
        above, _ = self.above_wp2(wp1, wp2[-1]["pos"])
        self.goto(above, self.cfg["speed"] * 0.5, gripper=g_lock)
        self.goto(wp1, gripper=g_lock)

    def test_wp2(self, kind="wp2", step=False):
        """Dry runs with whatever is in the gripper (nothing saved to the datasets, <=30 % speed).
        kind=wp2: WP1 -> above WP2 -> down into WP2 -> up -> WP1.
        kind=regrasp: tape spot set-down + 90 deg re-grip (with the retry), diagnosed: every sub-step is logged with
        target vs actual joints, gripper, holding() and wrist/top frames to runs/regrip_test_<time>/ (steps.jsonl,
        summary.json, jpgs). step=True: pauses after every sub-step until NEXT (N)."""
        A = self.A
        self.active = True; A.running = True; self.stop_flag.clear(); A.stop.clear()
        self.state = "regrasp" if kind == "regrasp" else "capture"
        diag = False; self.sub = None
        try:
            if A.mode != "hold": A.set_mode("hold", force=True)
            g = A.cmd["gripper"]
            spd = self.cfg["speed"]; self.cfg["speed"] = min(spd, 0.3)
            fast = self.fast; self.fast = None                       # dry runs stay slow whatever the fast mode says
            try:
                if kind == "regrasp":
                    if not (self.slot("tape_above") and self.slot("tape_place")): raise RuntimeError("teach the tape spot (above + on) first")
                    self.freeze_tape()                   # validates every pose BEFORE the brick is set down
                    gmin, gmax = A.fl["gripper"]
                    g_open = max(self.wps("home")[0]["pos"]["gripper"], gmin + 0.35 * (gmax - gmin))
                    d = self.start_diag(step); diag = True
                    how = self.regrip_poses(True)[2]
                    self.say(f"TEST: set down on the tape spot, re-grip turned 90° ({how} pose)"
                             + (" · step by step: N after each step" if step else "") + f" · log: runs/{d.name}")
                    ok = self.regrasp_90(g, g_open)
                    if not ok: raise RuntimeError("re-grip test: brick NOT held after re-gripping ("
                                                  + self.why_text((self.regrip_info or {}).get("failed", [])) + ")")
                else:
                    wp1 = self.wps("wp1")[0]["pos"]; wp2 = self.wps("wp2")
                    self.say("TEST: to WP1"); self.goto(wp1, gripper=g)
                    self.say("TEST: above WP2, then straight down"); self.to_wp2(wp1, wp2, g)
                    for w in wp2[1:]: self.goto(w["pos"], gripper=g)
                    self.say("TEST: straight up, back to WP1"); self.from_wp2(wp1, wp2, g)
            finally:
                self.cfg["speed"] = spd; self.fast = fast
            self.state = "done"
            if kind == "regrasp":
                tries = (self.regrip_info or {}).get("attempts", 1)
                self.say(f"✓ test of the tape re-grip done: brick held again{f' (needed {tries} attempts)' if tries > 1 else ''}"
                         f" · log: runs/{self.diag['dir'].name}")
            else:
                self.say("✓ test of the WP1 → WP2 path done (nothing saved)")
            if diag: self.end_diag("held again", True)
        except Stopped as e:
            self.state = "stopped"
            m = f"STOPPED (step mode: {e}): arm holds its position" if isinstance(e, StepTimeout) else "STOPPED: arm holds its position"
            self.say(m + (f" · after {self.sub}" if diag and self.sub else ""), False)
            if diag: self.end_diag(f"stopped after {self.sub}" + (f" ({e})" if isinstance(e, StepTimeout) else ""), False)
        except Exception as e:
            self.state = "error"; self.say(f"ERROR: {e}", False)
            if diag: self.end_diag(f"error: {e}", False)
        finally:
            self.thaw_tape()
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
            signed = {n: (lt[n] - A.cmd[n]) / TICK_PER_DEG for n in BODY}       # leader - follower, degrees
            gaps = {n: abs(v) for n, v in signed.items()}
            self.info = {"gaps": {k: round(v, 1) for k, v in gaps.items()}, "signed": {k: round(v, 1) for k, v in signed.items()}}
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
        det = HandoffDetector((gmin, gmax), wp1, float(self.cfg.get("lift_frac", LIFT_FRAC))); reason = None
        rec.start()
        try:
            while True:
                self.check()
                now = time.time()
                g, c = A.pos["gripper"], A.goal["gripper"]
                gn = (c - gmin) / max(1, gmax - gmin) * 100
                if was_open and gn < 25: attempts += 1; was_open = False
                elif gn > 40: was_open = True
                handed = det.update(now, A.pos, A.goal)
                self.info = {"attempts": attempts, "t": round(now - t0, 1), "frames": len(buf["t"]),
                             "lift": round(min(1.0, det.lift / max(1e-6, det.frac)) * 100)}
                self.say(f"● REC  pick the brick, then lift it a little: the robot takes over  ·  {now - t0:.1f} s  ·  grasp attempts {attempts}"
                         + (f"  ·  lift {self.info['lift']} %" if det.g0 is not None else ""))
                if handed: reason = det.why; break
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
                self.state = "carry"; self.say("● REC  robot carries the brick to waypoint 1: let go of the leader and put it back in its rest position")
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
            self.recover(reason, "pick discarded (X)" if reason == "discard" else f"no grasp within {MAX_PICK_S} s")
            return False

        if not self.holding():
            meta["outcome"] = "missed"
            threading.Thread(target=self._write, args=(self.EP / self.session / ep_id, buf, meta), daemon=True).start()
            self.recover("missed", "the brick was not in the gripper at waypoint 1 (episode not used for training)")
            return False
        meta["outcome"] = "success"; meta["gripper_lock"] = round(g_lock, 1)
        threading.Thread(target=self._write, args=(self.EP / self.session / ep_id, buf, meta), daemon=True).start()
        self.n_done += 1          # counted the moment it is saved (a STOP / BLOCKED during the put-back must not lose it)

        # 2 PHASE 2: WP1 -> above WP2 -> down -> photo -> up -> tape spot: set down, re-grip turned 90 deg
        #   -> above WP2 -> down -> photo of the rotated brick -> up.  Clamp command identical throughout.
        n_img, on_tape = 0, False
        if self.cfg.get("phase2"):
            n_img, on_tape = self.phase2(wp1, wp2, g_lock, g_open, label, brick, idx)
            if not on_tape and self.g_relock is not None: g_lock = self.g_relock      # re-gripped: keep the new clamp
        self.info = {"images": n_img}
        self.reset(home, wp1, drops, grasp_pose, g_lock, g_open, on_tape)
        return True

    def phase2(self, wp1, wp2, g_lock, g_open, label, brick, idx):
        """OPTIONAL: WP1 -> above WP2 -> down -> photo -> up [-> tape spot re-grip 90 deg -> photo 2]."""
        self.state = "capture"; n_img = 0; self.g_relock = None
        cams = [c for c in self.cfg.get("inspect_cams", ["front"]) if c in self.S.live]
        tape = self.cfg.get("regrasp") and self.slot("tape_above") and self.slot("tape_place")
        if tape:
            try: self.freeze_tape()              # this episode's tape poses, validated before anything is set down
            except Exception as e:
                tape = False; self.say(f"phase 2: re-grip skipped ({e})", False)

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
            rel_cams = [c for c in self.cfg.get("release_cams", ["wrist"]) if c in self.S.live]

            def released():       # the brick lies on the tape spot, gripper open above it: top-view photo (training data)
                nonlocal n_img
                if not rel_cams: return
                time.sleep(0.25); self.check()
                self.say("phase 2: photo of the released brick from above (tape spot)")
                n_img += self.ctx.capture(rel_cams, label, brick, "tape_top", idx, self.session, kind="release")
            try: ok = self.regrasp_90(g_lock, g_open, on_released=released)
            finally: self.thaw_tape()
            if ok:
                g_lock = self.g_relock          # the turned brick can have another width: keep the new clamp
                self.state = "capture"
                self.to_wp2(wp1, wp2, g_lock)
                # a retry may have spun or tipped the brick: its orientation is not verified -> marked in the view name
                # (meta.jsonl "view"), so these photos can be kept but told apart
                retried = (self.regrip_info or {}).get("attempts", 1) > 1
                n_img += shoot("_r90retry" if retried else "_r90", "(turned 90°" + (", after a retry: orientation not verified)" if retried else ")"))
                self.up_wp2(wp1, wp2, g_lock)
            else:
                on_tape = True
                self.say("re-grip FAILED (" + self.why_text((self.regrip_info or {}).get("failed", [])) + "): the brick is left on "
                         "the tape spot (put it back on the orange plate)", False)
        return n_img, on_tape

    def reset(self, home, wp1, drops, grasp_pose, g_lock, g_open, on_tape):
        """Collection only: put the brick back on the orange plate so the next pick starts from a NEW pose
        (what the pick policy needs: many different start positions and orientations, each picked by the human).
        Reset spots taught -> set it down at a random one. Otherwise -> back where it was picked (that path is known
        to be clear and the height is right), shifted sideways a little and turned clearly, set down slowly."""
        A = self.A; self.state = "reset_auto"
        if on_tape:
            self.goto(home, gripper=g_open); time.sleep(1.0); return
        J = self.cfg["roll_jitter"]
        turn = random.choice((-1, 1)) * random.uniform(PUT_MIN_TURN * J, J) if J > 0 else 0.0     # degrees
        leader_rest = " · meanwhile put the leader back in its rest position"

        def step(i, txt):
            self.info = {"step": f"{i}/4 {txt}", "turn": round(turn)}
            self.say(f"putting the block back {i}/4: {txt}{leader_rest}")
        step(1, "up to waypoint 1")
        self.goto(wp1, gripper=g_lock)
        if drops:
            d = random.choice(drops)["pos"]
            step(2, f"over a random reset spot, turning the block {turn:+.0f}°")
            self.goto(d, gripper=g_lock, wrist_roll=d["wrist_roll"] + turn * TICK_PER_DEG)
            step(3, "setting it down")
            self.goto(dict(A.cmd), 0.4, gripper=g_open); time.sleep(0.2)                       # release
            step(4, "letting go, back to the start pose")
            self.rise(wp1)
        else:
            pan = grasp_pose["shoulder_pan"]
            shift = random.uniform(-1, 1) * PUT_PAN_DEG * TICK_PER_DEG
            shift += 0.3 * max(-PUT_PAN_DEG * TICK_PER_DEG, min(PUT_PAN_DEG * TICK_PER_DEG, wp1["shoulder_pan"] - pan))  # drift toward the plate centre (WP1), never off the edge
            spot = {**grasp_pose, "shoulder_pan": pan + shift, "wrist_roll": grasp_pose["wrist_roll"] + turn * TICK_PER_DEG}
            above = {**spot, "shoulder_lift": wp1["shoulder_lift"], "elbow_flex": wp1["elbow_flex"], "wrist_flex": wp1["wrist_flex"]}
            step(2, f"over the pick spot (moved {shift / TICK_PER_DEG:+.0f}° sideways), turning the block {turn:+.0f}°")
            self.goto(above, gripper=g_lock)
            step(3, "lowering it slowly onto the plate")
            self.goto(spot, self.cfg["speed"] * 0.3, gripper=g_lock)                        # gently down
            self.goto(dict(A.cmd), 0.4, gripper=g_open); time.sleep(0.2)                      # let go
            step(4, "letting go, straight up, back to the start pose")
            self.goto(above, self.cfg["speed"] * 0.6, gripper=g_open)                       # straight up
        self.goto(home, gripper=g_open)
        self.info = {}

    # ------------------------------------------------------------------ recovery: a failed episode pauses, it never ends the run
    WHY = {"blocked": "BLOCKED", "missed": "MISSED", "discard": "DISCARDED", "timeout": "TIMED OUT", "error": "ERROR"}

    def recover(self, why, detail):
        """Pause the run so the human can fix the scene: the follower copies the leader (creeping slowly at first),
        N continues with the next episode. Only STOP (Stopped) or a lost arm/leader (exception) ends the run."""
        A = self.A; self.fails += 1
        self.state = "reset"; self.flags["next"].clear(); self.info = {"why": why, "detail": detail}
        self.say(f"{self.WHY.get(why, why.upper())}: {detail} · the robot copies the leader slowly: fix the scene, "
                 "put the brick back on the orange plate, put the leader in its rest position, then N", False)
        if A.f is None: raise RuntimeError("the arm is disconnected")
        A.set_mode("mirror", force=True)
        while not self.flags["next"].wait(0.1): self.check()
        A.set_mode("hold", force=True)
        self.info = {}

    # ------------------------------------------------------------------ session (resumable: session.json + episodes on disk)
    def session_info(self, name):
        d = self.EP / name
        try: si = json.loads((d / "session.json").read_text())
        except Exception: return None
        done = 0; n_eps = 0
        for md in d.glob("*/meta.json"):
            n_eps += 1
            try: done += json.loads(md.read_text()).get("outcome") == "success"
            except Exception: pass
        return {"session": name, "brick": si.get("brick"), "label": si.get("label"), "done": done,
                "target": int(si.get("target") or 0), "episodes": n_eps}

    def resumable(self):
        """The most recent session, if it ended before reaching its target (survives server restarts)."""
        if self.active: return None
        for d in sorted((x for x in self.EP.glob("*") if (x / "session.json").exists()), key=lambda x: x.name, reverse=True)[:1]:
            si = self.session_info(d.name)
            if si and si["done"] < si["target"]: return {k: si[k] for k in ("session", "brick", "label", "done", "target")}
        return None

    def run(self, n, brick, label, resume=None):
        self.active = True; self.A.running = True; self.stop_flag.clear(); self.A.stop.clear()
        self.fails = 0; self.info = {}
        si = self.session_info(resume) if resume else None
        if si:
            self.session = resume; self.n_done = si["done"]; idx = si["episodes"]
            brick, label = si["brick"] or brick, si["label"] or label
        else:
            self.session = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{brick}"; self.n_done = 0; idx = 0
        self.target = n
        sj = self.EP / self.session / "session.json"; sj.parent.mkdir(parents=True, exist_ok=True)
        try: started = json.loads(sj.read_text()).get("started")
        except Exception: started = None
        sj.write_text(json.dumps({"brick": brick, "label": label, "target": n,
                                  "started": started or datetime.now().isoformat(timespec="seconds")}, indent=1))
        try:
            while self.n_done < n:
                idx += 1; self.attempt = idx
                try:
                    self.episode(idx, brick, label)       # counts its own success in n_done when it is saved
                except Stopped: raise
                except Exception as e:                       # BLOCKED, motion error, ...: pause for the human, then go on
                    m = str(e)
                    self.recover("blocked" if "BLOCKED" in m else "error", m.replace("BLOCKED: ", "", 1))
            self.state = "done"; self.say(f"✓ DONE: {self.n_done} successful episodes in {self.session}"
                                          + (f" ({self.fails} failed attempt{'s' * (self.fails != 1)} skipped)" if self.fails else ""))
        except Stopped:
            self.state = "stopped"; self.say(f"STOPPED: {self.n_done} of {self.target} episodes saved, arm holds its position · RESUME continues this session", False)
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
                "export_running": export["proc"] is not None, "export_log": export["log"][-4000:],
                "resume": C.resumable(), "fails": C.fails, "attempt": C.attempt, "regrip_pose": C.regrip_status()}

    @r.get("/collect")
    def page(): return RedirectResponse("/record")      # the UI lives at / now (ui.py, web/)

    @r.get("/collect/counts")
    def counts():
        """Front-camera anomaly images per brick and label (originals only, not _depth/_masked)."""
        out = {}
        for lab in ("good", "defect"):
            for f in (DATA / "front" / "inhand" / lab).glob("*.png"):
                if f.stem.endswith(("_depth", "_crop")): continue
                parts = f.stem.split("_")
                b = parts[3] if len(parts) > 4 else "?"
                out.setdefault(b, {"good": 0, "defect": 0})[lab] += 1
        return out

    @r.get("/collect/status")
    def st(): return status()

    @r.post("/collect/start")
    def start(n: int = 10, brick: str = "", label: str = "good", resume: str = ""):
        if C.active: C.say("already running", False); return status()
        if A.f is None: C.say("connect the arm first (step 1, Connect)", False); return status()
        if resume:
            si = C.session_info(Path(resume).name)
            if not si: C.say(f"can't resume '{resume}': no such session", False); return status()
            brick, label, resume = si["brick"] or brick, si["label"] or label, si["session"]
        if not brick.strip(): C.say("type the brick name first", False); return status()
        try:
            for k in ("home", "wp1"): C.wps(k)
            if C.cfg.get("phase2"): C.wps("wp2")
            if C.cfg.get("phase2") and C.cfg.get("regrasp") and not (C.slot("tape_above") and C.slot("tape_place")):
                raise RuntimeError("re-grip is on: teach the tape spot (above + on) first, or switch the re-grip off")
            if C.cfg.get("phase2") and C.cfg.get("regrasp") and C.regrip_check(): raise RuntimeError(C.regrip_check())
        except Exception as e:
            C.say(str(e), False); return status()
        threading.Thread(target=C.run, args=(max(1, n), brick.strip(), label, resume or None), daemon=True).start()
        time.sleep(0.1); return status()

    @r.post("/collect/test_wp2")
    def test_wp2(kind: str = "wp2", step: bool = False):
        if C.active or A.running: C.say("busy: stop first", False); return status()
        if A.f is None: C.say("connect the arm first", False); return status()
        threading.Thread(target=C.test_wp2, args=(kind, step), daemon=True).start(); time.sleep(0.1); return status()

    @r.get("/collect/regrip/frame")
    def regrip_frame(i: int = 0, cam: str = "wrist"):
        """A camera frame saved by the re-grip test at sub-step i (0 = the latest one); 204 if there is none."""
        d = C.diag
        rows = [x for x in (d["rows"] if d else []) if x["frames"] and (not i or x["i"] == i)]
        if not rows: return Response(status_code=204)
        fn = next((f for f in rows[-1]["frames"] if f.endswith(f"_{cam}.jpg")), None)
        if fn is None or not (d["dir"] / fn).exists(): return Response(status_code=404)      # never another camera's frame
        p = d["dir"] / fn
        return Response(p.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @r.post("/collect/stop")
    def stop():
        C.stop_flag.set(); A.stop.set(); C.flags["next"].set()
        time.sleep(0.06)
        if A.f:
            with A.io: A.mode = "hold"; A.goal = dict(A.cmd)
        C.say(f"STOPPED: {C.n_done} of {C.target} episodes saved, arm holds its position · RESUME continues this session"
              if C.session and C.target and C.n_done < C.target else "STOPPED: arm holds its position", False)
        return status()

    @r.post("/collect/flag/{flag}")
    def flag(flag: str):
        if flag in ("grasped", "discard", "skip"): C.flags[flag] = True
        elif flag == "next": C.flags["next"].set()
        return status()

    @r.post("/collect/cfg")
    def cfg(home: str = "", wp1: str = "", wp2: str = "", drops: str = "", speed: float = 0,
            roll_jitter: float = -1, inspect_cams: str = "", phase2: str = "", regrasp: str = "",
            release_cams: str = None, lift_frac: float = -1, release_ticks: float = -1, regrip_contact_deg: float = -1,
            step_timeout_s: float = -1):
        if phase2: C.cfg["phase2"] = phase2 == "true"
        if regrasp: C.cfg["regrasp"] = regrasp == "true"
        if home: C.cfg["home"] = home
        if wp1: C.cfg["wp1"] = wp1
        if wp2: C.cfg["wp2"] = wp2
        if drops: C.cfg["drops"] = drops
        if speed: C.cfg["speed"] = min(1.0, max(0.1, speed))
        if roll_jitter >= 0: C.cfg["roll_jitter"] = min(90, roll_jitter)
        if lift_frac >= 0: C.cfg["lift_frac"] = min(0.8, max(0.05, lift_frac))
        if release_ticks >= 0: C.cfg["release_ticks"] = min(1000.0, max(60.0, release_ticks))
        if regrip_contact_deg >= 0: C.cfg["regrip_contact_deg"] = min(30.0, regrip_contact_deg)      # 0 = off
        if step_timeout_s >= 0: C.cfg["step_timeout_s"] = min(3600.0, max(0.2, step_timeout_s))
        if inspect_cams: C.cfg["inspect_cams"] = [c for c in inspect_cams.split(",") if c]
        if release_cams is not None: C.cfg["release_cams"] = [c for c in release_cams.split(",") if c]
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
    def view(cam: str, full: bool = False):
        """Exactly what the policy sees (top/wrist) - or, for front, the depth-masked in-hand view.
        full=true: the whole clean frame (no crop, no overlays), used by the crop editor on /arm."""
        f = S.frames.get(cam)
        if f is None: return Response(status_code=204)
        if full:
            img = cv2.resize(f, (960, 960 * f.shape[0] // f.shape[1]), interpolation=cv2.INTER_AREA)
        elif cam == "front" and getattr(S.cams.get("front"), "depth", None) is not None:
            m, _ = prep.inhand_mask(S.cams["front"].depth)
            m = prep.complete_mask(f, m)
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
                "--speedup", str(speedup), "--max-attempts", str(max_attempts), "--episodes", str(C.EP)]
        if name: args += ["--name", name]
        if bricks: args += ["--bricks"] + [b for b in bricks.split(",") if b]
        if dry: args += ["--dry-run"]
        export["proc"] = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        export["log"] = " ".join(args[2:]) + "\n"

        def pump(p=export["proc"]):
            for line in p.stdout: export["log"] += line
        threading.Thread(target=pump, daemon=True).start()
        return status()

    r.collector = C          # sorter.py reuses the collector's motion helpers (WP2 approach, tape re-grip)
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
<span class=small>brick</span><input id=brick placeholder="e.g. pink" style="width:100px">
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
