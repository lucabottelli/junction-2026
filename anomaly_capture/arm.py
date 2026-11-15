"""Teach-and-replay arm routine with automatic in-hand image capture (Junction 2026, SO-101).

Served by capture.py at http://127.0.0.1:7870/arm (same process, so it shares the locked cameras).

Idea: the robot captures its own training data with EXACTLY the poses it will use when sorting.
  1. TEACH  - "mirror leader": the follower copies the leader arm live. Move to a pose, click "add waypoint"
              (tick 📷 on poses where the front camera should take a picture). Gripper open/close is part of
              each waypoint. Alternative: "limp" = follower torque off, guide it by hand (hold it, it drops).
  2. RUN    - replays the waypoints N times with smooth (minimum-jerk) motion and saves an averaged,
              full-resolution frame from the chosen cameras at every 📷 waypoint.
              If the routine ends by putting the brick back on the pick spot, it can loop unattended.
  3. STOP   - Space/Esc or the STOP button: the arm freezes and holds its position (torque stays on).

Positions are raw servo ticks of the follower (no calibration file needed: the routine replays exactly what was
taught). Leader->follower mirroring uses the same convention as LeRobot/Studio ("degrees" mode: offset from the
middle of each joint's recorded range; gripper: fraction of its range), read from each servo's EEPROM limits.
"""
import json, os, random, threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

import prep
if os.environ.get("JUNCTION_MOCK") == "1":      # hardware-free mock station: simulated servos (mock_hw.py, docs/DEV.md)
    from mock_hw import FeetechMotorsBus, Motor, MotorNormMode
else:
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY = JOINTS[:-1]
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_{}-if00"
FOLLOWER, LEADER = "5B3D040988", "5B14111392"
HERE = Path(__file__).resolve().parent
ROUTINES = HERE / "routines.json"
GRIP_FLOOR = HERE / "gripper_floor.json"      # {"calibrated_min", "servo_min"}: written by Arm.set_grip_floor
VMAX = 1500          # ticks/s at 100 % speed (~130 deg/s); every command is rate-limited to the current speed
MIRROR_VMAX = 1200   # live leader following
ENGAGE_VMAX = 250    # first 3 s of mirroring: creep to the leader's pose instead of jumping
TOL = 40             # ticks (~3.5 deg): "arrived" for body joints (P-gain 16 leaves some gravity sag)
BLOCKED = 160        # ticks (~14 deg): still this far off after settling = something is in the way -> stop
TICK_PER_DEG = 4096 / 360
FAST_VMAX = 1.5      # x VMAX: the most a fast-mode cruise may ask for (~195 deg/s)
CORNER = 120         # ticks (~10 deg): Arm.path rounds each via point over this much travel before and after it; the
                     # path then passes about 0.35 x this (~3.5 deg) inside the corner instead of stopping on it


def profile(d, vmax, ramp=0.0, cruise=None):
    """Time law of one move over d ticks (the joint that travels furthest) -> (T seconds, f(t) -> share of the way).
    Default: minimum jerk. Peak speed = vmax, mean = vmax / 1.875 (it never stays at its speed).
    ramp > 0 (fast mode): the speed rises smoothly to `cruise` ticks/s in `ramp` s (smoothstep: the acceleration
    starts and ends at 0, peak 1.5 * cruise / ramp), stays there, and falls the same way; T = d / cruise + ramp.
    Not for a move of the jaws alone (Arm.goto). Used only when the move is long enough to reach the cruise speed and comes out shorter than minimum jerk:
    short moves (the corrected steps of a pick, a set-down) keep the minimum-jerk law at their own speed."""
    T = max(0.25, 1.875 * d / vmax)
    v = cruise or vmax
    if ramp > 0 and d >= v * ramp and d / v + ramp < T:
        T = d / v + ramp; k = v * ramp / d                 # share of the way spent in the two ramps together
        def f(t):
            if t >= T: return 1.0
            if t < ramp: u = max(0.0, t) / ramp; return k * (u ** 3 - u ** 4 / 2)
            if t > T - ramp: u = (T - t) / ramp; return 1 - k * (u ** 3 - u ** 4 / 2)
            return k / 2 + (t - ramp) * v / d
        return T, f, v
    def f(t):
        s = min(1.0, max(0.0, t / T)); return s ** 3 * (10 - 15 * s + 6 * s * s)
    return T, f, vmax


EDIT_BUSY = "the arm is running a routine / collection / sort: STOP it first, then change waypoints"   # poses in use


def _open(serial):
    b = FeetechMotorsBus(port=PORT.format(serial),
                         motors={n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(JOINTS)})
    b.connect(handshake=False)
    return b


def _limits(b):
    return {n: (b.read("Min_Position_Limit", n, normalize=False), b.read("Max_Position_Limit", n, normalize=False))
            for n in JOINTS}


class Interrupted(Exception):
    """A move was cut short on purpose (Arm.intr). Not a stop: the caller decides where the arm goes next."""


class Stopped(Exception):
    pass


class Arm:
    def __init__(self):
        self.f = self.l = None
        self.fl = self.ll = None; self.g_floor = None
        self.io = threading.Lock()
        self.mode = "off"             # off | hold | limp | mirror
        self.pos, self.goal, self.cmd, self.lpos = {}, {}, {}, {}
        self.vmax = VMAX * 0.5
        self.fast = None              # fast mode for the next goto: (speed scale, ramp seconds), set by Collector.goto
        self.mirror_t = 0.0
        self.engaged = False
        self.bus_errors = 0
        self.stop = threading.Event()
        self.running = False
        self.prog = {}

    # ---------------------------------------------------------------- connection / modes
    def connect(self):
        if self.f: return "already connected"
        f = _open(FOLLOWER)
        tq = f.sync_read("Torque_Enable", normalize=False)
        if not any(tq.values()):      # same servo tuning LeRobot applies on every connect (needs torque off)
            for m in JOINTS:
                f.write("P_Coefficient", m, 16, normalize=False)
                f.write("I_Coefficient", m, 0, normalize=False)
                f.write("D_Coefficient", m, 32, normalize=False)
            f.write("Max_Torque_Limit", "gripper", 500, normalize=False)   # don't cook the gripper on a brick
            f.write("Protection_Current", "gripper", 250, normalize=False)
            f.write("Overload_Torque", "gripper", 25, normalize=False)
        self.fl = _limits(f)
        self.g_floor = self.fl["gripper"][0]         # lowest gripper command the loop sends (see set_grip_floor)
        try:
            cal = json.loads(GRIP_FLOOR.read_text())["calibrated_min"]
            if self.fl["gripper"][0] < cal:          # the servo was allowed to close past its calibrated end:
                self.fl["gripper"] = (cal, self.fl["gripper"][1])      # every rule and normalisation keeps the calibrated one
        except Exception: pass
        p = f.sync_read("Present_Position", normalize=False)
        f.sync_write("Goal_Position", p, normalize=False)    # hold exactly where it is: no jump on torque-on
        f.enable_torque()
        self.pos, self.goal, self.cmd = dict(p), dict(p), {k: float(v) for k, v in p.items()}
        self.f, self.mode = f, "hold"
        threading.Thread(target=self._loop, daemon=True).start()
        return "follower connected, holding its current pose"

    def set_grip_floor(self, extra):
        """Let the gripper be COMMANDED `extra` ticks past its calibrated closed end (0 = back to the calibration).
        Why: a block held across a thin side stalls the fingers only ~25 ticks above that end, so the clamp could
        not squeeze it (the squeeze is the gap between command and position). Only the servo's Min_Position_Limit
        changes; self.fl keeps the calibrated end, so grasp rules, leader mapping and dataset normalisation are
        untouched. The gripper's torque limit / overload protection (connect) still bound the force.
        The gripper goes limp for a moment (EEPROM needs torque off): call it with the clamp empty."""
        if not self.f: raise RuntimeError("connect the arm first")
        cal = int(self.fl["gripper"][0]); want = cal - max(0, min(250, int(extra)))
        with self.io:
            self.f.disable_torque("gripper")
            self.f.write("Min_Position_Limit", "gripper", want, normalize=False)
            self.f.enable_torque("gripper")
            got = int(self.f.read("Min_Position_Limit", "gripper", normalize=False))
        self.g_floor = got
        GRIP_FLOOR.write_text(json.dumps({"calibrated_min": cal, "servo_min": got}, indent=1))
        return got

    def set_mode(self, m, force=False):
        if not self.f: raise RuntimeError("connect the arm first")
        if self.running and not force: raise RuntimeError("a routine/collection is running: press STOP first")
        with self.io:
            if m == "limp":
                self.f.disable_torque(); self.mode = "limp"; return "LIMP: follower torque off - hold the arm, it will drop"
            if m == "mirror" and not self.l:
                self.l = _open(LEADER); self.ll = _limits(self.l)
                self.lpos = self.l.sync_read("Present_Position", normalize=False)
            p = self.f.sync_read("Present_Position", normalize=False)
            keepg = self.cmd.get("gripper") if self.mode in ("hold", "mirror") else None
            self.pos, self.goal, self.cmd = dict(p), dict(p), {k: float(v) for k, v in p.items()}
            if keepg is not None:          # keep squeezing a held brick (actual position = brick width)
                self.goal["gripper"] = self.cmd["gripper"] = keepg
            if self.mode == "limp":
                self.f.sync_write("Goal_Position", p, normalize=False); self.f.enable_torque()
            self.mirror_t = time.time(); self.engaged = False
            self.mode = m
        return {"hold": "HOLD: follower stiff at its current pose",
                "mirror": "MIRROR: follower creeps to the leader's pose, then copies it live"}[m]

    def _leader_to_follower(self, lp):
        """Studio's convention for this project (robot unit = "normalized"): each joint's position as a fraction
        of its calibrated range on the leader -> same fraction of the follower's range. Recording the follower
        target in these units means the dataset's actions are exactly what Studio/ACT expects."""
        out = {}
        for n in JOINTS:
            lmin, lmax = self.ll[n]; fmin, fmax = self.fl[n]
            fr = min(1.0, max(0.0, (lp[n] - lmin) / max(1, lmax - lmin)))
            out[n] = fmin + fr * (fmax - fmin)
        return out

    def leader_target(self):
        return self._leader_to_follower(self.lpos) if self.l and self.lpos else None

    def ensure_leader(self):
        with self.io:
            if not self.l:
                self.l = _open(LEADER); self.ll = _limits(self.l)
                self.lpos = self.l.sync_read("Present_Position", normalize=False)

    def teach_pose(self):
        """Body joints: where the arm actually is. Gripper: what it is COMMANDED to (when it holds a brick the
        actual position is the brick's width; replaying that would hold the brick with zero force)."""
        p = {k: int(v) for k, v in self.pos.items()}
        if self.mode in ("hold", "mirror") and self.cmd: p["gripper"] = int(round(self.cmd["gripper"]))
        return p

    def norm(self, ticks):
        """Follower ticks -> Studio normalized units (body -100..100, gripper 0..100)."""
        out = {}
        for n in JOINTS:
            fmin, fmax = self.fl[n]; fr = min(1.0, max(0.0, (ticks[n] - fmin) / max(1, fmax - fmin)))
            out[n] = fr * 100 if n == "gripper" else fr * 200 - 100
        return out

    def _loop(self):
        """50 Hz: the only place that talks to the servos while connected. Every command is rate-limited."""
        dt = 0.02
        while True:
            t0 = time.time()
            try:
                with self.io:
                    self.pos = self.f.sync_read("Present_Position", normalize=False)
                    if self.l:
                        self.lpos = self.l.sync_read("Present_Position", normalize=False)
                    if self.mode == "mirror":
                        self.goal = self._leader_to_follower(self.lpos)
                        if not self.engaged and max(abs(self.goal[n] - self.cmd[n]) for n in JOINTS) < 60:
                            self.engaged = True          # caught up with the leader: now follow at full speed
                        v = MIRROR_VMAX if self.engaged else ENGAGE_VMAX
                    else:
                        v = self.vmax
                    if self.mode in ("hold", "mirror"):
                        step = v * dt * 1.05
                        for n in JOINTS:
                            g = min(self.fl[n][1], max(self.g_floor if n == "gripper" else self.fl[n][0], self.goal[n]))
                            self.cmd[n] += max(-step, min(step, g - self.cmd[n]))
                        self.f.sync_write("Goal_Position", {n: int(round(c)) for n, c in self.cmd.items()},
                                          normalize=False)
                self.bus_errors = 0
            except Exception as e:
                self.bus_errors += 1
                if self.bus_errors in (1, 25): print("arm bus error:", e)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    # ---------------------------------------------------------------- motion
    intr = None      # threading.Event or None: while set, a running goto / path raises Interrupted (flow.py: a photo judged
                     # in the background turned out not clean). The arm then holds where it is; nothing else changes.

    def _check(self):
        if self.stop.is_set(): raise Stopped()
        if self.intr is not None and self.intr.is_set(): raise Interrupted()

    def goto(self, tgt, speed):
        """Move of all joints from the current command to `tgt`, all arriving together. Minimum jerk (peak speed =
        speed*VMAX); in fast mode (self.fast) a long move ramps up to its cruise speed, holds it and ramps down
        (profile())."""
        vmax = max(50.0, VMAX * speed)
        start = dict(self.cmd)
        d = max(abs(tgt[n] - start[n]) for n in JOINTS)
        scale, ramp = self.fast or (1.0, 0.0)
        if max(JOINTS, key=lambda n: abs(tgt[n] - start[n])) == "gripper": ramp = 0.0     # closing / opening the jaws: they meet
                                                                                          # the block at the eased speed, not at cruise
        T, f, vpk = profile(d, vmax, ramp, min(VMAX * FAST_VMAX, vmax * scale))
        self.vmax = vpk * 1.2                    # rate limiter slightly above the profile: never the bottleneck
        t0 = time.time()
        while True:
            self._check()
            t = time.time() - t0; m = f(t)
            self.goal = {n: start[n] + (tgt[n] - start[n]) * m for n in JOINTS}
            if t >= T: break
            time.sleep(0.02)
        return self._settle(tgt)

    def path(self, pts, speed, corner=CORNER):
        """Move through the poses `pts` WITHOUT stopping at the ones in between: one time law (profile()) over the
        whole way, so the arm speeds up once, keeps moving through every via point and slows down once, into
        pts[-1], where it settles like goto(). The via points are rounded (see CORNER): they are passed close by, not
        hit, so use them for clearance poses (above a plate, above WP2), never for a pose where something happens.
        The stop check runs every tick; BLOCKED is tested at the end."""
        P = [dict(self.cmd)]
        for p in pts:
            q = {n: float(p[n]) for n in JOINTS}
            if max(abs(q[n] - P[-1][n]) for n in JOINTS) >= 1: P.append(q)
        if len(P) <= 2: return self.goto({n: float(pts[-1][n]) for n in JOINTS}, speed)
        seg = [max(abs(P[i + 1][n] - P[i][n]) for n in JOINTS) for i in range(len(P) - 1)]
        cum = [0.0]
        for d in seg: cum.append(cum[-1] + d)
        L = cum[-1]
        def at(s):
            s = min(L, max(0.0, s)); i = 0
            while i < len(seg) - 1 and s > cum[i + 1]: i += 1
            u = (s - cum[i]) / seg[i]
            return {n: P[i][n] + (P[i + 1][n] - P[i][n]) * u for n in JOINTS}
        vmax = max(50.0, VMAX * speed); scale, ramp = self.fast or (1.0, 0.0)
        T, f, vpk = profile(L, vmax, ramp, min(VMAX * FAST_VMAX, vmax * scale))
        self.vmax = vpk * 1.2
        t0 = time.time()
        while True:
            self._check()
            t = time.time() - t0; s = L * f(t); r = min(corner, s, L - s)
            a, b, c = at(s - r), at(s), at(s + r)
            self.goal = {n: (a[n] + 2 * b[n] + c[n]) / 4 for n in JOINTS}
            if t >= T: break
            time.sleep(0.02)
        self.goal = dict(P[-1])
        return self._settle(P[-1])

    def _settle(self, tgt):
        """Wait for the arm at the end of a move -> largest body-joint error (ticks). Raises BLOCKED."""
        # settle: body joints within TOL and gripper stopped moving (it stalls on the brick when closing)
        t1, lastg, e = time.time(), None, 0
        while time.time() - t1 < 1.5:
            self._check()
            p = self.pos
            e = max(abs(p[n] - tgt[n]) for n in BODY)
            g = p["gripper"]
            g_ok = abs(g - tgt["gripper"]) <= TOL or (lastg is not None and abs(g - lastg) <= 2)
            if e <= TOL and g_ok and time.time() - t1 > 0.1: return e
            lastg = g; time.sleep(0.1)
        if e > BLOCKED:
            raise RuntimeError(f"BLOCKED: arm is {e / TICK_PER_DEG:.0f} deg away from the waypoint - something is in the way")
        return e


# ======================================================================== routine + capture
def load_routines():
    try: return json.loads(ROUTINES.read_text())
    except Exception: return {}


def store_routines(r):
    ROUTINES.write_text(json.dumps(r, indent=1))


def make_router(S, DATA, AVG, say_recent):
    """S = capture.py camera state (frames, averaged(), saved undo stack)."""
    A = Arm()
    R = {"routines": load_routines(), "cur": None, "msg": "click CONNECT ARM", "ok": True}
    if R["routines"]: R["cur"] = next(iter(R["routines"]))
    r = APIRouter()

    def say(msg, ok=True): R["msg"], R["ok"] = msg, ok

    def wps(name="", create=False):
        """Waypoints of routine `name` (the fixed slots home/wp1/wp2/drops/...), or of the selected routine."""
        if name:
            if name not in R["routines"]:
                if not create: raise RuntimeError(f"'{name}' has no poses yet: teach it first")
                R["routines"][name] = {"waypoints": [], "created": datetime.now().isoformat(timespec="seconds")}
            return R["routines"][name]["waypoints"]
        if R["cur"] is None: raise RuntimeError("create a routine first")
        return R["routines"][R["cur"]]["waypoints"]

    def status():
        return {"connected": A.f is not None, "mode": A.mode, "running": A.running, "prog": A.prog,
                "pos": A.pos, "bus_errors": A.bus_errors, "msg": R["msg"], "ok": R["ok"],
                "routines": list(R["routines"]), "cur": R["cur"],
                "waypoints": R["routines"][R["cur"]]["waypoints"] if R["cur"] else [],
                "live": S.live, "all": R["routines"]}

    def act(fn):
        try:
            m = fn()
            if m: say(m)
        except Exception as e:
            say(f"ERROR: {e}", False)
        return status()

    @r.get("/arm/status")
    def st(): return status()

    @r.post("/arm/connect")
    def connect(): return act(A.connect)

    @r.post("/arm/mode")
    def mode(m: str): return act(lambda: A.set_mode(m))

    @r.post("/arm/stop")
    def stop():
        A.stop.set(); time.sleep(0.06)
        if A.f:
            if A.mode == "mirror": A.mode = "hold"
            A.goal = dict(A.cmd)              # freeze at the current commanded pose
        say("STOPPED: arm holds its position", False); return status()

    # ---- routine editing (every change is saved to routines.json immediately)
    @r.post("/arm/routine/new")
    def new(name: str):
        def f():
            n = name.strip() or f"routine{len(R['routines']) + 1}"
            R["routines"].setdefault(n, {"waypoints": [], "created": datetime.now().isoformat(timespec="seconds")})
            R["cur"] = n; store_routines(R["routines"]); return f"routine '{n}' selected"
        return act(f)

    @r.post("/arm/routine/select")
    def select(name: str):
        def f():
            if name not in R["routines"]: raise RuntimeError("no such routine")
            R["cur"] = name; return f"routine '{name}' selected"
        return act(f)

    @r.post("/arm/routine/delete")
    def delete_routine():
        def f():
            if A.running: raise RuntimeError(EDIT_BUSY)
            n = R["cur"]; R["routines"].pop(n, None); store_routines(R["routines"])
            R["cur"] = next(iter(R["routines"]), None); return f"routine '{n}' deleted"
        return act(f)

    @r.post("/arm/wp/add")
    def add(capture: bool = False, routine: str = "", single: bool = False):
        def f():
            if not A.f: raise RuntimeError("connect the arm first")
            if A.running: raise RuntimeError(EDIT_BUSY)
            w = wps(routine, create=True)
            pose = {"name": routine if single else (f"{routine}-{len(w) + 1}" if routine else f"{'cam' if capture else 'p'}{len(w) + 1}"),
                    "pos": A.teach_pose(), "capture": capture}
            if single: w[:] = [pose]
            else: w.append(pose)
            store_routines(R["routines"])
            return f"{routine or 'routine'}: pose {'saved' if single else f'{len(w)} added'}{' (📷)' if capture else ''}"
        return act(f)

    @r.post("/arm/wp/{i}/{op}")
    def wp_op(i: int, op: str, v: str = "", routine: str = ""):
        def f():
            w = wps(routine)
            if not 0 <= i < len(w): raise RuntimeError("no such waypoint")
            if A.running and op in ("delete", "update", "up", "down"): raise RuntimeError(EDIT_BUSY)
            if op == "delete": w.pop(i)
            elif op == "update":
                if not A.f: raise RuntimeError("connect the arm first")
                w[i]["pos"] = A.teach_pose()
            elif op == "capture": w[i]["capture"] = v == "true"
            elif op == "rename": w[i]["name"] = v.strip()[:30] or w[i]["name"]
            elif op in ("up", "down"):
                j = i - 1 if op == "up" else i + 1
                if 0 <= j < len(w): w[i], w[j] = w[j], w[i]
            elif op == "goto":
                if A.running: raise RuntimeError("a routine is running")
                if A.mode != "hold": A.set_mode("hold")
                tgt = {k: float(x) for k, x in w[i]["pos"].items()}
                A.stop.clear()
                threading.Thread(target=lambda: _safe(lambda: A.goto(tgt, 0.25)), daemon=True).start()
                return f"moving slowly to waypoint {i + 1}"
            else: raise RuntimeError("bad op")
            store_routines(R["routines"]); return f"waypoint {i + 1}: {op} done"
        return act(f)

    def _safe(fn):
        try: fn()
        except Stopped: pass
        except Exception as e: say(f"ERROR: {e}", False)

    # ---- run
    def capture(cams, label, brick, view, cycle, run_id, kind="inhand", shots=None, depth=None, joints=None, out=None):
        """kind="inhand": brick held in front of the camera (depth-masked crop).
        kind="release": brick lying released on the tape spot, seen from above (fixed crop, prep.release_crop).
        shots / depth / joints: frames grabbed EARLIER (S.averaged result, {cam: depth of that moment or None}, the
        joints then): only the processing and saving happens now, possibly in another thread while the arm already
        moves on (flow.py). out: a list that gets the meta records written."""
        grabbed = shots is not None
        if shots is None: shots = S.averaged(cams)
        group = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        files = []
        for k, (f, first, last) in shots.items():
            d = DATA / k / "inhand" / label; d.mkdir(parents=True, exist_ok=True)
            p = d / f"{group}_{brick or 'nobrick'}_{view}.png"; cv2.imwrite(str(p), f); files.append(p)
            motion = float(np.abs(cv2.blur(first, (5, 5)) - cv2.blur(last, (5, 5))).mean())
            dep = (depth or {}).get(k) if grabbed else getattr(S.cams[k], "depth", None); dinfo = None
            dcfg = S.cams[k].depth_info() if kind == "inhand" and hasattr(S.cams[k], "depth_info") else None   # provenance
            if kind == "inhand" and dcfg is not None:          # why there is no mask, instead of a silent None
                if prep.TUNING: dep, dinfo = None, "depth auto-tune running"
                elif dep is None: dinfo = "no depth"
                elif not grabbed and time.time() - S.cams[k].depth_t >= 1.0: dinfo = f"stale depth ({time.time() - S.cams[k].depth_t:.1f} s old)"
            if dep is not None and not grabbed and time.time() - S.cams[k].depth_t >= 1.0: dep = None
            if kind == "release":
                crop, cinfo = prep.release_crop(f, k)             # the exact anomaly-model input (sorter.py uses the same)
            else:
                if dep is not None:
                    dp = d / f"{group}_{brick or 'nobrick'}_{view}_depth.png"; cv2.imwrite(str(dp), dep); files.append(dp)
                    if prep.INHAND.get("brick"): mask, dinfo = prep.brick_mask(f, prep.INHAND["brick"])   # colour mask
                    else:
                        mask, dinfo = prep.inhand_mask(dep)
                        mask = prep.complete_mask(f, mask)
                    if mask is not None:
                        mp = d / f"{group}_{brick or 'nobrick'}_{view}_masked.jpg"
                        cv2.imwrite(str(mp), prep.apply_mask(f, mask), [cv2.IMWRITE_JPEG_QUALITY, 90]); files.append(mp)
                crop, cinfo = prep.inspect_crop(f, dep)           # the exact anomaly-model input (sorter.py uses the same)
            if cinfo.get("masked"):
                cp = d / f"{group}_{brick or 'nobrick'}_{view}_crop.png"; cv2.imwrite(str(cp), crop); files.append(cp)
            with open(DATA / "meta.jsonl", "a") as m:
                rec = {"file": str(p.relative_to(DATA)), "stage": kind, "cam": k, "group": group,
                                    "label": label, "brick": brick, "view": view, "cycle": cycle, "run": run_id,
                                    "routine": R["cur"], "joints": joints or {n: int(x) for n, x in A.pos.items()},
                                    "motion": round(motion, 2), "frames_averaged": AVG[k], "time": group,
                                    "depth": dinfo if isinstance(dinfo, dict) else None,
                                    "depth_mask_why": dinfo if isinstance(dinfo, str) else None,
                                    "depth_cfg": dcfg,
                                    "inhand_cfg": (cinfo.get("inhand_cfg") or dict(prep.INHAND)) if kind == "inhand" else None,
                                    "cam_settings": S.cams[k].settings}
                m.write(json.dumps(rec) + "\n")
                if out is not None: out.append(rec)
            say_recent(f"{k} · {view}", f)
        S.saved.append((files, len(files)))
        return len(files)

    def run(cycles, label, brick, speed, jitter, cams, save, pause, routine=""):
        w = [dict(x) for x in wps(routine)]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S"); n_saved = 0
        A.running = True; A.stop.clear()
        try:
            for c in range(1, cycles + 1):
                for i, wp in enumerate(w):
                    A.prog = {"cycle": c, "cycles": cycles, "step": i + 1, "steps": len(w), "wp": wp["name"],
                              "saved": n_saved, "waiting": False}
                    tgt = {k: float(x) for k, x in wp["pos"].items()}
                    if wp["capture"] and jitter > 0:
                        for n in ("shoulder_pan", "wrist_flex", "wrist_roll"):
                            tgt[n] += random.uniform(-jitter, jitter) * TICK_PER_DEG
                    first = c == 1 and i == 0
                    A.goto(tgt, speed * (0.4 if first else 1.0))      # first move from wherever: slower
                    if wp["capture"]:
                        time.sleep(0.25)                                # let the brick stop swinging
                        A._check()
                        if save: n_saved += capture(cams, label, brick, wp["name"], c, run_id)
                    say(f"cycle {c}/{cycles} · {i + 1}/{len(w)} {wp['name']} · {n_saved} images saved")
                if pause and c < cycles:
                    A.prog["waiting"] = True; R["go"] = threading.Event()
                    say(f"cycle {c} done: reposition the brick, then click CONTINUE")
                    while not R["go"].wait(0.1): A._check()
            say(f"✓ DONE: {cycles} cycle(s), {n_saved} images saved" if save else f"✓ test run done ({cycles} cycle(s), nothing saved)")
        except Stopped:
            say(f"STOPPED after {n_saved} saved images: arm holds its position", False)
        except Exception as e:
            say(f"ERROR: {e} (arm holds its position, {n_saved} images saved)", False)
        finally:
            A.goal = dict(A.cmd); A.running = False; A.prog = {}

    @r.post("/arm/run")
    def start(cycles: int = 1, label: str = "good", brick: str = "", speed: float = 0.5, jitter: float = 0.0,
              cams: str = "front", save: bool = True, pause: bool = False, routine: str = ""):
        def f():
            if not A.f: raise RuntimeError("connect the arm first")
            if A.running: raise RuntimeError("already running")
            if label not in ("good", "defect"): raise RuntimeError("label must be good/defect")
            w = wps(routine)
            if not w: raise RuntimeError("teach at least 1 pose")
            cl = [c for c in cams.split(",") if c in S.live]
            if save and not any(x["capture"] for x in w): raise RuntimeError("no 📷 waypoint: tick 📷 on the inspection poses")
            if save and not cl: raise RuntimeError("no live camera selected")
            if save and not brick.strip(): raise RuntimeError("type the brick name first (e.g. red, blue, yellow)")
            if A.mode == "limp": A.set_mode("hold")
            elif A.mode == "mirror": A.set_mode("hold")
            threading.Thread(target=run, args=(max(1, cycles), label, brick.strip(), min(1.0, max(0.05, speed)),
                                               max(0.0, jitter), cl, save, pause, routine), daemon=True).start()
            return f"{'running' if save else 'TEST run (no saving)'}: {cycles} cycle(s) of '{routine or R['cur']}'"
        return act(f)

    @r.post("/arm/continue")
    def cont():
        if R.get("go"): R["go"].set()
        return status()

    @r.get("/arm/routines")
    def page(): return HTMLResponse(PAGE)        # old generic routine editor (the main page is /arm, ui.py)

    class Ctx: pass
    ctx = Ctx(); ctx.R, ctx.capture, ctx.say = R, capture, say
    return r, A, ctx


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Arm Auto-Capture</title><style>
:root{--bg:#121417;--panel:#1b1f24;--line:#2c323a;--text:#e8eaed;--muted:#9aa3ad;--good:#2fa36b;--bad:#d2582a;--ok:#5fd38d;--err:#ff6b6b;--acc:#3b82f6}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:12px 16px}
button{font:600 14px system-ui;padding:8px 12px;border-radius:7px;border:0;cursor:pointer;color:#fff;background:#3d4854}
button:disabled{opacity:.4;cursor:default}button.on{background:var(--acc)}button.g{background:var(--good)}button.big{font-size:18px;padding:12px 20px}
#stop{background:#c62828;font:800 22px system-ui;padding:14px 30px;position:sticky;top:8px;z-index:3}
input,select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px;font-size:14px}
#msg{font-size:17px;font-weight:600;padding:9px 12px;border-radius:7px;margin:8px 0;background:var(--panel)}
.grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:12px}@media(max-width:1000px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
h3{margin:0 0 10px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}.cams img{width:100%;border-radius:6px;display:block}
.cams figure{margin:0;font-size:12px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left}
td input[type=text]{width:110px}tr.cur{background:#24405f}.small{font-size:12px;color:var(--muted)}
ol{margin:4px 0 0 18px;padding:0;font-size:14px;line-height:1.5}
#prog{height:8px;background:#2a3038;border-radius:4px;margin-top:6px}#prog i{display:block;height:8px;border-radius:4px;background:var(--good)}
</style></head><body>
<div class=row><button id=stop onclick="p('/arm/stop')">■ STOP (Space)</button>
<button class=big onclick="p('/arm/connect')" id=conn>CONNECT ARM</button>
<span class=small>mode: <b id=mode>-</b></span><a href="/" style="color:#8ab4f8;margin-left:auto">← manual capture page</a></div>
<div id=msg>loading…</div>
<div class=grid><div>
 <div class=card><h3>1 · teach</h3>
  <div class=row>
   <button id=m-mirror onclick="p('/arm/mode?m=mirror')">MIRROR LEADER</button>
   <button id=m-hold onclick="p('/arm/mode?m=hold')">HOLD</button>
   <button id=m-limp onclick="p('/arm/mode?m=limp')" title="follower torque off: hold it, it drops">LIMP (hand-guide)</button>
  </div>
  <div class=row>
   <select id=rsel onchange="p('/arm/routine/select?name='+encodeURIComponent(this.value))"></select>
   <input id=rname placeholder="new routine name" style="width:150px"><button onclick="p('/arm/routine/new?name='+encodeURIComponent($('rname').value))">+ new routine</button>
   <button onclick="if(confirm('delete this routine?'))p('/arm/routine/delete')">delete routine</button>
  </div>
  <div class=row>
   <button class=g onclick="p('/arm/wp/add?capture=false')">[W] add waypoint</button>
   <button class=g onclick="p('/arm/wp/add?capture=true')">[C] add 📷 waypoint</button>
  </div>
  <table id=wps></table>
 </div>
 <div class=card><h3>2 · run</h3>
  <div class=row>brick <input id=brick placeholder="e.g. red" style="width:100px">
   <select id=label><option value=good>good</option><option value=defect>defect</option></select>
   cycles <input id=cyc type=number value=1 min=1 style="width:70px">
   speed <input id=spd type=range min=10 max=100 value=40><span id=spdv class=small></span></div>
  <div class=row>cameras:
   <label><input type=checkbox id=c-front checked> front</label><label><input type=checkbox id=c-top> top</label><label><input type=checkbox id=c-wrist> wrist</label>
   · jitter ±<input id=jit type=number value=0 min=0 max=8 step=0.5 style="width:60px">°
   <label title="robot waits after each cycle until you click CONTINUE"><input type=checkbox id=pause> wait for me between cycles</label></div>
  <div class=row><button onclick="run(false)">TEST RUN (no saving)</button>
   <button class="g big" onclick="run(true)">▶ RUN + CAPTURE</button><button id=cont onclick="p('/arm/continue')" style="display:none">CONTINUE ▶</button></div>
  <div id=ptxt class=small></div><div id=prog><i style="width:0"></i></div>
 </div>
 <div class=card><h3>how to</h3><ol>
  <li>CONNECT ARM, then MIRROR LEADER: the follower copies the leader arm.</li>
  <li>New routine (e.g. <i>red</i>). Walk through the task with the leader and click a button at each pose:
   above pick spot → down + open → close (grip) → lift → <b>📷</b> side A → <b>📷</b> side B → <b>📷</b> bottom → set down turned 90° → open → grip → lift → <b>📷</b> side C → <b>📷</b> side D → put back on pick spot → open → up.</li>
  <li>HOLD, then TEST RUN at low speed with your hand near STOP. Fix a pose with <i>set</i> (moves the waypoint to the arm's current pose).</li>
  <li>Type the brick name, set cycles (e.g. 25), RUN + CAPTURE. Ending on the pick spot lets it loop unattended; tick "wait for me" to nudge the brick between cycles for grip variety.</li>
 </ol></div>
</div><div>
 <div class=card><h3>cameras</h3><div class=cams>
  <figure><img id=pv-front>front</figure><figure><img id=pv-top>top</figure><figure><img id=pv-wrist>wrist</figure></div></div>
 <div class=card><h3>last saved</h3><div id=recent style="display:flex;flex-wrap:wrap;gap:6px"></div></div>
 <div class=card><h3>joints (raw ticks)</h3><table id=jt></table></div>
</div></div>
<script>
const $=id=>document.getElementById(id);let q=Promise.resolve(),S=null;
function p(u){q=q.then(()=>fetch(u,{method:'POST'}).then(r=>r.json()).then(show).catch(e=>{$('msg').textContent='ERROR: '+e;$('msg').style.color='var(--err)'}))}
function run(save){const cams=['front','top','wrist'].filter(c=>$('c-'+c).checked).join(',');
 p(`/arm/run?cycles=${$('cyc').value}&label=${$('label').value}&brick=${encodeURIComponent($('brick').value.trim())}&speed=${$('spd').value/100}&jitter=${$('jit').value}&cams=${cams}&save=${save}&pause=${$('pause').checked}`)}
function live(id,url,ms){const im=$(id);const next=()=>setTimeout(()=>{im.src=url+'?t='+Date.now()},ms);im.onload=next;im.onerror=()=>setTimeout(next,1000);next()}
['front','top','wrist'].forEach(c=>live('pv-'+c,'/snap/'+c,150));
$('spd').oninput=()=>$('spdv').textContent=$('spd').value+'%';$('spd').oninput();
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let wsig='';
function show(s){S=s;const m=$('msg');m.textContent=s.msg;m.style.color=s.ok?'var(--ok)':'var(--err)';
 $('mode').textContent=!s.connected?'not connected':s.running?'RUNNING':s.mode;$('conn').style.display=s.connected?'none':'';
 ['mirror','hold','limp'].forEach(k=>{$('m-'+k).classList.toggle('on',s.mode==k);$('m-'+k).disabled=!s.connected||s.running});
 const rs=s.routines.map(r=>`<option ${r==s.cur?'selected':''}>${esc(r)}</option>`).join('');if($('rsel').innerHTML!=rs)$('rsel').innerHTML=rs||'<option>(no routine yet)</option>';
 const cur=s.running&&s.prog.step?s.prog.step-1:-1,sig=JSON.stringify(s.waypoints.map(w=>[w.name,w.capture]))+cur;
 if(sig!=wsig&&document.activeElement.tagName!='INPUT'){wsig=sig;
  $('wps').innerHTML='<tr><th>#</th><th>name</th><th>📷</th><th></th></tr>'+s.waypoints.map((w,i)=>`<tr class="${i==cur?'cur':''}"><td>${i+1}</td>
  <td><input type=text value="${esc(w.name)}" onchange="p('/arm/wp/${i}/rename?v='+encodeURIComponent(this.value))"></td>
  <td><input type=checkbox ${w.capture?'checked':''} onchange="p('/arm/wp/${i}/capture?v='+this.checked)"></td>
  <td><button onclick="p('/arm/wp/${i}/goto')" title="move there slowly">go</button> <button onclick="p('/arm/wp/${i}/update')" title="replace with the arm's current pose">set</button>
  <button onclick="p('/arm/wp/${i}/up')">↑</button><button onclick="p('/arm/wp/${i}/down')">↓</button> <button onclick="p('/arm/wp/${i}/delete')">✕</button></td></tr>`).join('')}
 const pr=s.prog||{};$('cont').style.display=pr.waiting?'':'none';
 if(pr.cycles){const f=((pr.cycle-1)*pr.steps+pr.step)/(pr.cycles*pr.steps);$('prog').firstChild.style.width=(100*f)+'%';
  $('ptxt').textContent=`cycle ${pr.cycle}/${pr.cycles} · step ${pr.step}/${pr.steps} (${pr.wp}) · ${pr.saved} saved`}else{$('ptxt').textContent='';$('prog').firstChild.style.width='0'}
 $('jt').innerHTML=Object.entries(s.pos||{}).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('')+(s.bus_errors?`<tr><td colspan=2 style="color:var(--err)">bus errors: ${s.bus_errors}</td></tr>`:'')}
let rc='';function recent(){fetch('/status').then(r=>r.json()).then(s=>{const k=JSON.stringify(s.recent);if(k!=rc){rc=k;
 $('recent').innerHTML=s.recent.map((l,i)=>`<figure style="margin:0;font-size:11px;color:var(--muted)"><img style="height:90px;border-radius:4px;display:block" src="/recent/${i}?${Date.now()}">${esc(l)}</figure>`).join('')}})}
document.onkeydown=e=>{if(e.key==' '||e.key=='Escape'){e.preventDefault();p('/arm/stop');return}
 if(e.target.tagName=='INPUT'&&e.target.type!='checkbox'&&e.target.type!='range')return;const k=e.key.toLowerCase();
 if(k=='w')p('/arm/wp/add?capture=false');else if(k=='c')p('/arm/wp/add?capture=true')};
setInterval(()=>fetch('/arm/status').then(r=>r.json()).then(show),300);setInterval(recent,1000);
fetch('/arm/status').then(r=>r.json()).then(show);
</script></body></html>"""
