"""Hardware-free mock station: fake cameras, fake Feetech servo bus, fake Anomalib (Junction 2026, SO-101).

    JUNCTION_MOCK=1 python anomaly_capture/capture.py        (or: python anomaly_capture/capture.py --mock, or: make mock)

Why: the control tool needs 2 RealSense D415, a USB wrist camera and two Feetech arms. This module lets the REAL
server (capture.py + arm.py + collect.py + sorter.py + ui.py, unchanged logic) run anywhere, so the UI can be
clicked through, flows can be regression-tested (tests/) and teammates can develop without the station.

How (everything is faked at the lowest level, so as much real code as possible runs):
  * FeetechMotorsBus / Motor / MotorNormMode: drop-in for lerobot's bus (arm.py imports these in mock mode). Servos
    have simple first-order dynamics with a speed limit (moves take time, STOP mid-move works), torque on/off, EEPROM
    limits, a gripper that STALLS on a brick between the fingers (so the "holding" logic works), optional obstacle
    (BLOCKED detection), bus errors. The leader arm is driven by a scripted "human" (auto: follows the follower when
    not mirroring so MATCH succeeds, picks the brick off the input plate during a pick, hand-feeds a brick into a wide
    open gripper held high, e.g. at WP1 for the sorter's hand-feed) or set by hand through /mock/leader.
  * FakeCam: top / front (colour + aligned depth in mm) / wrist frames at 1920x1080 ~30 fps, rendered from a small 3D
    scene (table, orange input plate, tape spot, good + defect plates, the arm from simple kinematics, bricks with an
    optional black pen cross on one face). The scene reacts to the arm: bricks are gripped, carried, released, fall on
    the plate/tape/plates. At WP2 the front depth isolates gripper + brick (prep.inhand_mask finds it).
  * Fake Anomalib: a tiny `anomalib.deploy` package (written under the data root) that the REAL score_worker.py
    imports; its score = how much pen ink the crop shows (deterministic, follows the scene's ground truth).
  * Data isolation: in mock mode every data dir and settings file goes to a separate root (env JUNCTION_DATA_ROOT,
    default <repo>/mock_data): training_images, episodes, runs, models, cam/collect/sort/inspect settings, routines.
    The root is seeded with pre-taught mock waypoints (every flow runs immediately) and mock models.
  * /mock/* API (state, scene, leader, human, fault, camera, reset, ik) to drive and inspect the fake world.

Not simulated: real optics/noise/lighting, servo load/temperature, physics beyond "grip, carry, release, fall"
(no collisions except the optional obstacle), Studio/LeRobot export, real Anomalib models. See docs/DEV.md.
"""
import json, math, os, sys, threading, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY = JOINTS[:-1]
TPD = 4096 / 360
W, H = 1920, 1080
FPS = 30
LEADER_SERIAL = "5B14111392"          # = arm.LEADER (not imported: arm.py imports this module)


def enabled():
    return os.environ.get("JUNCTION_MOCK") == "1"


# ======================================================================== kinematics (simple, self-consistent)
H0, L1, L2, L3 = 0.12, 0.15, 0.16, 0.10       # shoulder height, upper arm, forearm, wrist -> grasp point (m)
ROLL0 = 90.0                                   # deg: wrist_roll 1024 ticks -> gripper +x (free side) points radially out
G_MIN, G_MAX, G_SPAN = 1400, 3200, 0.10        # gripper ticks closed / open and the finger gap (m) at G_MAX
LIMITS = {"shoulder_pan": (300, 3800), "shoulder_lift": (300, 3800), "elbow_flex": (300, 3800),
          "wrist_flex": (200, 3900), "wrist_roll": (0, 4095), "gripper": (G_MIN, G_MAX)}
SERVO_V, SERVO_K = 3000.0, 25.0                # ticks/s max, 1/s proportional gain (small lag like P=16)


def _deg(t): return (t - 2048) / TPD
def _tick(d): return 2048 + d * TPD
def unit(v): v = np.asarray(v, float); return v / max(1e-12, np.linalg.norm(v))
def gap_of(g): return (g - G_MIN) / (G_MAX - G_MIN) * G_SPAN
def grip_of(gap): return G_MIN + gap / G_SPAN * (G_MAX - G_MIN)


class FK:
    """Joint ticks -> points (base, shoulder, elbow, wrist, tip), tool direction d and gripper frame x, y, z.
    Gripper frame as in coverage.py: z = palm (up when picked from above), y = finger axis, x = free side."""
    def __init__(self, p):
        th = math.radians(_deg(p["shoulder_pan"])); a1 = math.radians(_deg(p["shoulder_lift"]))
        a12 = a1 + math.radians(_deg(p["elbow_flex"])); phi = a12 + math.radians(_deg(p["wrist_flex"]))
        c, s = math.cos(th), math.sin(th)
        pt = lambda r, z: np.array([r * c, r * s, z])
        r1, z1 = L1 * math.sin(a1), H0 + L1 * math.cos(a1)
        r2, z2 = r1 + L2 * math.sin(a12), z1 + L2 * math.cos(a12)
        r3, z3 = r2 + L3 * math.sin(phi), z2 + L3 * math.cos(phi)
        self.base, self.shoulder = np.zeros(3), np.array([0, 0, H0])
        self.elbow, self.wrist, self.tip = pt(r1, z1), pt(r2, z2), pt(r3, z3)
        self.d = np.array([math.sin(phi) * c, math.sin(phi) * s, math.cos(phi)])
        rho = math.radians(_deg(p["wrist_roll"]) + ROLL0); y0 = np.array([-s, c, 0.0])
        self.y = unit(y0 * math.cos(rho) + np.cross(self.d, y0) * math.sin(rho))
        self.z = -self.d; self.x = np.cross(self.y, self.z)
        self.R = np.stack([self.x, self.y, self.z], 1)           # gripper -> world
        self.gap = gap_of(p["gripper"])


def ik(x, y, z, phi=180.0, roll=1024, grip=2650):
    """Grasp point (x, y, z) in m, tool angle phi from vertical (180 = pointing down, 90 = horizontal, forward),
    wrist_roll ticks, gripper ticks -> joint ticks (elbow up). Raises ValueError if out of reach."""
    th = math.atan2(y, x); r = math.hypot(x, y); ph = math.radians(phi)
    rw, zw = r - L3 * math.sin(ph), z - L3 * math.cos(ph) - H0
    c2 = (rw * rw + zw * zw - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1: raise ValueError(f"({x:.3f}, {y:.3f}, {z:.3f}) out of reach")
    q2 = math.acos(c2); a1 = math.atan2(rw, zw) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    q3 = ph - a1 - q2
    out = {"shoulder_pan": _tick(math.degrees(th)), "shoulder_lift": _tick(math.degrees(a1)),
           "elbow_flex": _tick(math.degrees(q2)), "wrist_flex": _tick(math.degrees(q3)), "wrist_roll": roll, "gripper": grip}
    out = {k: int(round(v)) for k, v in out.items()}
    for k, v in out.items():
        lo, hi = LIMITS[k]
        if not lo <= v <= hi: raise ValueError(f"{k}={v} outside {LIMITS[k]} for ({x:.3f}, {y:.3f}, {z:.3f})")
    return out


def roll_for(yaw, pan_rad):
    """wrist_roll ticks (pointing down) that put the finger axis along a brick axis with world yaw `yaw` (rad)."""
    best = None
    for k in range(-4, 5):                     # any multiple of 90 deg grips a box
        rho = math.pi / 2 + pan_rad - yaw + k * math.pi / 2
        t = _tick(math.degrees(rho) - ROLL0)
        if 300 <= t <= 3800 and (best is None or abs(t - 2048) < abs(best - 2048)): best = t
    return int(round(best if best is not None else 1024))


# ======================================================================== the scene
TABLE = (-0.30, 0.75, -0.55, 0.55)                       # x0, x1, y0, y1 (table top z = 0)
PLATES = {"input": ((0.21, 0.04), 0.08, (40, 120, 235)),  # centre, radius, BGR (orange input plate)
          "good": ((0.06, -0.19), 0.065, (80, 165, 80)),
          "defect": ((0.05, 0.20), 0.065, (70, 70, 175))}
PLATE_Z = 0.008
TAPE = ((0.15, -0.12), 0.028)                            # tape spot centre, half side
REAL_SET = ("pink", "green", "blue")   # the station's blocks (2026-11-14: pink 2x2 replaced the red ridge roof)
BRICK_COLORS = {"pink": (125, 135, 232), "red": (40, 45, 200), "green": (60, 150, 60), "yellow": (40, 190, 225), "blue": (170, 90, 30)}
SIZES = {"2x2": (0.0318, 0.0318, 0.024), "2x4": (0.0636, 0.0318, 0.024), "2x3": (0.0477, 0.0318, 0.024)}
AXES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}
INK = (18, 18, 22)
# where the mock station's fixed cameras are (world: x forward from the robot base, y left, z up)
WP2_TIP = (0.20, 0.03, 0.20)


def ext(R, size, a):
    """Extent of a box (rotation R, size) along world direction a."""
    return float(sum(abs(R[:, i] @ a) * size[i] for i in range(3)))


def rest_pose(R):
    """Rotate R minimally so that the brick axis closest to vertical becomes exactly vertical (it lies flat)."""
    i = int(np.argmax(np.abs(R[2, :]))); a = R[:, i]; t = np.array([0, 0, np.sign(a[2]) or 1.0])
    v = np.cross(a, t); s, c = np.linalg.norm(v), float(a @ t)
    if s < 1e-9: return R.copy()
    k = v / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + math.sin(math.atan2(s, c)) * K + (1 - c) * K @ K) @ R


def rotz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Brick:
    _n = 0

    def __init__(self, defect=None, size="2x2", color="red", pos=None, yaw=0.0):
        Brick._n += 1; self.id = Brick._n
        self.size_name = size if size in SIZES else "2x2"; self.size = np.array(SIZES[self.size_name])
        self.color_name = color if color in BRICK_COLORS else "red"; self.color = BRICK_COLORS[self.color_name]
        self.defect = defect if defect in AXES else None
        self.R = rotz(yaw); self.pos = np.array(pos if pos is not None else (*PLATES["input"][0], 0.0), float)
        self.state = "input"; self.Rg = np.eye(3); self.pg = np.zeros(3); self.slip = False
        self.settle()

    def settle(self, where=None):
        """Lie flat on whatever is below (input/good/defect plate, tape spot, table)."""
        self.R = rest_pose(self.R)
        x, y = self.pos[:2]; self.state, z0 = where or "table", 0.0
        if where is None:
            for k, (c, r, _) in PLATES.items():
                if math.hypot(x - c[0], y - c[1]) < r: self.state, z0 = k, PLATE_Z
            (tx, ty), hs = TAPE
            if abs(x - tx) < hs + 0.01 and abs(y - ty) < hs + 0.01: self.state = "tape"
            if not (TABLE[0] < x < TABLE[1] and TABLE[2] < y < TABLE[3]): self.state = "floor"
        self.pos[2] = z0 + ext(self.R, self.size, np.array([0, 0, 1.0])) / 2

    def info(self):
        return {"id": self.id, "state": self.state, "defect": self.defect, "size": self.size_name, "color": self.color_name,
                "pos": [round(float(v), 4) for v in self.pos]}


# ======================================================================== world: servos + bricks + the human
class World:
    def __init__(self):
        self.lock = threading.RLock()
        rest = {"shoulder_pan": 2048, "shoulder_lift": 1500, "elbow_flex": 3200, "wrist_flex": 2900, "wrist_roll": 1024,
                "gripper": 1500}
        self.servo = {r: {"pos": {k: float(v) for k, v in rest.items()}, "goal": dict(rest), "torque": False,
                          "t": None, "eeprom": {}} for r in ("follower", "leader")}
        self.connected = {"follower": False, "leader": False}
        self.arm = self.collector = self.sorter = None   # attached by capture.py in mock mode
        self.faults = {"obstacle": False, "regrip_fail": 0, "bus_errors": 0, "unplugged": []}
        self.cam_state = {k: "ok" for k in ("top", "front", "wrist")}; self.depth_on = True
        self.human = {"mode": "auto", "target": None, "feed": True, "phase": None, "since": 0.0, "last": "idle"}
        self.queue = []; self.bricks = []; self.active = None; self.n_spawn = 0; self.events = []
        self.tips = []; self.version = 0
        self.reset_scene()

    # ---- scene
    def log(self, m):
        self.events = (self.events + [f"{time.strftime('%H:%M:%S')} {m}"])[-40:]

    def spawn(self, where="input", spec=None):
        spec = spec or (self.queue.pop(0) if self.queue else {})
        self.n_spawn += 1
        rng = np.random.default_rng(self.n_spawn)
        c, r, _ = PLATES["input"]; a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.35 * r)
        b = Brick(defect=spec.get("defect"), size=spec.get("size", "2x2"),
                  color=spec.get("color", REAL_SET[(self.n_spawn - 1) % 3]),
                  pos=(c[0] + rr * math.cos(a), c[1] + rr * math.sin(a), 0), yaw=float(rng.uniform(-0.6, 0.6)))
        self.bricks = [x for x in self.bricks if x.state in ("good", "defect")][-10:] + [b]; self.active = b
        self.log(f"brick {b.id} ({b.color_name} {b.size_name}, defect {b.defect or 'none'}) on the {where} plate")
        return b

    def reset_scene(self, spec=None, empty=False):
        with self.lock:
            self.bricks = []; self.active = None
            if not empty: self.spawn(spec=spec)
            self.version += 1

    # ---- servos
    def _stall_tick(self, k):
        """Lowest gripper position possible right now (a brick between the fingers stops them)."""
        b = self.active
        if b is None: return G_MIN, None
        if b.state in ("held", "fed") or self._between(b, k):
            w = ext(b.R, b.size, k.y)
            return grip_of(w), b
        return G_MIN, None

    def _between(self, b, k):
        if b.state in ("held", "fed", "floor") or b.slip: return False
        if k.d[2] > -0.7: return False                            # fingers must point down at a brick lying flat
        v = b.pos - k.tip
        if abs(v @ k.x) > 0.016 or abs(v @ k.z) > 0.016: return False
        return abs(v @ k.y) + ext(b.R, b.size, k.y) / 2 <= k.gap / 2 + 0.006

    def _servo_step(self, role, t):
        sv = self.servo[role]
        if sv["t"] is None: sv["t"] = t
        dt = min(2.0, t - sv["t"]); sv["t"] = t          # servos keep moving between reads
        if dt <= 0 or role == "leader": return
        if not sv["torque"]: return                               # limp: stays where it is (nobody moves it)
        p = sv["pos"]; n = max(1, int(dt / 0.01)); h = dt / n
        for _ in range(n):
            for j in JOINTS:
                v = max(-SERVO_V, min(SERVO_V, SERVO_K * (sv["goal"][j] - p[j])))
                p[j] += v * h
            if self.faults["obstacle"]: p["shoulder_lift"] = min(p["shoulder_lift"], 2100.0)
            k = FK(p); lo, b = self._stall_tick(k)
            if p["gripper"] < lo:
                p["gripper"] = lo
                if b is not None and b.state not in ("held",):
                    if b.state == "tape" and self.faults["regrip_fail"] > 0:
                        self.faults["regrip_fail"] -= 1; b.slip = True; self.log("re-grip slipped (fault)")
                    else: self._grasp(b, k)
        self._world_step(FK(p), t)

    def _grasp(self, b, k):
        was = b.state
        b.Rg = k.R.T @ b.R; v = k.R.T @ (b.pos - k.tip); v[1] = 0.0; b.pg = v      # fingers centre it
        b.state = "held"; self.log(f"brick {b.id} gripped ({'from ' + was})"); self.version += 1

    def release(self, b, k=None, why="released"):
        b.state = "falling"; b.settle(); b.slip = False; self.version += 1
        self.log(f"brick {b.id} {why} -> {b.state}")
        if b.state in ("good", "defect"):
            threading.Timer(1.0, self._next_after_sort, args=(b,)).start()

    def _next_after_sort(self, b):
        with self.lock:
            if self.active is b and b.state in ("good", "defect"): self.spawn()

    def _world_step(self, k, t):
        b = self.active
        self.tips = [(tt, p) for tt, p in self.tips if t - tt < 0.6] + [(t, k.tip.copy())]
        if b is None: return
        if b.slip and k.gap > ext(b.R, b.size, k.y) + 0.01: b.slip = False
        if b.state == "fed" and not self.human["feed"]: self.release(b, k, "dropped by the human")
        if b.state in ("held", "fed"):
            b.R = k.R @ b.Rg; b.pos = k.tip + k.R @ b.pg
            if b.state == "held" and k.gap > ext(b.R, b.size, k.y) + 0.004: self.release(b, k)
            return
        # hand-feed: wide-open gripper held still and high -> the human puts the brick between the fingers
        still = len(self.tips) > 20 and max(np.linalg.norm(p - k.tip) for _, p in self.tips) < 0.002
        if (self.human["feed"] and k.gap > 0.075 and k.tip[2] > 0.13 and still and k.d[2] < -0.7
                and self.human["mode"] != "off"):
            if b.state in ("good", "defect"): b = self.spawn("hand")
            b.R = k.R.copy(); b.Rg = np.eye(3); b.pg = np.zeros(3); b.state = "fed"
            b.pos = k.tip.copy(); self.version += 1; self.log(f"human holds brick {b.id} between the open fingers")

    def drop(self):
        with self.lock:
            b = self.active
            if b is not None and b.state in ("held", "fed"): self.release(b, None, "DROPPED (fault)"); return True
        return False

    # ---- bus access (called by FeetechMotorsBus)
    def read_pos(self, role):
        with self.lock:
            if role == "follower" and self.faults["bus_errors"] > 0:
                self.faults["bus_errors"] -= 1; raise ConnectionError("mock bus error (fault injection)")
            t = time.time()
            if role == "leader": self._human_step(t)
            else: self._servo_step(role, t)
            return {j: int(round(v)) for j, v in self.servo[role]["pos"].items()}

    def write_goal(self, role, values):
        with self.lock:
            if role == "follower": self._servo_step(role, time.time())
            for j, v in values.items():
                lo, hi = LIMITS[j]; self.servo[role]["goal"][j] = float(min(hi, max(lo, v)))

    def torque(self, role, on):
        with self.lock:
            self._servo_step(role, time.time()); self.servo[role]["torque"] = on
            if on: self.servo[role]["goal"] = dict(self.servo[role]["pos"])

    # ---- the human at the leader arm
    def _picking(self):
        for o in (self.collector, self.sorter):
            if o is not None and getattr(o, "active", False) and getattr(o, "state", "") == "pick": return True
        return False

    def _human_targets(self, b):
        x, y, z = b.pos; th = math.atan2(y, x)
        yaw = math.atan2(b.R[1, 0], b.R[0, 0]) if abs(b.R[2, 0]) < 0.9 else math.atan2(b.R[1, 1], b.R[0, 1])
        roll = roll_for(yaw, th)
        try:
            return [ik(x, y, z + 0.07, 180, roll, 2750), ik(x, y, z, 180, roll, 2750), ik(x, y, z, 180, roll, 1450)]
        except ValueError:
            return None

    def _human_step(self, t):
        sv = self.servo["leader"]; hm = self.human; lp = sv["pos"]
        dt = 0.0 if sv["t"] is None else min(0.1, t - sv["t"]); sv["t"] = t
        tgt = None; arm = self.arm
        if hm["mode"] == "manual" and hm["target"]:
            tgt = hm["target"]
        elif hm["mode"] == "auto" and arm is not None:
            fp = self.servo["follower"]["pos"]
            if arm.mode != "mirror":
                tgt = dict(fp); hm["phase"] = hm["plan"] = None; hm["last"] = "following the robot (matching)"
            elif self._picking() and self.active is not None and self.active.state not in ("fed", "floor"):
                held = self.active.state == "held"
                plan = hm.get("plan") if held else self._human_targets(self.active)   # gripped: keep squeezing
                hm["plan"] = plan
                if plan:
                    ph = hm["phase"] if hm["phase"] is not None else 0
                    tgt = plan[min(ph, 2)]
                    near = (all(abs(lp[j] - tgt[j]) < 10 for j in JOINTS)
                            and all(abs(fp[j] - tgt[j]) < 45 for j in BODY))
                    if hm["phase"] is None: hm["phase"], hm["since"] = 0, t
                    elif near and ph < 2 and t - hm["since"] > 0.3: hm["phase"], hm["since"] = ph + 1, t
                    hm["last"] = ["approach above the brick", "down to the brick", "close the gripper"][min(ph, 2)]
            else:
                hm["last"] = "holding the leader still"
        if tgt and dt > 0:
            for j in JOINTS:
                v = 1200.0 if j == "gripper" else 900.0
                lp[j] += max(-v * dt, min(v * dt, tgt[j] - lp[j]))

    # ---- observation
    def snapshot(self):
        with self.lock:
            fp = {j: round(v) for j, v in self.servo["follower"]["pos"].items()}
            bs = [(b.id, b.pos.copy(), b.R.copy(), b.size, b.color, b.defect, b.state) for b in self.bricks]
            return fp, bs

    def truth(self, cam_pos=None):
        """Ground truth: is the active brick's defect face visible from a camera at cam_pos right now? (65 deg cone,
        not within 35 deg of the finger axis while held: the same physical model as sim_test.py)"""
        with self.lock:
            b = self.active
            if b is None or b.defect is None or cam_pos is None: return False
            i, sg = AXES[b.defect]; n = b.R[:, i] * sg
            v = unit(np.asarray(cam_pos) - b.pos)
            if n @ v < math.cos(math.radians(65)): return False
            if b.state in ("held", "fed"):
                k = FK(self.servo["follower"]["pos"])
                if abs(n @ k.y) >= math.cos(math.radians(35)): return False
            return True

    def state(self):
        with self.lock:
            fp = self.servo["follower"]["pos"]; k = FK(fp)
            cams = {"top": VIEWS["top"].pos, "front": VIEWS["front"].pos, "wrist": wrist_view(k).pos}
            return {"bricks": [b.info() for b in self.bricks], "active": self.active.info() if self.active else None,
                    "truth": {c: self.truth(p) for c, p in cams.items()},
                    "tip": [round(float(v), 4) for v in k.tip], "gap_mm": round(k.gap * 1000, 1),
                    "follower": {j: round(v) for j, v in fp.items()},
                    "leader": {j: round(v) for j, v in self.servo["leader"]["pos"].items()},
                    "torque": self.servo["follower"]["torque"], "human": {k2: v for k2, v in self.human.items() if k2 not in ("since", "plan")},
                    "faults": dict(self.faults), "cams": dict(self.cam_state), "depth": self.depth_on,
                    "queue": list(self.queue), "events": self.events[-12:]}


WORLD = World()


# ======================================================================== lerobot-compatible fake bus
class MotorNormMode:
    RANGE_M100_100 = "range_m100_100"; RANGE_0_100 = "range_0_100"; DEGREES = "degrees"


class Motor:
    def __init__(self, id, model, norm_mode):
        self.id, self.model, self.norm_mode = id, model, norm_mode


class FeetechMotorsBus:
    """Same calls arm.py makes on lerobot's FeetechMotorsBus (raw ticks only: normalize=False everywhere)."""
    def __init__(self, port, motors, **kw):
        self.port, self.motors = port, motors
        self.role = "leader" if LEADER_SERIAL in str(port) else "follower"

    def connect(self, handshake=True):
        if self.role in WORLD.faults["unplugged"]: raise ConnectionError(f"mock: no {self.role} arm on {self.port}")
        WORLD.connected[self.role] = True

    def disconnect(self, *a, **k): WORLD.connected[self.role] = False

    def read(self, reg, motor, normalize=True):
        if reg == "Min_Position_Limit": return LIMITS[motor][0]
        if reg == "Max_Position_Limit": return LIMITS[motor][1]
        if reg == "Present_Position": return self.sync_read(reg)[motor]
        return WORLD.servo[self.role]["eeprom"].get((reg, motor), 0)

    def write(self, reg, motor, value, normalize=True):
        if reg == "Goal_Position": WORLD.write_goal(self.role, {motor: value}); return
        if reg == "Torque_Enable": WORLD.torque(self.role, bool(value)); return
        WORLD.servo[self.role]["eeprom"][(reg, motor)] = value

    def sync_read(self, reg, motors=None, normalize=True):
        if reg == "Present_Position": return WORLD.read_pos(self.role)
        if reg == "Torque_Enable": return {m: int(WORLD.servo[self.role]["torque"]) for m in self.motors}
        if reg in ("Min_Position_Limit", "Max_Position_Limit"): return {m: self.read(reg, m) for m in self.motors}
        return {m: WORLD.servo[self.role]["eeprom"].get((reg, m), 0) for m in self.motors}

    def sync_write(self, reg, values, normalize=True):
        if reg == "Goal_Position": WORLD.write_goal(self.role, values)
        else:
            for m, v in values.items(): self.write(reg, m, v)

    def enable_torque(self, motors=None, num_retry=0): WORLD.torque(self.role, True)
    def disable_torque(self, motors=None, num_retry=0): WORLD.torque(self.role, False)


# ======================================================================== rendering
class View:
    """Pinhole camera: pos, forward, right (world), focal length in px."""
    def __init__(self, pos, fwd, right, f, w=W, h=H):
        self.pos = np.asarray(pos, float); fw = unit(fwd); r = unit(np.asarray(right, float) - fw * (np.asarray(right) @ fw))
        self.M = np.stack([r, np.cross(fw, r), fw]); self.f, self.w, self.h = f, w, h

    def cam(self, P): return (np.asarray(P, float) - self.pos) @ self.M.T

    def px(self, Q): return np.stack([self.w / 2 + self.f * Q[:, 0] / Q[:, 2], self.h / 2 + self.f * Q[:, 1] / Q[:, 2]], 1)


# front: 36 cm in front of WP2, 3 cm lower, pitched up 10 deg -> the table is never inside the 900 mm depth band
VIEWS = {"top": View((0.16, 0.0, 0.66), (0, 0, -1), (0, -1, 0), 1380),
         "front": View((0.56, WP2_TIP[1], WP2_TIP[2] - 0.03), (-math.cos(0.1745), 0, math.sin(0.1745)), (0, 1, 0), 1380)}


def wrist_view(k):
    """Behind the jaws, beside the gripper body, looking along the fingers (jaws at the image edge)."""
    pos = k.tip - 0.09 * k.d + 0.06 * k.x
    return View(pos, k.d - 0.15 * k.x, k.y, 900)


NEAR = 0.03


def clip_near(Q):
    """Sutherland-Hodgman clip of a camera-frame polygon against z = NEAR."""
    out = []
    for i in range(len(Q)):
        a, b = Q[i], Q[(i + 1) % len(Q)]; ia, ib = a[2] >= NEAR, b[2] >= NEAR
        if ia: out.append(a)
        if ia != ib:
            t = (NEAR - a[2]) / (b[2] - a[2]); out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None


def box_faces(c, R, size):
    """[(4 corners world, outward normal, local axis key)] of a box."""
    out = []
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        for sg in (1, -1):
            pts = []
            for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                l = np.zeros(3); l[i] = sg * size[i] / 2; l[j] = u * size[j] / 2; l[k] = v * size[k] / 2
                pts.append(c + R @ l)
            out.append((np.array(pts), R[:, i] * sg, ("+" if sg > 0 else "-") + "xyz"[i]))
    return out


def link_faces(a, b, t):
    u = unit(np.asarray(b) - np.asarray(a)); v = unit(np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.95 else np.cross(u, [1.0, 0, 0]))
    R = np.stack([u, v, np.cross(u, v)], 1)
    return box_faces((np.asarray(a) + np.asarray(b)) / 2, R, (np.linalg.norm(np.asarray(b) - np.asarray(a)) + t, t, t))


LIGHT = unit((0.3, 0.2, 1.0))


def shade(col, n, amb=0.62):
    s = amb + (1 - amb) * max(0.0, float(n @ LIGHT))
    return tuple(int(min(255, c * s)) for c in col)


def disc(c, r, z, n=40):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n, z)], 1)


def static_polys():
    x0, x1, y0, y1 = TABLE
    out = [(np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], float), (118, 136, 150), 0)]
    for k, (c, r, col) in PLATES.items():
        out.append((disc(c, r + 0.006, 0.004), tuple(int(v * 0.7) for v in col), 1))
        out.append((disc(c, r, PLATE_Z), col, 2))
    (tx, ty), hs = TAPE
    out.append((np.array([[tx - hs, ty - hs, 0.0005], [tx + hs, ty - hs, 0.0005], [tx + hs, ty + hs, 0.0005],
                          [tx - hs, ty + hs, 0.0005]]), (60, 200, 235), 1))
    return out


def fill(img, view, P, col, depth=None, cross=False):
    Q = clip_near(view.cam(P))
    if Q is None: return
    uv = view.px(Q)
    if np.abs(uv).max() > 1e5: return
    pts = np.round(uv * 4).astype(np.int32)
    cv2.fillConvexPoly(img, pts, col, cv2.LINE_AA, shift=2)
    if depth is not None:
        cv2.fillConvexPoly(depth, np.round(uv).astype(np.int32), int(min(65535, Q[:, 2].mean() * 1000)))
    if cross and len(uv) == 4:
        side = float(np.linalg.norm(uv[0] - uv[2]))
        th = max(2, int(side / 26))
        for a, b in ((0, 2), (1, 3)):
            p0 = uv[a] + 0.22 * (uv[b] - uv[a]); p1 = uv[a] + 0.78 * (uv[b] - uv[a])
            cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), INK, th, cv2.LINE_AA)


def dynamic_faces(fp, bricks):
    """[(corners, colour, normal, is_cross)] for the arm, the gripper and the bricks."""
    k = FK(fp); out = []
    arm_col, grip_col = (222, 224, 226), (72, 72, 78)
    for a, b, t in ((k.base, k.shoulder, 0.07), (k.shoulder, k.elbow, 0.045), (k.elbow, k.wrist, 0.04),
                    (k.wrist, k.tip - 0.085 * k.d, 0.035)):
        out += [(P, arm_col, n, False) for P, n, _ in link_faces(a, b, t)]
    body = k.tip - 0.06 * k.d
    out += [(P, grip_col, n, False) for P, n, _ in box_faces(body, k.R, (0.034, max(0.05, k.gap + 0.02), 0.035))]
    for sg in (1, -1):
        fc = k.tip + k.y * sg * (k.gap / 2 + 0.004) - 0.012 * k.d
        out += [(P, grip_col, n, False) for P, n, _ in box_faces(fc, k.R, (0.02, 0.008, 0.05))]
    for _, pos, R, size, col, defect, state in bricks:
        if state == "floor": continue
        for P, n, ax in box_faces(pos, R, size):
            out.append((P, col, n, ax == defect))
    return out, k


class Renderer:
    """Renders top / front (+depth) / wrist. Static background per fixed camera is cached."""
    def __init__(self):
        self.base, self.depth0 = {}, {}

    def _background(self, view, name):
        img = np.zeros((view.h, view.w, 3), np.uint8)
        g = np.linspace(60, 92, view.h)[:, None, None]; img[:] = np.concatenate([g, g + 4, g + 8], 2).astype(np.uint8)
        for P, col, _ in static_polys(): fill(img, view, P, col)
        if name != "wrist":
            rng = np.random.default_rng({"top": 1, "front": 2}.get(name, 3))
            img = np.clip(img.astype(np.int16) + rng.integers(-5, 6, img.shape[:2], dtype=np.int16)[..., None], 0, 255).astype(np.uint8)
        return img

    def _static_depth(self, view):
        """Per-pixel distance (mm, along the optical axis) to the table top, else a wall at 2.5 m."""
        s = 4; u, v = np.meshgrid(np.arange(view.w // s) * s + s / 2, np.arange(view.h // s) * s + s / 2)
        rays = np.stack([(u - view.w / 2) / view.f, (v - view.h / 2) / view.f, np.ones_like(u)], -1) @ view.M
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -view.pos[2] / rays[..., 2]
        hit = view.pos + rays * t[..., None]
        x0, x1, y0, y1 = TABLE
        ok = (rays[..., 2] < 0) & (hit[..., 0] > x0) & (hit[..., 0] < x1) & (hit[..., 1] > y0) & (hit[..., 1] < y1)
        d = np.where(ok, t * 1000, 2500.0).clip(0, 65535).astype(np.uint16)
        return cv2.resize(d, (view.w, view.h), interpolation=cv2.INTER_NEAREST)

    def render(self, name, fp, bricks, want_depth=False):
        faces, k = dynamic_faces(fp, bricks)
        view = wrist_view(k) if name == "wrist" else VIEWS[name]
        if name == "wrist": img = self._background(view, name)
        else:
            if name not in self.base: self.base[name] = self._background(view, name)
            img = self.base[name].copy()
        depth = None
        if want_depth:
            if name not in self.depth0: self.depth0[name] = self._static_depth(view)
            depth = self.depth0[name].copy()
        items = []
        for P, col, n, cross in faces:
            c = P.mean(0)
            if n @ (view.pos - c) <= 1e-6: continue                 # back face
            z = float(view.cam(c[None])[0, 2])
            if z < NEAR: continue
            items.append((z, P, shade(col, n), cross))
        for z, P, col, cross in sorted(items, key=lambda x: -x[0]):
            fill(img, view, P, col, depth, cross)
        return img, depth


RENDER = Renderer()


class FakeCam:
    """Stands in for capture.RSCam / WristCam: read() -> BGR 1920x1080 at ~30 fps; front also has .depth (mm)."""
    def __init__(self, name):
        self.name = name; self.n = 0; self.depth, self.depth_t = None, 0.0
        self.settings = {"mock": True, "exposure": 166, "gain": 64.0, "white_balance": 4500}
        self._next = time.time(); self._key = None; self._img = None; self._dep = None
        print(f"{name} camera locked (mock):", self.settings)

    def read(self):
        st = WORLD.cam_state.get(self.name, "ok")
        if st != "ok":
            time.sleep(0.1); return None                            # frozen/offline: capture.py reconnects after 3 s
        now = time.time()
        if self._next > now: time.sleep(self._next - now)
        self._next = max(self._next + 1 / FPS, time.time() - 0.1)
        fp, bs = WORLD.snapshot()
        key = (tuple(fp.values()), tuple((i, tuple(np.round(p, 4)), tuple(np.round(R, 3).ravel()), s) for i, p, R, _, _, _, s in bs))
        depth_cam = self.name == "front" and WORLD.depth_on
        if key != self._key:
            self._img, self._dep = RENDER.render(self.name, fp, bs, want_depth=depth_cam); self._key = key
        self.n += 1
        if depth_cam and self.n % 3 == 0: self.depth, self.depth_t = self._dep, time.time()
        if not depth_cam: self.depth = None
        return self._img


def open_cam(name):
    """capture.open_cam in mock mode. JUNCTION_MOCK_CAMS=top,front leaves the wrist camera 'unplugged'."""
    want = os.environ.get("JUNCTION_MOCK_CAMS", "top,front,wrist").split(",")
    if name not in want or WORLD.cam_state.get(name) == "offline":
        print(f"!! {name} camera failed to open: mock camera offline"); return None
    return FakeCam(name)


# ======================================================================== fake Anomalib (used by the REAL score_worker.py)
def ink_pixels(img):
    """Pen-cross pixels: near black (masked background is grey 128, the gripper is dark grey ~70)."""
    return (img.max(2) < 45) if img is not None and img.ndim == 3 else np.zeros((1, 1), bool)


def fake_predict(image_path, model_path=""):
    """-> (score 0..1, anomaly map). Deterministic: ~0.05 for a clean crop, ~0.95 when a pen cross is visible."""
    img = cv2.imread(str(image_path))
    m = ink_pixels(img); n = int(m.sum())
    s = 0.05 + 0.9 * (1 - math.exp(-n / 250.0))
    amap = cv2.resize(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3), (64, 64))[None]
    return float(s), amap


FAKE_DEPLOY = '''"""Fake anomalib.deploy written by mock_hw.py (mock station only): scores = visible pen ink."""
import sys, types
import numpy as np
sys.path.insert(0, %r)
import mock_hw


class OpenVINOInferencer:
    def __init__(self, path=None, device="AUTO", **kw): self.path = str(path)

    def predict(self, image=None, **kw):
        s, amap = mock_hw.fake_predict(image, self.path)
        return types.SimpleNamespace(pred_score=np.array([s]), pred_label=np.array([s > 0.5]), anomaly_map=amap)


TorchInferencer = OpenVINOInferencer
'''


# ======================================================================== data root + pre-taught waypoints
def data_root():
    r = Path(os.environ.get("JUNCTION_DATA_ROOT") or (REPO / "mock_data")).expanduser().resolve()
    if r in (REPO, HERE) or HERE.is_relative_to(r) or r == Path.home():
        raise SystemExit(f"JUNCTION_DATA_ROOT={r} would mix mock data with the station's: use a separate folder")
    return r


def mock_routines():
    """Pre-taught waypoints for the mock geometry (ticks from ik()), same slot names as on the station."""
    (ix, iy), _, _ = PLATES["input"]; (tx, ty), _ = TAPE; x2, y2, z2 = WP2_TIP
    bz = PLATE_Z + 0.012; tz = 0.0005 + 0.012
    wp = lambda name, p, cap=False: {"name": name, "pos": p, "capture": cap}
    one = lambda name, p: {"waypoints": [wp(name, p)], "created": "mock"}
    r_plate = roll_for(0.0, math.atan2(iy, ix))
    return {
        "home": one("home", ik(0.10, 0.0, 0.20, 180, 1024, 2650)),
        "wp1": one("wp1", ik(ix, iy, 0.16, 180, r_plate, 2650)),
        "drops": {"waypoints": [wp(f"drops-{i + 1}", ik(ix + dx, iy + dy, bz + 0.03, 180, r_plate, 2000))
                                for i, (dx, dy) in enumerate(((-0.02, -0.02), (0.025, 0.02), (0.0, 0.03)))], "created": "mock"},
        "wp2_above": one("wp2_above", ik(x2, y2, z2 + 0.04, 180, 1024, 2000)),
        "wp2": {"waypoints": [wp("wp2-1", ik(x2, y2, z2, 180, 1024, 2000), True), wp("wp2-2", ik(x2, y2, z2, 180, 3072, 2000), True),
                              wp("wp2-3", ik(x2, y2, z2, 90, 1024, 2000), True)], "created": "mock"},
        "tape_above": one("tape_above", ik(tx, ty, 0.10, 180, 2048, 2000)),
        "tape_place": one("tape_place", ik(tx, ty, tz, 180, 2048, 2000)),
        **{f"sort_{b}{s}": one(f"sort_{b}{s}", ik(*PLATES[b][0], z, 180, 1024, 2000))
           for b in ("good", "defect") for s, z in (("_above", 0.14), ("", PLATE_Z + 0.04))},
    }


def seed(root):
    """Create the mock data root (idempotent: never overwrites what a mock session already changed)."""
    st = root / "settings"
    for d in ("training_images", "episodes", "runs", st): (root / d).mkdir(parents=True, exist_ok=True)
    files = {"routines.json": mock_routines(),
             "collect_settings.json": {"speed": 0.8},
             "sort_settings.json": {"pick": "handfeed", "judge": "manual", "faces": {"wp2-1": ["front"], "wp2-2": ["back"], "wp2-3": ["down"]},
                                    "device": "CPU", "release": True, "release_cam": "wrist", "retries": 1},
             "inspect_settings.json": {"release_roi": {"wrist": [0.2, 0.15, 0.8, 0.85]}}}
    for n, v in files.items():
        if not (st / n).exists(): (st / n).write_text(json.dumps(v, indent=1))
    for cam in ("front", "wrist"):
        for b in ("all", "all_efficientad"):
            d = root / "models" / cam / b
            if not (d / "weights" / "openvino" / "model.xml").exists():
                (d / "weights" / "openvino").mkdir(parents=True, exist_ok=True)
                (d / "weights" / "openvino" / "model.xml").write_text("<!-- MOCK anomaly model (mock_hw.fake_predict) -->\n")
                (d / "thresholds.json").write_text(json.dumps({"t_low": 0.3, "t_high": 0.6, "mock": True}))
    pk = root / "_mock_pkgs" / "anomalib" / "deploy"; pk.mkdir(parents=True, exist_ok=True)
    (pk.parent / "__init__.py").write_text('"""Fake anomalib (mock station)."""\n')
    (pk / "__init__.py").write_text(FAKE_DEPLOY % str(HERE))
    return root


def install(real_root):
    """Called by capture.py at import time in mock mode: redirect every data dir / settings file of every module to
    the mock root and point the anomaly worker at the fake anomalib. Returns the mock root (capture's new ROOT)."""
    root = seed(data_root()); st = root / "settings"
    if os.environ.get("JUNCTION_MOCK_ANOMALIB") != "real":
        os.environ["ANOMALIB_PY"] = sys.executable
        os.environ["PYTHONPATH"] = os.pathsep.join([str(root / "_mock_pkgs"), str(HERE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    import arm, collect, prep, sorter                      # noqa: E401  (arm imports this module's fake bus)
    arm.ROUTINES = st / "routines.json"
    prep.INSPECT_SETTINGS = st / "inspect_settings.json"
    sorter.HERE = st                                        # only used for sort_settings.json
    orig = collect.make_router

    def make_router(*a, **k):                               # Collector reads/writes Path(__file__).parent/collect_settings.json
        real = collect.__file__; collect.__file__ = str(st / "collect.py")
        try: return orig(*a, **k)
        finally: collect.__file__ = real
    collect.make_router = make_router
    print(f"MOCK STATION: no hardware, data root {root} (real station data under {real_root} is never touched)")
    return root


# ======================================================================== /mock API
def attach(app, S, A, C, T):
    """Wire the world to the real objects and add /mock/* routes (capture.py calls this in mock mode only)."""
    from fastapi import APIRouter, HTTPException
    WORLD.arm, WORLD.collector, WORLD.sorter = A, C, T
    r = APIRouter()

    @r.get("/mock/state")
    def mstate(): return WORLD.state()

    @r.post("/mock/scene")
    def scene(defect: str = "", size: str = "2x2", color: str = "", queue: str = None, reset: bool = True, empty: bool = False):
        """reset=1: a new brick on the input plate (defect '+x'|'-x'|'+y'|'-y'|'+z'|'-z'|'' = clean), bins cleared.
        queue='+x,,-z': defects of the NEXT bricks spawned (hand-feed / after a sort). empty=1: no brick at all."""
        face = lambda q: (q if q[:1] in "+-" else "+" + q) if q.strip() else None      # '+x' arrives as ' x' unless %2B-encoded
        if queue is not None: WORLD.queue = [{"defect": face(q.strip()), "size": size, **({"color": color} if color else {})} for q in queue.split(",")]
        if reset or empty:
            spec = {"defect": face(defect.strip()), "size": size, **({"color": color} if color else {})}
            WORLD.reset_scene(spec, empty=empty)
        return WORLD.state()

    @r.post("/mock/human")
    def human(mode: str = "", feed: int = -1):
        if mode in ("auto", "manual", "off"): WORLD.human["mode"] = mode; WORLD.human["phase"] = None
        if feed in (0, 1): WORLD.human["feed"] = bool(feed)
        return WORLD.state()

    @r.post("/mock/leader")
    def leader(shoulder_pan: float = None, shoulder_lift: float = None, elbow_flex: float = None, wrist_flex: float = None,
               wrist_roll: float = None, gripper: float = None, x: float = None, y: float = None, z: float = None,
               phi: float = 180.0, roll: int = 1024):
        """Move the leader arm (the human, mode becomes 'manual'): joint ticks, or a grasp point x/y/z (m) via ik()."""
        with WORLD.lock:
            tgt = dict(WORLD.human["target"] or {j: round(v) for j, v in WORLD.servo["leader"]["pos"].items()})
            if x is not None and y is not None and z is not None:
                try: tgt.update({j: v for j, v in ik(x, y, z, phi, roll, int(tgt["gripper"])).items() if j != "gripper"})
                except ValueError as e: raise HTTPException(400, str(e))
            for j, v in (("shoulder_pan", shoulder_pan), ("shoulder_lift", shoulder_lift), ("elbow_flex", elbow_flex),
                         ("wrist_flex", wrist_flex), ("wrist_roll", wrist_roll), ("gripper", gripper)):
                if v is not None: tgt[j] = float(v)
            WORLD.human.update({"mode": "manual", "target": tgt})
        return WORLD.state()

    @r.post("/mock/fault")
    def fault(obstacle: int = -1, drop: bool = False, regrip_fail: int = -1, bus_errors: int = -1, unplug: str = None):
        if obstacle in (0, 1): WORLD.faults["obstacle"] = bool(obstacle)
        if regrip_fail >= 0: WORLD.faults["regrip_fail"] = regrip_fail
        if bus_errors >= 0: WORLD.faults["bus_errors"] = bus_errors
        if unplug is not None: WORLD.faults["unplugged"] = [x for x in unplug.split(",") if x]
        if drop: WORLD.drop()
        return WORLD.state()

    @r.post("/mock/camera")
    def camera(cam: str, state: str = "", depth: int = -1):
        if cam in WORLD.cam_state and state in ("ok", "frozen", "offline"): WORLD.cam_state[cam] = state
        if depth in (0, 1): WORLD.depth_on = bool(depth)
        return WORLD.state()

    @r.post("/mock/reset")
    def reset():
        WORLD.faults.update({"obstacle": False, "regrip_fail": 0, "bus_errors": 0, "unplugged": []})
        WORLD.cam_state = {k: "ok" for k in WORLD.cam_state}; WORLD.depth_on = True
        WORLD.human.update({"mode": "auto", "target": None, "feed": True, "phase": None}); WORLD.queue = []
        WORLD.reset_scene(); return WORLD.state()

    @r.get("/mock/paths")
    def paths():
        """Every place this server writes to (all must be under the mock data root)."""
        import arm, prep
        main = sys.modules.get("__main__")
        return {"root": str(data_root()), "data": str(C.DATA), "episodes": str(C.EP), "collect_cfg": str(C.cfg_path),
                "runs": str(T.RUNS), "sort_cfg": str(T.cfg_path), "models": str(T.scorer.dir), "routines": str(arm.ROUTINES),
                "inspect_cfg": str(prep.INSPECT_SETTINGS), "cam_settings": str(getattr(main, "SETTINGS", "")),
                "capture_data": str(getattr(main, "DATA", "")), "anomalib_py": os.environ.get("ANOMALIB_PY", "")}

    @r.get("/mock/ik")
    def mik(x: float, y: float, z: float, phi: float = 180.0, roll: int = 1024, grip: int = 2650):
        try: return ik(x, y, z, phi, roll, grip)
        except ValueError as e: raise HTTPException(400, str(e))

    app.include_router(r)
    return r
