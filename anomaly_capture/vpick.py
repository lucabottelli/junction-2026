"""Vision pick: the top camera finds the bricks on the orange plate, the arm's kinematics turn the chosen brick's
position and angle into a grasp, and a scripted move picks it. Deterministic alternative to pick = ACT.

    detect(img)            bricks on the plate: centre, angle, size (pixels of the FULL top frame), colour
    Kin                    SO-101 forward / inverse kinematics in servo ticks (MuJoCo model + IK from sim/)
    Calib                  pixel -> table (x, y) in the kinematic model's frame + grasp height, fitted from data
    VisionPick(...).run()  one pick with retries -> (clamp command g_lock, info), like policy.ActPick

Why the calibration is fitted in the model's own frame: the model's joint zeros are not measured, so its absolute
positions are a few cm off. Fitting pixel -> FK(joints that really held / released a brick there) makes the same
model error appear on both sides, where it cancels.

    python vpick.py --fit-demos      first calibration from the recorded teleop picks (episodes/)
    python vpick.py --show IMG       draw the detections on a saved top frame
"""
import json, math, random, sys, tempfile, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CALIB = HERE / "vpick_calib.json"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# ---------------------------------------------------------------------------------------------- detection
PLATE_HSV = ((3, 140, 80), (20, 255, 255))     # the orange plate (OpenCV H 0-180)
PLATE_HUE = (0, 23)                            # plate, its shadows and its glare all keep this hue
MIN_SAT, MIN_VAL = 70, 45                      # below: glare (white) or noise, hue means nothing
COLOURS = {"pink": (160, 180), "green": (24, 75), "blue": (85, 130)}     # hue classes; anything else = "other"
SPOT_GREEN_MIN = 34                            # look at the re-clamp spot (detect(at=)): the yellow plate beside it has hue 22-32
                                               # and read as green; the green block's top is 41-57 (flow 08:49: the block
                                               # and the plate's rim were one 122 mm blob, the fetch closed on the rim 3x)
PINK_MIN_SAT = 108                             # the salmon brick has S ~134; plate glare with a pink hue S 76-101
MIN_SHORT_MM = 14.0                            # thinner than any brick face: a glare streak
MIN_AREA_MM2, MAX_AREA_MM2 = 250.0, 4500.0     # a 2x2 seen from above is ~1000 mm2, a 2x4 ~2000
PLATE_R_MM = 96.0                              # outer radius of the orange plate (194 px at 0.497 mm/px)


def plate_mask(img):
    """The orange plate as a filled convex hull (bricks leave holes in the colour mask), rim eroded off.
    -> (mask uint8, (cx, cy, r) of the hull in pixels) or (None, None)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(PLATE_HSV[0]), np.array(PLATE_HSV[1]))
    k = max(3, int(round(img.shape[1] / 250)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None, None
    i = 1 + int(np.argmax(st[1:, 4]))
    pts = np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(pts)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    out = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(out, hull, 255)
    e = max(3, int(r * 0.10)) | 1                      # the rim is ~8 % of the radius: bricks can't lie there
    return cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))), (float(cx), float(cy), float(r))


def detect(img, roi=None, mm_per_px=None, at=None):
    """Bricks on the orange plate. img = top-camera frame (BGR); roi = (x0, y0, x1, y1) if img is that crop of the
    full frame, resized to img's size (the recorded episodes are). All pixel values returned are in FULL-frame
    pixels. -> (list of {"c": (u, v), "angle": deg of the long side in the image (0..180), "size": (long, short) px,
    "area": px2, "colour": name, "box": 4 corners}, plate (cx, cy, r) or None)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi or (0, 0, w, h)
    sx, sy = (x1 - x0) / w, (y1 - y0) / h              # full-frame px per image px
    if at is not None:      # at = (u, v, radius px), full frame, needs mm_per_px: bricks in that circle wherever it is
        pm = np.zeros((h, w), np.uint8); cv2.circle(pm, (int(round(at[0])), int(round(at[1]))), int(round(at[2])), 255, -1)
        plate = (float(at[0]), float(at[1]), float(at[2]))      # (the re-clamp spot on the table; wood and tape have the plate's hue)
    else:
        pm, plate = plate_mask(img)
        if pm is None: return [], None
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fg = (pm > 0) & (S >= MIN_SAT) & (V >= MIN_VAL) & ~((H >= PLATE_HUE[0]) & (H <= PLATE_HUE[1]))
    mmpp = mm_per_px or PLATE_R_MM / plate[2]          # rough scale from the plate's outer radius, mm per image px
    k = max(3, int(round(2.5 / mmpp)) | 1)
    out = []
    for name, (lo, hi) in list(COLOURS.items()) + [("other", (None, None))]:
        if lo is None:
            if at is not None: continue
            m = fg & ~np.any([(H >= a) & (H <= b) for a, b in COLOURS.values()], axis=0)
        else:
            m = fg & (H >= (max(lo, SPOT_GREEN_MIN) if at is not None and name == "green" else lo)) & (H <= hi)
            if name == "pink": m &= S >= PINK_MIN_SAT
        m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))   # studs, highlights
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            a_mm = cv2.contourArea(c) * mmpp * mmpp
            if not MIN_AREA_MM2 <= a_mm <= MAX_AREA_MM2: continue
            c = cv2.convexHull(c).astype(np.float32)
            c[:, 0, 0] = x0 + c[:, 0, 0] * sx; c[:, 0, 1] = y0 + c[:, 0, 1] * sy       # -> full-frame pixels
            (cu, cv_), (rw, rh), ang = cv2.minAreaRect(c)
            if rw < rh: rw, rh, ang = rh, rw, ang + 90
            if rh / ((sx + sy) / 2) * mmpp < MIN_SHORT_MM: continue
            out.append({"c": (float(cu), float(cv_)), "angle": float(ang % 180), "size": (float(rw), float(rh)),
                        "area": float(cv2.contourArea(c)), "area_mm2": float(a_mm), "colour": name,
                        "box": cv2.boxPoints(((cu, cv_), (rw, rh), ang)).tolist()})
    return out, (x0 + plate[0] * sx, y0 + plate[1] * sy, plate[2] * sx)


def draw(img, bricks, plate=None, roi=None, pick=None):
    """Debug picture: detections drawn on img (same roi convention as detect)."""
    h, w = img.shape[:2]; x0, y0, x1, y1 = roi or (0, 0, w, h); sx, sy = w / (x1 - x0), h / (y1 - y0)
    P = lambda p: (int(round((p[0] - x0) * sx)), int(round((p[1] - y0) * sy)))
    out = img.copy()
    if plate: cv2.circle(out, P(plate[:2]), int(plate[2] * sx), (255, 255, 255), 1)
    for i, b in enumerate(bricks):
        col = (0, 255, 255) if pick is not None and i == pick else (255, 255, 255)
        cv2.polylines(out, [np.array([P(p) for p in b["box"]], np.int32)], True, col, 2 if col[0] == 0 else 1)
        cv2.circle(out, P(b["c"]), 3, col, -1)
        cv2.putText(out, f"{b['colour']} {b['angle']:.0f}", (P(b["c"])[0] + 6, P(b["c"])[1] - 6), 0, 0.45, col, 1)
    return out


# ---------------------------------------------------------------------------------------------- kinematics
class Kin:
    """SO-101 kinematics in servo ticks. Model + IK come from sim/ (station.py, the plugin's ArmIK); positions are
    in the model's frame (base at the origin, metres), which is only self-consistent, not measured (see Calib)."""
    def __init__(self, joint_map=None):
        """joint_map = the ticks <-> radians map a calibration was fitted with (Calib.d["joint_map"]); None = the
        current one of sim/calibration.json. A stored calibration must keep using its own map."""
        sys.path.insert(0, str(ROOT / "sim"))
        import logging, warnings
        import mujoco, station
        from physicalai_mujoco_plugin.compose import compose_scene
        from physicalai_mujoco_plugin.conveyor_demo import ArmIK
        from physicalai_mujoco_plugin.profiles.registry import get_profile
        logging.getLogger("physicalai_mujoco_plugin").setLevel(logging.WARNING)
        self.mj, self.st = mujoco, station
        self.cal = station.load_calib()
        if joint_map: self.cal = {**self.cal, **joint_map}
        self.joint_map = {k: self.cal[k] for k in ("follower_limits", "joints", "gripper")}
        with warnings.catch_warnings(), tempfile.TemporaryDirectory() as td:      # own scene file: sim/ is not written
            warnings.simplefilter("ignore")
            self.m = compose_scene(station.build_xml(self.cal, Path(td) / "station.xml"), get_profile("so101")).model
        self.ik = ArmIK(self.m)

    def rad(self, ticks):
        """ticks: dict by joint name or array (.., 6) -> radians (6,)"""
        if isinstance(ticks, dict): ticks = [ticks[j] for j in JOINTS]
        return self.st.ticks_to_rad(np.asarray(ticks, np.float64), self.cal)[0]

    def ticks(self, q5, gripper):
        t = self.st.rad_to_ticks(np.r_[np.asarray(q5, np.float64)[:5], 0.0], self.cal)[0]
        return {**{j: float(t[i]) for i, j in enumerate(JOINTS[:5])}, "gripper": float(gripper)}

    def fk(self, q):
        """q radians (>= 5) -> (tcp xyz m, jaw closing yaw rad, tilt of the fingers from straight down rad)"""
        ik = self.ik; ik.data.qpos[ik.qpos_adr] = np.asarray(q)[:5]; self.mj.mj_kinematics(self.m, ik.data)
        return ik.tcp(ik.data).copy(), ik.gripper_yaw(ik.data), ik.tilt(ik.data)

    def pivot(self, ticks, deg):
        """The pose `ticks` (dict) with the wrist flexed by deg while the grasp point stays where it is: shoulder
        lift and elbow are re-solved, pan / roll / gripper untouched. What is held between the fingers turns in
        place, so it keeps its position in a fixed camera's picture. -> (ticks dict, position error m)"""
        ik, mj = self.ik, self.mj
        q = self.rad(ticks)[:5].copy(); p0 = self.fk(q)[0]
        q[3] += math.radians(deg); e = np.zeros(3)
        for _ in range(60):
            ik.data.qpos[ik.qpos_adr] = q; mj.mj_kinematics(self.m, ik.data); mj.mj_comPos(self.m, ik.data)
            p = ik.tcp(ik.data); e = p0 - p
            if np.linalg.norm(e) < 2e-5: break
            mj.mj_jac(self.m, ik.data, ik._jacp, ik._jacr, p, ik.body)
            q[[1, 2]] += np.clip(np.linalg.lstsq(ik._jacp[:, ik.dof_adr][:, [1, 2]], e, rcond=None)[0], -0.15, 0.15)
        out = self.ticks(q, ticks["gripper"]); out["shoulder_pan"] = float(ticks["shoulder_pan"]); out["wrist_roll"] = float(ticks["wrist_roll"])
        return out, float(np.linalg.norm(e))

    def elevation(self, q):
        """Angle (deg) of the finger direction above horizontal for joints q (rad); negative = pointing down.
        The model's joint zeros are not measured, so the absolute value can be tens of degrees off; differences
        between poses are what to rely on."""
        self.fk(q); return math.degrees(math.asin(float(np.clip(-self.ik.data.xmat[self.ik.body].reshape(3, 3)[2, 2], -1, 1))))

    def pose_at(self, seed_ticks, target, elev_deg, roll_ticks=None):
        """Joints (ticks dict) that put the grasp point at `target` (model frame, m) with the fingers at elev_deg
        above horizontal: pan, shoulder lift, elbow and wrist flex are solved; roll and gripper come from
        seed_ticks (roll_ticks overrides the roll). -> (ticks, position error m)"""
        ik, mj = self.ik, self.mj
        seed = dict(seed_ticks)
        if roll_ticks is not None: seed["wrist_roll"] = roll_ticks
        q = self.rad(seed)[:5].copy(); target = np.asarray(target, np.float64); e = np.ones(4)
        for _ in range(300):
            ik.data.qpos[ik.qpos_adr] = q; mj.mj_kinematics(self.m, ik.data); mj.mj_comPos(self.m, ik.data)
            p = ik.tcp(ik.data).copy(); el = self.elevation(q); e = np.r_[target - p, math.radians(elev_deg - el)]
            if np.linalg.norm(e[:3]) < 2e-5 and abs(e[3]) < 1e-3: break
            ik.data.qpos[ik.qpos_adr] = q; mj.mj_kinematics(self.m, ik.data); mj.mj_comPos(self.m, ik.data)
            mj.mj_jac(self.m, ik.data, ik._jacp, ik._jacr, p, ik.body); Jp = ik._jacp[:, ik.dof_adr][:, :4].copy()
            Je = np.zeros(4)
            for k in range(4):
                q2 = q.copy(); q2[k] += 1e-4; Je[k] = math.radians(self.elevation(q2) - el) / 1e-4
            q[:4] += np.clip(np.linalg.lstsq(np.vstack([Jp, Je]), e, rcond=None)[0], -0.08, 0.08)
        out = self.ticks(q, seed["gripper"]); out["wrist_roll"] = float(seed["wrist_roll"])
        return out, float(np.linalg.norm(e[:3]))

    def lowest(self, ticks):
        """Height (m, model frame: table = 0) of the lowest point of the arm's links at the pose `ticks` (dict)."""
        mj, m, d = self.mj, self.m, self.ik.data
        if not hasattr(self, "_arm_geoms"):
            names = ("upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")
            ids = {m.body(n).id for n in names}
            self._arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
        q = self.rad(ticks); d.qpos[self.ik.qpos_adr] = q[:5]; d.qpos[m.joint("gripper").qposadr[0]] = q[5]
        mj.mj_kinematics(m, d); z = 9.0
        for g in self._arm_geoms:
            k = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            z = min(z, float((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()))
        return z

    def solve(self, q0, target, yaw, iters=80):
        """Joint angles (5,) putting the grasp point at target with the jaws closing along yaw (mod 180 deg: both
        flips are tried, the smaller wrist turn wins). -> (q, position error m, tilt rad, yaw used)"""
        best = None
        for flip in (0.0, math.pi):
            q, err = self.ik.solve(np.asarray(q0, np.float64)[:5].copy(), np.asarray(target, np.float64), yaw + flip, iters=iters)
            _, got, tilt = self.fk(q)
            dyaw = abs((got - yaw - flip + math.pi) % (2 * math.pi) - math.pi)
            score = (err > 0.004) * 10 + (dyaw > math.radians(10)) * 3 + abs(q[4] - q0[4]) * 0.3 + tilt * 0.2
            if best is None or score < best[0]: best = (score, q, float(err), float(tilt), yaw + flip, float(dyaw))
        return best[1:]


# ---------------------------------------------------------------------------------------------- calibration
class Calib:
    """pixel (full top frame) -> grasp point (x, y, z) in the model frame. H = 3x3 homography (plate plane),
    z = a + b*x + c*y (the grasp height the kinematic model needs there, absorbs the model's lift/elbow error)."""
    def __init__(self, d=None):
        d = d or json.loads(CALIB.read_text())
        self.d = d; self.H = np.array(d["H"], np.float64); self.z = np.array(d["z"], np.float64)

    def xy(self, u, v):
        p = self.H @ np.array([u, v, 1.0]); return p[:2] / p[2]

    def px(self, x, y):
        """Model (x, y) on the plate plane -> pixel (u, v) of the full top frame."""
        p = np.linalg.inv(self.H) @ np.array([x, y, 1.0]); return float(p[0] / p[2]), float(p[1] / p[2])

    def xyz(self, u, v):
        x, y = self.xy(u, v); return np.array([x, y, self.z[0] + self.z[1] * x + self.z[2] * y])

    def dir_world(self, u, v, angle_deg):
        """Image direction (deg, as detect() gives it) at pixel (u, v) -> world yaw (rad)."""
        a = math.radians(angle_deg); d = 20.0
        p0, p1 = self.xy(u - d * math.cos(a), v - d * math.sin(a)), self.xy(u + d * math.cos(a), v + d * math.sin(a))
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def mm_per_px(self, u, v):
        return float(np.linalg.norm(self.xy(u + 10, v) - self.xy(u, v)) * 100)

    @staticmethod
    def fit(px, xyz, ransac_mm=12.0):
        """px (N, 2) full-frame pixels, xyz (N, 3) model-frame metres -> (dict for Calib, residuals mm (N,), inliers)"""
        px, xyz = np.asarray(px, np.float64), np.asarray(xyz, np.float64)
        H, inl = cv2.findHomography(px, xyz[:, :2] * 1000, cv2.RANSAC, ransac_mm)       # mm for conditioning
        inl = inl.ravel().astype(bool)
        H = H / 1000.0; H[2] *= 1000.0                                                  # -> metres out
        A = np.column_stack([np.ones(inl.sum()), xyz[inl, 0], xyz[inl, 1]])
        z, *_ = np.linalg.lstsq(A, xyz[inl, 2], rcond=None)
        c = Calib({"H": H.tolist(), "z": z.tolist()})
        res = np.array([np.linalg.norm(c.xy(*p) - q[:2]) for p, q in zip(px, xyz)]) * 1000
        return c.d, res, inl


# ---------------------------------------------------------------------------------------------- grasp planning
REF_W = 0.032            # brick width the calibration's "TCP at the grasp" belongs to (a 2x2 held across 32 mm)
FIX_CLEAR = 0.009        # gap left between the fixed jaw and the brick: the moving jaw pushes the brick across it
MOV_CLEAR = 0.014        # gap on the moving jaw's side when open
SQUARE = 1.28            # long / short side below this: both jaw directions are tried
LONG_BRICKS = {"green"}  # blocks that are NOT square whatever the camera measures (green = the 2x3, 48 x 32 mm): seen
                         # at an angle or with a side face in the blob its outline can read below SQUARE
TICKS_SHUT, TICKS_PER_M = 1930.0, 9700.0     # gripper ticks with the jaws shut on nothing; ticks per metre of gap
MAX_GRASP_W = 0.052
HOVER = 0.055            # m above the grasp point where the arm lines up before going down
RG_HOVER_FAST = 0.038    # re-clamp with fast moves on: hover this far above the release height instead. The fingertips
                         # are then ~42 mm above the table, ~18 mm over the studs of a block (24 mm high)
RG_STEPS_FAST = 3        # ... and go down in 3 corrected steps of 14 mm instead of 6 of 10 mm
RG_CLOSE_FAST = 0.5      # ... and close the jaws at this speed instead of 0.3 (what the fast pick uses)
YAW_BIAS_DEG = -11.5     # FK jaw yaw minus the brick axis in 99 human grasps (median; IQR -17..-6): the model reads the jaws 11.5 deg off
Z_SEEK = 0.012           # how far below the calibrated grasp height the descent may go looking for the plate (contact stops it)
EMPTY_TICKS = 25         # fingers stopped fewer ticks than this above the closed end: nothing between them
HELD_TICKS = 25          # fingers at least this far short of the squeeze command at waypoint 1: still holding
SURE_TICKS = 40      # this far short of the squeeze command at WP1 = held for sure, whatever the top camera sees
RIM_SLACK = 0.02         # m a finger may stand outside the plate's flat part before a grasp counts as blocked
MISS_PX, MISS_S = 45.0, 180.0     # a brick still within this many px of where a pick just failed (in the last MISS_S s) is
LAST_MISS = {}                    # dragged to the plate's middle before the next try: {"px": (u, v), "t": time, "colour": name}
DRAG_GAP = 0.065         # m the jaws are opened to straddle a brick that is dragged off the rim
FINGER = 0.011           # half thickness of a jaw, for the clearance to the neighbours
# What a grasp tuner may change per pick (grasp_tuner.Tuner.suggest); these values change nothing:
NEUTRAL = {"dz_mm": 0.0,             # grasp height offset
           "dxy_mm": 0.0,            # shift along the jaw axis (+ = toward the moving jaw)
           "yaw_off_deg": 0.0,       # jaw direction offset
           "open_frac": 1.0,         # multiplier on the opening margin around the brick
           "approach_speed": 1.0}    # multiplier on the descent speed


def plan_grasps(bricks, calib, kin, q_seed, roll_limits, plate=None, both=False):
    """Every way to grasp every brick, best first. Bricks are ordered nearest to the robot first (the user's rule);
    within a brick, the jaw direction with the most room to the neighbours and the plate rim wins.
    -> list of {"i": brick index, "xyz": TCP target, "yaw": jaw yaw (fixed -> moving jaw), "q": joints rad,
    "open": gripper ticks, "width": m, "clear": m, "dist": m from the robot base}
    both=True (re-clamp, sorter.regrip_vision): a brick that is not square is also offered across its LONG side
    (a 2x3 is 48 mm that way), so a grasp turned 90 deg exists for every brick."""
    W = []
    for b in bricks:
        u, v = b["c"]; mm = calib.mm_per_px(u, v) / 1000.0
        W.append({"c": calib.xy(u, v), "long": b["size"][0] * mm, "short": b["size"][1] * mm,
                  "th": calib.dir_world(u, v, b["angle"])})
    pc = calib.xy(*plate[:2]) if plate else None
    pr = (plate[2] * calib.mm_per_px(*plate[:2]) / 1000.0 * 0.80) if plate else None      # flat part of the dish
    out = []
    for i, w in enumerate(W):
        square = w["long"] / max(1e-6, w["short"]) < SQUARE
        opts = [(w["th"] + math.pi / 2, w["short"], False)]
        if square or both: opts.append((w["th"], w["long"], not square))
        for yaw0, width, long_side in opts:
            if long_side: width = min(width, MAX_GRASP_W - 0.002)            # across the long side of a 2x3: its real length
            else: width = min(width, 0.040) if width < MAX_GRASP_W else width      # side faces inflate the blob a little
            if width > MAX_GRASP_W: continue
            for flip in (0.0, math.pi):
                yaw = yaw0 + flip; x = np.array([math.cos(yaw), math.sin(yaw)])
                shift = FIX_CLEAR + max(0.0, width - REF_W) / 2
                c = w["c"] - shift * x                                        # TCP: fixed jaw FIX_CLEAR off the brick
                fingers = [w["c"] - (width / 2 + FIX_CLEAR + FINGER) * x, w["c"] + (width / 2 + MOV_CLEAR + FINGER) * x]
                clear = 1.0
                for j, o in enumerate(W):
                    if j == i: continue
                    clear = min(clear, min(np.linalg.norm(f - o["c"]) for f in fingers) - math.hypot(o["long"], o["short"]) / 2 - FINGER)
                rim = 1.0                                                     # m the outer finger stays inside the flat part
                if pc is not None:
                    rim = float(min(pr - np.linalg.norm(f - pc) for f in fingers))
                    clear = min(clear, rim + RIM_SLACK)
                z = calib.z[0] + calib.z[1] * c[0] + calib.z[2] * c[1]
                yaw = yaw + math.radians(calib.d.get("yaw_bias_deg", YAW_BIAS_DEG))    # model yaw that really lines the jaws up
                q, err = kin.ik.solve(np.asarray(q_seed, np.float64)[:5].copy(), np.array([c[0], c[1], z]), yaw, iters=80)
                _, got, tilt = kin.fk(q)
                roll = kin.ticks(q, 0)["wrist_roll"]
                if err > 0.004 or abs((got - yaw + math.pi) % (2 * math.pi) - math.pi) > math.radians(12): continue
                if not roll_limits[0] + 30 <= roll <= roll_limits[1] - 30: continue
                gap = width + FIX_CLEAR + MOV_CLEAR
                out.append({"i": i, "xyz": [float(c[0]), float(c[1]), float(z)], "yaw": float(yaw), "q": q.tolist(),
                            "open": TICKS_SHUT + TICKS_PER_M * gap, "width": float(width), "clear": float(clear),
                            "tilt": float(tilt), "dist": float(np.linalg.norm(w["c"])), "centre": w["c"].tolist(), "rim": rim,
                            "turn": float(abs(q[4] - q_seed[4]))})
    # nearest brick first; a blocked grasp (fingers would land on a neighbour) goes behind every free one
    out.sort(key=lambda g: (g["clear"] < 0.004, round(g["dist"] / 0.015), -min(g["clear"], 0.02), g["turn"]))
    return out


# ---------------------------------------------------------------------------------------------- the pick
class VisionPick:
    """One pick with retries, same contract as policy.ActPick: run() -> (clamp command g_lock, info) or raises
    PickFailed / Stopped. A = arm.Arm, C = collect.Collector, S = capture state (frames, frame_t)."""
    def __init__(self, A, C, S, kin, calib, retries=2, speed=1.0, check=None, say=None, debug_dir=None, target=None, smooth=False, fast=False,
                 on_event=None, tuner=None, force=None):
        import policy as pol
        self.pol = pol
        self.A, self.C, self.S, self.kin, self.calib = A, C, S, kin, calib
        self.retries, self.speed = int(retries), float(speed)
        self.fast = bool(fast)        # fast: 3 descent steps instead of 6, quicker travel to the hover and up to WP1
        self.smooth = bool(smooth)       # one continuous descent instead of six corrected steps (see attempt, step 2)
        self.check = check or (lambda: None); self.say = say or (lambda m, ok=True: None)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.target = target                    # colour to pick; "random" = one of the colours seen, drawn once;
        self.colour = None                      # "" / None = the brick nearest to the robot, whatever its colour
        self.on_event = on_event or (lambda name, info: None)     # "start" (plan made, about to move), "grasped"
        self.tuner = tuner; self.tune = dict(NEUTRAL)             # grasp_tuner.Tuner (optional): suggest(colour)
        self.attempts = []; self.grasp_pose = None; self.live = {}; self.tried = set(); self.g_lock = None; self.z_rel = 0.0
        self.plate = None; self.seen = []
        # re-clamp (sorter.regrip_vision): {"px": (u, v) where the brick was set down, "yaw": model jaw yaw (rad, mod
        # 360) the grasp must have, "max_px": how far from px the brick may lie, "tol_deg": allowed yaw difference}
        self.force = force
        self._spot_off = None    # px: where the brick is seen on the re-clamp spot minus where the calibration puts the release pose

    # ---- small helpers
    def _check(self):
        self.check()
        if self.A.stop.is_set(): raise self.pol.Stopped()

    def _q_now(self): return self.kin.rad(self.A.pos)

    def _goto(self, q5, gripper, speed, off=None):
        t = self.kin.ticks(q5, gripper)
        if off:
            for n, v in off.items(): t[n] += v
        self._check(); self.C.goto(t, self.C.cfg["speed"] * speed * self.speed); return t

    def _servo_in(self, q5, gripper, tol=6, iters=4):
        """Make the MEASURED joints equal q5: the servos sag a few degrees under gravity, so the command is pushed
        past the target by what is still missing. -> the offsets (ticks) that hold the arm there."""
        tgt = self.kin.ticks(q5, gripper); off = {n: 0.0 for n in self.pol.BODY}
        for _ in range(iters):
            self._goto(q5, gripper, 0.35, off); time.sleep(0.12 if self.fast else 0.2)
            err = {n: tgt[n] - self.A.pos[n] for n in self.pol.BODY}
            if max(abs(e) for e in err.values()) <= tol: break
            for n in off: off[n] += max(-80.0, min(80.0, err[n]))
        return off

    def look(self):
        """A fresh top frame with the arm at home (out of the picture) -> (frame, bricks, plate)."""
        t0 = time.time()
        while self.S.frame_t.get("top", 0) < t0 + 0.12:
            self._check()
            if time.time() - t0 > 3: raise self.pol.PickFailed("the top camera gives no fresh frame", retry=False)
            time.sleep(0.03)
        f = self.S.frames["top"].copy()
        bricks, plate = detect(f)
        return f, bricks, plate

    def _settle(self, max_s=0.8):
        """Wait until the measured body joints stop moving (A.goto returns within 3.5 deg of the target, which is
        still several mm at the fingertips)."""
        t0 = time.time(); last = dict(self.A.pos)
        while time.time() - t0 < max_s:
            time.sleep(0.1); self._check(); now = dict(self.A.pos)
            if max(abs(now[n] - last[n]) for n in self.pol.BODY) <= 2: return
            last = now

    def _snap(self, name):
        """Top + wrist frames to the debug folder (where the fingers really are relative to the brick)."""
        if self.debug_dir is None: return
        for cam in ("top", "wrist"):
            f = self.S.frames.get(cam)
            if f is not None: self._save(f"{name}_{cam}.jpg", cv2.resize(f, (960, 540)))

    def _save(self, name, img):
        if self.debug_dir is not None:
            try: cv2.imwrite(str(self.debug_dir / f"{time.strftime('%H%M%S')}_{name}"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])   # one set per trial
            except Exception: pass

    # ---- a brick on the rim: drag it to the flat middle first
    def drag_in(self, b, plate, tag=""):
        """The brick lies on / against the plate's raised rim (often tipped): a grasp there puts a finger on the rim,
        clamps the plate and misses. Straddle it with wide-open jaws, the rear jaw on the rim side, and drag it toward
        the plate's centre. Nothing is closed. -> True if the drag was made (look again after it)."""
        A, C, kin = self.A, self.C, self.kin
        c = self.calib.xy(*b["c"]); pc = self.calib.xy(*plate[:2])
        R = plate[2] * self.calib.mm_per_px(*plate[:2]) / 1000.0
        v = pc - c; dist = float(np.linalg.norm(v))
        if dist < 1e-3: return False
        u = v / dist; travel = min(0.075, max(0.03, dist - 0.30 * R))
        z = float(self.calib.xyz(*b["c"])[2])
        g_open = min(A.fl["gripper"][1], TICKS_SHUT + TICKS_PER_M * DRAG_GAP)
        q0 = kin.rad(C.wps("wp1")[0]["pos"]); best = None
        for back in (0.014, DRAG_GAP - 0.046):          # rear jaw = the fixed one (yaw along u) / the moving one (yaw against u)
            yaw = math.atan2(u[1], u[0]) + (0.0 if back == 0.014 else math.pi)
            start = c - u * back if back == 0.014 else c + u * back
            q, err = kin.ik.solve(np.asarray(q0, np.float64)[:5].copy(), np.array([start[0], start[1], z + HOVER]), yaw, iters=80)
            _, got, _ = kin.fk(q); roll = kin.ticks(q, 0)["wrist_roll"]
            ok = err <= 0.004 and abs((got - yaw + math.pi) % (2 * math.pi) - math.pi) <= math.radians(12) \
                and A.fl["wrist_roll"][0] + 30 <= roll <= A.fl["wrist_roll"][1] - 30
            if ok and (best is None or abs(q[4] - q0[4]) < best[3]): best = (q, yaw, start, abs(q[4] - q0[4]))
        if best is None: return False
        q, yaw, start, _ = best
        self.say("the brick is on the plate's rim: dragging it to the middle first")
        self._goto(q, g_open, 1.0); off = self._servo_in(q, g_open); zz = z + HOVER
        for s_ in range(1, 6):                                            # down in steps; stop where the fingers stand on the rim
            zz = z + HOVER * (1 - s_ / 5)
            q, _ = kin.ik.solve(q, np.array([start[0], start[1], zz]), yaw, iters=30)
            self._goto(q, g_open, 0.45, off); self._settle(0.4)
            if kin.fk(self._q_now())[0][2] - zz > 0.006: zz = float(kin.fk(self._q_now())[0][2]) + 0.002; break
        self._snap(f"{tag}drag0")
        for s_ in range(1, 5):                                            # the drag, at that height
            p = start + u * travel * s_ / 4
            q, _ = kin.ik.solve(q, np.array([p[0], p[1], zz]), yaw, iters=30)
            self._goto(q, g_open, 0.3, off)
        self._snap(f"{tag}drag1")
        q, _ = kin.ik.solve(q, np.array([p[0], p[1], z + HOVER + 0.02]), yaw, iters=40)
        self._goto(q, g_open, 0.7, off)
        return True

    # ---- one attempt
    def attempt(self, k, tag=""):
        A, C, kin = self.A, self.C, self.kin
        gmin = A.fl["gripper"][0]
        f, bricks, plate = self.look()
        every = bricks
        fo = self.force; spot = fo.get("at") if fo else None
        if spot:                                 # the brick lies off the plate, on the re-clamp spot: look there
            bricks = detect(f, mm_per_px=self.calib.mm_per_px(spot[0], spot[1]), at=spot)[0]
        if self.target == "random":
            if self.colour is None:
                seen = sorted({b["colour"] for b in bricks if b["colour"] in COLOURS})
                self.colour = random.choice(seen) if seen else ""
            want = self.colour
        else: want = self.target
        if want: bricks = [b for b in bricks if b["colour"] == want]
        where = "the re-clamp spot" if spot else "the orange plate"
        if want and not bricks: raise self.pol.PickFailed(f"no {want} brick found on {where}", retry=False)
        if not bricks: raise self.pol.PickFailed(f"no brick found on {where}", retry=False)
        wp1 = kin.rad(C.wps("wp1")[0]["pos"])
        if fo:                                   # only the brick that was just set down, only a grasp turned 90 deg
            near = min(bricks, key=lambda x: math.hypot(x["c"][0] - fo["px"][0], x["c"][1] - fo["px"][1]))
            off_px = math.hypot(near["c"][0] - fo["px"][0], near["c"][1] - fo["px"][1])
            if off_px > fo.get("max_px", 130):
                raise self.pol.PickFailed(f"the {near['colour']} brick is not where it was set down ({off_px:.0f} px away)", retry=False)
            bricks = [near]
        plan_b = bricks
        if spot:
            # The calibration is fitted on the plate; at the re-clamp spot it was ~37 mm off (07:42: the jaws closed
            # beside the brick twice). So the arm goes by its own release pose: the brick seen right after the release
            # IS where the jaws let go (fo["rel_xy"]); that pixel difference corrects this and every later look.
            if fo.get("rel_xy") is not None and self._spot_off is None:
                cp = self.calib.px(fo["rel_xy"][0], fo["rel_xy"][1])
                self._spot_off = (near["c"][0] - cp[0], near["c"][1] - cp[1])
            d_ = self._spot_off or fo.get("off_px") or (0.0, 0.0)
            plan_b = [{**near, "c": (near["c"][0] - d_[0], near["c"][1] - d_[1])}]
        plans = plan_grasps(plan_b, self.calib, kin, wp1, A.fl["wrist_roll"], None if spot else plate, both=bool(fo))
        turn = lambda x: (0.0, 0)
        if fo and fo.get("yaw") is not None:
            # fo["yaw"] = jaw yaw at the release. The new grasp must be turned 90 deg from it, +90 if that is in reach,
            # else -90 (the user's rule, 2026-11-15 07:40); which one it was is reported (info["forced"]["turn_deg"]).
            wrap = lambda a_: abs((a_ + math.pi) % (2 * math.pi) - math.pi)
            turn = lambda x: min((wrap(x["yaw"] - fo["yaw"] - s_ * math.pi / 2), -s_) for s_ in (1, -1))
            n_all = len(plans)
            plans = sorted([x for x in plans if turn(x)[0] <= math.radians(fo.get("tol_deg", 30))], key=lambda x: (turn(x)[1], turn(x)[0]))
            if n_all and not plans:              # the brick turned when it was let go: which faces were gripped is not known
                devs = sorted(round(math.degrees(turn(x)[0])) for x in plan_grasps(plan_b, self.calib, kin, wp1, A.fl["wrist_roll"], None, both=True))
                raise self.pol.PickFailed(f"the brick does not lie as it was set down (nearest grasp is {devs[0] if devs else '?'}° from a 90° turn, "
                                          f"limit {fo.get('tol_deg', 30)}°)", retry=False)
        if not plans:
            self._save(f"{tag}look.jpg", draw(f, bricks, plate)); return "unreachable", {"bricks": len(bricks)}
        key = lambda x: (round(x["centre"][0] / 0.01), round(x["centre"][1] / 0.01), round(math.degrees(x["yaw"]) / 20) % 9)
        g = next((x for x in plans if key(x) not in self.tried), plans[0])    # a retry takes another jaw direction / brick
        self.tried.add(key(g))
        b = bricks[g["i"]]
        if plate is not None and not fo and not getattr(self, "_dragged", False):   # on the rim: one drag to the middle, then a fresh look
            lm = LAST_MISS                                                 # the last pick failed on this very brick
            if lm and time.time() - lm["t"] < MISS_S and lm["colour"] == b["colour"] \
                    and math.hypot(b["c"][0] - lm["px"][0], b["c"][1] - lm["px"][1]) < MISS_PX:
                self._dragged = True; self._save(f"{tag}look_rim.jpg", draw(f, bricks, plate, pick=g["i"]))
                if self.drag_in(b, plate, tag):
                    C.goto(C.wps("home")[0]["pos"], gripper=A.cmd["gripper"]); self.tried.discard(key(g))
                    return self.attempt(k, tag)
        self.plate = plate; self.seen = [x["c"] for x in every if x is not b]      # for setting it down again
        self._save(f"{tag}look.jpg", draw(f, bricks, plate, pick=g["i"]))
        mm_b = self.calib.mm_per_px(b["c"][0], b["c"][1])
        info = {"brick": b["colour"], "px": [round(v, 1) for v in b["c"]], "angle": round(b["angle"], 1),
                "size_mm": [round(b["size"][0] * mm_b, 1), round(b["size"][1] * mm_b, 1)], "xyz": [round(v, 4) for v in g["xyz"]],
                "yaw": g["yaw"], "width_mm": round(g["width"] * 1000, 1), "clear_mm": round(g["clear"] * 1000, 1),
                "n_bricks": len(bricks), "tilt_deg": round(math.degrees(g["tilt"]), 1), "rim_mm": round(g.get("rim", 1.0) * 1000, 1),
                "dragged": bool(getattr(self, "_dragged", False))}
        if fo: info["forced"] = {"release_yaw_deg": None if fo.get("yaw") is None else round(math.degrees(fo["yaw"]), 1), "turn_deg": -90 * turn(g)[1],
                                 "yaw_err_deg": round(math.degrees(turn(g)[0]), 1), "moved_px": round(off_px, 1),
                                 "set_down_px": [round(float(v), 1) for v in fo["px"]], "spot": bool(spot),
                                 "spot_off_px": None if not spot else [round(float(v), 1) for v in d_]}
        self.live = {"attempt": k + 1, "brick": b["colour"], "bricks": len(bricks)}
        self.say(f"vision pick {k + 1}/{self.retries + 1}: {b['colour']} brick, {len(bricks)} on the plate")
        if self.tuner is not None:
            try: self.tune = {**NEUTRAL, **(self.tuner.suggest(b["colour"]) or {})}
            except Exception as e: self.tune = dict(NEUTRAL); info["tuner_error"] = str(e)
        tu = self.tune; info["tune"] = dict(tu)
        yaw = g["yaw"] + math.radians(tu["yaw_off_deg"]); x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        p = np.array(g["xyz"]) + tu["dxy_mm"] / 1000.0 * x + [0, 0, tu["dz_mm"] / 1000.0]
        gap = g["width"] + (FIX_CLEAR + MOV_CLEAR) * max(0.5, min(2.0, tu["open_frac"]))
        g_open = min(A.fl["gripper"][1], max(TICKS_SHUT + TICKS_PER_M * gap, gmin + 200))
        v_down = 0.45 * max(0.4, min(2.0, tu["approach_speed"]))
        self.on_event("start", info)
        # 1. above the brick, jaws open; measured joints servoed onto the IK pose
        q_hover, err = kin.ik.solve(np.array(g["q"]), p + [0, 0, HOVER], yaw, iters=60)
        self._goto(q_hover, g_open, 1.8 if self.fast else 1.0)
        off = self._servo_in(q_hover, g_open)
        self._settle()
        z0 = kin.fk(self._q_now())[0][2] - (p[2] + HOVER)              # what "on target" reads at the hover
        self._snap(f"{tag}hover")
        info["hover_err_mm"] = [round(v * 1000, 1) for v in (kin.fk(self._q_now())[0] - (p + [0, 0, HOVER]))]
        info["off_ticks"] = {n: round(v) for n, v in off.items()}
        # 2. straight down in steps; stop early if the fingers stand on something (measured height stops following)
        q = q_hover; n = 3 if self.fast else 6; touched = None; z_low = p[2] - Z_SEEK; steps = []; info["dz_steps_mm"] = steps
        info["smooth"] = self.smooth; info["fast"] = self.fast
        if self.smooth:
            # Smooth: one move from the hover to the grasp depth, then one sag correction without stopping in between.
            # No stepwise sag correction and no contact stop on the way, so the depth can be a few mm less exact.
            q, _ = kin.ik.solve(q, np.array([p[0], p[1], z_low]), yaw, iters=60)
            self._goto(q, g_open, v_down, off); self._settle(0.3)
            tgt = kin.ticks(q, g_open)
            for n_ in off: off[n_] += max(-40.0, min(40.0, tgt[n_] - A.pos[n_]))
            self._goto(q, g_open, 0.3, off); self._settle(0.3)
            steps.append(round((kin.fk(self._q_now())[0][2] - z_low) * 1000, 1))
        for s in ([] if self.smooth else range(1, n + 1)):
            z = p[2] + HOVER + (z_low - p[2] - HOVER) * s / n
            q, _ = kin.ik.solve(q, np.array([p[0], p[1], z]), yaw, iters=30)
            self._goto(q, g_open, v_down, off); self._settle(0.5)
            tgt = kin.ticks(q, g_open)                                   # sag changes as the arm reaches down: correct
            for n_ in off: off[n_] += max(-40.0, min(40.0, tgt[n_] - A.pos[n_]))   # the joints again at every step
            self._goto(q, g_open, 0.3, off); self._settle(0.5)
            dz = kin.fk(self._q_now())[0][2] - z; steps.append(round(dz * 1000, 1))
            if dz > 0.005: touched = (z, dz); break                      # still held up after the correction: contact
        if fo and fo.get("seek") and touched is None and not self.smooth:
            # the re-clamp spot is on the table, lower than the plate the heights are calibrated for: go on down in
            # small steps until the fingers stand on the table (same contact test)
            z = z_low
            while z > z_low - fo["seek"] + 1e-6:
                z -= 0.004
                q, _ = kin.ik.solve(q, np.array([p[0], p[1], z]), yaw, iters=30)
                self._goto(q, g_open, 0.3, off); self._settle(0.4)
                dz = kin.fk(self._q_now())[0][2] - z; steps.append(round(dz * 1000, 1))
                if dz > 0.005: touched = (z, dz); break
        info["touch_mm"] = None if touched is None else [round((touched[0] - p[2]) * 1000, 1), round(touched[1] * 1000, 1)]
        self._snap(f"{tag}low")
        if touched and touched[0] - p[2] > 0.014:                        # stopped high: standing on the brick
            C.goto(dict(A.cmd), 0.5, gripper=g_open); C.rise(C.wps("wp1")[0]["pos"])
            return "landed", info
        if touched:                                                      # on the plate: come up a little
            q, _ = kin.ik.solve(q, np.array([p[0], p[1], touched[0] + touched[1] + 0.002]), yaw, iters=30)
            self._goto(q, g_open, 0.3, off)
        tcp = kin.fk(self._q_now())[0]; info["tcp_err_mm"] = [round(v * 1000, 1) for v in (tcp - p)]
        info["tcp"] = [round(float(v), 4) for v in tcp]; self.z_rel = float(tcp[2] - p[2])   # height it is held at
        # 3. close until the fingers stall, then squeeze a little past that width
        self.grasp_pose = dict(A.cmd)
        C.goto(dict(A.cmd), 0.5 if self.fast else 0.3, gripper=gmin); time.sleep(0.15)
        g_stall = A.pos["gripper"]; info["g_stall"] = round(g_stall)
        # Shut on nothing = the fingers reached the command (reads gmin + ~6). Held = they stopped short of it:
        # a 2x2 clamped by this pick stalls only ~52 ticks above gmin (1927), so the old "< 120" threw good grasps away.
        if g_stall - gmin < EMPTY_TICKS:
            self._snap(f"{tag}shut")
            C.goto(dict(A.cmd), 0.5, gripper=g_open); C.rise(C.wps("wp1")[0]["pos"])
            return "empty", info
        from collect import REGRIP_SQUEEZE
        g_lock = max(getattr(A, 'g_floor', None) or gmin, g_stall - REGRIP_SQUEEZE)   # may squeeze past the calibrated end
        C.goto(dict(A.cmd), 0.5, gripper=g_lock); time.sleep(0.15)
        self.on_event("grasped", info)
        # 4. up, to waypoint 1, still held?
        wp1t = C.wps("wp1")[0]["pos"]
        info["g_squeezed"] = round(A.pos["gripper"])
        C.rise(wp1t)                                 # same lift in fast mode: 0.55 was tried 11-15 08:09 and saved no time
        info["g_risen"] = round(A.pos["gripper"]); self._snap(f"{tag}risen")
        C.goto(wp1t, 0.8 if self.fast else None, gripper=g_lock); time.sleep(0.3); info["g_wp1"] = round(A.pos["gripper"])
        info["g_lock"] = round(g_lock)
        # Held = the fingers still stand short of the squeeze command. An absolute width fails here: a 2x2 reads
        # 1872-1899 ticks depending on how it sits in the jaws, while empty jaws run on to the command / touch each other.
        if A.pos["gripper"] - g_lock < HELD_TICKS: return "dropped", info
        # The ticks alone are not enough: empty jaws that jam on each other read 1860-1865, a held brick 1867-1899
        # (two empty "holds" on 11-15 03:40/03:44). At WP1 the arm is clear of the plate, so look: if a brick of the
        # picked colour still lies on the plate, the gripper is empty.
        try:
            fr, left, _ = self.look()
            if spot: left = detect(fr, mm_per_px=self.calib.mm_per_px(spot[0], spot[1]), at=spot)[0]     # still on the re-clamp spot?
            info["seen_after"] = sorted({x["colour"] for x in left})
            info["seen_after_px"] = [[x["colour"], round(float(x["c"][0])), round(float(x["c"][1]))] for x in left]
            # The camera only decides when the ticks are ambiguous. With the jaws standing SURE_TICKS short of the
            # squeeze command the brick is in them, and the top camera then sometimes sees that held brick at WP1 and
            # calls it "still on the plate" (3 good grasps thrown away on 11-15 07:10-07:18, margins 51-54 ticks;
            # the empty jams of 03:40/03:44 had margins under 30).
            sure = A.pos["gripper"] - g_lock >= SURE_TICKS
            if not sure and b["colour"] in COLOURS and any(x["colour"] == b["colour"] for x in left): return "dropped", info
        except self.pol.PickFailed: pass
        self.g_lock = g_lock
        return "held", info

    def rg_quick(self):
        """Re-clamp shortcuts on? They follow the fast-moves switch (sorter cfg motion_fast -> Collector.fast)."""
        return bool(getattr(self.C, "fast", None))

    def rg_hover(self):
        """Height above the release point at which a re-clamp turns the jaws (also how far it rises after letting go)."""
        return RG_HOVER_FAST if self.rg_quick() else HOVER

    def grasp_world(self, centre, z, yaw, width, tag="rg_", seek=0.004, turn=0.0):
        """Grip a brick whose place is known from the ARM'S OWN POSE, not from the camera (re-clamp on the tape spot,
        sorter.regrip_vision): centre = brick centre (model x, y), z = TCP height it was let go at, yaw = MODEL jaw
        yaw wanted (fixed -> moving jaw), width = brick size along the jaws (m). Same hover, stepwise descent with
        sag correction, contact stop and close-until-stall as attempt(). Ends lifted back to the hover height.
        -> ("held" | "empty" | "landed" | "dropped" | "unreachable", info); the clamp command is self.g_lock."""
        A, C, kin = self.A, self.C, self.kin; gmin, gmax = A.fl["gripper"]; lo, hi = A.fl["wrist_roll"]
        wrap = lambda a_: abs((a_ + math.pi) % (2 * math.pi) - math.pi)
        quick = self.rg_quick(); hover = self.rg_hover()                               # fast moves: lower hover, 3 steps, quicker close
        ax = yaw - math.radians(self.calib.d.get("yaw_bias_deg", YAW_BIAS_DEG))        # where the jaws really point
        x = np.array([math.cos(ax), math.sin(ax), 0.0])
        p = np.array([centre[0], centre[1], z], np.float64) - (FIX_CLEAR + max(0.0, width - REF_W) / 2) * x
        info = {"xyz": [round(float(v), 4) for v in p], "yaw_deg": round(math.degrees(yaw), 1), "width_mm": round(width * 1000, 1),
                "quick": quick, "hover_mm": round(hover * 1000)}
        # turn = how far the jaws turn from where they are (rad): the IK is also started with the wrist roll already
        # turned that way round or the other, since a 180 deg turn has two solutions and only one is inside the limits
        q0 = np.array(self._q_now()[:5], np.float64); ok = False
        for dr in (0.0, turn, -turn):
            qs = q0.copy(); qs[4] += dr
            q_hover, e1 = kin.ik.solve(qs, p + [0, 0, hover], yaw, iters=80)
            q_low, e2 = kin.ik.solve(np.array(q_hover).copy(), p, yaw, iters=60)
            _, got, _ = kin.fk(q_hover); roll = kin.ticks(q_hover, 0)["wrist_roll"]
            if max(e1, e2) <= 0.004 and wrap(got - yaw) <= math.radians(12) and lo + 30 <= roll <= hi - 30: ok = True; break
            if not turn: break
        if not ok:
            info["why"] = f"err {max(e1, e2) * 1000:.1f} mm, yaw off {math.degrees(wrap(got - yaw)):.0f} deg, roll {roll:.0f}"
            return "unreachable", info
        g_open = min(gmax, max(TICKS_SHUT + TICKS_PER_M * (width + FIX_CLEAR + MOV_CLEAR), gmin + 200))
        self._goto(q_hover, g_open, 1.0); off = self._servo_in(q_hover, g_open); self._settle()
        self._snap(f"{tag}hover")
        q = q_hover; n = RG_STEPS_FAST if quick else 6; touched = None; z_low = p[2] - seek; steps = []; info["dz_steps_mm"] = steps
        for s in range(1, n + 1):
            zz = p[2] + hover + (z_low - p[2] - hover) * s / n
            q, _ = kin.ik.solve(q, np.array([p[0], p[1], zz]), yaw, iters=30)
            self._goto(q, g_open, 0.45, off); self._settle(0.5)
            tgt = kin.ticks(q, g_open)
            for n_ in off: off[n_] += max(-40.0, min(40.0, tgt[n_] - A.pos[n_]))
            self._goto(q, g_open, 0.3, off); self._settle(0.5)
            dz = kin.fk(self._q_now())[0][2] - zz; steps.append(round(dz * 1000, 1))
            if dz > 0.005: touched = (zz, dz); break
        info["touch_mm"] = None if touched is None else [round((touched[0] - p[2]) * 1000, 1), round(touched[1] * 1000, 1)]
        self._snap(f"{tag}low")
        if touched and touched[0] - p[2] > 0.014:                        # stopped high: standing on the brick
            self._goto(q_hover, g_open, 0.5, off); return "landed", info
        if touched:
            q, _ = kin.ik.solve(q, np.array([p[0], p[1], touched[0] + touched[1] + 0.002]), yaw, iters=30)
            self._goto(q, g_open, 0.3, off)
        tcp = kin.fk(self._q_now())[0]; info["tcp_err_mm"] = [round(v * 1000, 1) for v in (tcp - p)]
        C.goto(dict(A.cmd), RG_CLOSE_FAST if quick else 0.3, gripper=gmin); time.sleep(0.15)
        g_stall = A.pos["gripper"]; info["g_stall"] = round(g_stall)
        if g_stall - gmin < EMPTY_TICKS:
            self._snap(f"{tag}shut")
            C.goto(dict(A.cmd), 0.5, gripper=g_open); self._goto(q_hover, g_open, 0.5, off); return "empty", info
        from collect import REGRIP_SQUEEZE
        g_lock = max(getattr(A, "g_floor", None) or gmin, g_stall - REGRIP_SQUEEZE)
        C.goto(dict(A.cmd), 0.5, gripper=g_lock); time.sleep(0.15)
        self._goto(q_hover, g_lock, 0.4, off); time.sleep(0.25)         # straight up
        info["g_lock"] = round(g_lock); info["g_risen"] = round(A.pos["gripper"]); self._snap(f"{tag}risen")
        if A.pos["gripper"] - g_lock < HELD_TICKS: return "dropped", info
        self.g_lock = g_lock
        return "held", info

    def turn_yaw(self, u, v, margin=150, dz=0.0):
        """Jaw yaw for setting the held brick down at pixel (u, v) such that gripping it again there TURNED 90 DEG
        (+90 or -90) is in reach (re-clamp, sorter.regrip_vision). Checks what place() and the forced pick will
        solve: position error, jaw direction and the wrist-roll limits (with `margin` ticks to spare) for both.
        dz: the spot's height relative to the plate plane (m). -> (yaw to pass to place(), roll room in ticks) or None."""
        kin, A, C = self.kin, self.A, self.C
        seed = kin.rad(C.wps("wp1")[0]["pos"]); lo, hi = A.fl["wrist_roll"]; best = None
        p = self.calib.xyz(u, v); p[2] += self.z_rel + 0.003 + dz
        for k in range(8):
            yaw = -math.pi / 2 + k * math.pi / 8
            q, err, tilt, used, dyaw = kin.solve(seed, p, yaw)
            r1 = kin.ticks(q, 0)["wrist_roll"]
            if err > 0.004 or dyaw > math.radians(10) or not lo + 30 <= r1 <= hi - 30: continue
            for s_ in (1, -1):
                want = used + s_ * math.pi / 2
                q2, e2 = kin.ik.solve(np.asarray(seed, np.float64)[:5].copy(), np.asarray(p, np.float64), want, iters=80)
                _, got, _ = kin.fk(q2); r2 = kin.ticks(q2, 0)["wrist_roll"]
                room = min(r2 - lo, hi - r2, r1 - lo, hi - r1)
                if e2 > 0.004 or abs((got - want + math.pi) % (2 * math.pi) - math.pi) > math.radians(12) or room < margin: continue
                if best is None or room > best[1]: best = (yaw, float(room))
        return best

    def place(self, u, v, yaw, g_lock, dz=0.0, careful=False):
        """Set the held brick down so the grasp point is at pixel (u, v) of the top frame, jaws along yaw (either
        flip), let go and come straight up. -> {"xyz", "yaw"} = FK of the MEASURED joints at the release: with the
        brick's pixel in the next look this is one exact calibration pair (same IK, same sag handling as a pick)."""
        A, C, kin = self.A, self.C, self.kin
        gmin, gmax = A.fl["gripper"]
        p = self.calib.xyz(u, v); p[2] += self.z_rel + 0.003 + dz        # dz: spot lower / higher than the plate (m)
        q, err, tilt, yaw, dyaw = kin.solve(kin.rad(C.wps("wp1")[0]["pos"]), p, yaw)
        roll = kin.ticks(q, 0)["wrist_roll"]
        if err > 0.004 or not A.fl["wrist_roll"][0] + 30 <= roll <= A.fl["wrist_roll"][1] - 30:
            raise self.pol.PickFailed("the place pose is out of reach", retry=False)
        q_h, _ = kin.ik.solve(q, p + [0, 0, HOVER], yaw, iters=60)
        self._goto(q_h, g_lock, 1.0); off = self._servo_in(q_h, g_lock)
        z0 = kin.fk(self._q_now())[0][2] - (p[2] + HOVER); n = 5; qd = q_h
        stop_mm = 0.0
        for s_ in range(1, n + 1):
            z = p[2] + HOVER * (1 - s_ / n)
            qd, _ = kin.ik.solve(qd, np.array([p[0], p[1], z]), yaw, iters=30)
            if careful:
                # like the pick's descent: settle, correct the sag at every step, then test for contact. Without the
                # correction the arm lagging behind read as "standing on the table" and the brick was let go 39 mm
                # up (re-clamp spot, 2026-11-15 07:40), fell and turned 34 deg.
                self._goto(qd, g_lock, 0.4, off); self._settle(0.4)
                tgt = kin.ticks(qd, g_lock)
                for n_ in off: off[n_] += max(-40.0, min(40.0, tgt[n_] - A.pos[n_]))
                self._goto(qd, g_lock, 0.3, off); self._settle(0.4)
                stop_mm = (kin.fk(self._q_now())[0][2] - p[2]) * 1000
                if kin.fk(self._q_now())[0][2] - z > 0.005: break
                continue
            self._goto(qd, g_lock, 0.4, off); time.sleep(0.08)
            stop_mm = (kin.fk(self._q_now())[0][2] - p[2]) * 1000
            if kin.fk(self._q_now())[0][2] - z - z0 > 0.004: break       # the brick stands on the plate
        time.sleep(0.25)
        tcp, got, _ = kin.fk(self._q_now())
        g_open = min(gmax, max(A.pos["gripper"] + 330, gmin + 480))
        C.goto(dict(A.cmd), 0.4, gripper=g_open); time.sleep(0.2)
        x = np.array([math.cos(got), math.sin(got), 0.0])
        q2, _ = kin.ik.solve(self._q_now()[:5], tcp - 0.006 * x, got, iters=30)       # fixed jaw off the brick's face
        self._goto(q2, g_open, 0.3, off)
        q3, _ = kin.ik.solve(q2, tcp - 0.006 * x + [0, 0, HOVER + 0.02], got, iters=40)
        self._goto(q3, g_open, 0.7, off)
        return {"xyz": [float(v) for v in tcp], "yaw": float(got), "target_px": [float(u), float(v)],
                "above_target_mm": round(float(stop_mm), 1), "steps_done": s_}

    def run(self):
        A, C = self.A, self.C
        home = C.wps("home")[0]["pos"]; gmin, gmax = A.fl["gripper"]
        g_home = max(home["gripper"], gmin + 0.35 * (gmax - gmin)); reasons = []
        t_run = time.time()
        try:
            for k in range(self.retries + 1):
                self._check(); t0 = time.time()
                self.say(f"vision pick {k + 1}/{self.retries + 1}: robot to the start pose, looking at the plate")
                C.goto(home, gripper=g_home)
                res, info = self.attempt(k, tag=f"a{k + 1}_")
                self.attempts.append({"attempt": k + 1, "result": res, "t_s": round(time.time() - t0, 2), **info})
                if res == "held":
                    LAST_MISS.clear(); return self.g_lock, self.info(True, t_run)
                if res in ("empty", "dropped") and info.get("px"):
                    LAST_MISS.clear(); LAST_MISS.update(px=tuple(info["px"]), t=time.time(), colour=info.get("brick"))
                reasons.append({"landed": "the fingers came down on the brick", "empty": "closed on nothing",
                                "dropped": "lost it on the way to waypoint 1", "unreachable": "no reachable grasp"}[res])
                if k < self.retries:
                    self.say(f"vision pick {k + 1} failed ({reasons[-1]}): trying again", False)
                    C.goto(dict(A.cmd), 0.4, gripper=g_home)
            raise self.pol.PickFailed(f"vision pick failed {len(reasons)}x: {'; '.join(reasons)}", self.info(False, t_run))
        except self.pol.PickFailed as e:
            if not e.info: e.info = self.info(False, t_run, str(e))
            raise
        except self.pol.Stopped:
            raise
        except Exception as e:
            raise self.pol.PickFailed(f"vision pick stopped: {e}", self.info(False, t_run, str(e)), retry=False) from e
        finally:
            try: C.hold_here()
            except Exception: pass

    def info(self, ok, t0, error=None):
        return {"ok": ok, "error": error, "mode": "vision", "attempts": self.attempts, "n_attempts": len(self.attempts),
                "t_s": round(time.time() - t0, 2), "calib": {k: self.calib.d.get(k) for k in ("source", "n", "res_mm_med", "fitted")}}


PAIRS = HERE / "vpick_pairs.jsonl"     # one line per brick the robot set down and then saw: {"px", "xyz", ...}


def add_pair(px, xyz, **extra):
    with open(PAIRS, "a") as f: f.write(json.dumps({"px": list(px), "xyz": list(xyz), "t": time.strftime("%F %T"), **extra}) + "\n")


def refit(calib, min_robot=10):
    """Refit pixel -> table from the bricks the robot itself set down (exact pairs). The teleop pairs are kept only
    until there are min_robot of those: human grasps are off-centre by several mm and use other wrist angles.
    The grasp height plane is left as it is. -> (new Calib or None, text)"""
    try: rob = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    except Exception: rob = []
    if len(rob) < 4: return None, f"{len(rob)} robot pairs: too few to refit"
    demo = [x for x in calib.d.get("pairs", []) if x.get("src") == "demo"]
    use = rob if len(rob) >= min_robot else rob * 3 + demo                # few robot pairs: they count 3x
    d, res, inl = Calib.fit([x["px"] for x in use], [[*x["xyz"][:2], 0.0] for x in use])
    rr = np.array([np.linalg.norm(Calib(dict(d, z=calib.d["z"])).xy(*x["px"]) - np.array(x["xyz"][:2])) for x in rob]) * 1000
    new = dict(calib.d); new.update({"H": d["H"], "source": f"{len(rob)} robot placements" + ("" if len(rob) >= min_robot else f" + {len(demo)} teleop"),
                                     "n": len(use), "res_mm_med": float(np.median(rr)), "fitted": time.strftime("%F %T")})
    CALIB.write_text(json.dumps(new, indent=1))
    return Calib(new), f"refit on {new['source']}: robot pairs fit to {np.median(rr):.1f} mm median, {rr.max():.1f} mm max"


def grasp_frame(d, meta):
    """Index of the frame where the human's fingers stalled on the brick (the last such stall before the hand-off:
    earlier ones are the leader resting shut, or a first try that slipped)."""
    lo = meta["follower_limits"]["gripper"][0]; h = min(meta["handoff_frame"], len(d["t"]) - 1)
    t, g, c = d["t"], d["state"][:, 5], d["action"][:, 5]

    def stalled(i):
        w = (t > t[i] - 0.25) & (t <= t[i])
        return g[w].max() - g[w].min() <= 4 and g[i] - lo > 150 and c[i] < g[i] - 10
    i = h
    while i > 8 and not stalled(i): i -= 1
    if i <= 8: return None
    while i > 8 and stalled(i - 1): i -= 1
    return i


def demo_pairs(kin, episodes=ROOT / "episodes"):
    """(brick pixel in the first top frame, FK of the joints that held it) for every successful teleop pick."""
    rows = []
    for mp in sorted(Path(episodes).glob("*/*/meta.json")):
        m = json.loads(mp.read_text())
        if m.get("outcome") != "success" or "act_roi" not in m: continue
        d = np.load(mp.parent / "data.npz"); i = grasp_frame(d, m)
        img = cv2.imread(str(mp.parent / "top" / "000000.jpg"))
        if i is None or img is None: continue
        bricks, _ = detect(img, tuple(m["act_roi"]["top"]))
        mine = [b for b in bricks if b["colour"] == m.get("brick")]
        if len(mine) != 1: continue
        b = mine[0]; q = kin.rad(d["state"][i]); xyz, yaw, tilt = kin.fk(q)
        rows.append({"ep": f"{mp.parent.parent.name}/{mp.parent.name}", "brick": m["brick"], "px": b["c"], "angle": b["angle"],
                     "size": b["size"], "xyz": xyz.tolist(), "yaw": yaw, "tilt": tilt, "ticks": d["state"][i].tolist(),
                     "g_hold": float(d["state"][i][5])})
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-demos", action="store_true"); ap.add_argument("--show", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.show:
        img = cv2.imread(a.show); bricks, plate = detect(img)
        for b in bricks: print(b["colour"], [round(x) for x in b["c"]], f"angle {b['angle']:.0f}", [round(x) for x in b["size"]], f"{b['area_mm2']:.0f} mm2")
        cv2.imwrite(a.out or "/dev/null", draw(img, bricks, plate))
    if a.fit_demos:
        kin = Kin(); rows = demo_pairs(kin)
        d, res, inl = Calib.fit([r["px"] for r in rows], [r["xyz"] for r in rows])
        print(f"{len(rows)} demos, {inl.sum()} inliers, residual median {np.median(res[inl]):.1f} mm, p90 {np.percentile(res[inl], 90):.1f} mm")
        d.update({"source": "teleop demos", "n": int(inl.sum()), "res_mm_med": float(np.median(res[inl])), "fitted": time.strftime("%F %T"),
                  "joint_map": kin.joint_map, "pairs": [{"px": r["px"], "xyz": r["xyz"], "src": "demo"} for r, ok in zip(rows, inl) if ok]})
        CALIB.write_text(json.dumps(d, indent=1)); print("wrote", CALIB)
