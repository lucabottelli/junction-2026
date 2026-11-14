"""Simulated twin of the Junction 2026 station: SO-101 follower, orange input plate, Duplo bricks, top + wrist cameras.

Frame: robot base at the origin, table top at z = 0, +x = robot forward, +y = robot left (image right in the
top camera). Geometry comes from calibration.json (see calib.py); the arm is attached by
physicalai-mujoco-plugin's composer, so joint names/limits match the plugin's SO-101 twin.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = HERE / "calibration.json"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = JOINTS[:5]
TICKS_PER_RAD = 4096 / (2 * math.pi)

# Duplo: stud pitch 16 mm, a 2-wide brick is 31.7 mm, body height 19.2 mm, studs 9.4 mm wide x 4.5 mm high.
PITCH, BODY_H, STUD_R, STUD_H = 0.0159, 0.0192, 0.0047, 0.0045


RIDGE_H, RIDGE_SKIRT = 0.022, 0.006  # ridge roof: total height, vertical skirt below the two 45-ish deg slopes


@dataclass(frozen=True)
class Brick:
    name: str
    nx: int
    ny: int
    rgb: tuple[float, float, float]
    mass: float
    shape: str = "brick"  # "brick" (box + studs) or "ridge" (tent-shaped roof, no studs)


# The real set (user, 2026-11-14): red 2x4 ridge roof ("triangle"), green 2x3 brick, blue 2x2 brick.
BRICKS = (
    Brick("red", 4, 2, (0.78, 0.06, 0.06), 0.014, "ridge"),
    Brick("green", 3, 2, (0.45, 0.78, 0.18), 0.009),
    Brick("blue", 2, 2, (0.05, 0.42, 0.85), 0.006),
)
BRICK = {b.name: b for b in BRICKS}


def brick_vertices(b: Brick) -> np.ndarray:
    """Corner points of the brick in its body frame (origin = bounding-box centre of the body, studs excluded)."""
    sx, sy = b.nx * PITCH / 2 - 0.0002, b.ny * PITCH / 2 - 0.0002
    if b.shape == "ridge":
        h = RIDGE_H / 2
        base = [(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-h, -h + RIDGE_SKIRT)]
        return np.array(base + [(-sx, 0, h), (sx, 0, h)])
    h = BODY_H / 2
    box = [(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-h, h)]
    studs = [(x, y, h + STUD_H) for x in (-sx + 0.004, sx - 0.004) for y in (-sy + 0.004, sy - 0.004)]
    return np.array(box + studs)


DEFAULT_CALIB = {
    # top camera: optical centre above table point (xc, yc), height H, yaw about z, small tilts; fy at 1080p
    "top": {"xc": 0.18, "yc": -0.123, "H": 0.68, "yaw_deg": -25.3, "tilt_x_deg": 0.0, "tilt_y_deg": 0.0,
            "fy_1080": 1367.0},
    # wrist camera in the gripper frame: behind the jaws on the -y side, looking down the fingers
    "wrist": {"pos": [0.0, -0.045, -0.030], "tilt_deg": 12.0, "fovy": 78.0, "flip": False},
    "plates": {  # centre x, y, outer radius (m)
        "orange": [0.21, 0.10, 0.077],
        "yellow": [0.262, -0.057, 0.077],
        "blue": [-0.015, -0.189, 0.077],
    },
    # real ticks -> sim rad:  rad = sign * (ticks - mid) / TICKS_PER_RAD + offset,  mid = calibrated midpoint
    "joints": {j: {"sign": 1.0, "offset": 0.0} for j in ARM},
    # gripper: two (real ticks, sim rad) points: closed empty, and holding a 31.7 mm brick
    "gripper": {"ticks": [1979.0, 2630.9], "rad": [0.14, 1.14]},  # servo slope 2pi/4096, anchored on a held 31.7 mm brick
    "follower_limits": {"shoulder_pan": [745, 2814], "shoulder_lift": [825, 3222], "elbow_flex": [865, 3092],
                        "wrist_flex": [850, 3235], "wrist_roll": [0, 4090], "gripper": [1875, 3395]},
}


def load_calib() -> dict:
    cal = json.loads(json.dumps(DEFAULT_CALIB))
    if CALIB.exists():
        for k, v in json.loads(CALIB.read_text()).items():
            cal[k] = {**cal[k], **v} if isinstance(v, dict) and isinstance(cal.get(k), dict) else v
    return cal


# ---------------------------------------------------------------------------- units

def ticks_to_rad(ticks: np.ndarray, cal: dict) -> np.ndarray:
    """Real follower ticks (N, 6) -> sim joint angles (N, 6) in radians."""
    t = np.atleast_2d(np.asarray(ticks, dtype=np.float64)); out = np.empty_like(t)
    lim = cal["follower_limits"]
    for i, j in enumerate(ARM):
        mid = (lim[j][0] + lim[j][1]) / 2; c = cal["joints"][j]
        out[:, i] = c["sign"] * (t[:, i] - mid) / TICKS_PER_RAD + c["offset"]
    (t0, t1), (r0, r1) = cal["gripper"]["ticks"], cal["gripper"]["rad"]; out[:, 5] = r0 + (t[:, 5] - t0) * (r1 - r0) / (t1 - t0)
    return out


def rad_to_ticks(rad: np.ndarray, cal: dict) -> np.ndarray:
    """Sim joint angles (N, 6) in radians -> real follower ticks (N, 6)."""
    q = np.atleast_2d(np.asarray(rad, dtype=np.float64)); out = np.empty_like(q)
    lim = cal["follower_limits"]
    for i, j in enumerate(ARM):
        mid = (lim[j][0] + lim[j][1]) / 2; c = cal["joints"][j]
        out[:, i] = mid + (q[:, i] - c["offset"]) * TICKS_PER_RAD / c["sign"]
    (t0, t1), (r0, r1) = cal["gripper"]["ticks"], cal["gripper"]["rad"]
    out[:, 5] = t0 + (q[:, 5] - r0) * (t1 - t0) / (r1 - r0)
    return out


# ---------------------------------------------------------------------------- scene XML

def _wxyz(rot) -> str:
    q = rot.as_quat()
    return f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}"


def _plate_xml(name: str, x: float, y: float, r: float, rgb: tuple[float, float, float], segs: int = 20) -> str:
    """A shallow dish: flat base disc plus a ring of tilted rim boxes (bricks can lean against the rim)."""
    base_r, rim_h, rim_tilt = r * 0.80, 0.018, 0.55
    c = f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1"
    from scipy.spatial.transform import Rotation as R
    seg_len = 2 * math.pi * ((base_r + r) / 2) / segs * 0.56
    out = [f'<body name="plate_{name}" pos="{x:.4f} {y:.4f} 0">',
           f'  <geom name="plate_{name}_base" type="cylinder" size="{base_r:.4f} 0.002" pos="0 0 0.002" rgba="{c}" '
           f'friction="0.6 0.005 0.0001"/>']
    for k in range(segs):
        a = 2 * math.pi * k / segs; rr = (base_r + r) / 2
        out.append(f'  <geom type="box" size="0.0025 {seg_len:.4f} {rim_h / 2 / math.cos(rim_tilt):.4f}" '
                   f'pos="{rr * math.cos(a):.4f} {rr * math.sin(a):.4f} {rim_h / 2 + 0.002:.4f}" '
                   f'quat="{_wxyz(R.from_euler("ZY", [a, -rim_tilt]))}" rgba="{c}" friction="0.6 0.005 0.0001"/>')
    out.append('</body>')
    return "\n".join(out)


def _brick_xml(b: Brick, pos: tuple[float, float, float]) -> str:
    sx, sy = b.nx * PITCH / 2 - 0.0002, b.ny * PITCH / 2 - 0.0002
    c = f"{b.rgb[0]:.3f} {b.rgb[1]:.3f} {b.rgb[2]:.3f} 1"
    out = [f'<body name="brick_{b.name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">',
           f'  <freejoint name="brick_{b.name}:joint"/>']
    if b.shape == "ridge":
        out.append(f'  <geom name="brick_{b.name}" type="mesh" mesh="ridge_{b.name}" mass="{b.mass}" rgba="{c}" '
                   f'friction="0.9 0.01 0.0002" solref="0.004 1"/>')
        out.append('</body>')
        return "\n".join(out)
    out.append(f'  <geom name="brick_{b.name}" type="box" size="{sx:.4f} {sy:.4f} {BODY_H / 2:.4f}" mass="{b.mass}" '
               f'rgba="{c}" friction="0.9 0.01 0.0002" solref="0.004 1"/>')
    for i in range(b.nx):
        for j in range(b.ny):
            px, py = -sx + PITCH * (i + 0.5), -sy + PITCH * (j + 0.5)
            out.append(f'  <geom type="cylinder" size="{STUD_R} {STUD_H / 2}" pos="{px:.4f} {py:.4f} '
                       f'{BODY_H / 2 + STUD_H / 2:.4f}" mass="0.0003" rgba="{c}" friction="0.9 0.01 0.0002"/>')
    out.append('</body>')
    return "\n".join(out)


def _mesh_assets() -> str:
    out = []
    for b in BRICKS:
        if b.shape == "ridge":
            v = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in brick_vertices(b))
            out.append(f'<mesh name="ridge_{b.name}" vertex="{v}"/>')
    return "\n    ".join(out)


def top_camera_pose(cal: dict) -> tuple[list[float], list[float]]:
    """World position and quaternion (wxyz) of the top camera from the calibration."""
    from scipy.spatial.transform import Rotation as R
    t = cal["top"]
    # MuJoCo camera looks along its -z with +y up in the image. Image right = world +y rotated by yaw,
    # image down = world +x rotated by yaw (robot at the top of the image).
    yaw = math.radians(t["yaw_deg"])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # image +u in world
    down = np.array([math.cos(yaw), math.sin(yaw), 0.0])    # image +v in world
    cam_x, cam_y = right, -down
    cam_z = np.cross(cam_x, cam_y)  # points up (camera looks along -z = down)
    rot = R.from_matrix(np.column_stack([cam_x, cam_y, cam_z]))
    rot = R.from_euler("xy", [t["tilt_x_deg"], t["tilt_y_deg"]], degrees=True) * rot
    q = rot.as_quat()  # xyzw
    return [t["xc"], t["yc"], t["H"]], [q[3], q[0], q[1], q[2]]


def top_fovy(cal: dict) -> float:
    return math.degrees(2 * math.atan(540 / cal["top"]["fy_1080"]))


def build_xml(cal: dict, path: Path | None = None) -> Path:
    pos, quat = top_camera_pose(cal)
    plates = cal["plates"]
    ox, oy, _ = plates["orange"]
    bricks = "\n".join(_brick_xml(b, (ox + 0.3 + 0.07 * k, oy, BODY_H / 2)) for k, b in enumerate(BRICKS))
    xml = f"""<?xml version="1.0"?>
<mujoco model="junction_station">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast" timestep="0.002" cone="elliptic" iterations="20" ls_iterations="50" impratio="10"/>
  <visual>
    <headlight diffuse="0.25 0.25 0.25" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <quality shadowsize="4096" offsamples="4"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <default><geom condim="4"/></default>
  <asset>
    <texture name="wood" type="2d" builtin="gradient" rgb1="0.72 0.58 0.42" rgb2="0.60 0.46 0.32" width="512" height="512" random="0.04" mark="random" markrgb="0.50 0.38 0.26"/>
    <material name="table" texture="wood" texrepeat="3 3" texuniform="true" reflectance="0"/>
    {_mesh_assets()}
    <texture type="skybox" builtin="flat" rgb1="0.15 0.15 0.17" rgb2="0.1 0.1 0.1" width="64" height="64"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.2 0.0 1.6" dir="0 0 -1" directional="false" cutoff="60" diffuse="0.55 0.55 0.52" castshadow="true"/>
    <light name="fill" pos="-0.5 0.5 1.5" dir="0.3 -0.3 -1" directional="true" diffuse="0.25 0.25 0.27" castshadow="false"/>
    <geom name="table" type="box" size="0.8 0.7 0.02" pos="0.25 0 -0.02" material="table" friction="0.6 0.005 0.0001"/>
    {_plate_xml("orange", *plates["orange"], (0.93, 0.36, 0.08))}
    {_plate_xml("yellow", *plates["yellow"], (0.95, 0.80, 0.15))}
    {_plate_xml("blue", *plates["blue"], (0.62, 0.66, 0.92))}
    {bricks}
    <camera name="top" mode="fixed" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" fovy="{top_fovy(cal):.3f}"/>
    <frame name="robot_mount"/>
  </worldbody>
</mujoco>
"""
    path = path or HERE / "scene" / "station.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return path


def load(cal: dict | None = None):
    """Compose the station with the plugin's SO-101. Returns (model, data, cal)."""
    import logging
    import warnings

    import mujoco
    from physicalai_mujoco_plugin.compose import compose_scene
    from physicalai_mujoco_plugin.profiles.registry import get_profile

    logging.getLogger("physicalai_mujoco_plugin").setLevel(logging.WARNING)
    cal = cal or load_calib()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = compose_scene(build_xml(cal), get_profile("so101")).model
    data = mujoco.MjData(model)
    set_wrist_camera(model, data, cal)
    return model, data, cal


def set_wrist_camera(model, data, cal: dict) -> None:
    """Place the wrist camera (body camera_mount) at cal['wrist'], given in the gripper frame."""
    import mujoco
    from scipy.spatial.transform import Rotation as R
    w = cal["wrist"]; mujoco.mj_forward(model, data)
    g, cam = model.body("gripper").id, model.camera("wrist").id
    mb = model.cam_bodyid[cam]
    Rg, Rm = data.xmat[g].reshape(3, 3), data.xmat[mb].reshape(3, 3)
    t = math.radians(w["tilt_deg"])
    view = np.array([0.0, -math.sin(t), -math.cos(t)])        # mostly down the fingers, tilted away from them
    up = np.array([0.0, -math.cos(t), math.sin(t)])           # image up = away from the fingers (-y)
    x = np.cross(up, -view) if not w.get("flip") else -np.cross(up, -view)
    cam_g = np.column_stack([x, up if not w.get("flip") else up, -view])  # camera axes in the gripper frame
    rel = Rm.T @ Rg  # gripper -> mount
    model.cam_pos[cam] = Rm.T @ (data.xpos[g] + Rg @ np.array(w["pos"]) - data.xpos[mb])
    q = R.from_matrix(rel @ cam_g).as_quat()
    model.cam_quat[cam] = [q[3], q[0], q[1], q[2]]
    model.cam_fovy[cam] = w["fovy"]
    mujoco.mj_forward(model, data)
