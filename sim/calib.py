"""Fit the sim top camera (and optionally joint offsets) so the sim arm's FK lands on the real arm in the top image.

Keypoints: (episode frame, pixel u, v in the 640x480 policy crop, body point) read off real top frames.
The camera height comes from the measured scale (2x2 Duplo top 31.7 mm = 66 px at 1080p), so only
xc, yc, yaw (+ small tilts, + optional pan offset) are fitted.

    python calib.py            -> writes calibration.json and calib/overlay_top_*.jpg
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from scipy.optimize import least_squares

import station

EP = station.Path.home() / "junction26/episodes/20261114_032524_blue/ep0001_032524"
# frame, u, v, what   (tcp = grasp point between the jaw tips, i.e. the held brick's centre)
KEYPOINTS = [
    (515, 495, 182, "tcp"),
    (560, 498, 150, "tcp"),
    (631, 510, 135, "tcp"),
    (420, 500, 205, "tcp"),
    (0, 375, 45, "shoulder"),
    (515, 375, 45, "shoulder"),
]


def project(model, data, cam: str, p: np.ndarray) -> np.ndarray:
    c = model.camera(cam).id
    pos, rot = data.cam_xpos[c], data.cam_xmat[c].reshape(3, 3)
    pc = rot.T @ (p - pos)  # camera frame: x right, y up, looks along -z
    fy = 240 / np.tan(np.deg2rad(model.cam_fovy[c]) / 2)
    return np.array([320 + fy * pc[0] / -pc[2], 240 - fy * pc[1] / -pc[2]])


def main() -> None:
    model, data, cal = station.load()
    from physicalai_mujoco_plugin.conveyor_demo import ArmIK
    ik = ArmIK(model)
    st = np.load(EP / "data.npz")["state"]
    jid = [model.joint(j).qposadr[0] for j in station.JOINTS]
    cam = model.camera("top").id

    def apply(x):
        t = dict(cal["top"]); t.update(xc=x[0], yc=x[1], yaw_deg=x[2], tilt_x_deg=x[3], tilt_y_deg=x[4])
        c2 = {**cal, "top": t}
        pos, quat = station.top_camera_pose(c2)
        model.cam_pos[cam], model.cam_quat[cam] = pos, quat
        return c2

    def resid(x):
        c2 = apply(x); c2["joints"] = {**cal["joints"], "shoulder_pan": {"sign": 1.0, "offset": x[5]}}
        q = station.ticks_to_rad(st, c2); r = []
        for f, u, v, what in KEYPOINTS:
            data.qpos[jid] = q[f]; mujoco.mj_forward(model, data)
            p = ik.tcp(data) if what == "tcp" else data.xpos[model.body(what).id]
            r.extend(project(model, data, "top", p) - (u, v))
        return np.array(r + [x[3] * 20, x[4] * 20, x[5] * 20])  # strong priors: top-down, small pan offset

    t = cal["top"]
    x0 = [t["xc"], t["yc"], t["yaw_deg"], 0, 0, 0]
    sol = least_squares(resid, x0)
    r = resid(sol.x)[: 2 * len(KEYPOINTS)].reshape(-1, 2)
    print("fit xc %.3f yc %.3f yaw %.1f tilt %.1f %.1f pan_off %.2f deg" % (*sol.x[:5], np.rad2deg(sol.x[5])))
    print("residual px:", np.round(r, 1).tolist())
    c2 = apply(sol.x)
    out = {"top": c2["top"], "joints": {**cal["joints"], "shoulder_pan": {"sign": 1.0, "offset": float(sol.x[5])}},
           "fit": {"keypoints": KEYPOINTS, "rms_px": float(np.sqrt((r ** 2).mean()))}}
    # plates: invert pixel centres at table height through the fitted camera
    mujoco.mj_forward(model, data)
    pos, rot = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
    fy = 240 / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)

    def ray_to_table(u, v, z=0.0):
        d = rot @ np.array([(u - 320) / fy, -(v - 240) / fy, -1.0])
        s = (z - pos[2]) / d[2]
        return (pos + s * d)[:2]

    plates = {}
    for name, (u, v, rpx) in {"orange": (493, 185, 62), "yellow": (397, 277, 64), "blue": (204, 120, 60)}.items():
        c = ray_to_table(u, v); edge = ray_to_table(u + rpx, v)
        plates[name] = [round(float(c[0]), 4), round(float(c[1]), 4), round(float(np.linalg.norm(edge - c)), 4)]
    out["plates"] = plates
    print("plates:", plates)
    station.CALIB.write_text(json.dumps(out, indent=1))
    print("wrote", station.CALIB)


if __name__ == "__main__":
    main()
