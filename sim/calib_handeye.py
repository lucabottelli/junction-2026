"""Refit the sim top camera on real grasps (replaces the 6 hand-clicked keypoints of calib.py).

Correspondences: target brick pixel in the first top frame <-> follower FK grasp point (lowest TCP before the
hand-off), from handeye_fit.py's inliers. Fits camera xc, yc, yaw, tilts and height (priors: near top-down,
H ~ 0.68 m from the measured scale). Plates are re-derived from their pixels. Old fit: calib/calibration_v1_6keypoints.json.

    MUJOCO_GL=egl ~/physical-ai-studio/application/backend/.venv/bin/python calib_handeye.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

import calib
import expert
import real_nearest as rn
import station

ROI = [388, 113, 1677, 1080]  # top act_roi of every training session since 12:42 (03:25 used the full frame)


def correspondences(model, cal):
    sc = expert.Scene(model, mujoco.MjData(model), cal); px, pts = [], []
    for md in sorted(rn.EP.glob("*/*/meta.json")):
        m = json.loads(md.read_text()); d = md.parent
        if m.get("outcome") != "success" or not m.get("handoff_frame") or not (d / "data.npz").exists(): continue
        if (m.get("act_roi") or {}).get("top") != ROI: continue  # only the policy crop the training episodes use
        img = cv2.imread(str(d / "top" / "000000.jpg")); pm = rn.plate_mask(img)
        c = [(u, v) for u, v, a in rn.detect(img).get(m.get("brick"), []) if pm[int(v), int(u)] and a < 4000]
        if len(c) != 1: continue
        st = np.load(d / "data.npz")["state"]; h = min(int(m["handoff_frame"]), len(st) - 1); best = None
        for f in range(max(0, h - 120), h + 1):
            q = station.ticks_to_rad(st[f], cal)[0]; sc.ik.data.qpos[sc.ik.qpos_adr] = q[:5]
            mujoco.mj_kinematics(model, sc.ik.data); t = sc.ik.tcp(sc.ik.data).copy()
            if best is None or t[2] < best[2]: best = t
        px.append(c[0]); pts.append(best)
    return np.array(px), np.array(pts)


def main() -> None:
    model, data, cal = station.load()
    px, pts = correspondences(model, cal)
    H, inl = cv2.findHomography(px, pts[:, :2], cv2.RANSAC, 0.012); inl = inl.ravel().astype(bool)
    px, pts = px[inl], pts[inl]
    pts[:, 2] += 0.01  # the pixel is the brick's TOP face centre; the grasp point is ~45% up the brick
    # + the 6 hand-clicked keypoints of calib.py (03:25 episode, centre-4:3 view of the full frame), converted to the ROI view:
    #   full-res x = 240 + u * 2.25, y = v * 2.25 ; ROI u' = (x - 388) * 640 / 1289, v' = (y - 113) * 480 / 967
    st0 = np.load(calib.EP / "data.npz")["state"]; q0 = station.ticks_to_rad(st0, cal); d0 = mujoco.MjData(model)
    sc0 = expert.Scene(model, d0, cal); kp_px, kp_pts = [], []
    for f, u, v, what in calib.KEYPOINTS:
        X, Y = 240 + u * 2.25, v * 2.25
        kp_px.append(((X - ROI[0]) * 640 / (ROI[2] - ROI[0]), (Y - ROI[1]) * 480 / (ROI[3] - ROI[1])))
        d0.qpos[sc0.qadr] = q0[f]; mujoco.mj_forward(model, d0)
        kp_pts.append(sc0.ik.tcp(d0).copy() if what == "tcp" else d0.xpos[model.body(what).id].copy())
    n_grasp = len(px)
    if os.environ.get("USE_KEYPOINTS"):  # off: they disagree with the grasps by 30-80 px (joint map wrong away from the plate)
        px = np.vstack([px, kp_px]); pts = np.vstack([pts, kp_pts])
    cam = model.camera("top").id

    def apply(x):
        t = dict(cal["top"]); t.update(xc=x[0], yc=x[1], yaw_deg=x[2], tilt_x_deg=x[3], tilt_y_deg=x[4], H=x[5])
        c2 = {**cal, "top": t}; pos, quat = station.top_camera_pose(c2)
        model.cam_pos[cam], model.cam_quat[cam] = pos, quat
        model.cam_fovy[cam] = station.top_fovy(c2)
        mujoco.mj_forward(model, data); return c2

    def resid(x):
        apply(x)
        r = np.concatenate([calib.project(model, data, "top", p) - uv for p, uv in zip(pts, px)])
        return np.concatenate([r, [x[3] * 20, x[4] * 20, (x[5] - 0.68) * 400]])  # strong priors: near top-down, H from scale

    t = cal["top"]; x0 = [t["xc"], t["yc"], t["yaw_deg"], t["tilt_x_deg"], t["tilt_y_deg"], t["H"]]
    r0 = resid(np.array(x0))[: 2 * len(px)].reshape(-1, 2)
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=5.0)
    r = resid(sol.x)[: 2 * len(px)].reshape(-1, 2)
    print(f"{len(px)} grasps. before: rms {np.sqrt((r0**2).sum(1).mean()):.1f} px; after: rms {np.sqrt((r**2).sum(1).mean()):.1f} px,"
          f" median {np.median(np.linalg.norm(r, axis=1)):.1f} px")
    print("camera: xc %.3f yc %.3f yaw %.1f tilt %.1f %.1f H %.3f" % tuple(sol.x))
    c2 = apply(sol.x)
    pos, rot = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3); fy = 240 / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)

    def ray_to_table(u, v, z=0.0):
        d = rot @ np.array([(u - 320) / fy, -(v - 240) / fy, -1.0]); s = (z - pos[2]) / d[2]; return (pos + s * d)[:2]

    # table-plane error of the fit, in mm (pixel -> table at the grasp height vs FK grasp point)
    e = np.array([np.linalg.norm(ray_to_table(u, v, p[2]) - p[:2]) for (u, v), p in zip(px[:n_grasp], pts[:n_grasp])])
    print("keypoint residuals px:", np.round(r[n_grasp:], 1).tolist())
    print(f"table-plane error: median {np.median(e)*1000:.1f} mm, p90 {np.percentile(e, 90)*1000:.1f} mm")
    plates = {}
    for name, (u, v, rpx) in {"orange": (540, 148, 90), "yellow": (395, 268, 97), "blue": (110, 87, 90)}.items():  # measured in the ROI view, 19:55
        c = ray_to_table(u, v); edge = ray_to_table(u + rpx, v)
        plates[name] = [round(float(c[0]), 4), round(float(c[1]), 4), round(float(np.linalg.norm(edge - c)), 4)]
    print("plates:", plates, " (was", cal["plates"], ")")
    full = json.loads(station.CALIB.read_text())
    full["top"] = c2["top"]; full["plates"] = plates
    full["fit"] = {"method": "calib_handeye.py: real grasps (brick pixel <-> FK grasp point)", "n": int(len(px)),
                   "rms_px": float(np.sqrt((r ** 2).sum(1).mean())), "table_err_median_mm": round(float(np.median(e)) * 1000, 1),
                   "table_err_p90_mm": round(float(np.percentile(e, 90)) * 1000, 1)}
    station.CALIB.write_text(json.dumps(full, indent=1)); print("wrote", station.CALIB)


if __name__ == "__main__":
    main()
