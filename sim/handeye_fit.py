"""Pixel (top-camera policy crop, 640x480) -> table xy (robot base frame, m), fitted from real teleop picks.

For every real episode: the target brick's pixel centre in the first top frame (sim/real_nearest.detect, inside
the orange plate) and the follower's joint ticks at the hand-off frame (fingers closed on the brick) -> FK
(station.ticks_to_rad + ArmIK TCP) -> where the gripper actually was. A homography maps pixel -> xy; its
leave-one-out residual says how well we can aim at a brick seen by the top camera.

    MUJOCO_GL=egl ~/physical-ai-studio/application/backend/.venv/bin/python handeye_fit.py  -> calib/handeye.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import expert
import real_nearest as rn
import station

ROI = [388, 113, 1677, 1080]  # top act_roi of every training session since 12:42 (03:25 used the full frame)

EP = Path.home() / "junction26/episodes"


def main() -> None:
    model, data, cal = station.load(); sc = expert.Scene(model, data, cal); cam = rn.TopCam()
    px, xy, cam_xy, names = [], [], [], []
    for md in sorted(EP.glob("*/*/meta.json")):
        m = json.loads(md.read_text()); d = md.parent
        if m.get("outcome") != "success" or not m.get("handoff_frame") or not (d / "data.npz").exists(): continue
        if (m.get("act_roi") or {}).get("top") != ROI: continue  # only the policy crop the training episodes use
        img = cv2.imread(str(d / "top" / "000000.jpg")); pm = rn.plate_mask(img)
        cands = [(u, v) for u, v, a in rn.detect(img).get(m.get("brick"), []) if pm[int(v), int(u)] and a < 4000]
        if len(cands) != 1: continue  # target not found, or ambiguous
        st = np.load(d / "data.npz")["state"]; h = min(int(m["handoff_frame"]), len(st) - 1)
        # the hand-off fires after the lift: take the LOWEST gripper point in the 4 s before it (= the grasp)
        best = None
        for f in range(max(0, h - 120), h + 1):
            q = station.ticks_to_rad(st[f], cal)[0]
            sc.ik.data.qpos[sc.ik.qpos_adr] = q[:5]; mujoco.mj_kinematics(model, sc.ik.data)
            t = sc.ik.tcp(sc.ik.data).copy()
            if best is None or t[2] < best[2]: best = t
        tcp = best
        px.append(cands[0]); xy.append(tcp[:2]); cam_xy.append(cam.to_table(*cands[0])); names.append(f"{d.parent.name}/{d.name}")
    px, xy, cam_xy = np.array(px, np.float64), np.array(xy), np.array(cam_xy)
    print(f"{len(px)} episodes with a single detected target")
    err_cam = np.linalg.norm(cam_xy - xy, axis=1)
    print(f"camera model (calibration.json) vs FK grasp point: median {np.median(err_cam)*1000:.0f} mm, p90 {np.percentile(err_cam,90)*1000:.0f} mm,"
          f" mean offset {((cam_xy - xy).mean(0)*1000).round(0)} mm")
    H, inl = cv2.findHomography(px, xy, cv2.RANSAC, 0.012)
    inl = inl.ravel().astype(bool)
    loo = []
    for i in np.flatnonzero(inl):
        k = inl.copy(); k[i] = False
        Hi, _ = cv2.findHomography(px[k], xy[k], 0)
        p = cv2.perspectiveTransform(px[i][None, None], Hi)[0, 0]; loo.append(np.linalg.norm(p - xy[i]))
    loo = np.array(loo)
    print(f"homography: {inl.sum()}/{len(px)} inliers (12 mm), leave-one-out error median {np.median(loo)*1000:.1f} mm,"
          f" p90 {np.percentile(loo,90)*1000:.1f} mm, max {loo.max()*1000:.1f} mm")
    out = {"H_pixel_to_xy": H.tolist(), "image": "top policy crop 640x480 (act_roi of the 1080p frame)", "n": int(len(px)),
           "inliers": int(inl.sum()), "loo_median_mm": round(float(np.median(loo)) * 1000, 1), "loo_p90_mm": round(float(np.percentile(loo, 90)) * 1000, 1),
           "camera_model_median_mm": round(float(np.median(err_cam)) * 1000, 1),
           "note": "xy = FK TCP of the follower at hand-off (sim/station.ticks_to_rad + ArmIK.tcp), robot base frame, metres. "
                   "Errors include how far off-centre the human grasped and FK/joint-map error."}
    (station.HERE / "calib" / "handeye.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "H_pixel_to_xy"}))


if __name__ == "__main__":
    main()
