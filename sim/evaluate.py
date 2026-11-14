"""Closed-loop test of a trained policy export in the simulated station, with the same rendering as the training data.

    MUJOCO_GL=egl ~/physical-ai-studio/application/backend/.venv/bin/python evaluate.py \
        --model ~/.local/share/physicalai/models/<id>/exports/torch -n 50 [--device xpu] [--video 5]

The observation matches what Studio's model worker sends on the real arm:
    {"state": [1, 6] float32 normalized (-100..100, gripper 0..100), "images.<cam>": [1, 480, 640, 3] uint8 RGB, "task": [str]}
Seeds start at 900000, far from the generator's, so the layouts are unseen.
Metrics: grasped (target brick lifted > 4 cm), reached WP1 with it (success), dropped, timeouts.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np

import expert
import generate
import station


def normalize(ticks: np.ndarray, lim: dict) -> np.ndarray:
    out = np.empty(6, np.float32)
    for j, n in enumerate(station.JOINTS):
        lo, hi = lim[n]; fr = np.clip((ticks[j] - lo) / max(1, hi - lo), 0, 1)
        out[j] = fr * 100 if n == "gripper" else fr * 200 - 100
    return out


def denormalize(v: np.ndarray, lim: dict) -> np.ndarray:
    out = np.empty(6, np.float64)
    for j, n in enumerate(station.JOINTS):
        lo, hi = lim[n]; fr = v[j] / 100 if n == "gripper" else (v[j] + 100) / 200
        out[j] = lo + np.clip(fr, 0, 1) * (hi - lo)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--backend", default="auto"); ap.add_argument("--device", default="auto")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--video", type=int, default=3, help="write a video of the first N episodes")
    ap.add_argument("--bricks", nargs="*", help="only these target bricks")
    a = ap.parse_args()

    from physicalai.inference import InferenceModel
    policy = InferenceModel(a.model.expanduser(), backend=a.backend, device=a.device)
    feats = [f.name for f in policy.input_features]
    cams = [n.removeprefix("images.") for n in feats if n.startswith("images")]
    print("model inputs:", feats, "-> cameras", cams, "| chunk", policy.chunk_size)

    model, data, cal = station.load()
    sc = expert.Scene(model, data, cal)
    rz = generate.Randomizer(model, cal)
    rend = mujoco.Renderer(model, 480, 640)
    seg = mujoco.Renderer(model, 480, 640); seg.enable_segmentation_rendering()
    pools = generate.load_backgrounds(); table_id = model.geom("table").id
    lim = cal["follower_limits"]
    wp1_ticks = expert._routine("wp1")
    ik = sc.ik
    out_dir = station.HERE / "eval" / time.strftime("%Y%m%d_%H%M%S"); out_dir.mkdir(parents=True, exist_ok=True)
    stats, rows = collections.Counter(), []
    names = a.bricks or [b.name for b in station.BRICKS]

    for k in range(a.n):
        rng = np.random.default_rng(a.seed + k)
        target = str(rng.choice(names))
        vis = rz.apply(rng)
        sc.reset(rng, int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])), target)
        station.set_wrist_camera(model, data, {**cal, "wrist": vis["wrist"]})
        looks = {c: generate.sample_look(c, rng) for c in cams}
        bgs = {c: (pools[c][rng.integers(len(pools[c]))] if pools[c] and rng.uniform() < 0.85 else None) for c in cams}
        policy.reset()
        writer = None
        if k < a.video:
            writer = cv2.VideoWriter(str(out_dir / f"ep{k:03d}_{target}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     expert.FPS, (640 * len(cams), 480))
        bid = sc.bricks[target]; z0 = float(data.xpos[bid][2])
        grasped = reached = False; t_lift = None; max_z = z0
        for step in range(int(a.seconds * expert.FPS)):
            obs, imgs = {}, []
            for c in cams:
                rend.update_scene(data, camera=c)
                im = generate.degrade(rend.render(), c, looks[c], rng)
                if bgs[c] is not None:
                    seg.update_scene(data, camera=c)
                    im = generate.composite(im, seg.render(), c, bgs[c], table_id, rng)
                imgs.append(im)
                obs[f"images.{c}"] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[np.newaxis]
            ticks = station.rad_to_ticks(data.qpos[sc.qadr], cal)[0]
            obs["state"] = normalize(ticks, lim)[np.newaxis]
            if "task" in feats or True:
                obs["task"] = [f"pick up the {target} brick"]
            act = np.asarray(policy.select_action(obs)).reshape(-1)[:6]
            q = station.ticks_to_rad(denormalize(act, lim), cal)[0]
            lo, hi = model.jnt_range[[model.joint(j).id for j in station.JOINTS]].T
            data.ctrl[sc.act] = np.clip(q, lo, hi)
            for _ in range(expert.SUBSTEPS): mujoco.mj_step(model, data)
            if writer: writer.write(np.hstack(imgs))
            bz = float(data.xpos[bid][2]); max_z = max(max_z, bz)
            ik.data.qpos[ik.qpos_adr] = data.qpos[sc.qadr[:5]]; mujoco.mj_kinematics(model, ik.data)
            held = np.linalg.norm(ik.tcp(ik.data) - data.xpos[bid]) < 0.05
            if bz > z0 + 0.04 and held: grasped = True; t_lift = t_lift or step / expert.FPS
            near_wp1 = np.abs(ticks[:5] - wp1_ticks[:5]).max() < 150  # ~13 deg per joint
            if grasped and held and near_wp1 and bz > 0.06:
                reached = True; break
        if writer: writer.release()
        outcome = "success" if reached else ("dropped" if grasped else ("touched" if max_z > z0 + 0.01 else "no_grasp"))
        stats[outcome] += 1
        rows.append({"seed": a.seed + k, "brick": target, "outcome": outcome, "t_lift": t_lift,
                     "steps": step + 1})
        print(f"{k:3d} {target:6s} {outcome:9s} lift@{t_lift}  ({step + 1} steps)")
    rend.close(); seg.close(); policy.close()
    n = sum(stats.values())
    summary = {"model": str(a.model), "n": n, **{k: v for k, v in stats.items()},
               "success_rate": stats["success"] / n, "grasp_rate": (stats["success"] + stats["dropped"]) / n}
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary), "->", out_dir)


if __name__ == "__main__":
    main()
