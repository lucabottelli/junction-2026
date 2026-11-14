"""Render the sim top/wrist view at a real episode frame's joint state and blend it with the real image.

    python overlay.py --frames 0 515 631 [--episode DIR]   -> calib/overlay_<cam>_<frame>.jpg
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np

import station

EP = Path.home() / "junction26/episodes/20261114_032524_blue/ep0001_032524"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, default=EP)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 515, 631])
    ap.add_argument("--cams", nargs="+", default=["top"])
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    model, data, cal = station.load()
    st = np.load(a.episode / "data.npz")["state"]
    q = station.ticks_to_rad(st, cal)
    jid = [model.joint(j).qposadr[0] for j in station.JOINTS]
    r = mujoco.Renderer(model, 480, 640)
    out = station.HERE / "calib"; out.mkdir(exist_ok=True)
    for f in a.frames:
        data.qpos[jid] = q[f]; mujoco.mj_forward(model, data)
        for cam in a.cams:
            r.update_scene(data, camera=cam); sim = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            real = cv2.imread(str(a.episode / cam / f"{f:06d}.jpg"))
            blend = cv2.addWeighted(real, 1 - a.alpha, sim, a.alpha, 0)
            cv2.imwrite(str(out / f"overlay_{cam}_{f}.jpg"), np.hstack([real, sim, blend]))
    r.close()
    print(json.dumps({"written": [f"overlay_{c}_{f}.jpg" for f in a.frames for c in a.cams]}))


if __name__ == "__main__":
    main()
