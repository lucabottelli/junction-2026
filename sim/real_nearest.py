"""Which real teleop demos follow the "pick the brick closest to the robot" rule?

For each real episode: detect the bricks (blue, green, pink, red) by colour in the first top-camera frame,
project their pixel centres through the calibrated top camera onto the table, and compare their distances
to the robot base (origin). Episodes where the picked brick (meta "brick") was the nearest by >= MARGIN
follow the rule.

    MUJOCO_GL=egl ~/physical-ai-studio/application/backend/.venv/bin/python real_nearest.py [--write ../nearest_rule_exclude.txt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import station

EP = Path.home() / "junction26/episodes"
MARGIN = 0.015
PLATE = np.array(station.load_calib()["plates"]["orange"][:2]); PLATE_R = 0.08  # only blobs on the orange plate
Z_TOP = 0.03  # brick top surface height above the world origin's plane (plate + brick)
# HSV ranges (OpenCV H 0-180). Pink = salmon (lower saturation, bright); red = deep red (high saturation).
COLOURS = {
    "blue": [((95, 120, 80), (115, 255, 255))],
    "green": [((25, 55, 60), (60, 255, 255))],  # lime; pale when lit from the side
    "pink": [((168, 115, 140), (180, 175, 255))],  # salmon brick H~175 S~134; plate glare / tape: H 0-4, S 76-101
    "red": [((170, 176, 50), (180, 255, 255)), ((0, 176, 50), (3, 255, 200))],
}


class TopCam:
    def __init__(self):
        self.m, self.d, cal = station.load()
        mujoco.mj_forward(self.m, self.d)
        c = self.m.camera("top").id
        self.pos = self.d.cam_xpos[c].copy(); self.R = self.d.cam_xmat[c].reshape(3, 3).copy()
        self.f = 0.5 * 480 / np.tan(np.radians(self.m.cam_fovy[c]) / 2)

    def to_table(self, u: float, v: float, z: float = Z_TOP) -> np.ndarray:
        ray_cam = np.array([(u - 320) / self.f, -(v - 240) / self.f, -1.0])  # MuJoCo camera looks along -z, y up
        ray = self.R @ ray_cam
        t = (z - self.pos[2]) / ray[2]
        return (self.pos + t * ray)[:2]


def plate_mask(img: np.ndarray) -> np.ndarray:
    """The orange plate: the largest orange component, filled as its convex hull (bricks on it leave holes)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array((4, 150, 90)), np.array((18, 255, 255)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    if n > 1:
        k = 1 + int(np.argmax(st[1:, 4])); pts = np.column_stack(np.nonzero(lab == k))[:, ::-1]
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
        out = cv2.erode(out, np.ones((9, 9), np.uint8))  # not the rim
    return out


def detect(img: np.ndarray) -> dict[str, list[tuple[float, float, int]]]:
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    out = {}
    for name, ranges in COLOURS.items():
        m = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in ranges).clip(0, 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        out[name] = [(float(cen[i][0]), float(cen[i][1]), int(st[i][4])) for i in range(1, n) if st[i][4] > 250]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", type=Path); ap.add_argument("--sheet", type=Path)
    a = ap.parse_args()
    cam = TopCam(); rows = []; tiles = []
    for md in sorted(EP.glob("*/*/meta.json")):
        m = json.loads(md.read_text())
        if m.get("outcome") != "success": continue
        d = md.parent; name = f"{d.parent.name}/{d.name}"; tgt = m.get("brick")
        img = cv2.imread(str(d / "top" / "000000.jpg"))
        blobs = detect(img)
        pm = plate_mask(img)
        on_plate = lambda u, v: pm[int(v), int(u)] > 0
        dist = sorted((float(np.linalg.norm(cam.to_table(u, v))), c, (u, v)) for c, bl in blobs.items()
                      for u, v, area in bl if on_plate(u, v) and area < 4000)
        tgt_d = [x for x in dist if x[1] == tgt]
        if not tgt_d: verdict, lead = "target not found", None
        else:
            others = [x[0] for x in dist if x is not tgt_d[0]]
            lead = (min(others) - tgt_d[0][0]) if others else 1.0
            verdict = "nearest" if lead >= MARGIN else ("tie" if lead > -MARGIN else "not nearest")
        rows.append((name, tgt, verdict, lead, [(c, round(x * 100, 1)) for x, c, _ in dist]))
        if a.sheet:
            t = img.copy()
            for x, c, (u, v) in dist:
                cv2.circle(t, (int(u), int(v)), 9, (255, 255, 255) if c == tgt else (0, 0, 0), 2)
                cv2.putText(t, f"{x * 100:.0f}", (int(u) + 8, int(v)), 0, 0.45, (255, 255, 255), 1)
            t = cv2.resize(t[0:260, 330:640], (240, 200)); cv2.putText(t, f"{d.name[2:6]} {tgt} {verdict}", (3, 14), 0, 0.42, (255, 255, 255), 1)
            tiles.append(t)
    for r in rows: print(f"{r[0]:40s} {r[1]:6s} {r[2]:16s} lead={'' if r[3] is None else round(r[3] * 100, 1)}cm  {r[4]}")
    from collections import Counter
    print(Counter((r[1], r[2]) for r in rows))
    if a.sheet and tiles:
        while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(a.sheet), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    if a.write:
        a.write.write_text("# Real picks that don't follow the 'pick the brick closest to the robot' rule (sim/real_nearest.py)\n"
                           + "".join(f"{n}   # nearest rule: {v}\n" for n, _, v, *_ in rows if v != "nearest"))


if __name__ == "__main__":
    main()
