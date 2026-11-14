"""Scripted expert for the station: pick a brick lying anywhere on the orange plate and carry it to WP1.

Uses privileged state (brick poses) and the plugin's SO-101 IK (conveyor_demo.ArmIK). Runs at the real
recording rate (30 Hz control, 20 physics substeps of 1/600 s) and returns per-frame state + commanded action
in sim radians. Variation (speeds, hover height, pauses, approach error that gets corrected) makes the
demos less robotic, so ACT doesn't overfit one exact trajectory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import mujoco
import numpy as np
from physicalai_mujoco_plugin.conveyor_demo import ArmIK

import station

FPS, SUBSTEPS = 30, 20
GRIP_CLOSED = -0.15
GAP_PER_RAD = 0.08  # jaw gap (m) per rad of gripper opening above GRIP_CLOSED (plugin: 0.5 rad ~ 5.2 cm)
PARK = np.array([1.2, 0.0, 0.02])


def _routine(name: str) -> np.ndarray:
    r = json.loads((station.Path.home() / "junction26/anomaly_capture/routines.json").read_text())
    p = r[name]["waypoints"][0]["pos"]
    return np.array([p[j] for j in station.JOINTS], dtype=np.float64)


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


@dataclass
class Scene:
    model: object
    data: object
    cal: dict
    ik: ArmIK = field(init=False)

    def __post_init__(self):
        m = self.model
        m.opt.timestep = 1.0 / (FPS * SUBSTEPS)
        self.ik = ArmIK(m)
        self.qadr = np.array([m.joint(j).qposadr[0] for j in station.JOINTS])
        self.act = np.array([m.actuator(j).id for j in station.JOINTS])
        self.bricks = {b.name: m.body(f"brick_{b.name}").id for b in station.BRICKS}
        self.brick_jnt = {b.name: m.joint(f"brick_{b.name}:joint").qposadr[0] for b in station.BRICKS}
        self.brick_geom = {b.name: m.geom(f"brick_{b.name}").id for b in station.BRICKS}
        self.home = station.ticks_to_rad(_routine("home"), self.cal)[0]
        self.wp1 = station.ticks_to_rad(_routine("wp1"), self.cal)[0]
        lo, hi = m.jnt_range[m.joint("gripper").id]
        self.home[5] = min(max(self.home[5], lo), hi)

    # ------------------------------------------------------------------ reset
    def reset(self, rng: np.random.Generator, n_bricks: int, target: str) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        names = [target] + [n for n in rng.permutation(list(self.bricks)) if n != target][: n_bricks - 1]
        px, py, pr = self.cal["plates"]["orange"]
        placed = []
        for name in self.bricks:
            a = self.brick_jnt[name]
            if name not in names:
                d.qpos[a:a + 3] = PARK + [0, 0.1 * len(placed), 0]; d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
                continue
            for _ in range(100):
                r, t = pr * 0.70 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
                xy = np.array([px + r * np.cos(t), py + r * np.sin(t)])
                if all(np.linalg.norm(xy - q) > 0.068 for q in placed): break
            placed.append(xy)
            face = rng.choice(6, p=[0.45, 0.1, 0.1125, 0.1125, 0.1125, 0.1125])  # mostly studs up
            tilt = {0: (0, 0), 1: (np.pi, 0), 2: (np.pi / 2, 0), 3: (-np.pi / 2, 0), 4: (0, np.pi / 2), 5: (0, -np.pi / 2)}[face]
            q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([tilt[0], tilt[1], rng.uniform(-np.pi, np.pi)]), "xyz")
            qz = np.zeros(4); mujoco.mju_euler2Quat(qz, np.array([0, 0, rng.uniform(-np.pi, np.pi)]), "xyz")
            qq = np.zeros(4); mujoco.mju_mulQuat(qq, qz, q)
            d.qpos[a:a + 3] = [xy[0], xy[1], 0.05]; d.qpos[a + 3:a + 7] = qq
        d.qpos[self.qadr] = self.home + np.r_[rng.normal(0, 0.02, 5), 0]
        d.ctrl[self.act] = d.qpos[self.qadr]
        for _ in range(int(0.6 / m.opt.timestep)):  # let the bricks fall and settle
            mujoco.mj_step(m, d)

    # ------------------------------------------------------------------ grasp choice
    def grasp_axis(self, name: str) -> tuple[float, float, np.ndarray]:
        """(closing yaw, width across the jaws, grasp centre) for the brick as it lies now.

        Works for any shape: the brick's corner points are projected on horizontal closing directions every
        3 deg and the narrowest one wins (ties: the one closest to the brick's own axes, which grips flat faces).
        """
        d = self.data; bid = self.bricks[name]
        rot, pos = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        pts = pos + station.brick_vertices(station.BRICK[name]) @ rot.T
        best = None
        for deg in range(0, 180, 3):
            yaw = np.radians(deg); u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            pr = pts @ u; w = float(pr.max() - pr.min())
            if best is None or w < best[1] - 1e-4: best = (yaw, w, pr)
        yaw, width, pr = best
        u = np.array([np.cos(yaw), np.sin(yaw), 0.0]); v = np.array([-u[1], u[0], 0.0])
        pv = pts @ v
        centre = u * (pr.max() + pr.min()) / 2 + v * (pv.max() + pv.min()) / 2
        zlo, zhi = pts[:, 2].min(), pts[:, 2].max()
        centre[2] = zlo + 0.45 * (zhi - zlo)  # a bit below the middle: sloped/studded tops push the jaws up
        return float(yaw), width, centre

    def solve(self, q0, target, yaw):
        """Best of the two jaw flips: reachable, upright, smallest wrist turn."""
        best = None
        for flip in (0.0, np.pi):
            q, err = self.ik.solve(q0, target, yaw + flip, iters=60)
            self.ik.data.qpos[self.ik.qpos_adr] = q; mujoco.mj_kinematics(self.model, self.ik.data)
            tilt = self.ik.tilt(self.ik.data)
            score = (err > 0.004) * 10 + (tilt > np.radians(12)) * 5 + abs(q[4] - q0[4]) * 0.3 + tilt
            if best is None or score < best[0]: best = (score, q, err, tilt, yaw + flip)
        return best[1:]


def run_episode(sc: Scene, rng: np.random.Generator, target: str, on_frame=None) -> dict:
    """One pick -> WP1. on_frame(i) is called after each 30 Hz control tick (render there)."""
    m, d, ik = sc.model, sc.data, sc.ik
    states, actions, phases = [], [], []
    v = dict(speed=rng.uniform(0.7, 1.3), hover=rng.uniform(0.06, 0.10), pause=rng.uniform(0.1, 0.8),
             approach_err=rng.normal(0, 0.008, 2), close_s=rng.uniform(0.35, 0.6), dz=rng.uniform(0.0, 0.006))
    cmd = d.qpos[sc.qadr].copy()

    def tick(c, phase):
        nonlocal cmd
        cmd = np.asarray(c, dtype=np.float64).copy()
        d.ctrl[sc.act] = cmd
        for _ in range(SUBSTEPS): mujoco.mj_step(m, d)
        states.append(d.qpos[sc.qadr].copy()); actions.append(cmd.copy()); phases.append(phase)
        if on_frame: on_frame(len(states) - 1)

    def move_joints(q_to, secs, phase):
        q_from = cmd.copy(); n = max(2, int(secs * FPS / v["speed"]))
        for i in range(1, n + 1): tick(q_from + (q_to - q_from) * min_jerk(i / n), phase)

    def move_cart(p_from, p_to, yaw, grip, secs, phase):
        n = max(2, int(secs * FPS / v["speed"])); q = cmd[:5].copy()
        for i in range(1, n + 1):
            p = p_from + (p_to - p_from) * min_jerk(i / n)
            q, _ = ik.solve(q, p, yaw, iters=10)
            tick(np.r_[q, grip], phase)

    for _ in range(int(v["pause"] * FPS)): tick(cmd, "start")
    yaw, width, centre = sc.grasp_axis(target)
    if width > 0.055: return {"ok": False, "why": f"too wide ({width * 1000:.0f} mm)"}
    grip_open = GRIP_CLOSED + (width + rng.uniform(0.015, 0.03)) / GAP_PER_RAD
    grasp = centre + [0, 0, v["dz"]]
    grasp[2] = max(grasp[2], 0.012)
    hover = grasp + np.r_[v["approach_err"], v["hover"]]
    q_hover, err, tilt, yaw = sc.solve(sc.wp1[:5], hover, yaw)
    if err > 0.006: return {"ok": False, "why": f"unreachable ({err * 1000:.0f} mm)"}
    # 1. home -> above the brick (joint space, opening the gripper on the way)
    move_joints(np.r_[q_hover, grip_open], rng.uniform(1.4, 2.2), "approach")
    # 2. correct the approach error, then descend
    ik.data.qpos[ik.qpos_adr] = cmd[:5]; mujoco.mj_kinematics(m, ik.data); p_now = ik.tcp(ik.data).copy()
    above = grasp + [0, 0, v["hover"]]
    move_cart(p_now, above, yaw, grip_open, rng.uniform(0.3, 0.6), "align")
    move_cart(above, grasp, yaw, grip_open, rng.uniform(0.7, 1.1), "descend")
    # 3. close (command past the brick so the servo squeezes)
    n = int(v["close_s"] * FPS)
    for i in range(1, n + 1): tick(np.r_[cmd[:5], grip_open + (GRIP_CLOSED - grip_open) * min_jerk(i / n)], "close")
    for _ in range(4): tick(cmd, "close")
    ik.data.qpos[ik.qpos_adr] = d.qpos[sc.qadr[:5]]; mujoco.mj_kinematics(m, ik.data)
    diag = {"g_after_close": round(float(d.qpos[sc.qadr[5]]), 3),
            "tcp_err_mm": np.round((ik.tcp(ik.data) - grasp) * 1000, 1).tolist(),
            "brick_vs_grasp_mm": np.round((d.xpos[sc.bricks[target]] - grasp) * 1000, 1).tolist(),
            "tilt_deg": round(float(np.degrees(ik.tilt(ik.data))), 1), "width_mm": round(width * 1000, 1)}
    # 4. lift straight up, then joint-space carry to WP1 keeping the squeeze
    move_cart(grasp, grasp + [0, 0, 0.08], yaw, GRIP_CLOSED, rng.uniform(0.6, 0.9), "lift")
    move_joints(np.r_[sc.wp1[:5], GRIP_CLOSED], rng.uniform(1.0, 1.6), "carry")
    for _ in range(int(0.3 * FPS)): tick(cmd, "wp1")
    bz = d.xpos[sc.bricks[target]][2]
    ik.data.qpos[ik.qpos_adr] = d.qpos[sc.qadr[:5]]; mujoco.mj_kinematics(m, ik.data)
    dist = float(np.linalg.norm(ik.tcp(ik.data) - d.xpos[sc.bricks[target]]))
    ok = bz > 0.06 and dist < 0.05
    return {"ok": ok, "why": "" if ok else f"dropped (z={bz:.3f}, dist={dist:.3f}) {diag}", "diag": diag, "state": np.array(states),
            "action": np.array(actions), "phase": phases, "width": width, "grip_open": grip_open,
            "gripper_hold": float(np.mean(np.array(states)[-8:, 5]))}


if __name__ == "__main__":
    import argparse
    import collections
    import time
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model, data, cal = station.load(); sc = Scene(model, data, cal)
    rng = np.random.default_rng(a.seed); res = collections.Counter(); t0 = time.time(); holds = []
    for i in range(a.n):
        tgt = rng.choice([b.name for b in station.BRICKS]); sc.reset(rng, int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])), tgt)
        r = run_episode(sc, rng, tgt)
        res[(tgt, r["ok"])] += 1
        if r["ok"]: holds.append((tgt, round(r["width"] * 1000), round(r["gripper_hold"], 3), len(r["state"])))
        else: print(i, tgt, r["why"])
    n_ok = sum(v for (t, ok), v in res.items() if ok)
    print(f"success {n_ok}/{a.n} = {100 * n_ok / a.n:.0f}%  ({time.time() - t0:.1f} s)", dict(res))
    print("width mm / gripper rad when holding / frames:", holds[:12])
