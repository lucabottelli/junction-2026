"""Closed-loop check of the ACT pick runner (anomaly_capture/policy.py) in the MuJoCo twin, no hardware, no model.

    .venv/bin/python sim/act_loop.py [-n 10] [--mode chunk|async|ensemble] [--seed 0]

The scripted expert (expert.py) is wrapped as a "policy": before each pick it plans the whole pick -> WP1
trajectory on a copy of the scene (privileged state), then answers every inference with the next chunk of that
trajectory in Studio normalized units, indexed by sim time. Everything else is the real runtime code:
policy.ActPick (30 Hz loop, chunk queue + blending, targets through an arm-like object that rate-limits them at
50 Hz like arm.py, the collector's stall hand-off, TAIL_S, scripted rise + carry to WP1 via collect.Collector,
the holding() check). Physics decides whether the brick really is in the gripper, so the check is: does the
runner's verdict (hand-off + held at WP1) agree with the physics (brick lifted, between the fingers)?
Images: MuJoCo rendering needs EGL/OSMesa; the expert doesn't look at images, so the cameras are flat grey frames
(the observation path still runs: crop, JPEG round trip, RGB). Needs physicalai-mujoco-plugin (as generate.py).
"""
from __future__ import annotations

import argparse, json, sys, threading, time, types
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
AC = HERE.parent / "anomaly_capture"
sys.path.insert(0, str(AC))
try:
    import lerobot.motors.feetech.feetech  # noqa: F401
except Exception:                                   # not on the station PC: stub the servo bus (as sim_test.py)
    for name in ["lerobot", "lerobot.motors", "lerobot.motors.feetech", "lerobot.motors.feetech.feetech", "lerobot.motors.motors_bus"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["lerobot.motors.feetech.feetech"].FeetechMotorsBus = object
    sys.modules["lerobot.motors.motors_bus"].Motor = object
    sys.modules["lerobot.motors.motors_bus"].MotorNormMode = types.SimpleNamespace(RANGE_M100_100=0)

import arm, collect, export_act, policy as pol  # noqa: E402
import expert, station  # noqa: E402

ROUTINES = json.loads((AC / "routines.json").read_text())
expert._routine = lambda n: np.array([ROUTINES[n]["waypoints"][0]["pos"][j] for j in station.JOINTS], float)   # repo copy
DT = 0.02                                            # arm.py's servo loop period


class SimArm:
    """arm.Arm's interface (pos, cmd, goal, fl, io, stop, mode, vmax, goto, set_mode) on the MuJoCo twin."""
    def __init__(self, sc):
        self.sc, self.m, self.d, self.cal = sc, sc.model, sc.data, sc.cal
        self.fl = {n: tuple(v) for n, v in self.cal["follower_limits"].items()}
        self.io = threading.Lock(); self.stop = threading.Event(); self.mode = "hold"; self.running = False
        self.vmax = arm.VMAX * 0.5; self.t = 0.0
        self.sub = max(1, round(DT / self.m.opt.timestep))
        self.lo, self.hi = self.m.jnt_range[[self.m.joint(j).id for j in station.JOINTS]].T
        self.sync()

    def ticks(self): return station.rad_to_ticks(self.d.qpos[self.sc.qadr], self.cal)[0]

    def sync(self):
        p = self.ticks(); self.pos = {n: float(p[j]) for j, n in enumerate(station.JOINTS)}
        self.cmd = dict(self.pos); self.goal = dict(self.pos)

    def tick(self):
        if self.mode == "hold":
            for n in station.JOINTS:
                g = min(self.fl[n][1], max(self.fl[n][0], self.goal[n]))
                self.cmd[n] += max(-self.vmax * DT, min(self.vmax * DT, g - self.cmd[n]))
        q = station.ticks_to_rad(np.array([self.cmd[n] for n in station.JOINTS]), self.cal)[0]
        self.d.ctrl[self.sc.act] = np.clip(q, self.lo, self.hi)
        for _ in range(self.sub): mujoco.mj_step(self.m, self.d)
        p = self.ticks(); self.pos = {n: float(p[j]) for j, n in enumerate(station.JOINTS)}; self.t += DT

    def goto(self, tgt, speed):
        """arm.Arm.goto: minimum-jerk move from the current command, then settle."""
        vmax = max(50.0, arm.VMAX * speed); start = dict(self.cmd)
        T = max(0.25, 1.875 * max(abs(tgt[n] - start[n]) for n in station.JOINTS) / vmax); self.vmax = vmax * 1.2
        for k in range(1, int(np.ceil(T / DT)) + 1):
            if self.stop.is_set(): raise arm.Stopped()
            s = min(1.0, k * DT / T); mj = s ** 3 * (10 - 15 * s + 6 * s * s)
            self.goal = {n: start[n] + (tgt[n] - start[n]) * mj for n in station.JOINTS}; self.tick()
        for _ in range(15): self.tick()

    def set_mode(self, m, force=False): self.mode = m


class SimClock:
    def __init__(self, a): self.a = a
    def now(self): return self.a.t
    def sleep_until(self, t):
        while self.a.t < t - 1e-9: self.a.tick()


class ExpertPolicy:
    """expert.run_episode planned on a snapshot of the scene, replayed as chunks of normalized actions."""
    def __init__(self, sc, a, clock, target, rng, chunk=30):
        self.sc, self.a, self.clock, self.target, self.rng, self.n = sc, a, clock, target, rng, chunk
        self.cams, self.image_keys, self.takes_task = ["top", "wrist"], ["images.top", "images.wrist"], False
        self.export_dir, self.backend, self.device, self.runtime, self.load_ms = Path("expert"), "scripted", "CPU", "sim", 0.0
        self.label = "scripted expert"; self.plan = None; self.t0 = None

    def replan(self):
        m, d = self.sc.model, self.sc.data
        snap = mujoco.MjData(m); mujoco.mj_copyData(snap, m, d)
        r = expert.run_episode(self.sc, self.rng, self.target)
        mujoco.mj_copyData(d, m, snap)
        if not r.get("ok") and "action" not in r: raise RuntimeError(f"expert can't plan this pick: {r['why']}")
        ticks = station.rad_to_ticks(r["action"], self.sc.cal)
        self.plan = export_act.normalize(ticks, self.a.fl); self.expert_ok = bool(r.get("ok"))

    def reset(self): self.t0 = None

    def chunk(self, obs):
        assert obs["state"].shape == (1, 6) and obs["images.top"].shape == (1, 480, 640, 3)
        if self.t0 is None: self.t0 = self.clock.now(); self.replan()       # plan from where the arm starts now
        i = int(round((self.clock.now() - self.t0) * pol.FPS))
        idx = np.minimum(np.arange(i, i + self.n), len(self.plan) - 1)
        return self.plan[idx].astype(np.float32)

    def close(self): pass


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="chunk", choices=pol.MODES); ap.add_argument("--bricks", nargs="*")
    a = ap.parse_args()
    model, data, cal = station.load(); sc = expert.Scene(model, data, cal); A = SimArm(sc); clock = SimClock(A)
    grey = np.full((1080, 1920, 3), 128, np.uint8)
    S = types.SimpleNamespace(frames={"top": grey, "wrist": grey}, frame_t={}, live=["top", "wrist"], cams={})
    tmp = Path(__import__("tempfile").mkdtemp(prefix="act_loop_"))
    C = collect.Collector(S, A, types.SimpleNamespace(R={"routines": ROUTINES}), tmp, tmp); C.cfg["speed"] = 0.8
    rng = np.random.default_rng(a.seed); names = a.bricks or [b.name for b in station.BRICKS]; rows = []
    for k in range(a.n):
        target = str(rng.choice(names)); sc.reset(rng, 1, target); A.sync(); A.stop.clear()
        p = ExpertPolicy(sc, A, clock, target, np.random.default_rng(a.seed * 1000 + k))
        run = pol.ActPick(A, C, S, p, max_s=12, retries=0, mode=a.mode, clock=clock)
        bid = sc.bricks[target]; t0 = A.t
        try:
            g_lock, info = run.run(); verdict, why = "picked", ""
        except pol.PickFailed as e:
            info, verdict, why = e.info, "failed", str(e)
        bz = float(data.xpos[bid][2])
        sc.ik.data.qpos[sc.ik.qpos_adr] = data.qpos[sc.qadr[:5]]; mujoco.mj_kinematics(model, sc.ik.data)
        dist = float(np.linalg.norm(sc.ik.tcp(sc.ik.data) - data.xpos[bid]))
        truth = bz > 0.06 and dist < 0.05
        att = (info.get("attempts") or [{}])[-1]
        rows.append({"k": k, "brick": target, "runner": verdict, "physics_held": truth, "agree": (verdict == "picked") == truth,
                     "handoff": att.get("result"), "expert_ok": getattr(p, "expert_ok", None), "sim_s": round(A.t - t0, 1),
                     "z_mm": round(bz * 1000), "inferences": info.get("inferences"), "infer_ms_p50": info.get("infer_ms_p50")})
        print(json.dumps(rows[-1]) + (f"  {why}" if why else ""))
    n = len(rows)
    print(f"\n{sum(r['runner'] == 'picked' for r in rows)}/{n} picked by the runner, {sum(r['physics_held'] for r in rows)}/{n} held "
          f"per physics, verdict agrees with physics in {sum(r['agree'] for r in rows)}/{n} (mode {a.mode})")


if __name__ == "__main__":
    main()
