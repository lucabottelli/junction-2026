"""Offline comparison of SmolVLA exports on real recorded picks (same frames for every model).
For each episode: observation LEAD frames before the grasp (lowest TCP before the hand-off).
  aim     |predicted grasp (action LEAD -> FK) - actual grasp|  for the episode's own instruction
  instr   max shift of the predicted grasp when the instruction is changed to the other colours
  spread  std of the predicted grasp over SAMPLES repeated calls on the same observation (sampling = unsureness)
  jerk    mean step between consecutive actions inside one predicted chunk (mm per 33 ms tick)
    python compare_smolvla.py <export_dir> [<export_dir> ...]"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import cv2, mujoco, numpy as np
import evaluate, station, expert
from physicalai.inference import InferenceModel

LEAD, SAMPLES = 40, 4
model, data, cal = station.load(); sc = expert.Scene(model, data, cal)

def tcp(t):
    q = station.ticks_to_rad(t, cal)[0]; sc.ik.data.qpos[sc.ik.qpos_adr] = q[:5]; mujoco.mj_kinematics(model, sc.ik.data)
    return sc.ik.tcp(sc.ik.data).copy()

def episodes(root, sessions, k):
    out = []
    for s in sessions:
        for d in sorted((Path.home() / root / s).glob("ep*")):
            m = json.loads((d / "meta.json").read_text())
            if m.get("outcome") == "success": out.append(d)
            if len([o for o in out if o.parent.name == s]) >= k: break
    return out

SETS = {"new (tDataset2)": episodes("junction26/tDataset2", ["20261114_223316_pink", "20261114_224845_blue", "20261114_230057_green"], 4),
        "old (15:00 sessions)": episodes("junction26/episodes", ["20261114_150142_pink", "20261114_131414_blue", "20261114_142128_green"], 4)}
PREP = {}
for name, eps in SETS.items():
    for d in eps:
        m = json.loads((d / "meta.json").read_text()); st = np.load(d / "data.npz")["state"]; h = min(int(m["handoff_frame"]), len(st) - 1)
        g = min((tcp(st[f])[2], f) for f in range(max(0, h - 120), h + 1))[1]; f0 = max(0, g - LEAD)
        o = {"state": evaluate.normalize(st[f0], m["follower_limits"])[None]}
        for c in ("top", "wrist"): o[f"images.{c}"] = cv2.cvtColor(cv2.imread(str(d / c / f"{f0:06d}.jpg")), cv2.COLOR_BGR2RGB)[None]
        PREP[d] = (m, o, tcp(st[g])[:2])

for exp in sys.argv[1:]:
    exp = Path(exp).expanduser(); pol = InferenceModel(exp, backend="openvino", device="GPU")
    mdir = exp.parent.parent; vl = [round(float(r["val/loss"]), 3) for r in csv.DictReader(open(mdir / "version_0/metrics.csv")) if r.get("val/loss")]
    print(f"\n=== {mdir.name[:8]}  val/loss per epoch {vl}")
    def run(o, task, lim):
        o = {**o, "task": [task]}; pol.reset(); acts = np.array([np.asarray(pol.select_action(o)).reshape(-1)[:6] for _ in range(LEAD + 1)])
        P = np.array([tcp(evaluate.denormalize(a, lim)) for a in acts]); return P[-1][:2], np.linalg.norm(np.diff(P, axis=0), axis=1).mean()
    for name, eps in SETS.items():
        aim, instr, spread, jerk = [], [], [], []
        for d in eps:
            m, o, actual = PREP[d]; lim = m["follower_limits"]; own = f"pick up the {m['brick']} brick"
            samples = [run(o, own, lim) for _ in range(SAMPLES)]
            pts = np.array([s[0] for s in samples]); aim.append(np.linalg.norm(pts.mean(0) - actual)); spread.append(pts.std(0).mean())
            jerk.append(np.mean([s[1] for s in samples]))
            others = [run(o, f"pick up the {c} brick", lim)[0] for c in ("pink", "green", "blue") if c != m["brick"]]
            instr.append(max(np.linalg.norm(p - pts.mean(0)) for p in others))
        f = lambda v: f"{np.median(v) * 1000:5.1f}"
        print(f"  {name:22s} n={len(eps):2d}  aim {f(aim)} mm   instr-shift {f(instr)} mm   spread {f(spread)} mm   jerk {f(jerk)} mm/tick")
    pol.close() if hasattr(pol, "close") else None
