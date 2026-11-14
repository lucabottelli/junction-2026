"""Open-loop probe: does the policy aim at the brick?  For training episodes (real + sim), feed the recorded
observation from LEAD frames before the grasp and compare the policy's predicted gripper position at the grasp
(action chunk index LEAD -> FK) with where the recorded episode actually grasped (FK of the lowest TCP).
Also reports how much the predictions vary between episodes (a policy that ignores the image predicts ~one spot)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, mujoco, numpy as np
import station, expert, evaluate

LEAD = 40
model_dir = Path(sys.argv[1]).expanduser()
from physicalai.inference import InferenceModel
pol = InferenceModel(model_dir, backend="openvino", device="CPU")
model, data, cal = station.load(); sc = expert.Scene(model, data, cal)

def tcp(ticks):
    q = station.ticks_to_rad(ticks, cal)[0]; sc.ik.data.qpos[sc.ik.qpos_adr] = q[:5]; mujoco.mj_kinematics(model, sc.ik.data)
    return sc.ik.tcp(sc.ik.data).copy()

def probe(d: Path):
    m = json.loads((d / "meta.json").read_text()); z = np.load(d / "data.npz"); st, lim = z["state"], m["follower_limits"]
    h = min(int(m.get("handoff_frame") or len(st) - 1), len(st) - 1)
    lows = [(tcp(st[f])[2], f) for f in range(max(0, h - 120), h + 1)]; g = min(lows)[1]
    f0 = max(0, g - LEAD); obs = {"state": evaluate.normalize(st[f0], lim)[None]}
    for c in ("top", "wrist"):
        obs[f"images.{c}"] = cv2.cvtColor(cv2.imread(str(d / c / f"{f0:06d}.jpg")), cv2.COLOR_BGR2RGB)[None]
    obs["task"] = [m.get("task", "pick up the brick")]
    pol.reset(); chunk = []
    for k in range(LEAD + 1):   # chunk_size=1 export: step through the internal queue
        chunk.append(np.asarray(pol.select_action(obs)).reshape(-1)[:6])
    pred = evaluate.denormalize(chunk[-1], lim)
    return tcp(st[g])[:2], tcp(pred)[:2]

for label, paths in (("real", [l.split()[0] for l in []]), ):
    pass
excl = {l.split('#')[0].strip() for f in ("qualia_exclude.txt", "nearest_rule_exclude.txt")
        for l in open(Path.home() / "junction26" / f) if l.strip() and not l.startswith('#')}
real = [md.parent for md in sorted((Path.home() / "junction26/episodes").glob("2026*/*/meta.json"))
        if json.loads(md.read_text()).get("outcome") == "success" and f"{md.parent.parent.name}/{md.parent.name}" not in excl]
real_other = [md.parent for md in sorted((Path.home() / "junction26/episodes").glob("2026*/*/meta.json"))
              if json.loads(md.read_text()).get("outcome") == "success" and f"{md.parent.parent.name}/{md.parent.name}" in excl][:12]
sim = [p.parent for p in sorted((Path.home() / "junction26/episodes_sim_snap/gen5").glob("*/meta.json"))][:12]
sim_new = [p.parent for p in sorted((Path.home() / "junction26/episodes_sim/gen5").glob("*/meta.json"))][-12:]
for label, eps in (("sim (trained on)", sim), ("sim (unseen)", sim_new), ("real (trained on)", real), ("real (not trained on)", real_other)):
    act, pred = map(np.array, zip(*[probe(d) for d in eps]))
    err = np.linalg.norm(pred - act, axis=1) * 1000
    print(f"{label:22s} n={len(eps):2d}  pred-vs-actual grasp: median {np.median(err):5.1f} mm  |  spread of actual {act.std(0).mean()*1000:5.1f} mm,"
          f" spread of predicted {pred.std(0).mean()*1000:5.1f} mm,  corr x {np.corrcoef(act[:,0],pred[:,0])[0,1]:+.2f} y {np.corrcoef(act[:,1],pred[:,1])[0,1]:+.2f}")
