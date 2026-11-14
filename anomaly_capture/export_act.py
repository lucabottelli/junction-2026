"""Clean the raw pick episodes and write them as a Studio (LeRobot v3) dataset for ACT training.

Run with Studio's Python (the capture page's EXPORT button does this):
    ~/physical-ai-studio/application/backend/.venv/bin/python export_act.py [--dry-run] [--speedup 1.5] ...

Why: a human teleoperator is slow and makes mistakes, and ACT copies whatever it sees - including hesitation.
The policy should learn the BEST version of each pick, so every raw episode is cleaned:
  1. REJECT   failed/missed/discarded picks, unticked episodes, fumbles (more grasp attempts than allowed),
              very slow outliers (> --slow-factor x median pick time), camera drop-outs.
  2. TRIM     the idle time before the human starts moving.
  3. SMOOTH   hand tremor out of the commanded joint targets (Gaussian, sigma 1.5 frames; gripper untouched).
  4. RETIME   - pauses (arm nearly still) are compressed 10x,
              - free-space motion is sped up by --speedup,
              - the final approach + grasp (last --precision-s before the gripper closes) keeps full detail,
              - the robot's own automatic carry to WP1 (after the grasp) is kept as recorded (already optimal),
              - no step may move a joint more than --max-step-deg (keeps the sped-up motion trackable).
              Frames are SELECTED from the real recording (images and joint states stay genuine), so the
              policy simply learns to take bigger steps where the human dawdled.
Nothing in the raw episodes is changed; you can re-export with other settings any time.
"""
import argparse, json, sys, urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EP = ROOT / "episodes"
STUDIO_SRC = Path.home() / "physical-ai-studio/application/backend/src"
API = "http://127.0.0.1:7860/api"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POLICY_CAMS = ["top", "wrist"]
TICK_PER_DEG = 4096 / 360


def gauss_smooth(x, sigma):
    if sigma <= 0 or len(x) < 3: return x.copy()
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def normalize(ticks, lim):
    out = np.empty_like(ticks, dtype=np.float32)
    for j, n in enumerate(JOINTS):
        lo, hi = lim[n]; fr = np.clip((ticks[:, j] - lo) / max(1, hi - lo), 0, 1)
        out[:, j] = fr * 100 if n == "gripper" else fr * 200 - 100
    return out


def plan(meta, d, a, args):
    """Return (selected frame indices, report) for one episode."""
    fps = meta["fps"]; N = len(d["t"]); lim = meta["follower_limits"]
    act = d["action"].astype(np.float64)
    body = slice(0, 5)
    act_s = act.copy(); act_s[:, body] = gauss_smooth(act[:, body], args.smooth)
    deg = act_s[:, body] / TICK_PER_DEG
    # 2. trim idle start
    moved = np.nonzero(np.abs(deg - deg[0]).max(1) > 1.5)[0]
    start = max(0, int(moved[0]) - 3) if moved.size else 0
    # precision zone: from precision_s before the final gripper close starts, to the end
    glo, ghi = lim["gripper"]; gn = (act[:, 5] - glo) / max(1, ghi - glo) * 100
    h = meta.get("handoff_frame") or N
    opened = np.nonzero(gn[:h] > 40)[0]
    close_start = int(opened[-1]) if opened.size else h
    p0 = max(start, close_start - int(args.precision_s * fps))
    # 4. retime: cost of each original frame in the new timeline
    v = np.r_[0, np.abs(np.diff(deg, axis=0)).max(1)]            # deg per original frame
    v_s = gauss_smooth(v[:, None], 2)[:, 0]
    w = np.ones(N)
    free = np.arange(N) < p0
    w[free] = 1.0 / args.speedup
    w[free & (v_s < args.pause_deg)] = 0.1                       # pauses nearly vanish
    w = np.maximum(w, v / args.max_step_deg)                     # never more than max_step_deg per new step
    w[:start] = 0
    T = np.cumsum(w) - w[start]
    idx, k = [], 0.0
    for i in range(start, N):
        if T[i] >= k - 1e-9:
            idx.append(i); k = np.floor(T[i]) + 1
    if idx[-1] != N - 1: idx.append(N - 1)
    rep = {"orig_s": round(N / fps, 2), "new_s": round(len(idx) / fps, 2), "trim_s": round(start / fps, 2),
           "pause_s": round(float((free & (v_s < args.pause_deg))[start:].sum()) / fps, 2),
           "precision_s": round((N - p0) / fps, 2), "frames": len(idx)}
    return idx, act_s, rep


def load_episodes(args):
    eps, rej = [], []
    for md in sorted(EP.glob("*/*/meta.json")):
        m = json.loads(md.read_text()); d = md.parent; name = f"{d.parent.name}/{d.name}"
        if args.sessions and d.parent.name not in args.sessions: continue
        if args.bricks and m.get("brick") not in args.bricks: continue
        why = None
        if m.get("outcome") != "success": why = f"outcome {m.get('outcome')}"
        elif m.get("keep") is False: why = "unticked by you"
        elif (m.get("grasp_attempts") or 0) > args.max_attempts: why = f"{m['grasp_attempts']} grasp attempts (fumble)"
        elif (m.get("max_cam_age_s") or 0) > 0.5: why = f"camera stalled {m['max_cam_age_s']} s"
        elif not (d / "data.npz").exists(): why = "data not written"
        if why: rej.append((name, why)); continue
        eps.append((name, m, d))
    if eps:
        med = float(np.median([m["pick_s"] for _, m, _ in eps]))
        keep = []
        for name, m, d in eps:
            if m["pick_s"] > args.slow_factor * med: rej.append((name, f"slow outlier {m['pick_s']} s (median {med:.1f} s)"))
            else: keep.append((name, m, d))
        eps = keep
    return eps, rej


def studio_ids():
    projects = json.load(urllib.request.urlopen(f"{API}/projects", timeout=5))
    p = projects[0]
    envs = json.load(urllib.request.urlopen(f"{API}/projects/{p['id']}/environments", timeout=5))
    return p["id"], envs[0]["id"]


def main():
    global EP
    ap = argparse.ArgumentParser()
    ap.add_argument("--speedup", type=float, default=1.5)
    ap.add_argument("--max-attempts", type=int, default=1)
    ap.add_argument("--slow-factor", type=float, default=2.5)
    ap.add_argument("--precision-s", type=float, default=1.0)
    ap.add_argument("--pause-deg", type=float, default=0.15, help="deg/frame below which the arm counts as paused")
    ap.add_argument("--max-step-deg", type=float, default=6.0)
    ap.add_argument("--smooth", type=float, default=1.5)
    ap.add_argument("--bricks", nargs="*"); ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--name", default=""); ap.add_argument("--task", default="", help="override the per-episode task (default: 'pick up the <brick> brick')")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--episodes", default=str(ROOT / "episodes"), help="raw episodes folder")
    args = ap.parse_args()
    EP = Path(args.episodes)

    eps, rej = load_episodes(args)
    print(f"{len(eps)} episode(s) accepted, {len(rej)} rejected")
    for n, why in rej: print(f"  ✗ {n}: {why}")
    plans, tot_o, tot_n = [], 0, 0
    for name, m, d in eps:
        data = dict(np.load(d / "data.npz"))
        idx, act_s, rep = plan(m, data, None, args)
        plans.append((name, m, d, data, idx, act_s)); tot_o += rep["orig_s"]; tot_n += rep["new_s"]
        print(f"  ✓ {name}: {rep['orig_s']} s -> {rep['new_s']} s  (idle trimmed {rep['trim_s']} s, pauses {rep['pause_s']} s,"
              f" grasp + robot carry to WP1 {rep['precision_s']} s kept, {rep['frames']} frames)")
    if not plans: print("nothing to export"); return 1
    print(f"total {tot_o:.1f} s of demonstrations -> {tot_n:.1f} s ({100 * (1 - tot_n / tot_o):.0f}% shorter)")
    if args.dry_run: print("dry run: nothing written"); return 0

    sys.path.insert(0, str(STUDIO_SRC))
    from internal_datasets.utils import get_internal_recording_dataset
    from runtime_session.dataset import lerobot_features
    from runtime_session.store import FeatureSpec, ImageSpec
    from schemas.dataset import Dataset

    pid, eid = studio_ids()
    bricks = sorted({m.get("brick") or "?" for _, m, *_ in plans})
    task_of = lambda m: args.task or (m.get("task") if m.get("task") and m.get("brick") else "pick up the brick")
    tasks = sorted({task_of(m) for _, m, *_ in plans})
    name = args.name or f"pick-{'-'.join(bricks)}-x{len(plans)}"
    body = json.dumps({"name": name, "default_task": tasks[0] if len(tasks) == 1 else "pick up the brick",
                       "project_id": pid, "environment_id": eid}).encode()
    print(f"blocks: {bricks} · tasks: {tasks}")
    req = urllib.request.Request(f"{API}/dataset", data=body, headers={"Content-Type": "application/json"}, method="POST")
    ds = Dataset(**json.load(urllib.request.urlopen(req, timeout=10)))
    print(f"Studio dataset '{name}' created: {ds.id}")
    spec = FeatureSpec(joint_names=tuple(f"{n}.pos" for n in JOINTS),
                       observations={c: ImageSpec(480, 640) for c in POLICY_CAMS})
    mut = get_internal_recording_dataset(ds).start_recording_mutation(fps=30, features=lerobot_features(spec),
                                                                      robot_type="SO101_Follower")
    try:
        for name_, m, d, data, idx, act_s in plans:
            lim = m["follower_limits"]
            st = normalize(data["state"][idx], lim); ac = normalize(act_s[idx], lim)
            for r, i in enumerate(idx):
                obs = {f"{n}.pos": float(st[r, j]) for j, n in enumerate(JOINTS)}
                for c in POLICY_CAMS:
                    img = cv2.imread(str(d / c / f"{i:06d}.jpg"))
                    obs[c] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mut.add_frame(obs, {f"{n}.pos": float(ac[r, j]) for j, n in enumerate(JOINTS)}, task_of(m))
            mut.save_episode(); print(f"  wrote {name_} ({len(idx)} frames)")
    finally:
        mut.teardown()
    print(f"✓ done: dataset '{name}' is in Studio (Datasets). Train ACT on it there; cameras: {POLICY_CAMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
