"""Generate randomized simulated pick episodes in the SAME raw format as the real collector (~/junction26/episodes).

    MUJOCO_GL=egl ~/physical-ai-studio/application/backend/.venv/bin/python generate.py -n 1000 -j 12
    -> ~/junction26/episodes_sim/<session>/<ep>/{top,wrist}/000000.jpg, data.npz (ticks), meta.json (source=sim)

Then export together with the real demos:  export_act.py --sim-episodes ~/junction26/episodes_sim
Domain randomization per episode: bricks (which, how many, pose, face up, colour shade), lights, table/plate/arm
colours, top + wrist camera pose and FOV jitter, and per-camera image degradation matched to the real cameras
(top: noisy at max gain, slightly soft; wrist: dark, warm, vignetted).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np

OUT = Path.home() / "junction26/episodes_sim"
CAMS = ["top", "wrist"]


# ---------------------------------------------------------------------------- image degradation (sim -> real look)

def degrade(img: np.ndarray, cam: str, look: dict, rng: np.random.Generator) -> np.ndarray:
    """RGB uint8 render -> BGR uint8 that looks like the real camera."""
    x = img.astype(np.float32) / 255.0
    x = x ** look["gamma"] * look["gain"] * np.array(look["wb"], np.float32)  # exposure + white balance (RGB)
    if cam == "wrist":
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        x *= (1 - look["vignette"] * np.clip(r2, 0, 2) / 2)[..., None]
    if look["blur"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), look["blur"])
    x = x + rng.normal(0, look["noise"], x.shape).astype(np.float32)  # sensor noise (real top runs at max gain)
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def sample_look(cam: str, rng: np.random.Generator) -> dict:
    if cam == "top":
        return dict(gamma=rng.uniform(0.85, 1.15), gain=rng.uniform(0.8, 1.15),
                    wb=[rng.uniform(0.9, 1.05), rng.uniform(0.92, 1.05), rng.uniform(0.95, 1.2)],  # real top is bluish
                    vignette=0, blur=rng.uniform(0.3, 1.0), noise=rng.uniform(0.015, 0.04))
    return dict(gamma=rng.uniform(1.1, 1.6), gain=rng.uniform(0.45, 0.85),                    # real wrist: dark + warm
                wb=[rng.uniform(1.0, 1.15), rng.uniform(0.75, 0.95), rng.uniform(0.45, 0.7)],
                vignette=rng.uniform(0.3, 0.7), blur=rng.uniform(0.0, 0.8), noise=rng.uniform(0.01, 0.03))


# ---------------------------------------------------------------------------- real backgrounds (sim -> real)

def _crop43(img: np.ndarray) -> np.ndarray:
    """Full 16:9 frame -> centre 4:3 crop at 640x480 (what prep.py gives the policy for act_roi = full frame)."""
    h, w = img.shape[:2]
    if abs(w / h - 4 / 3) > 0.01:
        cw = int(round(h * 4 / 3)); x0 = (w - cw) // 2; img = img[:, x0:x0 + cw]
    return cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)


def _erase_input_plate(img: np.ndarray) -> np.ndarray:
    """Paint out the real orange plate and the real bricks on it (the sim draws its own plate and bricks there)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (3, 120, 90), (18, 255, 255))
    n, lab, st, cen = cv2.connectedComponentsWithStats(orange)
    mask = np.zeros(img.shape[:2], np.uint8)
    if n > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] > 2000:
            r = int(max(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]) * 0.62)
            cv2.circle(mask, tuple(int(v) for v in cen[k]), r, 255, -1)
    if not mask.any(): return img
    small = cv2.inpaint(cv2.resize(img, (160, 120)), cv2.resize(mask, (160, 120)), 7, cv2.INPAINT_TELEA)
    fill = cv2.resize(small, (640, 480), interpolation=cv2.INTER_LINEAR)
    out = img.copy(); out[mask > 0] = fill[mask > 0]
    return out


def load_backgrounds() -> dict[str, list[np.ndarray]]:
    """Real frames used behind the sim: top = table + clutter, wrist = room seen from the gripper."""
    root = Path.home() / "junction26"
    pools = {"top": [], "wrist": []}
    for c in pools:
        p = root / "training_images" / c / "background.png"  # (live /snap previews carry UI overlays: not used)
        if p.exists(): pools[c].append(_crop43(cv2.imread(str(p))))
    for ep in sorted((root / "episodes").glob("*/*/meta.json")):
        d = ep.parent
        for f in (0, 10, 20):
            p = d / "top" / f"{f:06d}.jpg"
            if p.exists(): pools["top"].append(_erase_input_plate(_crop43(cv2.imread(str(p)))))
        for f in range(0, 120, 15):
            p = d / "wrist" / f"{f:06d}.jpg"
            if p.exists(): pools["wrist"].append(_crop43(cv2.imread(str(p))))
    return pools


def composite(sim_bgr: np.ndarray, seg: np.ndarray, cam: str, bg: np.ndarray, table_id: int,
              rng: np.random.Generator) -> np.ndarray:
    """Replace the sim table (top) / empty sky (wrist) with a real frame, keeping the sim shadows on the table."""
    import mujoco
    if cam == "top":
        mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] == table_id)
    else:
        mask = seg[..., 0] < 0
    if not mask.any(): return sim_bgr
    bgj = np.clip(bg.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.normal(0, 6, 3), 0, 255)
    if cam == "top":
        lum = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref = np.percentile(lum[mask], 90) + 1e-3
        bgj *= np.clip(lum / ref, 0.35, 1.0)[..., None]  # sim shadow -> darkens the real table
    out = sim_bgr.copy(); out[mask] = bgj[mask].astype(np.uint8)
    return out


# ---------------------------------------------------------------------------- scene randomization

class Randomizer:
    def __init__(self, model, cal):
        import station
        self.m, self.cal, self.station = model, cal, station
        m = model
        self.cam_top = m.camera("top").id
        self.top_pos0, self.top_quat0 = m.cam_pos[self.cam_top].copy(), m.cam_quat[self.cam_top].copy()
        self.light_pos0 = m.light_pos.copy(); self.light_diffuse0 = m.light_diffuse.copy()
        self.mat_rgba0 = m.mat_rgba.copy(); self.geom_rgba0 = m.geom_rgba.copy()
        names = [m.geom(i).name for i in range(m.ngeom)]
        body = [m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)]
        self.brick_geoms = {b.name: [i for i in range(m.ngeom) if body[i] == f"brick_{b.name}"] for b in station.BRICKS}
        self.plate_geoms = {p: [i for i in range(m.ngeom) if body[i] == f"plate_{p}"] for p in ("orange", "yellow", "blue")}
        self.table = names.index("table")
        arm_bodies = {"base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"}
        self.arm_geoms = [i for i in range(m.ngeom) if body[i] in arm_bodies]
        self.jaw_geoms = [i for i in range(m.ngeom) if body[i] in ("gripper", "moving_jaw_so101_v1") and m.geom_group[i] == 2]
        for g in self.jaw_geoms:  # own colour: their material is shared with the rest of the arm
            m.geom_matid[g] = -1
        self.geom_rgba0 = m.geom_rgba.copy()

    def apply(self, rng: np.random.Generator) -> dict:
        import mujoco
        from scipy.spatial.transform import Rotation as R
        m = self.m
        m.mat_rgba[:] = self.mat_rgba0; m.geom_rgba[:] = self.geom_rgba0
        # lights
        m.light_pos[:] = self.light_pos0 + rng.normal(0, [0.25, 0.25, 0.1], self.light_pos0.shape)
        m.light_diffuse[:] = np.clip(self.light_diffuse0 * rng.uniform(0.6, 1.4), 0, 1)
        # colours: table (wood shades), plates, bricks
        tm = m.geom_matid[self.table]
        if tm >= 0: m.mat_rgba[tm, :3] = np.clip(rng.uniform(0.75, 1.25, 3) * rng.uniform(0.8, 1.15), 0, 2)
        for p, gs in self.plate_geoms.items():
            k = rng.uniform(0.85, 1.1, 3)
            for g in gs: m.geom_rgba[g, :3] = np.clip(self.geom_rgba0[g, :3] * k, 0, 1)
        for b, gs in self.brick_geoms.items():
            k = rng.uniform(0.8, 1.15, 3) * rng.uniform(0.85, 1.1)
            for g in gs: m.geom_rgba[g, :3] = np.clip(self.geom_rgba0[g, :3] * k, 0, 1)
        # arm: the real follower is white with black servos; recolour the plugin's yellow plastic
        white = rng.uniform(0.82, 0.97); tint = rng.uniform(0.97, 1.03, 3)
        for g in self.arm_geoms:
            mid = m.geom_matid[g]; rgba = m.mat_rgba[mid] if mid >= 0 else m.geom_rgba[g]
            if rgba[0] > 0.5 and rgba[1] > 0.4 and rgba[2] < 0.4:  # yellow-ish -> white
                if mid >= 0: m.mat_rgba[mid, :3] = np.clip(white * tint, 0, 1)
                else: m.geom_rgba[g, :3] = np.clip(white * tint, 0, 1)
        # gripper + jaws: the real ones are dark red / black
        jaw = np.array([rng.uniform(0.35, 0.7), rng.uniform(0.04, 0.14), rng.uniform(0.04, 0.14)]) * rng.uniform(0.6, 1.0)
        for g in self.jaw_geoms: m.geom_rgba[g, :3] = jaw
        # top camera pose jitter (+-1 cm, +-1.5 deg)
        m.cam_pos[self.cam_top] = self.top_pos0 + rng.normal(0, 0.007, 3)
        q0 = self.top_quat0; r0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]])
        r = R.from_rotvec(rng.normal(0, np.radians(1.0), 3)) * r0; q = r.as_quat()
        m.cam_quat[self.cam_top] = [q[3], q[0], q[1], q[2]]
        # wrist camera jitter (mount tolerance)
        w = dict(self.cal["wrist"])
        w["pos"] = list(np.array(w["pos"]) + rng.normal(0, 0.003, 3)); w["tilt_deg"] += rng.normal(0, 3)
        w["fovy"] += rng.normal(0, 3)
        return {"wrist": w}


# ---------------------------------------------------------------------------- worker

def worker(args):
    wid, seeds, out_dir, session = args
    import mujoco

    import expert
    import station
    model, data, cal = station.load()
    sc = expert.Scene(model, data, cal)
    rz = Randomizer(model, cal)
    rend = mujoco.Renderer(model, 480, 640)
    seg_r = mujoco.Renderer(model, 480, 640); seg_r.enable_segmentation_rendering()
    pools = load_backgrounds(); table_id = model.geom("table").id
    names = [b.name for b in station.BRICKS]
    done = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        target = str(rng.choice(names))
        vis = rz.apply(rng)
        sc.reset(rng, int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])), target)
        station.set_wrist_camera(model, data, {**cal, "wrist": vis["wrist"]})
        looks = {c: sample_look(c, rng) for c in CAMS}
        bgs = {c: (pools[c][rng.integers(len(pools[c]))] if pools[c] and rng.uniform() < 0.85 else None) for c in CAMS}
        frames = {c: [] for c in CAMS}

        def on_frame(i):
            for c in CAMS:
                rend.update_scene(data, camera=c)
                im = degrade(rend.render(), c, looks[c], rng)
                if bgs[c] is not None:
                    seg_r.update_scene(data, camera=c)
                    im = composite(im, seg_r.render(), c, bgs[c], table_id, rng)
                frames[c].append(im)

        r = expert.run_episode(sc, rng, target, on_frame=on_frame)
        if not r["ok"]:
            done.append((seed, False, r["why"])); continue
        ep = f"sim{seed:06d}"; d = out_dir / session / ep
        for c in CAMS:
            (d / c).mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(frames[c]):
                cv2.imwrite(str(d / c / f"{i:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        state = station.rad_to_ticks(r["state"], cal).astype(np.float32)
        action = station.rad_to_ticks(r["action"], cal).astype(np.float32)
        n = len(state)
        np.savez_compressed(d / "data.npz", t=np.arange(n) / expert.FPS, state=state, action=action, leader=action,
                            cam_age=np.zeros((n, len(CAMS)), np.float32), cams=np.array(CAMS))
        close = r["phase"].index("close") if "close" in r["phase"] else n
        meta = {"episode": ep, "session": session, "source": "sim", "seed": int(seed), "brick": target,
                "label": "good", "fps": expert.FPS, "cams": CAMS,
                "act_roi": {c: [0, 0, 640, 480] for c in CAMS}, "joints": station.JOINTS,
                "follower_limits": cal["follower_limits"], "task": f"pick up the {target} brick",
                "phase1": "scripted sim expert pick + carry to WP1", "start": datetime.now().isoformat(timespec="seconds"),
                "pick_s": round(close / expert.FPS, 2), "handoff_frame": close, "handoff": "sim",
                "grasp_attempts": 1, "n_frames": n, "wp1_frame": n - 1, "outcome": "success",
                "gripper_lock": float(action[close + 5, 5]) if close + 5 < n else None, "max_cam_age_s": 0.0,
                "grasp_width_mm": round(r["width"] * 1000, 1), "phases": sorted(set(r["phase"]))}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        done.append((seed, True, ""))
    rend.close(); seg_r.close()
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="episodes to attempt")
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 2) - 4), help="worker processes")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--session", default="")
    a = ap.parse_args()
    session = a.session or datetime.now().strftime("%Y%m%d_%H%M%S_sim")
    seeds = list(range(a.seed, a.seed + a.n))
    chunks = [(i, seeds[i::a.j], a.out, session) for i in range(a.j) if seeds[i::a.j]]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(chunks)) as pool:
        res = [x for part in pool.map(worker, chunks) for x in part]
    ok = sum(1 for _, k, _ in res if k)
    print(f"{ok}/{len(res)} episodes kept in {time.time() - t0:.0f} s -> {a.out / session}")
    for s, k, why in res:
        if not k: print(f"  ✗ seed {s}: {why}")


if __name__ == "__main__":
    main()
