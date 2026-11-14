"""Hardware-free test of the inspect + sort logic (sorter.py, coverage.py, the tape re-grip in collect.py).

    python3 anomaly_capture/sim_test.py            (needs numpy, opencv, fastapi; no arm, no cameras, no Anomalib)

A fake arm (instant moves, a gripper that stalls on the object), a fake camera whose picture is bright exactly when
the defect is physically visible in the current pose, and a fake Anomalib worker. The fake world knows the TRUE
geometry (where the defect is, which way the re-grip really turns), the sorter only knows the ticks on the page.
Checked properties:
  * an object with a defect ANYWHERE on its surface is never sorted as good - for random defect directions, for
    both re-grip directions, for cube / long box / triangular prism / tetrahedron / cylinder in random orientations
  * a clean object is sorted as good after the planned number of photos
  * uncertain photo, missing depth mask, unseeable surface -> defect plate; lost object, failed re-grip -> stop
"""
import json, os, sys, tempfile, threading, time, types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import lerobot.motors.feetech.feetech  # noqa: F401
except Exception:                          # not on the station PC: stub the servo bus
    for m in ["lerobot", "lerobot.motors", "lerobot.motors.feetech", "lerobot.motors.feetech.feetech", "lerobot.motors.motors_bus"]:
        sys.modules[m] = types.ModuleType(m)
    sys.modules["lerobot.motors.feetech.feetech"].FeetechMotorsBus = object
    sys.modules["lerobot.motors.motors_bus"].Motor = object
    sys.modules["lerobot.motors.motors_bus"].MotorNormMode = types.SimpleNamespace(RANGE_M100_100=0)

FAKE = Path(tempfile.mkdtemp(prefix="fake_anomalib_"))
(FAKE / "anomalib" / "deploy").mkdir(parents=True)
(FAKE / "anomalib" / "__init__.py").write_text("")
(FAKE / "anomalib" / "deploy" / "__init__.py").write_text(
    "import cv2, numpy as np, types\n"
    "class OpenVINOInferencer:\n"
    "    def __init__(self, path, device): self.fast = 'efficientad' in str(path)\n"
    "    def predict(self, image):\n"
    "        v = float(cv2.imread(image).mean()) / 255\n"
    "        if self.fast and 0.35 < v < 0.9: v = 0.5\n"          # the fast model is unsure about subtle cases
    "        return types.SimpleNamespace(pred_score=np.array([v]), pred_label=np.array([v > .5]), anomaly_map=np.random.rand(1, 32, 32))\n"
    "TorchInferencer = OpenVINOInferencer\n")
os.environ["ANOMALIB_PY"] = sys.executable
os.environ["PYTHONPATH"] = str(FAKE)

import arm, collect, coverage as cov, sorter  # noqa: E402
from arm import JOINTS  # noqa: E402

NOSLEEP = types.SimpleNamespace(sleep=lambda s: None, time=time.time, perf_counter=time.perf_counter, monotonic=time.monotonic)
TRUE = {"view_deg": 65.0, "finger_deg": 35.0}      # physical truth (the sorter's model keeps a 5 deg margin to this)


def wp(name, **kw):
    p = {n: 2048 for n in JOINTS}; p.update(kw); return {"name": name, "pos": p, "capture": False}


ROUTINES = {"home": {"waypoints": [wp("home", gripper=2800)]}, "wp1": {"waypoints": [wp("wp1", shoulder_lift=1500)]},
            "wp2": {"waypoints": [wp("wp2-1", shoulder_pan=2300), wp("wp2-2", shoulder_pan=2300, wrist_roll=3000),
                                  wp("wp2-3", shoulder_pan=2300, wrist_flex=1500)]},
            "wp2_above": {"waypoints": [wp("wp2_above", shoulder_pan=2300, shoulder_lift=1500)]},
            "tape_above": {"waypoints": [wp("tape_above", shoulder_pan=2600, shoulder_lift=1500, wrist_roll=1000)]},
            "tape_place": {"waypoints": [wp("tape_place", shoulder_pan=2600, wrist_roll=1000)]},
            "sort_good": {"waypoints": [wp("sort_good", shoulder_pan=1000)]},
            "sort_defect": {"waypoints": [wp("sort_defect", shoulder_pan=3000)]}}
FACEMAP = {"wp2-1": ["front"], "wp2-2": ["back"], "wp2-3": ["down"]}


class World:
    """True state: defect normals (object frame), true re-grip direction, object width per grip, failures."""
    def __init__(self, defects=(), sign=1, widths=(2000,), facemap=FACEMAP, subtle=False):
        self.defects = [np.asarray(d, float) / np.linalg.norm(d) for d in defects]
        self.sign, self.widths, self.facemap, self.subtle = sign, list(widths), facemap, subtle
        self.k = 0; self.held = False; self.in_reach = True; self.lose_at_photo = None; self.photos = 0
        self.regrip_ok = True; self.no_depth = False; self.uncertain_views = set(); self.view = None

    def width(self): return self.widths[self.k % len(self.widths)]

    def sees_defect(self):
        """Is a defect physically visible in the current view? view = ('inhand', sides) | ('release',) | None"""
        if self.view is None: return False
        if self.view[0] == "release":
            v = np.array([0, 0, 1.0]); fing = None
        else:
            R = cov.rotz(self.k, self.sign); v = R @ cov.view_dir(self.view[1]); fing = R @ np.array([0, 1.0, 0])
        for n in self.defects:
            if n @ v >= np.cos(np.radians(TRUE["view_deg"])) and (fing is None or abs(n @ fing) < np.cos(np.radians(TRUE["finger_deg"]))):
                return True
        return False


class FakeArm:
    def __init__(self, w, routines):
        self.w = w; self.R = routines; self.f = True
        self.fl = {n: (0, 4095) for n in JOINTS}; self.fl["gripper"] = (1000, 3000)
        self.pos = {n: 2048 for n in JOINTS}; self.cmd = {n: 2048.0 for n in JOINTS}; self.goal = dict(self.cmd); self.lpos = {}
        self.mode = "hold"; self.stop = threading.Event(); self.running = False; self.io = threading.Lock(); self.moves = []

    def set_mode(self, m, force=False): self.mode = m

    def _at(self, name, tgt, ignore=("gripper",)):
        return any(all(abs(tgt[n] - x["pos"][n]) < 1 for n in JOINTS if n not in ignore) and (i,) for i, x in enumerate(self.R[name]["waypoints"]))

    def goto(self, tgt, speed):
        if self.stop.is_set(): raise arm.Stopped()
        w = self.w; g = tgt["gripper"]
        at_tape = self._at("tape_place", tgt, ignore=("gripper", "wrist_roll"))
        if w.held and g >= w.width(): w.held = False                              # opened: released
        elif not w.held and g < w.width() and w.in_reach and (at_tape or not w.photos):
            if at_tape and not w.regrip_ok: pass                                     # fingers miss the object
            else:
                if at_tape and abs(tgt["wrist_roll"] - self.R["tape_place"]["waypoints"][0]["pos"]["wrist_roll"]) > 500: w.k += 1
                w.held = True
        self.cmd = {n: float(tgt[n]) for n in JOINTS}; self.goal = dict(self.cmd)
        self.pos = {n: int(tgt[n]) for n in JOINTS}
        if w.held: self.pos["gripper"] = max(int(g), w.width())
        # which view is the camera looking at now?
        w.view = None
        for i, x in enumerate(self.R["wp2"]["waypoints"]):
            if all(abs(tgt[n] - x["pos"][n]) < 1 for n in JOINTS if n != "gripper") and w.held:
                w.view = ("inhand", w.facemap.get(x["name"]) or ["front"])
        if self._at("tape_above", tgt, ignore=("gripper", "wrist_roll")) and not w.held: w.view = ("release",)
        self.moves.append(dict(tgt)); return 0


def make(world, routines=None, live=("front", "wrist")):
    routines = json.loads(json.dumps(routines or ROUTINES))
    A = FakeArm(world, routines); ctx = types.SimpleNamespace(R={"routines": routines}, capture=lambda *a, **k: 1)
    depth = np.full((120, 160), 2000, np.uint16); depth[20:100, 40:120] = 400

    def averaged(cams):
        world.photos += 1
        if world.lose_at_photo == world.photos: world.held = False; A.pos["gripper"] = int(A.cmd["gripper"])
        sides = world.view[1] if world.view and world.view[0] == "inhand" else ()
        v = 215 if world.sees_defect() else 128 if set(sides) & world.uncertain_views else 20
        if world.subtle and v == 215: v = 170
        img = np.full((120, 160, 3), v, np.uint8)
        return {c: (img, img.astype(np.float32), img.astype(np.float32)) for c in cams}
    cams = {"front": types.SimpleNamespace(depth=depth, depth_t=time.time() + 1e9, settings={}), "wrist": types.SimpleNamespace(settings={})}
    S = types.SimpleNamespace(live=list(live), frames={}, cams=cams, averaged=averaged)
    root = Path(tempfile.mkdtemp(prefix="simroot_")); (root / "training_images").mkdir()
    C = collect.Collector(S, A, ctx, root / "training_images", root); C.cfg_path = root / "c.json"; C.cfg["speed"] = 1.0
    T = sorter.Sorter(S, A, ctx, C, root / "training_images", root); T.cfg_path = root / "s.json"
    T.cfg.update({"pick": "handfeed", "judge": "manual", "faces": dict(world.facemap), "retries": 1, "release": True})

    class Flags(dict):
        def __getitem__(self, k): return True if k == "grasped" else dict.__getitem__(self, k)
    T.flags = Flags(T.flags)

    def auto_judge(crop, caption):          # stands in for the human: looks at the (fake) picture
        m = crop.mean()
        return "defect" if m > 150 else "unsure" if m > 80 else "clean"
    T._manual = auto_judge
    return A, C, T, root


def run(world, n=1, **cfg):
    A, C, T, root = make(world)
    T.cfg.update(cfg)
    sorter.time = NOSLEEP; collect.time = NOSLEEP
    if world.no_depth: T.S.cams["front"].depth = None

    def helper():                           # presses NEXT whenever the sorter asks for help
        while T.active or T.state == "idle":
            if T.state == "help": T.flags["next"].set()
            time.sleep(0.005)
    threading.Thread(target=helper, daemon=True).start()
    T.run(n)
    logs = [json.loads(l) for l in open(root / "runs" / T.session / "log.jsonl")] if (root / "runs" / T.session / "log.jsonl").exists() else []
    pan = [m["shoulder_pan"] for m in A.moves if m["shoulder_pan"] in (1000, 3000)]
    plate = {1000: "good", 3000: "defect"}.get(pan[-1]) if pan else None
    return T, logs, plate


def rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3))); return q * np.sign(np.linalg.det(q))


SHAPES = {"cube": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
          "triangular prism": [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0.866, -0.5, 0), (-0.866, -0.5, 0)],
          "tetrahedron": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
          "cylinder": [(0, 0, 1), (0, 0, -1)] + [(np.cos(a), np.sin(a), 0) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)],
          "wedge": [(0, 0, -1), (0, 1, 0), (0, -1, 0), (-1, 0, 0), (0.6, 0, 0.8)]}


def main():
    rng = np.random.default_rng(1); t0 = time.time(); ok = lambda c, m: (_ for _ in ()).throw(AssertionError(m)) if not c else None

    # 1 clean object, both re-grip directions, constant and changing width (long box)
    for sign in (1, -1):
        for widths in ((2000,), (1700, 2600)):
            T, logs, plate = run(World(sign=sign, widths=widths))
            ok(logs[0]["verdict"] == "good" and plate == "good", f"clean object not sorted good: {logs[0]}")
            ok(logs[0]["photos"] == T.plan()["photos"] and logs[0]["regrasps"] == 1, "clean object: photo count differs from the plan")
    print(f"clean object      -> GOOD after {logs[0]['photos']} photos, {logs[0]['regrasps']} re-grip: {[(s['grip'], s['view']) for s in logs[0]['steps']]}")

    # 2 defect anywhere: never good
    n_photos = []
    for i in range(400):
        d = rng.normal(size=3); sign = 1 if i % 2 else -1
        T, logs, plate = run(World([d], sign=sign))
        ok(logs[0]["verdict"] == "defect" and plate == "defect", f"DEFECT MISSED at normal {d / np.linalg.norm(d)} sign {sign}: {logs[0]['reason']}")
        ok("defect seen" in logs[0]["reason"], f"defect not seen but fail-safed: {logs[0]['reason']}")
        n_photos.append(logs[0]["photos"])
    print(f"defect anywhere   -> 400/400 random defect directions sent to DEFECT; photos needed: mean {np.mean(n_photos):.1f}, "
          f"1 photo in {np.mean(np.array(n_photos) == 1) * 100:.0f} % (clean object: {T.plan()['photos']})")

    # 3 real shapes, random orientation in the hand, each face defective in turn
    for name, normals in SHAPES.items():
        cnt = 0
        for _ in range(12):
            Rm = rand_rot(rng)
            for nrm in normals:
                sign = 1 if cnt % 2 else -1; cnt += 1
                T, logs, plate = run(World([Rm @ np.asarray(nrm, float)], sign=sign, widths=(1800, 2500)))
                ok(logs[0]["verdict"] == "defect" and plate == "defect", f"{name}: defect face missed")
        T, logs, plate = run(World(widths=(1800, 2500)))
        ok(plate == "good", f"{name}: clean not good")
        print(f"{name:17s} -> {cnt} defective-face cases all DEFECT, clean one GOOD")

    # 4 fail-safes
    w = World(); w.uncertain_views = {"back"}
    T, logs, plate = run(w); ok(plate == "defect" and "uncertain" in logs[0]["reason"], "uncertain not fail-safed")
    print("uncertain photo   ->", logs[0]["verdict"], "|", logs[0]["reason"])
    w = World(); w.regrip_ok = False
    T, logs, plate = run(w); ok(logs[0]["verdict"] == "aborted" and plate is None and logs[0].get("on_tape"), "failed re-grip not aborted")
    print("re-grip fails x2  ->", logs[0]["verdict"], "|", logs[0]["reason"])
    w = World(); w.lose_at_photo = 2
    T, logs, plate = run(w); ok(logs[0]["verdict"] == "aborted" and plate is None and logs[0].get("lost"), "lost object not detected")
    print("object dropped    ->", logs[0]["verdict"], "|", logs[0]["reason"])
    w = World(facemap={"wp2-1": ["front"], "wp2-2": ["back"], "wp2-3": ["back"]})      # nothing shows the bottom
    T, logs, plate = run(w, allow_incomplete=True)
    ok(plate == "defect" and "not verifiable" in logs[0]["reason"], "unseeable surface not fail-safed")
    print("bottom never seen ->", logs[0]["verdict"], "|", logs[0]["reason"])
    T, logs, plate = run(World(), release=False, allow_incomplete=True)
    ok(plate == "defect" and "U" in logs[0]["reason"], "missing top not fail-safed")
    print("no top photo      ->", logs[0]["verdict"], "|", logs[0]["reason"])

    # 5 the Anomalib path (fake worker): single model, cascade, missing depth mask
    def with_models(world, cascade=False, **cfg):
        A, C, T, root = make(world)
        for d in ("front/all", "wrist/all", "front/all_efficientad", "wrist/all_efficientad"):
            (root / "models" / d / "weights").mkdir(parents=True); (root / "models" / d / "weights" / "model.xml").write_text("x")
            T.scorer.save_thresholds(root / "models" / d, {"t_low": 0.3, "t_high": 0.6})
        T.cfg.update({"judge": "model", "cascade": cascade, **cfg})
        if world.no_depth: T.S.cams["front"].depth = None
        sorter.time = NOSLEEP; collect.time = NOSLEEP
        T.run(1); T.scorer.close()
        log = json.loads(open(root / "runs" / T.session / "log.jsonl").readline()) if T.state == "done" else {"verdict": T.state, "reason": T.msg}
        return T, log
    T, log = with_models(World()); ok(log["verdict"] == "good", f"model path, clean: {log}")
    T, log = with_models(World([(0, 1, 0.2)])); ok(log["verdict"] == "defect" and log["regrasps"] == 1, f"model path, finger-side defect: {log}")
    print(f"model judge       -> clean GOOD, defect under a finger found after the re-grip ({log['photos']} photos, {log['infer_ms_mean']} ms/inference)")
    T, log = with_models(World([(1, 0, 0)], subtle=True), cascade=True)
    st = log["steps"][0]["shots"][0]["stages"]
    ok(log["verdict"] == "defect" and [s["stage"] for s in st] == ["fast", "main"], f"cascade escalation wrong: {st}")
    T, log2 = with_models(World(), cascade=True)
    used = [sh["decided_by"] for s in log2["steps"] for sh in s["shots"]]
    ok(log2["verdict"] == "good" and set(used) == {"fast"}, f"cascade: clean photos should be settled by the fast model: {used}")
    print(f"cascade           -> clean photos settled by the fast model alone ({len(used)}/{len(used)}), a subtle defect escalated fast → main")
    w = World(); w.no_depth = True
    T, log = with_models(w); ok(log["verdict"] == "defect" and "uncertain" in log["reason"], f"missing depth mask not fail-safed: {log}")
    print("no depth mask     ->", log["verdict"], "|", log["reason"])

    # 6 threshold calibration
    r = sorter.thresholds_from([0.1, 0.15, 0.2, 0.22], [0.8, 0.7, 0.18])          # third defect brick: cross never faced the camera
    ok(r["t_low"] < r["t_high"] and 0.22 < r["t_low"] and r["t_high"] < 0.7 and r["defect_bricks_not_visible"] == 1, f"thresholds: {r}")
    r = sorter.thresholds_from(list(np.linspace(0.1, 0.5, 40)), [0.49, 0.9]); ok(r["overlap"] and r["t_high"] > 0.5 and r["t_low"] < 0.49, f"overlap: {r}")
    try: sorter.thresholds_from([0.1, 0.2, 0.3], [0.15]); ok(False, "no separation must raise")
    except RuntimeError: pass
    print("calibration       -> separated / overlapping / hidden-cross cases OK")

    # 7 HTTP API
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    A, C, T, root = make(World())
    app = FastAPI(); app.include_router(sorter.make_router(T.S, A, T.ctx, C, T.DATA, root)); c = TestClient(app)
    s = c.get("/sort/status").json(); ok(not s["plan"]["complete"], "default ticks (front only) must not be complete")
    r = c.post("/sort/start?n=1").json(); ok("never be fully seen" in r["msg"], r["msg"])
    for v, sd in FACEMAP.items(): s = c.post(f"/sort/faces?view={v}&sides={','.join(sd)}").json()
    ok(s["plan"]["complete"] and s["plan"]["regrasps"] == 1, f"plan: {s['plan']}")
    s = c.post("/sort/cfg?release=0").json(); ok(not s["plan"]["complete"] and "U" in s["plan"]["missing"], "no top photo -> U missing")
    s = c.post("/sort/cfg?release=1&judge=model").json(); r = c.post("/sort/start?n=1").json(); ok("no anomaly model" in r["msg"], r["msg"])
    print("API               -> plan / faces / cfg / start checks OK")
    print(f"\nALL CHECKS PASSED in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
