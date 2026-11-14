"""Unit tests of the mock station itself (anomaly_capture/mock_hw.py): kinematics, pre-taught waypoints, rendering,
depth masking (prep.py on mock frames), fake anomaly scores, servo dynamics, data-root safety. In-process, no server.

    .venv/bin/python -m pytest tests/test_mock_hw.py
"""
import importlib.util, json, math, sys, time
from pathlib import Path

import numpy as np
import pytest

AC = Path(__file__).resolve().parent.parent / "anomaly_capture"


def _load(name):
    """Import anomaly_capture/<name>.py as a private module (no global sys.path change)."""
    key = f"_mocktest_{name}"
    if key in sys.modules: return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, AC / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[key] = m; spec.loader.exec_module(m)
    return m


m = _load("mock_hw")
prep = _load("prep")


def held_world(routine, i=0, defect="+x", grip=1890.0):
    """A fresh World with the follower at a pre-taught waypoint, holding a brick fed from above."""
    w = m.World(); p = {k: float(v) for k, v in m.mock_routines()[routine]["waypoints"][i]["pos"].items()}
    p["gripper"] = grip; w.servo["follower"]["pos"] = p
    k = m.FK(p); b = w.active; b.defect = defect; b.R = k.R.copy(); b.pos = k.tip.copy(); b.state = "held"
    return w, k


def render(w, cam):
    fp, bs = w.snapshot()
    return m.RENDER.render(cam, fp, bs, want_depth=(cam == "front"))


def test_ik_fk_roundtrip():
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(300):
        x, y, z = rng.uniform(0.05, 0.25), rng.uniform(-0.2, 0.2), rng.uniform(0.01, 0.2)
        phi = rng.choice([180.0, 150.0, 120.0])
        try: t = m.ik(x, y, z, phi, 1024, 2000)
        except ValueError: continue
        k = m.FK(t); n += 1
        assert np.allclose(k.tip, (x, y, z), atol=0.002), (x, y, z, k.tip)
        assert abs(math.degrees(math.acos(max(-1, min(1, k.d[2])))) - phi) < 1.0
    assert n > 100


def test_gripper_frame_is_right_handed_and_matches_coverage_convention():
    k = m.FK(m.ik(*m.WP2_TIP, 180, 1024, 2000))
    assert np.allclose(k.R.T @ k.R, np.eye(3), atol=1e-9) and np.linalg.det(k.R) > 0.999
    assert k.z[2] > 0.99                                 # palm up when pointing down
    cam = m.VIEWS["front"].pos
    assert k.x @ m.unit(cam - k.tip) > 0.9               # wp2-1: the free side "front" (+x) faces the front camera
    k2 = m.FK(m.ik(*m.WP2_TIP, 180, 3072, 2000))
    assert k2.x @ m.unit(cam - k2.tip) < -0.9            # wp2-2 (wrist turned 180 deg): "back" faces it
    k3 = m.FK(m.ik(*m.WP2_TIP, 90, 1024, 2000))
    assert k3.d @ m.unit(cam - k3.tip) > 0.9             # wp2-3: fingertips ("down") point at it


def test_pretaught_waypoints_cover_every_slot_and_stay_in_limits():
    r = m.mock_routines()
    for slot in ("home", "wp1", "drops", "wp2_above", "wp2", "tape_above", "tape_place", "sort_good", "sort_defect",
                 "sort_good_above", "sort_defect_above"):
        assert r[slot]["waypoints"], slot
        for w in r[slot]["waypoints"]:
            assert set(w["pos"]) == set(m.JOINTS)
            for j, v in w["pos"].items(): assert m.LIMITS[j][0] <= v <= m.LIMITS[j][1], (slot, j, v)
    assert [w["capture"] for w in r["wp2"]["waypoints"]] == [True, True, True]
    tp = m.FK(r["tape_place"]["waypoints"][0]["pos"]).tip; (tx, ty), _ = m.TAPE
    assert abs(tp[0] - tx) < 0.003 and abs(tp[1] - ty) < 0.003 and tp[2] < 0.02
    # the tape re-grip turns the wrist +90 deg: must stay inside the roll range (collect.grip_on_tape)
    assert r["tape_place"]["waypoints"][0]["pos"]["wrist_roll"] + 1024 <= m.LIMITS["wrist_roll"][1] - 20
    # the home gripper opening must NOT trigger the hand-feed (only the sorter's 80 % opening does)
    assert m.gap_of(r["home"]["waypoints"][0]["pos"]["gripper"]) < 0.075 < m.gap_of(m.G_MIN + 0.8 * (m.G_MAX - m.G_MIN))


@pytest.mark.parametrize("cam", ["top", "front", "wrist"])
def test_frames_have_real_resolution(cam):
    w, _ = held_world("home")
    img, dep = render(w, cam)
    assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8 and img.std() > 10
    if cam == "front": assert dep.shape == (1080, 1920) and dep.dtype == np.uint16


def test_front_depth_isolates_the_held_brick_at_wp2():
    for i in range(3):
        w, k = held_world("wp2", i)
        img, dep = render(w, "front")
        mask, info = prep.inhand_mask(dep)
        assert mask is not None, info
        assert 300 < info["z_mm"] < 420, info
        crop, ci = prep.inspect_crop(img, dep)
        assert ci["masked"] and crop.shape[0] == crop.shape[1]
        # the brick centre projects inside the mask
        q = m.VIEWS["front"].cam(w.active.pos[None]); u, v = m.VIEWS["front"].px(q)[0].astype(int)
        assert mask[v, u] > 0


def test_nothing_held_far_from_the_front_camera_gives_no_mask():
    w, _ = held_world("home"); w.active.state = "input"; w.active.settle()
    _, dep = render(w, "front")
    mask, info = prep.inhand_mask(dep)
    assert mask is None or info["z_mm"] > 420


@pytest.mark.parametrize("routine,i,defect,visible", [
    ("wp2", 0, "+x", True), ("wp2", 0, "-x", False), ("wp2", 1, "-x", True), ("wp2", 2, "-z", True),
    ("wp2", 0, "+y", False), ("wp2", 0, None, False)])
def test_fake_score_follows_ground_truth(tmp_path, routine, i, defect, visible):
    w, _ = held_world(routine, i, defect)
    img, dep = render(w, "front")
    crop, _ = prep.inspect_crop(img, dep)
    p = tmp_path / "c.png"; import cv2; cv2.imwrite(str(p), crop)
    s, amap = m.fake_predict(p)
    assert w.truth(m.VIEWS["front"].pos) == visible
    assert (s > 0.6) == visible, s
    assert amap.ndim == 3


def test_release_photo_sees_the_top_face(tmp_path):
    import cv2
    w, _ = held_world("tape_above", defect="+z")
    b = w.active; b.R = np.eye(3); (tx, ty), _ = m.TAPE; b.pos = np.array([tx, ty, 0.0]); b.settle()
    assert b.state == "tape"
    w.servo["follower"]["pos"]["gripper"] = 2850.0                         # released: gripper open above the tape
    img, _ = render(w, "wrist")
    m.seed(tmp_path); roi = json.loads((tmp_path / "settings" / "inspect_settings.json").read_text())["release_roi"]["wrist"]
    H, W = img.shape[:2]; c = img[int(roi[1] * H):int(roi[3] * H), int(roi[0] * W):int(roi[2] * W)]
    cv2.imwrite(str(tmp_path / "r.png"), c)
    assert m.fake_predict(tmp_path / "r.png")[0] > 0.6
    b.defect = None
    img, _ = render(w, "wrist"); cv2.imwrite(str(tmp_path / "r2.png"), img)
    assert m.fake_predict(tmp_path / "r2.png")[0] < 0.3


def test_servo_moves_take_time_and_stall_on_a_brick():
    w = m.World(); w.torque("follower", True)
    start = dict(w.read_pos("follower"))
    tgt = dict(start); tgt["shoulder_pan"] = start["shoulder_pan"] + 800
    w.write_goal("follower", tgt)
    time.sleep(0.05); p1 = w.read_pos("follower")["shoulder_pan"]
    assert start["shoulder_pan"] < p1 < tgt["shoulder_pan"] - 300          # not a teleport
    time.sleep(1.0); assert abs(w.read_pos("follower")["shoulder_pan"] - tgt["shoulder_pan"]) < 5
    # gripper closing on a hand-fed brick stalls at the brick width and grips it
    k = m.FK(w.servo["follower"]["pos"]); b = w.active
    b.R = k.R.copy(); b.pos = k.tip.copy(); b.state = "fed"
    w.write_goal("follower", {"gripper": m.G_MIN}); time.sleep(0.8)
    g = w.read_pos("follower")["gripper"]
    assert abs(g - m.grip_of(b.size[1])) < 3 and b.state == "held"
    w.write_goal("follower", {"gripper": 2800}); time.sleep(0.8); w.read_pos("follower")
    assert b.state not in ("held", "fed")                                  # opened: released, fell somewhere
    # limp: torque off -> the goal is ignored
    w.torque("follower", False); p = w.read_pos("follower")["shoulder_pan"]
    w.write_goal("follower", {"shoulder_pan": p - 500}); time.sleep(0.2)
    assert w.read_pos("follower")["shoulder_pan"] == p


def test_obstacle_and_bus_errors():
    w = m.World(); w.torque("follower", True); w.faults["obstacle"] = True
    w.write_goal("follower", {"shoulder_lift": 2600}); time.sleep(0.6)
    assert w.read_pos("follower")["shoulder_lift"] <= 2100
    w.faults["bus_errors"] = 2
    for _ in range(2):
        with pytest.raises(ConnectionError): w.read_pos("follower")
    w.read_pos("follower")


def test_data_root_refuses_the_repo(monkeypatch):
    monkeypatch.setenv("JUNCTION_DATA_ROOT", str(AC.parent))
    with pytest.raises(SystemExit): m.data_root()
    monkeypatch.setenv("JUNCTION_DATA_ROOT", str(AC))
    with pytest.raises(SystemExit): m.data_root()


def test_seed_is_idempotent(tmp_path):
    m.seed(tmp_path)
    r = tmp_path / "settings" / "routines.json"
    assert set(json.loads(r.read_text())) >= {"home", "wp1", "wp2", "tape_place", "sort_good"}
    r.write_text("{}"); m.seed(tmp_path)
    assert r.read_text() == "{}"                                           # a mock session's own edits are kept
    assert (tmp_path / "models" / "front" / "all" / "weights" / "openvino" / "model.xml").exists()
    assert (tmp_path / "_mock_pkgs" / "anomalib" / "deploy" / "__init__.py").exists()


def test_fake_anomalib_package_works_like_the_real_inferencer(tmp_path):
    """score_worker.py talks to anomalib.deploy.OpenVINOInferencer: run the REAL worker against the fake package."""
    import cv2, os, subprocess
    m.seed(tmp_path)
    img = np.full((200, 200, 3), 128, np.uint8); cv2.rectangle(img, (60, 60), (140, 140), (40, 40, 200), -1)
    cv2.imwrite(str(tmp_path / "clean.png"), img)
    cv2.line(img, (70, 70), (130, 130), m.INK, 6); cv2.line(img, (70, 130), (130, 70), m.INK, 6)
    cv2.imwrite(str(tmp_path / "defect.png"), img)
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "_mock_pkgs")}
    reqs = "".join(json.dumps({"id": i, "model": str(tmp_path / "m.xml"), "image": str(tmp_path / f), "device": "CPU",
                               "heatmap": str(tmp_path / f"h{i}.jpg")}) + "\n" for i, f in enumerate(["clean.png", "defect.png"]))
    out = subprocess.run([sys.executable, "-u", str(AC / "score_worker.py")], input=reqs, capture_output=True, text=True,
                         cwd=str(AC), env=env, timeout=60)
    lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    assert lines[0] == {"ready": True}, out.stderr
    assert lines[1]["score"] < 0.3 and lines[2]["score"] > 0.6, lines
    assert (tmp_path / "h1.jpg").exists()
