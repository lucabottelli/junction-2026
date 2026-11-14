"""Hardware-free tests of the autonomous ACT pick (anomaly_capture/policy.py) and its wiring into sorter.py.

    .venv/bin/python -m pytest tests/test_policy.py -q

Self-contained (no conftest fixtures). A small simulated arm (rate-limited like arm.py's 50 Hz loop, a gripper
whose fingers stall on the object) runs on a simulated clock, so the policy loop executes tick by tick without
waiting. Covered: the observation equals what export_act.py writes into the Studio dataset, a real OpenVINO IR
(and ONNX) export with Studio's manifest layout loads and runs on CPU (through physicalai's InferenceModel when
installed, and through the direct fallback), chunk / async / ensemble execution, the stall hand-off, timeout ->
retry -> clean failure, STOP in the middle of the policy, BLOCKED, latency logging, the full autonomous cycle
(ACT pick -> WP1 -> inspect -> sort) and the HTTP API.
"""
import json, sys, threading, time, types
from pathlib import Path

import cv2
import numpy as np
import pytest

AC = Path(__file__).resolve().parents[1] / "anomaly_capture"
sys.path.insert(0, str(AC))
import sim_test  # noqa: E402  (stubs the servo bus when lerobot is missing; fake arm / world / camera of the sorter test)
import arm, collect, export_act, policy as pol, prep, sorter  # noqa: E402
from arm import JOINTS  # noqa: E402

RNG = np.random.default_rng(0)
LIM = {n: (0, 4095) for n in JOINTS}; LIM["gripper"] = (1000, 3000)


# ====================================================================== simulated arm + clock
class SimArm:
    """Like arm.Arm for the policy loop: goal -> cmd rate-limited at vmax (50 Hz), pos follows cmd; the fingers stall
    at the object's width when the arm is at the object (OBJ pose) and keep holding it while squeezing."""
    def __init__(self, obj=None, width=2000, follow=True, slip=False):
        self.f = True; self.fl = dict(LIM); self.mode = "hold"; self.stop = threading.Event(); self.io = threading.Lock()
        self.running = False; self.vmax = 750.0; self.obj, self.width, self.follow, self.slip = obj, width, follow, slip
        self.cmd = {n: 2048.0 for n in JOINTS}; self.cmd["gripper"] = 2800.0
        self.goal = dict(self.cmd); self.pos = dict(self.cmd); self.held = False; self.writes_after_stop = 0

    def _at_obj(self):
        return self.obj is not None and all(abs(self.cmd[n] - self.obj[n]) < 40 for n in arm.BODY)

    def _settle(self):
        g = self.cmd["gripper"]
        if self.held and g > self.width: self.held = False
        elif not self.held and g < self.width and self._at_obj(): self.held = True
        if self.follow: self.pos = {n: self.cmd[n] for n in arm.BODY}
        self.pos["gripper"] = max(g, self.width) if self.held else g

    def tick(self, dt=0.02):
        if self.stop.is_set() and self.goal != self.cmd: self.writes_after_stop += 1
        if self.mode == "hold":
            for n in JOINTS:
                g = min(self.fl[n][1], max(self.fl[n][0], self.goal[n]))
                self.cmd[n] += max(-self.vmax * dt, min(self.vmax * dt, g - self.cmd[n]))
        self._settle()

    def goto(self, tgt, speed):           # scripted moves: instant (the timing of those isn't under test here)
        if self.stop.is_set(): raise arm.Stopped()
        if self.slip: self.held = False                         # the object slips out on any scripted move
        self.cmd = {n: float(tgt[n]) for n in JOINTS}; self.goal = dict(self.cmd); self._settle()

    def set_mode(self, m, force=False): self.mode = m


class SimClock:
    def __init__(self, arm_, stop_at=None):
        self.t, self.arm, self.stop_at, self.frozen = 0.0, arm_, stop_at, None

    def now(self): return self.t

    def sleep_until(self, t):
        while self.t < t - 1e-9:
            self.t += 0.02; self.arm.tick()
            if self.stop_at is not None and self.t >= self.stop_at and not self.arm.stop.is_set():
                self.arm.stop.set()
                with self.arm.io: self.arm.mode = "hold"; self.arm.goal = dict(self.arm.cmd)    # what /sort/stop does
                self.frozen = dict(self.arm.goal)


class FreshTimes(dict):
    def get(self, k, d=None): return time.time()


def wp(name, **kw):
    p = {n: 2048 for n in JOINTS}; p.update(kw); return {"name": name, "pos": p, "capture": False}


ROUT = {"home": {"waypoints": [wp("home", gripper=2800)]}, "wp1": {"waypoints": [wp("wp1", shoulder_lift=1500)]}}
OBJ = {**{n: 2048 for n in JOINTS}, "shoulder_pan": 2300, "elbow_flex": 2400}


def station(arm_, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    frames = {c: RNG.integers(0, 255, (1080, 1920, 3), np.uint8) for c in ("top", "wrist", "front")}
    S = types.SimpleNamespace(frames=frames, frame_t=FreshTimes(), live=["top", "wrist", "front"], cams={})
    C = collect.Collector(S, arm_, types.SimpleNamespace(R={"routines": json.loads(json.dumps(ROUT))}), tmp, tmp)
    C.cfg_path = tmp / "c.json"; C.cfg["speed"] = 1.0
    return S, C


class FakePolicy:
    """Closed-loop fake ACT (normalized units in and out, like the real export): drives the body towards the object
    pose in 10-step chunks, closes the gripper once there. close=False never closes (-> timeouts)."""
    def __init__(self, target=OBJ, close=True, chunk=10, delay=0.0):
        self.cams, self.image_keys, self.takes_task = ["top", "wrist"], ["images.top", "images.wrist"], False
        self.export_dir, self.backend, self.device, self.runtime, self.load_ms, self.label = Path("fake"), "openvino", "CPU", "fake", 1.0, "fake"
        self.tgt = export_act.normalize(np.array([[target[n] for n in JOINTS]], float), LIM)[0]
        self.close, self.n, self.delay, self.calls, self.closed = close, chunk, delay, [], False

    def chunk(self, obs):
        assert obs["state"].shape == (1, 6) and obs["state"].dtype == np.float32
        assert all(obs[k].shape == (1, 480, 640, 3) and obs[k].dtype == np.uint8 for k in self.image_keys)
        if self.delay: time.sleep(self.delay)
        s = obs["state"][0].astype(np.float64); self.calls.append(s.copy())
        out = []
        for i in range(1, self.n + 1):
            a = s + (self.tgt - s) * min(1.0, i / 6); a[5] = 70.0
            if self.close and np.abs(s[:5] - self.tgt[:5]).max() < 1.0: a[5] = 0.0     # there: close the fingers
            out.append(a)
        return np.array(out, np.float32)

    def reset(self): pass
    def close(self): pass


def act_pick(policy, arm_=None, tmp=None, **kw):
    arm_ = arm_ or SimArm(OBJ); S, C = station(arm_, tmp)
    clock = kw.pop("clock", None) or SimClock(arm_)
    return pol.ActPick(arm_, C, S, policy, clock=clock, **kw), arm_, C


# ====================================================================== observation = training data
def test_observation_matches_export(tmp_path):
    frame = RNG.integers(0, 255, (1080, 1920, 3), np.uint8); roi = prep.fit_43([300, 100, 1500, 1000])
    old = cv2.imencode(".jpg", prep.act_frame(frame, roi), [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tobytes()
    assert prep.act_jpeg(frame, roi) == old                                   # collect._record writes identical bytes
    (tmp_path / "000000.jpg").write_bytes(old)
    trained = cv2.cvtColor(cv2.imread(str(tmp_path / "000000.jpg")), cv2.COLOR_BGR2RGB)     # export_act.py's read
    pos = {n: float(RNG.integers(0, 4095)) for n in JOINTS}; pos["gripper"] = 1700.0
    obs = pol.observation({"top": frame, "wrist": frame}, pos, LIM, ["images.top", "images.wrist"], ["top", "wrist"],
                          {"top": roi, "wrist": roi}, task="pick up the red brick")
    assert set(obs) == {"state", "images.top", "images.wrist", "task"} and obs["task"] == ["pick up the red brick"]
    assert obs["images.top"].shape == (1, 480, 640, 3) and obs["images.top"].dtype == np.uint8
    assert np.array_equal(obs["images.top"][0], trained)
    st = export_act.normalize(np.array([[pos[n] for n in JOINTS]]), LIM)
    assert obs["state"].dtype == np.float32 and obs["state"].shape == (1, 6) and np.array_equal(obs["state"], st)
    assert -100 <= obs["state"][0, :5].min() and obs["state"][0, :5].max() <= 100 and 0 <= obs["state"][0, 5] <= 100
    back = export_act.denormalize(st, LIM)[0]
    assert np.abs(back - [pos[n] for n in JOINTS]).max() < 1e-3                # actions map back to the same ticks
    with pytest.raises(RuntimeError, match="camera wrist"):
        pol.observation({"top": frame}, pos, LIM, ["images.top", "images.wrist"], ["top", "wrist"], {"top": roi, "wrist": roi})


# ====================================================================== a real export with Studio's layout
CH, NSTEPS, RES = 12, 8, (96, 96)


def expected_chunk(state, img_rgb):
    """What the fake network computes: action[t] = state + 10 * mean(letterboxed top image) + t."""
    lb = pol._letterbox(img_rgb[None], RES)
    return np.array([state + 10 * lb.mean() + t for t in range(CH)])[:NSTEPS]


def studio_manifest(backend, artifact):
    F = "physicalai.inference.data.features.InferenceFeature"
    feat = lambda ft, shape, name: {"class_path": F, "init_args": {"ftype": ft, "shape": shape, "name": name, "dtype": "float32"}}
    return {"format": "policy_package", "version": "1.0",
            "policy": {"name": "act", "source": {"repo_id": "", "class_path": "physicalai.policies.act.policy.ACT"}},
            "model": {"runner": {"class_path": "physicalai.inference.runners.single_pass.SinglePass", "init_args": {}},
                      "artifacts": {backend: artifact},
                      "preprocessors": [{"type": "to_float_tensor"}, {"type": "resize", "image_resolution": list(RES), "mode": "letterbox"}],
                      "postprocessors": [{"type": "action_chunk_trimmer", "n_action_steps": NSTEPS}],
                      "input_features": [feat("STATE", [6], "state"), feat("VISUAL", [3, 480, 640], "images.top"),
                                         feat("VISUAL", [3, 480, 640], "images.wrist")],
                      "output_features": [{"class_path": F, "init_args": {"ftype": "ACTION", "shape": [CH, 6], "name": "action", "dtype": "float32"}}]}}


def make_openvino_export(d):
    import openvino as ov, openvino.opset13 as ops
    d.mkdir(parents=True)
    st = ops.parameter([1, 6], np.float32, name="state")
    top = ops.parameter([1, 3, *RES], np.float32, name="images.top"); wr = ops.parameter([1, 3, *RES], np.float32, name="images.wrist")
    m = ops.reshape(ops.reduce_mean(top, [1, 2, 3], keep_dims=False), [1, 1], False)
    a = ops.add(st, ops.multiply(m, ops.constant(np.float32(10))))
    a = ops.add(ops.tile(ops.unsqueeze(a, [1]), [1, CH, 1]), ops.constant(np.arange(CH, dtype=np.float32).reshape(1, CH, 1)))
    a = ops.add(a, ops.multiply(ops.reshape(ops.reduce_mean(wr, [1, 2, 3], keep_dims=False), [1, 1, 1], False), ops.constant(np.float32(0))))   # uses wrist
    a.output(0).get_tensor().set_names({"action"})
    ov.save_model(ov.Model([a], [st, top, wr], "act"), str(d / "act.xml"))
    (d / "manifest.json").write_text(json.dumps(studio_manifest("openvino", "act.xml"), indent=2))
    return d


def make_onnx_export(d):
    import onnx
    from onnx import TensorProto as T, helper as h
    d.mkdir(parents=True)
    nodes = [h.make_node("ReduceMean", ["images.top"], ["m"], keepdims=0, axes=[1, 2, 3]),
             h.make_node("Mul", ["m", "ten"], ["m10"]), h.make_node("Unsqueeze", ["m10", "ax1"], ["m2"]),
             h.make_node("Add", ["state", "m2"], ["a"]), h.make_node("Unsqueeze", ["a", "ax1"], ["a3"]),
             h.make_node("Tile", ["a3", "reps"], ["at"]), h.make_node("Add", ["at", "ramp"], ["action"]),
             h.make_node("Identity", ["images.wrist"], ["w_unused"])]
    init = [h.make_tensor("ten", T.FLOAT, [], [10.0]), h.make_tensor("ax1", T.INT64, [1], [1]),
            h.make_tensor("reps", T.INT64, [3], [1, CH, 1]),
            h.make_tensor("ramp", T.FLOAT, [1, CH, 1], list(np.arange(CH, dtype=float)))]
    g = h.make_graph(nodes, "act", [h.make_tensor_value_info("state", T.FLOAT, [1, 6]),
                                    h.make_tensor_value_info("images.top", T.FLOAT, [1, 3, *RES]),
                                    h.make_tensor_value_info("images.wrist", T.FLOAT, [1, 3, *RES])],
                     [h.make_tensor_value_info("action", T.FLOAT, [1, CH, 6]), h.make_tensor_value_info("w_unused", T.FLOAT, None)], init)
    onnx.save(h.make_model(g, opset_imports=[h.make_opsetid("", 13)]), str(d / "act.onnx"))
    (d / "manifest.json").write_text(json.dumps(studio_manifest("onnx", "act.onnx"), indent=2))
    return d


def _obs_for(policy):
    frame = RNG.integers(0, 255, (1080, 1920, 3), np.uint8); pos = {n: 2048.0 for n in JOINTS}; pos["gripper"] = 2200.0
    roi = prep.default_act_roi("top")
    obs = pol.observation({"top": frame, "wrist": frame}, pos, LIM, policy.image_keys, policy.cams, {"top": roi, "wrist": roi})
    return obs, expected_chunk(obs["state"][0].astype(np.float64), obs["images.top"][0])


def _have_physicalai():
    try: import physicalai.inference  # noqa: F401
    except ImportError: return False
    return True


@pytest.mark.parametrize("runtime", ["direct", "physicalai"])
@pytest.mark.parametrize("backend", ["openvino", "onnx"])
def test_real_export_loads_and_runs(tmp_path, backend, runtime):
    if runtime == "physicalai" and not _have_physicalai(): pytest.skip("physicalai not installed (it is in Studio's Python)")
    model_dir = tmp_path / "models" / "0b5e-model-id"
    (make_openvino_export if backend == "openvino" else make_onnx_export)(model_dir / "exports" / backend)
    p = pol.Policy(model_dir, "CPU", "auto", runtime=runtime)                # a model dir: the export is found inside
    assert (p.backend, p.runtime, p.cams, p.image_keys) == (backend, runtime, ["top", "wrist"], ["images.top", "images.wrist"])
    obs, want = _obs_for(p)
    got = p.chunk(obs)
    assert got.shape == (NSTEPS, 6) and np.allclose(got, want, atol=1e-3), (got[0], want[0])
    p.reset(); p.close()


def test_find_exports_and_resolve(tmp_path, monkeypatch):
    studio = tmp_path / "studio_models"; root_models = tmp_path / "root_models"
    make_openvino_export(studio / "id-1" / "exports" / "openvino"); make_onnx_export(studio / "id-1" / "exports" / "onnx")
    make_openvino_export(root_models / "pick_red")
    (root_models / "front" / "red").mkdir(parents=True)                   # an anomaly model: not a policy package
    (root_models / "front" / "red" / "manifest.json").write_text('{"format": "anomalib"}')
    monkeypatch.setattr(pol, "STUDIO_MODELS", studio)
    ex = pol.find_exports([root_models], names=False)
    assert sorted((e["id"], e["backend"], e["source"] == "studio") for e in ex) == [("id-1", "onnx", True), ("id-1", "openvino", True), ("pick_red", "openvino", False)]
    assert all(e["chunk"] == NSTEPS and e["cams"] == ["top", "wrist"] for e in ex)
    assert pol.resolve(studio / "id-1")[1] == "openvino" and pol.resolve(studio / "id-1", "onnx")[1] == "onnx"
    with pytest.raises(FileNotFoundError): pol.resolve(studio / "id-1", "torch")


# ====================================================================== chunk execution
class CountPolicy:
    """chunk = the global time index of each action (state[0] carries the tick number in)."""
    def __init__(self, n=10, delay=0.0): self.n, self.delay, self.calls = n, delay, 0
    def chunk(self, obs):
        self.calls += 1
        if self.delay: time.sleep(self.delay)
        t = obs["t"]; return np.array([[t + i] * 6 for i in range(self.n)], np.float32)


def test_executor_chunk_ensemble_async():
    p = CountPolicy(10); ex = pol.Executor(p, "chunk"); out = [ex.step(lambda k=k: {"t": k})[0] for k in range(30)]
    assert out == list(range(30)) and p.calls == 6            # re-infers each time half the chunk (5) is left
    assert ex.stats()["inferences"] == 6 and ex.stats()["infer_ms_p50"] is not None
    p = CountPolicy(10); ex = pol.Executor(p, "ensemble"); out = [ex.step(lambda k=k: {"t": k})[0] for k in range(20)]
    assert np.allclose(out, range(20)) and p.calls == 20      # every tick; consistent predictions -> exact
    p = CountPolicy(10, delay=0.03); ex = pol.Executor(p, "async"); out = []
    for k in range(40):
        out.append(ex.step(lambda k=k: {"t": k})[0]); time.sleep(0.005)
    ex.close()
    assert np.allclose(out, range(40)), out                   # aligned to the frame each chunk was computed from
    lag = pol.Executor(CountPolicy(10), "chunk", lerp=5); lag.step(lambda: {"t": 0})
    lag.q = lag.q + 100                                       # stale queue: the new chunk is blended in, not jumped to
    a = [lag.step(lambda: {"t": 0})[0] for _ in range(8)]
    assert a[0] > 100 and a[-1] < 100


# ====================================================================== the pick
def test_stall_handoff_then_wp1(tmp_path):
    fp = FakePolicy(); ap, A, C = act_pick(fp, tmp=tmp_path, max_s=10, retries=0)
    g_lock, info = ap.run()
    assert A.held and C.holding() and abs(A.cmd["shoulder_lift"] - 1500) < 1     # carried to WP1, still held
    at = info["attempts"]
    assert len(at) == 1 and at[0]["result"] == "grasp" and at[0]["held_at_wp1"] and info["ok"]
    assert g_lock < A.width - collect.HOLD_GAP and len(fp.calls) > 3                # squeezing past the object
    assert at[0]["t_s"] < 10 and at[0]["hz"] and abs(at[0]["hz"] - pol.FPS) < 1      # ran at the dataset rate
    for k in ("infer_ms_p50", "infer_ms_p95", "obs_ms_p50", "obs_ms_p95"): assert info[k] is not None and info[k] >= 0


def test_timeout_retry_clean_failure(tmp_path):
    ap, A, C = act_pick(FakePolicy(close=False), tmp=tmp_path, max_s=3, retries=2)
    with pytest.raises(pol.PickFailed) as e: ap.run()
    assert [a["result"] for a in e.value.info["attempts"]] == ["timeout"] * 3 and e.value.retry
    assert "3x" in str(e.value) and not A.held and A.mode == "hold"
    # missed grasp: closes on nothing -> retried; a policy that grasps on the 2nd try succeeds
    A = SimArm(OBJ, slip=True); ap, A, C = act_pick(FakePolicy(), arm_=A, tmp=tmp_path / "b", max_s=3, retries=1)
    with pytest.raises(pol.PickFailed, match="closed on nothing") as e: ap.run()
    assert [(a["result"], a["held_at_wp1"]) for a in e.value.info["attempts"]] == [("grasp", False)] * 2


def test_stop_mid_policy(tmp_path):
    A = SimArm(OBJ); clk = SimClock(A, stop_at=0.5)
    ap, A, C = act_pick(FakePolicy(), arm_=A, tmp=tmp_path, clock=clk, max_s=10)
    with pytest.raises(arm.Stopped): ap.run()
    assert clk.frozen is not None and clk.frozen["shoulder_pan"] != OBJ["shoulder_pan"]     # stopped while moving
    for _ in range(20): A.tick()
    assert A.goal == clk.frozen and A.writes_after_stop == 0 and A.mode == "hold"   # nothing written after STOP


def test_blocked_and_stalled_camera(tmp_path):
    A = SimArm(OBJ, follow=False)                              # the arm doesn't move: something in the way
    ap, A, C = act_pick(FakePolicy(), arm_=A, tmp=tmp_path, max_s=10, retries=2)
    with pytest.raises(pol.PickFailed, match="BLOCKED") as e: ap.run()
    assert not e.value.retry
    ap, A, C = act_pick(FakePolicy(), tmp=tmp_path / "c", max_s=10)
    ap.S.frame_t = {"top": time.time() - 5, "wrist": time.time()}
    with pytest.raises(pol.PickFailed, match="camera top stalled") as e: ap.run()
    assert not e.value.retry


def test_targets_bounded(tmp_path):
    """A wild action can't become a long jump: body targets stay within MAX_LEAD_DEG of the arm, inside the limits."""
    ap, A, C = act_pick(FakePolicy(), tmp=tmp_path)
    t = ap._target(np.array([100, -100, 100, -100, 100, 0], np.float32))
    lead = pol.MAX_LEAD_DEG * arm.TICK_PER_DEG
    assert all(abs(t[n] - A.pos[n]) <= lead + 1e-6 for n in arm.BODY) and t["gripper"] == LIM["gripper"][0]


# ====================================================================== the whole autonomous cycle in the sorter
def run_sorter(world, policy, n=1, **cfg):
    A, C, T, root = sim_test.make(world)
    T.flags = {"grasped": False, "next": threading.Event()}  # sim_test's helper flags always say "grasped"
    T.S.frames.update({c: RNG.integers(0, 255, (1080, 1920, 3), np.uint8) for c in ("top", "wrist")}); T.S.live.append("top")
    T.cfg.update({"pick": "act", "act_model": "fake-export", **cfg})
    sorter.time = sim_test.NOSLEEP; collect.time = sim_test.NOSLEEP
    loads = []
    pol_load = pol.load
    pol.load = lambda path, device, backend: loads.append((path, device, backend)) or policy

    class FakeArmClock:                                      # sim_test's FakeArm moves instantly on goto()
        t = 0.0
        def now(self): return self.t
        def sleep_until(self, t):
            if t > self.t: self.t = t; A.goto(dict(A.goal), 1.0)
    T.clock = FakeArmClock()
    stop = threading.Event()

    def helper():
        while not stop.is_set():
            if T.state == "help": T.flags["next"].set()
            time.sleep(0.005)
    threading.Thread(target=helper, daemon=True).start()
    try: T.run(n)
    finally: pol.load = pol_load; stop.set()
    lp = root / "runs" / T.session / "log.jsonl"
    logs = [json.loads(l) for l in open(lp)] if lp.exists() else []
    pan = [m["shoulder_pan"] for m in A.moves if m["shoulder_pan"] in (1000, 3000)]
    return T, logs, ({1000: "good", 3000: "defect"}.get(pan[-1]) if pan else None), loads


def WorldPolicy(close=True):
    """Against sim_test's FakeArm (same limits as LIM): the object is between the fingers at the start pose."""
    return FakePolicy(target={**{n: 2048 for n in JOINTS}, "gripper": 2800}, close=close)


def test_full_cycle_act_pick_inspect_sort():
    T, logs, plate, loads = run_sorter(sim_test.World(), WorldPolicy())
    assert loads == [("fake-export", "GPU", "auto")] and T.state == "done", T.msg
    r = logs[0]
    assert r["pick"] == "act" and r["verdict"] == "good" and plate == "good", r["reason"]
    assert r["act"]["n_attempts"] == 1 and r["act"]["attempts"][0]["result"] == "grasp"
    assert r["act_infer_ms_p50"] is not None and r["act_infer_ms_p95"] >= r["act_infer_ms_p50"] and r["t_pick_s"] >= 0
    assert T.history[0]["act_infer_ms_p50"] == r["act_infer_ms_p50"]
    T, logs, plate, _ = run_sorter(sim_test.World([(1, 0, 0)]), WorldPolicy())
    assert logs[0]["verdict"] == "defect" and plate == "defect"


def test_full_cycle_pick_fails_asks_for_help_and_continues():
    T, logs, plate, _ = run_sorter(sim_test.World(), WorldPolicy(close=False), n=2, act_max_s=2, act_retries=1)
    assert T.state == "done" and len(logs) == 2 and plate is None, T.msg        # never crashed, both cycles logged
    for r in logs:
        assert r["verdict"] == "aborted" and r["pick_failed"] and "2x" in r["reason"] and r["act"]["n_attempts"] == 2
    assert T.totals["aborted"] == 2


def test_http_api(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    A, C, T, root = sim_test.make(sim_test.World())
    make_openvino_export(root / "models" / "pick_red" / "exports" / "openvino")
    monkeypatch.setattr(pol, "STUDIO_MODELS", tmp_path / "no_studio_here")
    app = FastAPI(); router = sorter.make_router(T.S, A, T.ctx, C, T.DATA, root); app.include_router(router)
    router.sorter.cfg_path = tmp_path / "s.json"
    c = TestClient(app)
    for v, sd in sim_test.FACEMAP.items(): c.post(f"/sort/faces?view={v}&sides={','.join(sd)}")
    s = c.post("/sort/cfg?pick=act").json(); assert s["cfg"]["pick"] == "act"
    r = c.post("/sort/start?n=1").json(); assert "choose a policy export" in r["msg"], r["msg"]
    ex = c.get("/sort/policies").json()
    assert [e["id"] for e in ex["exports"]] == ["pick_red"] and "CPU" in ex["devices"]
    path = ex["exports"][0]["path"]
    s = c.post("/sort/cfg", params={"act_model": path, "act_device": "CPU", "act_mode": "ensemble", "act_max_s": 15,
                                    "act_retries": 1, "act_speed": 0.5}).json()["cfg"]
    assert (s["act_model"], s["act_device"], s["act_mode"], s["act_max_s"], s["act_retries"], s["act_speed"]) == (path, "CPU", "ensemble", 15, 1, 0.5)
    s = c.post("/sort/cfg?act_device=TPU&act_mode=fast&act_speed=3").json()["cfg"]         # invalid values ignored
    assert (s["act_device"], s["act_mode"], s["act_speed"]) == ("CPU", "ensemble", 0.5)
    r = c.post("/sort/start?n=1").json(); assert "needs camera top" in r["msg"], r["msg"]     # sim_test has no top cam
    st = c.get("/sort/status").json(); assert "act" in st and st["act"]["loaded"] is None
