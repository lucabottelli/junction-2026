"""The tape-spot 90 deg re-grip (collect.place_on_tape / grip_on_tape / regrip, the optional taught `tape_regrip`
slot, the step-by-step diagnosis of "Test re-grip", the retry and the reasons the sorter logs).

    .venv/bin/python -m pytest tests/test_regrip.py -q                 (fast part, in-process, a few seconds)
    .venv/bin/python -m pytest tests/test_regrip.py -q -m slow         (the same on the mock station, ~3 min)

The fast part drives the real Collector / Sorter with a small fake arm (instant moves, a gripper that stalls on the
brick, a brick that is only gripped where the test says the jaws straddle it, e.g. only at the taught pose, the way
the SO-101's fixed jaw makes the computed turn miss on the real arm). The slow part runs the real server in mock mode.
"""
import json, sys, threading, time, types
from pathlib import Path

import numpy as np
import pytest

AC = Path(__file__).resolve().parents[1] / "anomaly_capture"
sys.path.insert(0, str(AC))
import sim_test  # noqa: E402  (stubs the servo bus when lerobot is missing; the sorter's fake world)
import arm, collect, sorter  # noqa: E402
from arm import JOINTS, TICK_PER_DEG  # noqa: E402

GMIN, GMAX, WIDTH = 1000, 3000, 1800        # gripper ticks: closed, open, where the fingers stall on the brick
TAPE_LIFT = 2048                            # shoulder_lift at table height (tape_place); tape_above is at 1500
R90 = int(round(90 * TICK_PER_DEG))


def wp(name, **kw):
    p = {n: 2048 for n in JOINTS}; p.update(kw); return {"name": name, "pos": p, "capture": False}


def routines(place_roll=1000, regrip=None, regrip_above=None):
    r = {"home": {"waypoints": [wp("home", gripper=2800)]}, "wp1": {"waypoints": [wp("wp1", shoulder_lift=1500)]},
         "wp2": {"waypoints": [wp("wp2-1", shoulder_pan=2300)]},
         "tape_above": {"waypoints": [wp("tape_above", shoulder_pan=2600, shoulder_lift=1500, wrist_roll=place_roll)]},
         "tape_place": {"waypoints": [wp("tape_place", shoulder_pan=2600, wrist_roll=place_roll)]}}
    if regrip: r["tape_regrip"] = {"waypoints": [wp("tape_regrip", **regrip)]}
    if regrip_above: r["tape_regrip_above"] = {"waypoints": [wp("tape_regrip_above", **regrip_above)]}
    return r


class FakeArm:
    """Instant moves. The brick is held (fingers stall at WIDTH) until the gripper opens past it; it is gripped again
    when the gripper closes at table height and grip_ok(target) says the jaws straddle it. stop_at=k: STOP arrives
    during the k-th move (raises Stopped like arm.Arm.goto, the move is not finished). lag: following error (ticks)
    on shoulder_lift at table height while the gripper is wide open (a jaw sitting on the brick)."""
    def __init__(self, grip_ok=None, held=True):
        self.fl = {n: (0, 4095) for n in JOINTS}; self.fl["gripper"] = (GMIN, GMAX)
        self.cmd = {n: 2048.0 for n in JOINTS}; self.cmd["gripper"] = 1700.0
        self.pos = {n: int(v) for n, v in self.cmd.items()}; self.goal = dict(self.cmd)
        self.held = held
        if held: self.pos["gripper"] = WIDTH
        self.mode = "hold"; self.stop = threading.Event(); self.running = False; self.io = threading.Lock(); self.f = True
        self.moves = []; self.speeds = []; self.grip_ok = grip_ok or (lambda tgt: True); self.stop_at = None; self.lag = 0
        self.gate = None                      # threading.Event: every move waits for it (lets a test act mid-motion)

    def set_mode(self, m, force=False): self.mode = m

    def goto(self, tgt, speed):
        if self.stop_at is not None and len(self.moves) + 1 >= self.stop_at: self.stop.set()
        if self.stop.is_set(): raise arm.Stopped()
        self.moves.append({n: float(tgt[n]) for n in JOINTS}); self.speeds.append(speed)
        g = tgt["gripper"]; low = tgt["shoulder_lift"] >= TAPE_LIFT - 10
        if self.held and g >= WIDTH + 50: self.held = False
        elif not self.held and g < WIDTH and self.cmd["gripper"] >= WIDTH and low and self.grip_ok(tgt): self.held = True   # closing from open
        self.cmd = {n: float(tgt[n]) for n in JOINTS}; self.goal = dict(self.cmd)
        self.pos = {n: int(round(tgt[n])) for n in JOINTS}
        if self.held: self.pos["gripper"] = max(int(g), WIDTH)
        if low and g > 2500 and self.lag: self.pos["shoulder_lift"] -= self.lag
        return 0


def make(tmp_path, R, A=None, **cfg):
    A = A or FakeArm(); tmp_path.mkdir(parents=True, exist_ok=True)
    img = np.full((48, 64, 3), 90, np.uint8)
    S = types.SimpleNamespace(live=["wrist", "top"], frames={"wrist": img, "top": img}, cams={}, averaged=None)
    ctx = types.SimpleNamespace(R={"routines": R}, capture=lambda *a, **k: 1)
    C = collect.Collector(S, A, ctx, tmp_path / "training_images", tmp_path)
    C.cfg_path = tmp_path / "c.json"; C.cfg.update({"speed": 1.0, **cfg})
    return A, C


@pytest.fixture(autouse=False)
def fast(monkeypatch):
    monkeypatch.setattr(collect, "time", sim_test.NOSLEEP)


def descent(A):
    """(approach move, descent move) of the LAST turned grip attempt: the gripper wide open, going down to the table."""
    i = max(k for k, m in enumerate(A.moves) if m["shoulder_lift"] >= TAPE_LIFT - 10 and m["gripper"] >= 2700
            and A.moves[k - 1]["shoulder_lift"] < TAPE_LIFT - 10)
    return A.moves[i - 1], A.moves[i]


def diag_dir(tmp_path):
    ds = sorted((tmp_path / "runs").glob("regrip_test_*")); assert ds, "no runs/regrip_test_* folder"
    return ds[-1]


# ================================================================== untaught: the old computed turn (plus safer set-down)
def test_untaught_uses_the_computed_turn(tmp_path, fast):
    A, C = make(tmp_path, routines())
    C.test_wp2("regrasp")
    assert C.state == "done" and C.ok and A.held, C.msg
    up, down = descent(A)
    assert down["wrist_roll"] == 1000 + 90 * TICK_PER_DEG and down["shoulder_pan"] == 2600 and down["shoulder_lift"] == TAPE_LIFT
    assert up["wrist_roll"] == down["wrist_roll"] and up["shoulder_lift"] == 1500          # turned while up, then straight down
    assert C.regrip_info["pose"] == "computed" and C.regrip_info["attempts"] == 1
    # safer set-down: at table height the gripper opens only RELEASE_TICKS past the brick, fully only once lifted
    rel = next(m for m in A.moves if m["gripper"] > WIDTH)
    assert rel["shoulder_lift"] == TAPE_LIFT and rel["gripper"] == WIDTH + collect.RELEASE_TICKS
    first_wide = next(m for m in A.moves if m["gripper"] >= 2800)
    assert first_wide["shoulder_lift"] == 1500
    # the lift off the tape starts slowly (a few mm), nothing is turned before the arm is up
    i = A.moves.index(rel)
    assert TAPE_LIFT > A.moves[i + 1]["shoulder_lift"] > 1500 and A.speeds[i + 1] < A.speeds[i + 2]
    assert all(m["wrist_roll"] == 1000 for m in A.moves[:i + 3])


# ================================================================== taught tape_regrip
def test_taught_regrip_pose_is_used_with_a_straight_descent(tmp_path, fast):
    taught = {"shoulder_pan": 2611, "shoulder_lift": TAPE_LIFT, "elbow_flex": 2040, "wrist_flex": 2060, "wrist_roll": 2048 - R90}
    ok_at = lambda t: t["shoulder_pan"] == 2611                 # the jaws only straddle the brick at the taught pose
    A, C = make(tmp_path, routines(place_roll=2048, regrip=taught), FakeArm(ok_at))
    st = C.regrip_status()
    assert st["taught"] and not st["above"] and st["problem"] is None and round(st["turn_deg"]) == -90 and st["pan_deg"] == 1.0
    C.test_wp2("regrasp")
    assert C.state == "done" and A.held, C.msg
    up, down = descent(A)
    assert all(down[n] == taught[n] for n in taught)
    # derived approach: the taught pose at tape_above's height (lift/elbow/wrist flex), with the taught pan + roll
    assert up["shoulder_pan"] == 2611 and up["wrist_roll"] == taught["wrist_roll"] and up["shoulder_lift"] == 1500
    assert C.regrip_info["pose"] == "taught" and C.regrip_info["attempts"] == 1


def test_taught_regrip_above_is_used_with_the_regrip_roll(tmp_path, fast):
    taught = {"shoulder_pan": 2611, "shoulder_lift": TAPE_LIFT, "wrist_roll": 1000 + R90}
    above = {"shoulder_pan": 2612, "shoulder_lift": 1400, "elbow_flex": 2100, "wrist_roll": 1000 + R90 + 300}
    A, C = make(tmp_path, routines(regrip=taught, regrip_above=above), FakeArm(lambda t: t["shoulder_pan"] == 2611))
    C.test_wp2("regrasp")
    assert C.state == "done", C.msg
    up, down = descent(A)
    assert (up["shoulder_pan"], up["shoulder_lift"], up["elbow_flex"]) == (2612, 1400, 2100)
    assert up["wrist_roll"] == down["wrist_roll"] == 1000 + R90                      # no roll during the descent


def test_computed_turn_misses_where_the_taught_pose_grips(tmp_path, fast):
    """The fixed-jaw case: the brick lies off the wrist axis, so only the taught pose straddles it."""
    A, C = make(tmp_path, routines(), FakeArm(lambda t: t["shoulder_pan"] == 2611))
    C.test_wp2("regrasp")
    assert C.state == "error" and not A.held
    assert "attempt 1: closed: the fingers closed on nothing" in C.msg and "attempt 2: closed" in C.msg, C.msg


@pytest.mark.parametrize("turn_deg,ok", [(90, True), (-90, True), (94, True), (-86, True), (96, False), (100, False), (75, False),
                                         (270, False), (0, False), (180, False), (-180, False), (45, False)])
def test_taught_turn_must_be_about_90(tmp_path, fast, turn_deg, ok):
    place_roll = 2048
    taught = {"shoulder_pan": 2600, "shoulder_lift": TAPE_LIFT, "wrist_roll": place_roll + int(round(turn_deg * TICK_PER_DEG))}
    A, C = make(tmp_path, routines(place_roll=place_roll, regrip=taught))
    assert (C.regrip_problem() is None) == ok
    C.test_wp2("regrasp")
    if ok: assert C.state == "done", C.msg
    else:
        assert C.state == "error" and "not a clean 90° turn" in C.msg and "wrist roll" in C.msg and A.moves == [], C.msg   # before moving
        assert not list((tmp_path / "runs").glob("regrip_test_*"))


# ================================================================== roll limits
def test_roll_near_the_upper_limit_turns_the_other_way(tmp_path, fast):
    A, C = make(tmp_path, routines(place_roll=4095 - 500))
    C.test_wp2("regrasp")
    assert C.state == "done", C.msg
    _, down = descent(A)
    assert down["wrist_roll"] == pytest.approx(4095 - 500 - 90 * TICK_PER_DEG)
    assert all(0 <= m["wrist_roll"] <= 4095 for m in A.moves)


def test_roll_with_no_room_either_way_is_refused(tmp_path, fast):
    A, C = make(tmp_path, routines(place_roll=2048))
    A.fl["wrist_roll"] = (1500, 2600)                       # less than 90 deg each way
    C.test_wp2("regrasp")
    assert C.state == "error" and "no room for a 90° turn" in C.msg, C.msg


def test_taught_roll_outside_the_limits_is_clamped(tmp_path, fast):
    A, C = make(tmp_path, routines(place_roll=3000, regrip={"shoulder_pan": 2600, "shoulder_lift": TAPE_LIFT, "wrist_roll": 3000 + R90}))
    A.fl["wrist_roll"] = (0, 3900)                          # 3000 + 1024 = 4024 > 3900
    C.test_wp2("regrasp")
    assert all(m["wrist_roll"] <= 3900 for m in A.moves)


# ================================================================== retry policy
def test_failed_grip_lets_go_lifts_and_retries_slower(tmp_path, fast):
    n = {"tries": 0}

    def second_time(t):
        n["tries"] += 1; return n["tries"] > 1
    A, C = make(tmp_path, routines(), FakeArm(second_time))
    C.test_wp2("regrasp")
    assert C.state == "done" and A.held and "needed 2 attempts" in C.msg, C.msg
    info = C.regrip_info
    assert info["attempts"] == 2 and info["held"] and info["failed"][0].startswith("closed: the fingers closed on nothing")
    # after the miss: reopened wide AT TABLE HEIGHT before lifting (never drags a half-gripped brick up)
    i = next(k for k, m in enumerate(A.moves) if m["gripper"] == GMIN)              # first close
    reopen = next(m for m in A.moves[i + 1:] if m["gripper"] >= 2700)
    assert reopen["shoulder_lift"] == TAPE_LIFT
    # the retry is not a blind repeat: it comes down and closes slower
    downs = [k for k, m in enumerate(A.moves) if m["shoulder_lift"] == TAPE_LIFT and m["gripper"] >= 2700 and A.moves[k - 1]["shoulder_lift"] < TAPE_LIFT]
    assert len(downs) == 2 and A.speeds[downs[1]] < A.speeds[downs[0]]
    closes = [k for k, m in enumerate(A.moves) if m["gripper"] == GMIN and A.moves[k - 1]["gripper"] >= 2700]   # close from wide open
    assert len(closes) == 2
    assert A.speeds[closes[1]] < A.speeds[closes[0]]
    # the turn direction is the same for both attempts (coverage.py assumes it)
    assert A.moves[downs[0]]["wrist_roll"] == A.moves[downs[1]]["wrist_roll"]
    rows = [json.loads(l) for l in open(diag_dir(tmp_path) / "steps.jsonl")]
    assert "reopened" in [r["step"] for r in rows]


def test_slipping_out_while_lifting_is_reported(tmp_path, fast):
    class Slippy(FakeArm):
        def goto(self, tgt, speed):
            r = super().goto(tgt, speed)
            if self.held and tgt["shoulder_lift"] < TAPE_LIFT - 10 and tgt["gripper"] < WIDTH and len(self.moves) > 8:
                self.held = False; self.pos["gripper"] = int(tgt["gripper"])     # falls out on the way up
            return r
    A, C = make(tmp_path, routines(), Slippy())
    C.test_wp2("regrasp")
    assert C.state == "error" and "lifted: the brick slipped out" in C.msg, C.msg
    # the retry after a slip does not open at height (holding() may have been a false negative): it goes down closed to
    # just above the brick (where the slow part of the lift starts) and opens there
    lifts = [k for k, m in enumerate(A.moves) if m["shoulder_lift"] == 1500 and m["gripper"] < WIDTH and k > 8]
    k0 = lifts[0]                                                  # end of attempt 1: up, closed, empty
    nxt = A.moves[k0 + 1:k0 + 4]
    assert nxt[0]["shoulder_lift"] == 1500 and nxt[0]["gripper"] < WIDTH                       # turn while up, still closed
    assert 1500 < nxt[1]["shoulder_lift"] < TAPE_LIFT and nxt[1]["gripper"] < WIDTH             # down to just above it
    assert nxt[2]["shoulder_lift"] == nxt[1]["shoulder_lift"] and nxt[2]["gripper"] >= 2700     # opens only there


def test_sorter_reports_the_failed_substep_in_the_log():
    w = sim_test.World(); w.regrip_ok = False
    T, logs, plate = sim_test.run(w)
    r = logs[0]["reason"]
    assert logs[0]["verdict"] == "aborted" and logs[0]["on_tape"] and plate is None
    assert r.startswith("re-grip failed twice (attempt 1: closed: the fingers closed on nothing") and "attempt 2: closed" in r, r
    rl = logs[0]["regrip_log"]
    assert rl and rl[-1]["pose"] == "computed" and rl[-1]["attempts"] == 2 and not rl[-1]["held"], rl


def test_contact_detection_is_off_by_default_and_works_when_set(tmp_path, fast):
    A, C = make(tmp_path, routines()); A.lag = int(8 * TICK_PER_DEG)       # 8 deg short at the bottom (jaw on the brick)
    C.test_wp2("regrasp")
    assert C.state == "done", C.msg                                         # off: lowers, closes, grips as before
    A, C = make(tmp_path, routines(), regrip_contact_deg=5.0); A.lag = int(8 * TICK_PER_DEG)
    C.test_wp2("regrasp")
    assert C.state == "error" and "contact detection" in C.msg and not any(m["gripper"] == GMIN for m in A.moves), C.msg


# ================================================================== STOP in every sub-step
def test_stop_during_every_move_leaves_the_arm_holding(tmp_path, fast):
    A0, C0 = make(tmp_path / "ref", routines())
    C0.test_wp2("regrasp"); n = len(A0.moves); assert C0.state == "done" and n >= 12
    seen_subs = set()
    for k in range(1, n + 1):
        A, C = make(tmp_path / f"k{k}", routines()); A.stop_at = k
        C.test_wp2("regrasp")
        assert C.state == "stopped" and "STOPPED" in C.msg and not C.active and not A.running, (k, C.msg)
        assert len(A.moves) == k - 1                           # no motion after the STOP
        assert A.goal == A.cmd and A.mode == "hold"            # holds exactly where it was commanded
        if A.held: assert A.cmd["gripper"] < WIDTH             # still squeezing the brick: never dropped by a STOP
        s = json.loads((diag_dir(tmp_path / f"k{k}") / "summary.json").read_text())
        assert s["ok"] is False and s["result"].startswith("stopped after")
        seen_subs.add(C.sub)
    assert set(collect.REGRIP_PLAN[:-1]) <= seen_subs | {None}, seen_subs


# ================================================================== step mode
def run_bg(C, **kw):
    t = threading.Thread(target=C.test_wp2, args=("regrasp",), kwargs=kw, daemon=True); t.start(); return t


def wait_for(fn, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v: return v
        time.sleep(0.01)
    raise AssertionError("timeout")


def test_step_mode_advances_only_on_next(tmp_path, fast):
    A, C = make(tmp_path, routines())
    t = run_bg(C, step=True)
    for i, name in enumerate(collect.REGRIP_PLAN, 1):
        wait_for(lambda: C.diag and C.diag["waiting"] and C.diag["i"] == i)
        n = len(A.moves); time.sleep(0.25)
        assert len(A.moves) == n and C.diag["i"] == i and C.diag["waiting"]   # nothing moves without NEXT
        v = C.info["regrip"]; assert v["rows"][-1]["step"] == name and v["waiting"] and v["step_mode"]
        assert f"STEP {i}:" in C.msg and "N = next" in C.msg
        C.flags["next"].set()
    t.join(5); assert not t.is_alive()
    assert C.state == "done" and A.held, C.msg
    assert [r["step"] for r in C.diag["rows"]] == list(collect.REGRIP_PLAN)
    assert C.info["regrip"]["ok"] is True and not C.info["regrip"]["active"]


def test_step_mode_times_out_cleanly(tmp_path, fast):
    A, C = make(tmp_path, routines(), step_timeout_s=0.3)
    t = run_bg(C, step=True); t.join(5)
    assert not t.is_alive() and C.state == "stopped" and "no NEXT for 0.3 s" in C.msg, C.msg
    assert len(A.moves) == 1 and A.goal == A.cmd and A.held                 # stopped after step 1, still holding
    s = json.loads((diag_dir(tmp_path) / "summary.json").read_text())
    assert "no NEXT" in s["result"] and s["ok"] is False


def test_stop_while_waiting_in_step_mode(tmp_path, fast):
    A, C = make(tmp_path, routines())
    t = run_bg(C, step=True)
    for i in (1, 2):
        wait_for(lambda: C.diag and C.diag["waiting"] and C.diag["i"] == i)
        C.flags["next"].set()
    wait_for(lambda: C.diag["waiting"] and C.diag["i"] == 3)                  # released: brick on the tape, gripper a little open
    n = len(A.moves)
    C.stop_flag.set(); A.stop.set(); C.flags["next"].set()                   # = POST /collect/stop
    t.join(5); assert not t.is_alive()
    assert C.state == "stopped" and "after released" in C.msg and len(A.moves) == n and A.goal == A.cmd, C.msg


# ================================================================== diagnostic files
def test_diagnostic_files(tmp_path, fast):
    A, C = make(tmp_path, routines())
    C.test_wp2("regrasp")
    d = diag_dir(tmp_path)
    rows = [json.loads(l) for l in open(d / "steps.jsonl")]
    assert [r["step"] for r in rows] == list(collect.REGRIP_PLAN)
    for r in rows:
        assert set(r["target"]) == set(JOINTS) and set(r["actual"]) == set(JOINTS) and set(r["err_deg"]) == set(arm.BODY)
        assert {"cmd", "pos", "open_pct", "jaw_deg_from_closed"} <= set(r["gripper"]) and isinstance(r["holding"], bool)
        assert r["frames"] == [f"{r['i']:02d}_{r['step']}_wrist.jpg", f"{r['i']:02d}_{r['step']}_top.jpg"]
        for f in r["frames"]: assert (d / f).stat().st_size > 0
    by = {r["step"]: r for r in rows}
    assert by["lowered"]["holding"] and not by["lifted"]["holding"] and by["lifted_held"]["holding"]
    assert by["closed"]["gripper"]["stall"] == WIDTH and by["closed"]["gripper"]["stall_minus_first_grip_ticks"] == 0
    assert by["lowered_open"]["pose"] == "computed" and "lift_err_deg" in by["lowered_open"]
    s = json.loads((d / "summary.json").read_text())
    assert s["ok"] and s["started_holding"] and s["first_grip_width_ticks"] == WIDTH and s["regrip"]["attempts"] == 1
    assert C.info["regrip"]["dir"] == f"runs/{d.name}"


def test_no_diagnostics_outside_the_test(tmp_path, fast):
    A, C = make(tmp_path, routines())
    C.place_on_tape(1700.0, 2800.0); held, g, whys = C.regrip(2800.0)
    assert held and g is not None and whys == [] and C.sub == "lifted_held"
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").glob("regrip_test_*"))


def test_badly_taught_pose_is_refused_by_sort_and_collect_start():
    R = json.loads(json.dumps(sim_test.ROUTINES))
    tp = R["tape_place"]["waypoints"][0]["pos"]
    R["tape_regrip"] = {"waypoints": [{"name": "tape_regrip", "pos": {**tp, "wrist_roll": tp["wrist_roll"] + 2048}, "capture": False}]}
    A, C, T, root = sim_test.make(sim_test.World(), routines=R)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI(); app.include_router(sorter.make_router(T.S, A, T.ctx, C, T.DATA, root))
    cr = collect.make_router(T.S, A, T.ctx, T.DATA, root); cr.collector.cfg_path = root / "c2.json"
    app.include_router(cr); c = TestClient(app)
    for v, sd in sim_test.FACEMAP.items(): c.post(f"/sort/faces?view={v}&sides={','.join(sd)}")
    r = c.post("/sort/start?n=1").json()
    assert not r["active"] and "not a clean 90° turn" in r["msg"], r["msg"]
    c.post("/collect/cfg?phase2=true&regrasp=true")
    r = c.post("/collect/start?n=1&brick=x").json()
    assert not r["active"] and "not a clean 90° turn" in r["msg"] and r["regrip_pose"]["turn_deg"] == 180, r["msg"]


# ================================================================== coverage chirality with a taught pose
@pytest.mark.parametrize("taught_turn", [1, -1])
@pytest.mark.parametrize("sign", [1, -1])
def test_coverage_with_a_taught_pose_either_way(taught_turn, sign):
    """The taught pose may turn either way: the sorter still finds a defect under the fingers and confirms a clean one."""
    R = json.loads(json.dumps(sim_test.ROUTINES))
    tp = R["tape_place"]["waypoints"][0]["pos"]
    R["tape_regrip"] = {"waypoints": [{"name": "tape_regrip", "pos": {**tp, "wrist_roll": tp["wrist_roll"] + taught_turn * 1024}, "capture": False}]}
    for defect, verdict in (((0, 1, 0.2), "defect"), ((0, -1, -0.1), "defect"), (None, "good")):
        w = sim_test.World([defect] if defect else (), sign=sign)
        A, C, T, root = sim_test.make(w, routines=R)
        sorter.time = sim_test.NOSLEEP; collect.time = sim_test.NOSLEEP
        T.run(1)
        log = json.loads(open(root / "runs" / T.session / "log.jsonl").readline())
        assert log["verdict"] == verdict and log["regrasps"] == 1, log["reason"]
        assert log["regrip_log"][0]["pose"] == "taught"
        assert any(m["wrist_roll"] == max(0, tp["wrist_roll"] + taught_turn * 1024) for m in A.moves)


# ================================================================== the same on the mock station (slow)
def _mock_hw():
    import importlib.util
    key = "_regrip_mock_hw"
    if key in sys.modules: return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, AC / "mock_hw.py")
    m = importlib.util.module_from_spec(spec); sys.modules[key] = m; spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def regrip_station(tmp_path_factory):
    """A mock station whose routines also have a taught tape_regrip turned -90 deg (the computed turn would be +90)."""
    from conftest import MockServer
    m = _mock_hw(); root = tmp_path_factory.mktemp("regrip_root"); st = root / "settings"; st.mkdir(parents=True)
    R = m.mock_routines(); tp = R["tape_place"]["waypoints"][0]["pos"]
    R["tape_regrip"] = {"waypoints": [{"name": "tape_regrip", "pos": {**tp, "wrist_roll": tp["wrist_roll"] - 1024}, "capture": False}],
                        "created": "test"}
    (st / "routines.json").write_text(json.dumps(R, indent=1))
    srv = MockServer(root)
    srv.connect()
    yield srv, tp
    srv.stop()


def _feed(srv):
    """Hand a brick into the gripper at WP1 (mirror + the mock's human), then hold."""
    st = srv.get("/arm/status")["all"]["wp1"]["waypoints"][0]["pos"]
    srv.post("/arm/stop")
    srv.post("/mock/scene", reset=True)
    srv.post("/mock/leader", **{k: v for k, v in st.items() if k != "gripper"}, gripper=2950)
    srv.post("/arm/mode", m="mirror")
    srv.wait(lambda: srv.get("/mock/state")["active"]["state"] == "fed", 30, "brick fed")
    srv.post("/mock/leader", gripper=1500)
    srv.wait(lambda: srv.get("/mock/state")["active"]["state"] == "held", 20, "brick gripped")
    srv.post("/arm/mode", m="hold"); srv.post("/mock/human", mode="auto")


def _idle(srv, timeout=180):
    return srv.wait(lambda: (lambda c: c if not c["active"] else None)(srv.get("/collect/status")), timeout, "test finished")


@pytest.mark.slow
def test_mock_taught_regrip_step_by_step_with_retry_and_files(regrip_station):
    srv, tp = regrip_station
    c = srv.get("/collect/status")
    assert c["regrip_pose"] == {"taught": True, "above": False, "turn_deg": -90.0, "pan_deg": 0.0, "problem": None}
    _feed(srv)
    srv.post("/mock/fault", regrip_fail=1)                    # the first close slips: the retry must get it
    srv.post("/collect/test_wp2", kind="regrasp", step=True)
    pressed = set()

    def step():
        c = srv.get("/collect/status"); d = (c.get("info") or {}).get("regrip")
        if d and d["waiting"] and d["rows"][-1]["i"] not in pressed:
            pressed.add(d["rows"][-1]["i"]); srv.post("/collect/flag/next")
        return c if not c["active"] else None
    c = srv.wait(step, 300, "step-by-step re-grip test")
    assert c["state"] == "done" and c["ok"] and "needed 2 attempts" in c["msg"], c["msg"]
    assert srv.get("/mock/state")["active"]["state"] == "held"
    d = c["info"]["regrip"]; steps = [r["step"] for r in d["rows"]]
    assert steps[:4] == ["above_tape", "lowered", "released", "lifted"] and "reopened" in steps and steps[-1] == "lifted_held"
    assert len(pressed) == len(steps)                           # it waited for N at every step
    folder = srv.root / d["dir"]
    rows = [json.loads(l) for l in open(folder / "steps.jsonl")]
    lo = [r for r in rows if r["step"] == "lowered_open"]
    assert lo and all(r["pose"] == "taught" and r["target"]["wrist_roll"] == tp["wrist_roll"] - 1024 for r in lo)
    assert all(r["frames"] and (folder / r["frames"][0]).exists() for r in rows)
    assert json.loads((folder / "summary.json").read_text())["regrip"]["attempts"] == 2
    r = srv.http.get("/collect/regrip/frame", params={"i": rows[-1]["i"]})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    # STOP in the middle of the next step-by-step test: arm holds, brick stays where it is
    srv.post("/collect/test_wp2", kind="regrasp", step=True)
    at = lambda name: (lambda d: d and d["active"] and d["waiting"] and d["rows"][-1]["step"] == name)(
        (srv.get("/collect/status").get("info") or {}).get("regrip"))
    srv.wait(lambda: at("above_tape"), 60, "step 1 (above the tape)")
    srv.post("/collect/flag/next")
    srv.wait(lambda: at("lowered"), 60, "step 2 (lowered)")
    s = srv.post("/collect/stop"); assert "STOPPED" in s["msg"]
    c = _idle(srv, 20)
    assert c["state"] == "stopped" and not c["info"]["regrip"]["active"], c["msg"]
    a = srv.get("/arm/status"); assert a["mode"] == "hold" and not a["running"]
    p1 = a["pos"]; time.sleep(0.5); p2 = srv.get("/arm/status")["pos"]
    assert max(abs(p1[k] - p2[k]) for k in p1) < 25
    assert srv.get("/mock/state")["active"]["state"] == "held"   # still squeezing the brick on the tape


@pytest.mark.slow
def test_mock_untaught_uses_the_computed_turn(regrip_station):
    srv, tp = regrip_station
    srv.post("/arm/wp/0/delete", routine="tape_regrip")
    assert srv.get("/collect/status")["regrip_pose"]["taught"] is False
    _feed(srv)
    srv.post("/collect/test_wp2", kind="regrasp")
    c = _idle(srv)
    assert c["state"] == "done" and c["ok"], c["msg"]
    rows = [json.loads(l) for l in open(srv.root / c["info"]["regrip"]["dir"] / "steps.jsonl")]
    lo = [r for r in rows if r["step"] == "lowered_open"]
    assert lo[0]["pose"] == "computed" and lo[0]["target"]["wrist_roll"] == tp["wrist_roll"] + 1024
    assert srv.get("/mock/state")["active"]["state"] == "held"


# ================================================================== review fixes (regressions)
@pytest.mark.parametrize("roll,pan,ok,names", [(90, 0, True, ""), (90, 4.9, True, ""), (92, 2.5, True, ""), (-87, -1.5, True, ""),
                                               (90, 6, False, "shoulder pan"), (93, 3, False, "wrist roll"), (75, 0, False, "wrist roll")])
def test_tolerance_is_the_coverage_margin_roll_plus_pan(tmp_path, fast, roll, pan, ok, names):
    """coverage.py credits each re-grip as exactly 90 deg with margin_deg: | |roll| - 90 | + |pan| must stay within it."""
    assert collect.TURN_TOL_DEG == collect.COV_DEFAULTS["margin_deg"] == 5.0
    taught = {"shoulder_pan": 2600 + pan * TICK_PER_DEG, "shoulder_lift": TAPE_LIFT, "wrist_roll": 2048 + roll * TICK_PER_DEG}
    A, C = make(tmp_path, routines(place_roll=2048, regrip=taught))
    prob = C.regrip_problem()
    assert (prob is None) == ok, prob
    if not ok:
        assert names in prob and "allowed 5°" in prob and C.regrip_status()["problem"] == prob
        if names == "shoulder pan": assert "wrist roll" not in prob
        C.test_wp2("regrasp"); assert C.state == "error" and A.moves == []


class FlakyWorld(sim_test.World):
    """The first re-grip attempt misses, the second succeeds (the brick may have moved in between)."""
    @property
    def regrip_ok(self):
        self._reads = getattr(self, "_reads", 0) + 1; return self._reads > 2     # attempt 1 = close + squeeze

    @regrip_ok.setter
    def regrip_ok(self, v): pass


def test_sorter_never_calls_good_after_a_retried_regrip():
    T, logs, plate = sim_test.run(FlakyWorld())
    lg = logs[0]
    assert lg["verdict"] == "defect" and plate == "defect" and lg.get("orientation_unverified"), lg["reason"]
    assert "needed a retry, orientation not verified" in lg["reason"] and lg["regrip_log"][0]["attempts"] == 2
    T, logs, plate = sim_test.run(FlakyWorld([(0, 1, 0.2)]))           # a defect found after the retry still counts
    assert logs[0]["verdict"] == "defect" and "defect seen" in logs[0]["reason"]
    T, logs, plate = sim_test.run(sim_test.World())                    # no retry: GOOD as before
    assert logs[0]["verdict"] == "good" and not logs[0].get("orientation_unverified")


def test_collection_marks_photos_after_a_retried_regrip(tmp_path, fast):
    n = {"tries": 0}

    def second_time(t):
        n["tries"] += 1; return n["tries"] > 1
    A, C = make(tmp_path, routines(), FakeArm(second_time))
    views = []
    C.ctx.capture = lambda cams, label, brick, view, idx, session, kind="inhand": views.append(view) or 1
    C.cfg.update({"regrasp": True, "release_cams": []})
    n_img, on_tape = C.phase2(C.wps("wp1")[0]["pos"], C.wps("wp2"), 1700.0, 2800.0, "good", "x", 1)
    assert not on_tape and views == ["wp2-1", "wp2-1_r90retry"], views
    A, C = make(tmp_path / "b", routines()); views.clear()
    C.ctx.capture = lambda cams, label, brick, view, idx, session, kind="inhand": views.append(view) or 1
    C.cfg.update({"regrasp": True, "release_cams": []})
    C.phase2(C.wps("wp1")[0]["pos"], C.wps("wp2"), 1700.0, 2800.0, "good", "x", 1)
    assert views == ["wp2-1", "wp2-1_r90"], views


def test_poses_are_frozen_for_the_whole_test(tmp_path, fast):
    """Teaching tape_regrip (other direction) while the test is paused must not change the turn mid-cycle."""
    R = routines(place_roll=2048)
    A, C = make(tmp_path, R)
    t = run_bg(C, step=True)
    for i in (1, 2, 3):
        wait_for(lambda: C.diag and C.diag["waiting"] and C.diag["i"] == i); C.flags["next"].set()
    wait_for(lambda: C.diag["waiting"] and C.diag["i"] == 4)
    R["tape_regrip"] = {"waypoints": [wp("tape_regrip", shoulder_pan=2600, wrist_roll=2048 - R90)]}   # taught meanwhile
    del R["tape_place"]                                                                                # ... and deleted
    while t.is_alive():
        if C.diag["waiting"]: C.flags["next"].set()
        time.sleep(0.01)
    assert C.state == "done", C.msg
    _, down = descent(A)
    assert down["wrist_roll"] == 2048 + R90 and C.regrip_info["pose"] == "computed"               # the frozen poses


def test_waypoint_edits_are_refused_while_the_arm_runs(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    rf = tmp_path / "routines.json"; rf.write_text(json.dumps(routines()))
    monkeypatch.setattr(arm, "ROUTINES", rf)
    S = types.SimpleNamespace(live=[], frames={}, cams={})
    r, A, ctx = arm.make_router(S, tmp_path, {}, lambda *a: None)
    A.f = object(); A.pos = {n: 2048 for n in JOINTS}; A.cmd = {n: 2048.0 for n in JOINTS}; A.mode = "hold"; A.running = True
    app = FastAPI(); app.include_router(r); c = TestClient(app)
    before = rf.read_text()
    for url in ("/arm/wp/add?routine=tape_regrip&single=true", "/arm/wp/0/update?routine=tape_place",
                "/arm/wp/0/delete?routine=tape_place", "/arm/wp/0/up?routine=tape_place"):
        m = c.post(url).json()["msg"]; assert "STOP it first" in m, (url, m)
    assert rf.read_text() == before and ctx.R["routines"]["tape_place"]["waypoints"]
    A.running = False
    assert "saved" in c.post("/arm/wp/add?routine=tape_regrip&single=true").json()["msg"]


def test_sorter_without_usable_regrip_poses_fails_safe_without_moving_to_the_tape():
    """A pose problem found at the start of the cycle (e.g. no room for 90 deg either way) is no error that stops the run:
    no re-grip, the unseen faces can't be verified -> defect plate, with the reason."""
    w = sim_test.World()
    A, C, T, root = sim_test.make(w)
    A.fl["wrist_roll"] = (900, 1100)                                  # tape_place roll 1000: no room either way
    sorter.time = sim_test.NOSLEEP; collect.time = sim_test.NOSLEEP
    T.run(1)
    lg = json.loads(open(root / "runs" / T.session / "log.jsonl").readline())
    assert T.state == "done" and lg["verdict"] == "defect" and "re-grip not possible" in lg["reason"], (T.state, T.msg)
    assert not any(m["shoulder_pan"] == 2600 for m in A.moves)       # never went to the tape spot


def test_sorter_regrip_error_after_release_is_aborted_on_tape(monkeypatch):
    w = sim_test.World()
    A, C, T, root = sim_test.make(w)
    orig = C.grip_on_tape

    def boom(*a, **k): raise RuntimeError("BLOCKED: arm is 20 deg away from the waypoint")
    monkeypatch.setattr(C, "grip_on_tape", boom)
    sorter.time = sim_test.NOSLEEP; collect.time = sim_test.NOSLEEP
    threading.Thread(target=lambda: [T.flags["next"].set() or time.sleep(0.01) for _ in range(500)], daemon=True).start()
    T.run(1)
    lg = json.loads(open(root / "runs" / T.session / "log.jsonl").readline())
    assert lg["verdict"] == "aborted" and lg["on_tape"] and "BLOCKED" in lg["reason"] and T.state == "done", (T.state, lg["reason"])


def test_summary_does_not_carry_the_previous_runs_regrip(tmp_path, fast):
    n = {"tries": 0}

    def second_time(t):
        n["tries"] += 1; return n["tries"] > 1
    A, C = make(tmp_path, routines(), FakeArm(second_time))
    C.test_wp2("regrasp"); assert C.regrip_info["attempts"] == 2
    A.stop_at = len(A.moves) + 2                                      # next test: STOP before any re-grip
    C.test_wp2("regrasp")
    s = json.loads((diag_dir(tmp_path) / "summary.json").read_text())
    assert s["regrip"] is None and s["result"].startswith("stopped after")


def test_stop_at_lowered_reports_still_holding_and_frames_by_camera(tmp_path, fast):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    A, C = make(tmp_path, routines())
    A.stop_at = 3                                                     # during the release, brick still held
    C.test_wp2("regrasp")
    v = C.diag_view()
    assert C.state == "stopped" and v["holding"] is True and v["ok"] is False and A.held
    assert json.loads((diag_dir(tmp_path) / "summary.json").read_text())["holding_at_end"] is True
    # frames: never another camera's frame under the wrist label
    cr = collect.make_router(C.S, A, C.ctx, tmp_path / "training_images", tmp_path); C2 = cr.collector
    C2.cfg_path = tmp_path / "c2.json"; C2.S.frames = {"top": C.S.frames["top"]}; A.stop_at = None
    C2.test_wp2("regrasp")
    c = TestClient((lambda a: (a.include_router(cr), a)[1])(FastAPI()))
    assert c.get("/collect/regrip/frame?cam=wrist").status_code == 404
    r = c.get("/collect/regrip/frame?cam=top"); assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_a_failing_log_write_does_not_end_the_test(tmp_path, fast):
    import shutil
    A, C = make(tmp_path, routines())
    t = run_bg(C, step=True)
    wait_for(lambda: C.diag and C.diag["waiting"] and C.diag["i"] == 1)
    shutil.rmtree(C.diag["dir"])                                      # disk gone: steps.jsonl / jpg / summary fail
    while t.is_alive():
        if C.diag["waiting"]: C.flags["next"].set()
        time.sleep(0.01)
    assert C.state == "done" and A.held, C.msg
    assert collect.STEP_TIMEOUT_S == 120
