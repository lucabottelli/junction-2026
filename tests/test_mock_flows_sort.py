"""End-to-end inspect & sort on the mock station (sorter.py + coverage.py + collect.py re-grip + the REAL
score_worker.py with the fake anomalib): hand-feed, judge MANUAL (answered from the mock's ground truth) and MODEL,
clean and defect bricks, plus STOP mid-cycle, a dropped brick and a failed re-grip, each with recovery.

    .venv/bin/python -m pytest tests/test_mock_flows_sort.py          (~5 min)
"""
import json, time

import pytest

from conftest import judge_by_truth

pytestmark = pytest.mark.slow


def feeder(srv):
    """The human at the sorter: presses G once the brick is between the fingers, judges photos, presses NEXT on help."""
    def on(s):
        if s["state"] == "handfeed" and "press G" in s["msg"] and srv.get("/mock/state")["active"]["state"] == "fed":
            srv.post("/sort/flag/grasped")
        if s["state"] == "help": srv.post("/sort/flag/next")
        judge_by_truth(srv, s)
    return on


def cycle(srv, judge, defect, timeout=200, extra=None):
    srv.post("/mock/scene", reset=True, defect=defect, queue=defect)
    srv.post("/sort/cfg", judge=judge, retries=1)
    s = srv.post("/sort/start", n=1); assert s["active"], s["msg"]
    on = feeder(srv)

    def done():
        s = srv.get("/sort/status"); on(s)
        if extra: extra(s)
        return s if not s["active"] else None
    s = srv.wait(done, timeout, f"sort cycle ({judge}, defect {defect or 'none'})")
    log = [json.loads(l) for l in open(srv.root / "runs" / s["session"] / "log.jsonl")]
    return s, log[-1]


@pytest.mark.parametrize("judge,defect,verdict,photos_max", [
    ("manual", "", "good", 6), ("manual", "+x", "defect", 1), ("model", "+y", "defect", 6), ("model", "", "good", 6),
    ("model", "+z", "defect", 4)])
def test_sort_cycle(connected, judge, defect, verdict, photos_max):
    srv = connected
    s, log = cycle(srv, judge, defect)
    assert s["state"] == "done", s["msg"]
    assert log["verdict"] == verdict, log["reason"]
    assert log["photos"] <= photos_max, log
    if verdict == "defect": assert "defect seen" in log["reason"], log["reason"]
    if defect == "+y": assert log["regrasps"] == 1                  # under a finger: only seen after the re-grip
    if defect == "+z": assert log["steps"][-1]["view"] == "tape_top"    # top face: seen in the release photo
    if judge == "model":
        assert log["infer_ms_mean"] is not None
        shots = [sh for st in log["steps"] for sh in st["shots"]]
        assert all("stages" in sh for sh in shots)
    st = srv.get("/mock/state")
    assert any(b["state"] == verdict for b in st["bricks"]), st["bricks"]      # physically on the right plate


def test_stop_mid_cycle_then_recover(connected):
    srv = connected
    srv.post("/mock/scene", reset=True, queue="")
    srv.post("/sort/cfg", judge="manual"); srv.post("/sort/start", n=1)
    on = feeder(srv)
    srv.wait(lambda: (lambda s: (on(s), s["state"] in ("inspect", "judge"))[1])(srv.get("/sort/status")), 90, "inspecting")
    s = srv.post("/sort/stop"); assert "STOPPED" in s["msg"]
    srv.wait(lambda: not srv.get("/sort/status")["active"], 10, "stopped")
    a = srv.get("/arm/status"); assert not a["running"] and a["mode"] == "hold"
    assert srv.get("/sort/status")["state"] == "stopped"
    # recovery: the brick is still held; open the gripper on the defect plate by hand, then a full cycle works
    srv.goto("sort_defect_above")
    s, log = cycle(srv, "manual", "+x")
    assert s["state"] == "done" and log["verdict"] == "defect"


def test_dropped_brick_aborts_and_asks_for_help(connected):
    srv = connected
    dropped = {"done": False}

    def drop(s):
        if not dropped["done"] and s["state"] == "inspect":
            srv.post("/mock/fault", drop=True); dropped["done"] = True
    s, log = cycle(srv, "manual", "", extra=drop)
    assert log["verdict"] == "aborted" and log.get("lost"), log


def test_failed_regrip_aborts_with_the_brick_on_the_tape(connected):
    srv = connected
    srv.post("/mock/fault", regrip_fail=2)
    s, log = cycle(srv, "manual", "")
    assert log["verdict"] == "aborted" and log.get("on_tape") and "re-grip failed" in log["reason"], log
    assert srv.get("/mock/state")["faults"]["regrip_fail"] == 0
