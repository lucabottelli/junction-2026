"""End-to-end collection flows on the mock station (collect.py + arm.py, the scripted human at the leader):
crop -> record episodes -> episodes list / keep -> export preview (dry run), phase 2 photos with the tape re-grip,
the WP2 / re-grip test buttons, discard + NEXT, STOP in the middle and recovery.

    .venv/bin/python -m pytest tests/test_mock_flows_collect.py        (~2-3 min)
"""
import time

import pytest

pytestmark = pytest.mark.slow


def run_until_idle(srv, timeout=180, on=None):
    def done():
        c = srv.get("/collect/status")
        if on: on(c)
        return c if not c["active"] and c["state"] not in ("idle",) else None
    return srv.wait(done, timeout, "collection finished")


def test_record_two_episodes_then_keep_and_export_preview(connected):
    srv = connected
    srv.post("/collect/roi/top", x0=400, y0=100, x1=1400, y1=850)
    srv.post("/collect/cfg", speed=1.0, phase2="false")
    s = srv.post("/collect/start", n=2, brick="red", label="good")
    assert s["active"], s["msg"]
    c = run_until_idle(srv)
    assert c["state"] == "done" and c["done"] == 2, c["msg"]
    eps = c["sessions"][0]["episodes"]
    assert [e["outcome"] for e in eps] == ["success", "success"] and all(e["handoff"] == "auto" for e in eps)
    assert all(e["grasp_attempts"] == 1 for e in eps), eps
    d = srv.root / "episodes" / c["sessions"][0]["session"] / eps[0]["episode"]
    assert (d / "data.npz").exists() and len(list((d / "top").glob("*.jpg"))) > 30
    srv.post("/collect/keep", session=c["sessions"][0]["session"], episode=eps[1]["episode"], on=False)
    assert srv.get("/collect/status")["sessions"][0]["episodes"][1]["keep"] is False
    srv.post("/collect/export", dry=True)
    c = srv.wait(lambda: (lambda x: x if not x["export_running"] and "exit code" in x["export_log"] else None)(srv.get("/collect/status")),
                 60, "export preview")
    log = c["export_log"]
    assert "--dry-run" in log and "1 episode(s) accepted, 1 rejected" in log and "unticked by you" in log, log
    assert "dry run: nothing written" in log and "[exit code 0]" in log, log
    assert str(srv.root / "episodes") in log                         # reads THIS root's episodes, not the station's


def test_phase2_photos_with_tape_regrip(connected):
    srv = connected
    srv.post("/collect/cfg", speed=1.0, phase2="true", regrasp="true", inspect_cams="front", release_cams="wrist")
    srv.post("/collect/start", n=1, brick="blue", label="defect")
    c = run_until_idle(srv, 240)
    assert c["state"] == "done", c["msg"]
    front = list((srv.root / "training_images" / "front" / "inhand" / "defect").glob("*"))
    crops = [f for f in front if f.name.endswith("_crop.png")]
    assert len(crops) == 6, sorted(f.name for f in front)            # 3 poses before + 3 after the 90 deg re-grip
    assert any(f.name.endswith("_depth.png") for f in front) and any("_r90_" in f.name for f in crops)
    assert list((srv.root / "training_images" / "wrist" / "inhand" / "defect").glob("*tape_top_crop.png"))
    assert srv.get("/collect/counts")["blue"]["defect"] >= 6
    srv.post("/collect/cfg", phase2="false", regrasp="false")


def test_test_buttons_wp2_and_regrasp(connected):
    srv = connected
    srv.goto("wp1")
    srv.post("/collect/test_wp2", kind="wp2")
    c = srv.wait(lambda: (lambda x: x if not x["active"] else None)(srv.get("/collect/status")), 90, "WP2 test")
    assert "test of the WP1" in c["msg"] and c["ok"], c["msg"]
    # re-grip test needs a brick in the gripper: hand-feed one at WP1 (gripper wide open, human feeds it)
    st = srv.get("/arm/status")["all"]["wp1"]["waypoints"][0]["pos"]
    srv.post("/arm/routine/new", name="feedpose"); srv.post("/arm/stop")
    srv.post("/mock/leader", **{k: v for k, v in st.items() if k != "gripper"}, gripper=2950)
    srv.post("/arm/mode", m="mirror")
    srv.wait(lambda: srv.get("/mock/state")["active"]["state"] == "fed", 20, "brick fed")
    srv.post("/mock/leader", gripper=1500)
    srv.wait(lambda: srv.get("/mock/state")["active"]["state"] == "held", 20, "brick gripped")
    srv.post("/arm/mode", m="hold"); srv.post("/mock/human", mode="auto")
    srv.post("/collect/test_wp2", kind="regrasp")
    c = srv.wait(lambda: (lambda x: x if not x["active"] else None)(srv.get("/collect/status")), 120, "re-grip test")
    assert c["ok"] and "re-grip" in c["msg"], c["msg"]
    assert srv.get("/mock/state")["active"]["state"] == "held"


def test_discard_next_and_stop_mid_pick_then_recover(connected):
    srv = connected
    srv.post("/mock/scene", reset=True); srv.post("/collect/cfg", speed=1.0, phase2="false")
    srv.post("/collect/start", n=1, brick="green")
    srv.wait(lambda: srv.get("/collect/status")["state"] == "pick", 30, "pick")
    srv.post("/collect/flag/discard")
    srv.wait(lambda: srv.get("/collect/status")["state"] == "reset", 10, "reset after discard")
    srv.post("/mock/scene", reset=True)                              # the human puts the brick back
    srv.post("/collect/flag/next")
    srv.wait(lambda: srv.get("/collect/status")["state"] == "pick", 30, "second pick")
    time.sleep(0.5)
    s = srv.post("/collect/stop")
    assert "STOPPED" in s["msg"]
    c = srv.wait(lambda: (lambda x: x if not x["active"] else None)(srv.get("/collect/status")), 10, "stopped")
    a = srv.get("/arm/status"); assert a["mode"] == "hold" and not a["running"]
    p1 = a["pos"]; time.sleep(0.5); p2 = srv.get("/arm/status")["pos"]
    assert max(abs(p1[k] - p2[k]) for k in p1) < 25                 # frozen, holding
    eps = [e for se in c["sessions"] for e in se["episodes"]]
    assert any(e["outcome"] == "discard" for e in eps), eps
    # recovery: a new run works
    srv.post("/mock/scene", reset=True)
    srv.post("/collect/start", n=1, brick="green")
    c = run_until_idle(srv)
    assert c["state"] == "done" and c["done"] == 1, c["msg"]


def test_blocked_carry_pauses_run_then_stop_and_resume(connected):
    """A failed episode (BLOCKED in the carry) pauses the run for N instead of ending it; STOP keeps the session
    resumable, and RESUME finishes the same session up to its target."""
    srv = connected
    srv.post("/mock/scene", reset=True); srv.post("/collect/cfg", speed=1.0, phase2="false")
    s = srv.post("/collect/start", n=2, brick="blue", label="good"); assert s["active"], s["msg"]
    srv.wait(lambda: srv.get("/collect/status")["state"] == "carry", 90, "carry")
    srv.post("/mock/fault", obstacle=1)
    c = srv.wait(lambda: (lambda x: x if x["state"] in ("reset", "error", "stopped", "done") else None)(srv.get("/collect/status")),
                 120, "paused after BLOCKED")
    assert c["state"] == "reset" and c["active"] and c["info"]["why"] == "blocked" and c["fails"] == 1, c
    srv.post("/mock/fault", obstacle=0); srv.post("/mock/scene", reset=True); srv.post("/collect/flag/next")
    srv.wait(lambda: srv.get("/collect/status")["state"] == "pick", 60, "next pick after N")
    time.sleep(0.5)
    srv.post("/collect/stop")
    c = srv.wait(lambda: (lambda x: x if not x["active"] else None)(srv.get("/collect/status")), 15, "stopped")
    r = c["resume"]; assert r and r["session"] == c["session"] and r["target"] == 2 and r["done"] == 0, c
    srv.post("/mock/scene", reset=True)
    s = srv.post("/collect/start", n=2, brick="", resume=r["session"]); assert s["active"] and s["session"] == r["session"], s["msg"]
    c = run_until_idle(srv, 300)
    assert c["state"] == "done" and c["done"] == 2 and c["resume"] is None, c["msg"]
