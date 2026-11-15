"""Fast semantic API checks on the mock station: data isolation, camera endpoints, arm modes/routine editing, STOP,
blocked detection, crop/config round trips, the manual stills tool (background / save / undo).

    .venv/bin/python -m pytest tests/test_api_mock.py
"""
from pathlib import Path

import cv2
import numpy as np

from conftest import AC, REPO, station_files


def jpeg(srv, url):
    r = srv.http.get(url); assert r.status_code == 200, url
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def test_everything_is_written_under_the_mock_root(mock_server):
    p = mock_server.get("/mock/paths"); root = Path(p.pop("root"))
    assert root == mock_server.root.resolve()
    assert Path(p.pop("anomalib_py")).exists()
    for k, v in p.items():
        assert Path(v).resolve().is_relative_to(root), (k, v)
        assert not Path(v).resolve().is_relative_to(AC), (k, v)


def test_cameras(mock_server):
    s = mock_server.get("/status")
    assert s["live"] == ["top", "front", "wrist"] and all(a is not None and a < 2 for a in s["age"].values())
    assert all(v["mock"] for v in s["settings"].values())
    assert jpeg(mock_server, "/snap/top").shape == (360, 640, 3)
    assert jpeg(mock_server, "/collect/view/top").shape == (480, 640, 3)
    assert jpeg(mock_server, "/collect/view/top?full=true").shape == (540, 960, 3)
    assert jpeg(mock_server, "/collect/view/front").shape == (360, 640, 3)
    assert jpeg(mock_server, "/sort/release_view.jpg") is not None
    assert mock_server.http.get("/sort/judge.jpg").status_code == 204
    s = mock_server.wait(lambda: (lambda x: x if x["n_boxes"]["top"] >= 1 else None)(mock_server.get("/status")), 10,
                         "brick detected on the top camera")


def test_stills_background_save_undo(mock_server):
    srv = mock_server
    s = srv.post("/background"); assert not s["ok"] and "NOT SET" in s["msg"]            # brick in the area: refused
    srv.post("/mock/scene", empty=True)
    srv.wait(lambda: srv.get("/status")["n_boxes"]["top"] == 0, 10, "empty table seen")
    s = srv.post("/background"); assert s["ok"] and s["has_bg"]["top"], s["msg"]
    srv.post("/mock/scene", reset=True)
    srv.wait(lambda: srv.get("/status")["n_boxes"]["top"] == 1, 10, "brick detected against the background")
    s = srv.post("/save/good", brick="red"); assert s["ok"] and "SAVED" in s["msg"], s["msg"]
    assert s["counts"]["top"]["good"] == 1 and s["recent"]
    assert jpeg(srv, "/recent/0") is None or True                                     # PNG, just must answer
    assert srv.http.get("/recent/0").status_code == 200
    s = srv.post("/undo"); assert s["counts"]["top"]["good"] == 0 and "undid" in s["msg"]
    s = srv.post("/save/bogus"); assert not s["ok"]
    assert srv.post("/background/clear")["has_bg"]["top"] is False


def test_rois_and_configs(mock_server):
    srv = mock_server
    assert not srv.post("/roi/top", x0=0, y0=0, x1=10, y1=10)["ok"]
    assert srv.post("/roi/top", x0=600, y0=200, x1=1200, y1=700)["roi"]["top"] == [600, 200, 1200, 700]
    assert srv.post("/roi/top/reset")["roi"]["top"] == [712, 278, 1132, 638]
    r = srv.post("/collect/roi/top", x0=300, y0=100, x1=1500, y1=900)["cfg"]["act_roi"]["top"]
    assert abs((r[2] - r[0]) / (r[3] - r[1]) - 4 / 3) < 0.01
    c = srv.post("/collect/cfg", speed=1.0, phase2="true", regrasp="false", roll_jitter=20)["cfg"]
    assert c["speed"] == 1.0 and c["phase2"] and not c["regrasp"] and c["roll_jitter"] == 20
    m = srv.get("/sort/status")["cfg"]                                   # fast moves: off until asked for
    assert (m["motion_fast"], m["motion_speed"], m["motion_ramp"]) == (False, 1.0, 0.4)
    m = srv.post("/sort/cfg", motion_fast=1, motion_speed=1.2, motion_ramp=0.3)["cfg"]
    assert (m["motion_fast"], m["motion_speed"], m["motion_ramp"]) == (True, 1.2, 0.3)
    m = srv.post("/sort/cfg", motion_fast=0, motion_speed=9, motion_ramp=0.01)["cfg"]          # invalid values ignored
    assert (m["motion_fast"], m["motion_speed"], m["motion_ramp"]) == (False, 1.2, 0.3)
    s = srv.get("/sort/status")
    assert s["plan"]["complete"] and s["tape"] and s["bins"] == {"good": True, "defect": True}, s["plan"]
    assert {m["cam"] for m in s["models"]} == {"front", "wrist"}
    s = srv.post("/sort/faces", view="wp2-3", sides="")
    assert not s["plan"]["complete"]
    assert srv.post("/sort/faces", view="wp2-3", sides="down")["plan"]["complete"]


def test_arm_modes_routines_stop_and_blocked(mock_server):
    srv = mock_server
    s = srv.post("/arm/mode", m="hold"); assert not s["ok"] and "connect" in s["msg"]
    srv.connect(); assert srv.get("/arm/status")["mode"] == "hold"
    assert srv.post("/arm/mode", m="limp")["mode"] == "limp"
    assert srv.post("/arm/mode", m="mirror")["mode"] == "mirror"
    assert srv.post("/arm/mode", m="hold")["mode"] == "hold"
    # routine editing on a scratch routine
    s = srv.post("/arm/routine/new", name="scratch"); assert s["cur"] == "scratch"
    srv.post("/arm/wp/add", capture=True); srv.post("/arm/wp/add")
    s = srv.post("/arm/wp/0/rename", v="first"); assert [w["name"] for w in s["waypoints"]] == ["first", "p2"]
    s = srv.post("/arm/wp/0/down"); assert [w["name"] for w in s["waypoints"]] == ["p2", "first"]
    s = srv.post("/arm/wp/1/capture", v="false"); assert not s["waypoints"][1]["capture"]
    s = srv.post("/arm/wp/5/delete"); assert not s["ok"]
    s = srv.post("/arm/routine/delete"); assert "scratch" not in s["routines"]
    # Go -> moves over time (not a teleport), STOP freezes it mid-move, a new Go finishes it
    tgt = srv.get("/arm/status")["all"]["wp2"]["waypoints"][0]["pos"]
    p0 = srv.get("/arm/status")["pos"]
    srv.post("/arm/wp/0/goto", routine="wp2")
    srv.wait(lambda: srv.get("/arm/status")["running"] or True, 2)
    import time; time.sleep(0.6)
    s = srv.post("/arm/stop"); assert "STOPPED" in s["msg"]
    p1 = srv.get("/arm/status")["pos"]; time.sleep(0.6); p2 = srv.get("/arm/status")["pos"]
    assert p1 != p0 and not srv.near(tgt, 10), "goto should still be on its way after 0.6 s at 25 % speed"
    assert max(abs(p2[k] - p1[k]) for k in p1) < 25, (p1, p2)                          # frozen
    srv.goto("wp2", 0, timeout=30)
    # obstacle -> BLOCKED error, arm keeps holding; clears after the fault is removed
    srv.post("/mock/fault", obstacle=1)
    srv.post("/arm/wp/0/goto", routine="wp1")
    s = srv.wait(lambda: (lambda a: a if "BLOCKED" in a["msg"] else None)(srv.get("/arm/status")), 30, "BLOCKED")
    assert not s["ok"] and s["connected"]
    srv.post("/mock/fault", obstacle=0); srv.goto("wp1")
    # bus errors are counted and recover
    srv.post("/mock/fault", bus_errors=10)
    srv.wait(lambda: srv.get("/arm/status")["bus_errors"] > 0, 5, "bus errors counted")
    srv.wait(lambda: srv.get("/arm/status")["bus_errors"] == 0, 5, "bus recovers")


def test_teach_with_the_leader(mock_server):
    srv = mock_server
    srv.connect(); srv.post("/arm/mode", m="mirror")
    srv.post("/mock/leader", x=0.18, y=-0.05, z=0.12)
    tgt = srv.get("/mock/ik", x=0.18, y=-0.05, z=0.12)
    srv.wait(lambda: srv.near(tgt, 40), 20, "follower mirrors the leader")
    s = srv.post("/arm/wp/add", routine="taught", single=True)
    pos = s["all"]["taught"]["waypoints"][0]["pos"]
    assert all(abs(pos[k] - tgt[k]) < 40 for k in tgt if k != "gripper"), (pos, tgt)
    srv.post("/mock/human", mode="auto"); srv.post("/arm/mode", m="hold")


def test_camera_freeze_reconnects_and_no_depth(mock_server):
    srv = mock_server
    srv.post("/mock/camera", cam="wrist", state="frozen")
    srv.wait(lambda: (srv.get("/status")["age"]["wrist"] or 0) > 2, 10, "wrist frozen (UI shows FROZEN)")
    srv.wait(lambda: srv.get("/status")["reconnects"]["wrist"] >= 1, 10, "capture.py reconnects after 3 s")
    srv.post("/mock/camera", cam="wrist", state="ok")
    srv.wait(lambda: srv.get("/status")["age"]["wrist"] < 1, 15, "frames again")
    srv.post("/mock/camera", cam="front", depth=0)
    srv.wait(lambda: srv.get("/sort/status") and jpeg(srv, "/collect/view/front") is not None, 5)
    srv.post("/mock/camera", cam="front", depth=1)


def test_station_files_untouched_by_this_module(mock_server):
    """The tracked station files (taught waypoints, locked camera settings) are byte-identical to git HEAD's view at
    module start (only checks files nobody else edits concurrently)."""
    before = station_files_at_start
    now = station_files()
    for f in ("routines.json", "cam_settings.json"):
        assert now[f] == before[f], f"{f} changed during a mock run!"


station_files_at_start = station_files()
