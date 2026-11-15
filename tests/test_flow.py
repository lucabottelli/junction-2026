"""Logic of the production sort flow (anomaly_capture/flow.py) on a fake station: no arm, no cameras, no models.
The photo verdicts and the re-clamp turns are scripted; checked are the order of the steps, the early stop, the
side bookkeeping (same turn direction) and the plate assignment."""
import sys, threading, types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "anomaly_capture"))
flow = pytest.importorskip("flow")
pol = flow.pol


class FakeC:
    def __init__(self, log):
        self.log = log; self.cfg = {"speed": 1.0}; self.stop_flag = threading.Event(); self.active = False
        self.pos = {"gripper": 3000}; self.stuck = False        # measured joints (shared with the fake arm)
        self.slots = {"plate_a": [{"pos": {"shoulder_pan": 1, "shoulder_lift": 500, "elbow_flex": 500, "wrist_flex": 500}}],
                      "plate_b": [{"pos": {"shoulder_pan": 2, "shoulder_lift": 500, "elbow_flex": 500, "wrist_flex": 500}}]}
    def wps(self, n):
        if n == "wp2": return [{"name": f"wp2-{i}", "pos": {"id": f"wp2-{i}"}} for i in (1, 2, 3)]
        return [{"name": n, "pos": {"id": n, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0}}]
    def slot(self, n): return self.slots.get(n)
    def goto(self, pos, *a, **k):
        self.log.append(("goto", pos.get("id") or pos.get("shoulder_pan"), k.get("gripper")))
        if not self.stuck: self.pos.update({n: v for n, v in pos.items() if n != "id"})
        if k.get("gripper") is not None: self.pos["gripper"] = k["gripper"]
    def rise(self, wp1): self.log.append(("rise",))
    def above_wp2(self, wp1, wp2pos): return {"id": "wp2_above"}, True
    def to_wp2(self, *a): self.log.append(("to_wp2",))
    def up_wp2(self, *a): self.log.append(("up_wp2",))
    def hold_here(self): pass


class FakeT:
    def __init__(self, log, turns, pick_fail=None, regrip_fail_at=None):
        self.log, self.turns, self.pick_fail, self.regrip_fail_at = log, list(turns), pick_fail, regrip_fail_at
        self.stop_flag = threading.Event(); self.state = ""; self.msg = ""; self.ok = True; self._vopt = None; self._vdir = None
        self.cfg = {"device": "GPU"}; self.act = types.SimpleNamespace(attempts=[{"brick": "pink"}]); self.signs = []
        self.scorer = types.SimpleNamespace(find=lambda cam, b: ("m", "root"), thresholds=lambda root: {"t_low": 0.5, "t_high": 0.8})
        self.active = False; self.session = None; self.picks = 0
    def check(self): pass
    def say(self, m, ok=True): self.msg, self.ok = m, ok
    def g_open(self): return 3000
    def spot_force(self, yaw): return {}
    def pick_vision(self, tag=""):
        self.picks += 1; self.log.append(("pick", tag))
        if self.pick_fail: raise pol.PickFailed(self.pick_fail, retry=False)
        return 2000
    def regrip_vision(self, g_lock, k, retries=2, sign=None, end=None):
        self.signs.append(sign); self.log.append(("reclamp", k, (end or {}).get("id")))
        if self.regrip_fail_at == k: raise pol.PickFailed("closed on nothing")
        t = self.turns.pop(0)
        return g_lock, {"turn_deg": t, "hidden": "the two long sides" if abs(t) == 180 else None}


def make(tmp_path, verdicts, turns=(90, 90, 90), **kw):
    """verdicts: scripted photo verdicts in order ('clean' | 'defect' | 'uncertain' | 'nomask'); then 'clean'."""
    log = []; C = FakeC(log); T = FakeT(log, turns, **kw); T.RUNS = tmp_path
    A = types.SimpleNamespace(stop=threading.Event(), cmd={"id": "here", "gripper": 2000}, running=False, mode="hold",
                              pos=C.pos, fl={"gripper": (1875, 3200)})
    F = types.SimpleNamespace(lift={}, lifted=lambda wp2, brick: wp2)
    FL = flow.Flow(None, A, None, C, T, F, None, tmp_path); FL.at_wait = 0
    FL.cfg.update({"accept": "plate_a", "reject": "plate_b", "max_reclamps": 4, "retakes": 1, "uncertain_to": "reject", "side_view": ""})
    FL.cfg_path = tmp_path / "flow_settings.json"
    seq = list(verdicts)
    def shot(brick, view, idx):
        log.append(("photo", view)); return {"view": view, "file": "x.png", "ok": True, "why": None}
    def judge(p, brick, model, t_low, t_high):
        p["verdict"] = seq.pop(0) if seq else "clean"; p["reason"] = p["verdict"]; return p
    FL.shot, FL.judge = shot, judge
    FL.session = "flow_test"; (tmp_path / "flow_test").mkdir(exist_ok=True)
    return FL, T, log


def photos(log): return [x[1] for x in log if x[0] == "photo"]
def plate(log):
    """pan value of the plate the arm went to (1 = plate_a, 2 = plate_b)."""
    return next((x[1] for x in log if x[0] == "goto" and x[1] in (1, 2)), None)


def test_clean_block_three_reclamps_accept(tmp_path):
    FL, T, log = make(tmp_path, [])
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "accept" and rec["reclamps"] == 3 and rec["sides_shown"] == 4
    assert photos(log) == ["wp2-1", "wp2-2", "wp2-3", "wp2-1", "wp2-1", "wp2-1"]     # 3 angles, then the side only
    assert T.signs == [None, 1, 1]                                                  # the first turn fixes the direction
    assert plate(log) == 1


def test_anomaly_on_first_photo_rejects_at_once(tmp_path):
    FL, T, log = make(tmp_path, ["defect"])
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "reject" and photos(log) == ["wp2-1"] and not T.signs and plate(log) == 2
    assert not rec.get("unconfirmed")


def test_anomaly_on_third_angle(tmp_path):
    FL, T, log = make(tmp_path, ["clean", "clean", "defect"])
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "reject" and photos(log) == ["wp2-1", "wp2-2", "wp2-3"] and not T.signs


def test_anomaly_after_second_reclamp(tmp_path):
    FL, T, log = make(tmp_path, ["clean"] * 4 + ["defect"])
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "reject" and rec["reclamps"] == 2 and plate(log) == 2


def test_negative_direction_is_kept(tmp_path):
    FL, T, log = make(tmp_path, [], turns=(-90, -90, -90))
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "accept" and T.signs == [None, -1, -1] and rec["turns_deg"] == [-90, -90, -90]


def test_mixed_direction_needs_one_more_reclamp(tmp_path):
    FL, T, log = make(tmp_path, [], turns=(90, -90, 90, 90))       # the 2nd turn shows side 1 again
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "reject" and rec["sides_shown"] == 3 and rec["reclamps"] == 4 and rec["unconfirmed"]


def test_doubt_is_retaken_then_follows_setting(tmp_path):
    FL, T, log = make(tmp_path, ["uncertain", "clean"])
    assert FL.cycle(1, "")["verdict"] == "accept"                   # the retake cleared it
    FL, T, log = make(tmp_path, ["uncertain", "uncertain"])
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "reject" and rec["unconfirmed"] and photos(log) == ["wp2-1", "wp2-1"]
    FL, T, log = make(tmp_path, ["uncertain", "uncertain"]); FL.cfg["uncertain_to"] = "accept"
    assert FL.cycle(1, "")["verdict"] == "accept"


def test_plates_can_be_swapped(tmp_path):
    FL, T, log = make(tmp_path, []); FL.cfg.update({"accept": "plate_b", "reject": "plate_a"})
    assert FL.cycle(1, "")["verdict"] == "accept" and plate(log) == 2
    FL, T, log = make(tmp_path, ["defect"]); FL.cfg.update({"accept": "plate_b", "reject": "plate_a"})
    assert FL.cycle(1, "")["verdict"] == "reject" and plate(log) == 1


def test_no_block_seen_in_the_jaws_aborts_and_is_not_counted_as_sorted(tmp_path):
    FL, T, log = make(tmp_path, ["nomask", "nomask"])
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "aborted" and rec["lost"] and plate(log) == 2      # the clamp is opened at the reject plate
    assert FL.totals == {"accept": 0, "reject": 0, "aborted": 1}


def test_failed_reclamp_fetches_and_rejects(tmp_path):
    FL, T, log = make(tmp_path, [], regrip_fail_at=2)
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "reject" and rec["unconfirmed"] and plate(log) == 2
    assert ("pick", "fetch2_") in log


def test_empty_plate_ends_the_run(tmp_path):
    FL, T, log = make(tmp_path, [], pick_fail="no brick found on the orange plate")
    FL.T.RUNS = tmp_path
    FL.run(5, "")
    assert T.picks == 1 and T.state == "done" and "empty" in T.msg and FL.totals == {"accept": 0, "reject": 0, "aborted": 0}


def test_long_block_is_turned_end_for_end_and_accepted_with_a_note(tmp_path):
    FL, T, log = make(tmp_path, [], turns=(180,))
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "accept" and rec["reclamps"] == 1 and rec["sides_shown"] == 2
    assert rec["sides_hidden"] and "not inspected" in rec["reason"] and plate(log) == 1
    FL, T, log = make(tmp_path, ["clean", "clean", "clean", "defect"], turns=(180,))
    assert FL.cycle(1, "")["verdict"] == "reject"


def test_pause_holds_before_the_next_step_and_resume_continues(tmp_path):
    import time
    FL, T, log = make(tmp_path, [])
    FL.pause.set()
    th = threading.Thread(target=lambda: log.append(("done", FL.cycle(1, "")["verdict"])), daemon=True); th.start()
    time.sleep(0.4)
    assert FL.paused and not photos(log) and ("pick", "") not in log and "PAUSED" in T.msg       # stands before the pick
    FL.pause.clear(); th.join(5)
    assert ("done", "accept") in log and not FL.paused


def test_stop_works_while_paused(tmp_path):
    import time
    FL, T, log = make(tmp_path, [])
    def check():
        if T.stop_flag.is_set(): raise flow.Stopped()
    T.check = check; FL.T.RUNS = tmp_path
    FL.pause.set = FL.pause.set                     # run() clears the pause at its start: ask for it right after
    th = threading.Thread(target=FL.run, args=(1, ""), daemon=True); th.start()
    time.sleep(0.05); FL.pause.set(); time.sleep(0.5)
    T.stop_flag.set(); th.join(5)
    assert not th.is_alive() and T.state in ("stopped", "done") and not FL.pause.is_set() and not FL.running


def opens(log):
    """Indices of the moves that open the clamp (gripper command 3000 = open in the fake)."""
    return [i for i, x in enumerate(log) if x[0] == "goto" and x[2] == 3000]


def test_clamp_opens_only_at_the_plate(tmp_path):
    for verdicts, pan in (([], 1), (["defect"], 2), (["clean"] * 4 + ["defect"], 2)):
        FL, T, log = make(tmp_path, verdicts)
        FL.cycle(1, "")
        first = next(i for i, x in enumerate(log) if x[0] == "pick")
        held = [i for i in opens(log) if i > first]                      # after the pick (before it the gripper is empty)
        assert held, "the block was never released"
        i = held[0]
        assert log[i][1] == "here" and log[i - 1][:2] == ("goto", pan) and log[i - 1][2] == 2000   # at the plate, clamp kept until then
        assert all(x[2] in (None, 2000) for x in log[first:i] if x[0] == "goto")


def test_arm_short_of_the_plate_keeps_the_block(tmp_path):
    FL, T, log = make(tmp_path, [])
    orig = FL.C.goto
    def goto(pos, *a, **k):                       # the arm stalls on the way down to the plate
        FL.C.stuck = pos.get("shoulder_pan") in (1, 2) and pos.get("shoulder_lift") == 500
        orig(pos, *a, **k)
    FL.C.goto = goto; FL.C.pos.update({"shoulder_pan": 9999, "shoulder_lift": 9999, "elbow_flex": 9999, "wrist_flex": 9999})
    with pytest.raises(RuntimeError, match="did not reach plate"):
        FL.cycle(1, "")
    first = next(i for i, x in enumerate(log) if x[0] == "pick")
    assert not [i for i in opens(log) if i > first]


def test_aborted_block_is_let_go_at_the_reject_plate(tmp_path):
    FL, T, log = make(tmp_path, ["nomask", "nomask"])
    rec = FL.cycle(1, "")
    first = next(i for i, x in enumerate(log) if x[0] == "pick")
    i = [k for k in opens(log) if k > first][0]
    assert rec["verdict"] == "aborted" and rec["released_at"] == "plate_b" and log[i - 1][:2] == ("goto", 2)


def test_run_with_a_closed_clamp_empties_it_at_the_reject_plate_first(tmp_path):
    FL, T, log = make(tmp_path, [], pick_fail="no brick found on the orange plate")
    FL.C.pos["gripper"] = 1900; FL.T.RUNS = tmp_path
    FL.run(1, "")
    i = opens(log)[0]
    assert ("rise",) in log and log[i - 1][:2] == ("goto", 2) and log.index(("rise",)) < i
    assert next(k for k, x in enumerate(log) if x[0] == "pick") > i


def test_no_detour_over_waypoint_1_between_the_steps(tmp_path):
    for verdicts in ([], ["defect"], ["clean"] * 4 + ["defect"]):
        FL, T, log = make(tmp_path, verdicts)
        FL.cycle(1, "")
        first = next(i for i, x in enumerate(log) if x[0] == "pick")
        assert not [x for x in log[first:] if x[0] == "goto" and x[1] == "wp1"]          # the pick itself ends at WP1; nothing after it goes there
        assert all(x[2] == "wp2_above" for x in log if x[0] == "reclamp")               # every re-clamp ends above WP2


def test_block_cut_by_the_frame_is_moved_into_it_instead_of_rejected(tmp_path):
    FL, T, log = make(tmp_path, [])
    FL.F.lifted = lambda wp2, brick: [{**w, "pos": {"id": w["name"] + "+" + str(int(FL.F.lift[brick][w["name"]]))}} for w in wp2]
    n = {"cut": 2}                                  # the first two photos after the re-clamp are cut by the bottom edge
    def shot(brick, view, idx):
        log.append(("photo", view)); cut = len(photos(log)) > 3 and n["cut"] > 0
        if cut: n["cut"] -= 1
        return {"view": view, "file": "x.png", "ok": True, "cut": cut, "why": "block cut by the bottom of the frame" if cut else None}
    def judge(p, brick, model, t_low, t_high):
        p["verdict"] = "uncertain" if p["cut"] else "clean"; p["reason"] = p["why"] or "clean"; return p
    FL.shot, FL.judge = shot, judge
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "accept"
    ups = [x[1] for x in log if x[0] == "goto" and str(x[1]).startswith("wp2-1+")]
    assert ups == ["wp2-1+22", "wp2-1+42", "wp2-1+62", "wp2-1+62", "wp2-1+62"]   # base lift 22 mm, then +20, +40; showings 3 and 4 start at +40
    assert rec["steps"][3]["lift_mm"] == 40 and len(rec["steps"][3]["shots"]) == 3
    assert FL.lift_mem[("pink", "wp2-1", True)] == 40                            # remembered for the next re-clamped showing


def test_cut_block_that_cannot_be_moved_into_the_frame_is_still_rejected(tmp_path):
    FL, T, log = make(tmp_path, [])
    def shot(brick, view, idx):
        log.append(("photo", view)); return {"view": view, "file": "x.png", "ok": True, "cut": True, "why": "block cut by the bottom of the frame"}
    def judge(p, brick, model, t_low, t_high):
        p["verdict"] = "uncertain"; p["reason"] = p["why"]; return p
    FL.shot, FL.judge = shot, judge
    def lifted(wp2, brick):                      # the base poses are fine, anything higher is out of reach
        if any(v > 30 for v in FL.F.lift[brick].values()): raise RuntimeError("out of reach")
        return wp2
    FL.F.lifted = lifted
    rec = FL.cycle(1, "")
    assert rec["verdict"] == "reject" and rec["unconfirmed"] and "cut by" in rec["reason"]
