"""arm.profile: the time law of one move. Minimum jerk by default; in fast mode a long move ramps up smoothly to
its cruise speed, holds it and ramps down, and a short move is left as it was."""
import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "anomaly_capture"))
import pytest

import sim_test  # noqa: E402,F401  (stubs the servo bus when lerobot is missing)
from arm import profile  # noqa: E402


def sample(T, f, d, dt=0.002):
    xs = [f(i * dt) * d for i in range(int(T / dt) + 3)]
    v = [(b - a) / dt for a, b in zip(xs, xs[1:])]
    a = [(b - c) / dt for c, b in zip(v, v[1:])]
    return xs, v, a


def test_default_is_minimum_jerk():
    T, f, vpk = profile(1500, 1200)
    assert T == pytest.approx(1.875 * 1500 / 1200) and vpk == 1200
    xs, v, _ = sample(T, f, 1500)
    assert xs[0] == 0 and xs[-1] == pytest.approx(1500) and max(v) == pytest.approx(1200, rel=0.01)


def test_fast_long_move_cruises_and_ramps_smoothly():
    d, v, ramp = 1500, 1500, 0.4
    T, f, vpk = profile(d, 1200, ramp, v)
    assert T == pytest.approx(d / v + ramp) and T < 0.65 * 1.875 * d / 1200 and vpk == v
    xs, vel, acc = sample(T, f, d)
    assert xs[0] == 0 and xs[-1] == pytest.approx(d) and all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))
    assert max(vel) == pytest.approx(v, rel=0.005)                                # never above the target speed
    assert f(T / 2) == pytest.approx(0.5)                                         # symmetric
    cruise = [x for i, x in enumerate(vel) if ramp + 0.01 < i * 0.002 < T - ramp - 0.01]
    assert min(cruise) == pytest.approx(v, rel=0.005)                             # it holds the speed
    assert max(abs(x) for x in acc) == pytest.approx(1.5 * v / ramp, rel=0.03)    # smoothstep ramp
    assert abs(vel[0]) < 0.02 * v and abs(vel[-1]) < 0.02 * v                     # starts and ends at rest
    assert max(abs(b - a) for a, b in zip(acc, acc[1:])) < 0.03 * 1.5 * v / ramp  # no jump in the acceleration


def test_fast_short_move_is_unchanged():
    for d in (20, 150, 400):                               # shorter than the two ramps: minimum jerk at its own speed
        T, f, vpk = profile(d, 450, 0.4, 1500)
        T0, f0, _ = profile(d, 450)
        assert T == T0 and vpk == 450 and f(T / 3) == f0(T / 3)


def test_fast_is_never_slower():
    for d in range(10, 4000, 37):
        for vm in (300, 600, 1200):
            assert profile(d, vm, 0.4, vm * 1.25)[0] <= profile(d, vm)[0] + 1e-9


def test_collector_hands_fast_mode_to_the_arm_per_move(tmp_path):
    import collect
    from arm import JOINTS

    class FakeArm:
        def __init__(self):
            self.fl = {n: (0, 4095) for n in JOINTS}; self.cmd = {n: 2000.0 for n in JOINTS}; self.seen = []; self.fast = None
        def goto(self, tgt, speed): self.seen.append((speed, self.fast)); return 0

    A = FakeArm(); S = types.SimpleNamespace(live=[], frames={}, cams={}, averaged=None)
    C = collect.Collector(S, A, types.SimpleNamespace(R={"routines": {}}, capture=lambda *a, **k: 1), tmp_path / "t", tmp_path)
    C.cfg_path = tmp_path / "c.json"; C.cfg["speed"] = 0.8
    pos = {n: 2100 for n in JOINTS}
    C.goto(pos); C.fast = (1.0, 0.4); C.goto(pos); C.goto(pos, 0.8 * 0.3)
    assert A.seen[0] == (0.8, None)                                    # off: as before
    assert A.seen[1] == (0.8, (pytest.approx(1.25), 0.4))              # on: a move at cfg speed cruises at the target
    assert A.seen[2] == (pytest.approx(0.24), (pytest.approx(1.25), 0.4))   # a gentle move keeps its ratio to it
    assert A.fast is None                                              # never left on for moves that bypass the collector


def test_spot_look_does_not_merge_a_green_block_with_the_yellow_plate():
    """flow 2026-11-15 08:49: the green block lay against the yellow plate, whose rim read as green; block and rim
    were one 122 mm blob and the fetch closed on the rim three times."""
    import cv2, numpy as np
    import vpick as vp
    hsv = np.zeros((600, 800, 3), np.uint8); hsv[:] = (15, 90, 150)                       # wood: the plate's hue
    cv2.circle(hsv, (560, 380), 190, (28, 110, 170), -1)                                  # yellow plate
    cv2.rectangle(hsv, (330, 250), (382, 346), (54, 95, 155), -1)                         # green block, touching it
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    got = vp.detect(img, mm_per_px=0.5, at=(300, 300, 200))[0]
    assert [b["colour"] for b in got] == ["green"]
    (u, v), (lg, sh) = got[0]["c"], got[0]["size"]
    assert abs(u - 356) < 4 and abs(v - 298) < 4 and abs(lg * 0.5 - 48) < 4 and abs(sh * 0.5 - 26) < 4


def test_jaws_alone_never_cruise():
    """Closing / opening the jaws keeps the eased law in fast mode (they meet the block at the eased speed); the same
    distance on a body joint cruises."""
    import time
    import arm
    def run(joint):
        A = arm.Arm(); A.cmd = {n: 2000.0 for n in arm.JOINTS}; tgt = dict(A.cmd); tgt[joint] += 700
        A.pos = dict(tgt); A.fast = (1.25, 0.4); t0 = time.time(); A.goto(tgt, 0.5); return time.time() - t0
    T_mj = 1.875 * 700 / 750
    assert run("gripper") >= T_mj                              # minimum jerk at its own speed
    assert run("shoulder_pan") < T_mj - 0.3                    # 700 / 937 + 0.4 s (+ the settle)
