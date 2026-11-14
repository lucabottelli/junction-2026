"""Close-range front depth + the in-hand depth mask (issue 13: the WP2 brick was inside the D415's Min-Z and
inhand_mask silently kept a wrong blob).

  * rs_depth: disparity-shift window math, config, auto-tune scoring, advanced-mode enable/reset handling (fake rs).
  * prep.inhand_mask / fill_holes on synthetic depth: the failure modes it must refuse or survive.
  * Regression on the one real WP2 capture (1280x720 depth: the brick has no depth -> must be None, not a blob).
  * The mock station: /depth/status, /depth/autotune, /depth/reset, /status, capture meta provenance.

    .venv/bin/python -m pytest tests/test_depth_mask.py
"""
import importlib.util, json, math, sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
AC = REPO / "anomaly_capture"
REAL = REPO / "training_images/front/inhand/good/20261114_032612_113_blue_wp2-1"
STATION_FRONT_ROI = [777, 270, 1579, 847]       # cam_settings.json "roi"/"front" when the real capture was taken


def _load(name):
    key = f"_depthtest_{name}"
    if key in sys.modules: return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, AC / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[key] = m; spec.loader.exec_module(m)
    return m


rd = _load("rs_depth")
prep = _load("prep")
H, W = 1080, 1920
FX = 1380.0


@pytest.fixture
def cfg():
    """Known INHAND config (no ROI = image centre, no expected z); restored afterwards."""
    old = dict(prep.INHAND)
    prep.configure_inhand(roi=None, fx=FX, expect_z_mm=None, min_mm=50)
    yield prep.INHAND
    prep.INHAND.clear(); prep.INHAND.update(old)


def scene(seed=0):
    return np.zeros((H, W), np.uint16), np.random.default_rng(seed)


def speckle(d, rng, frac=0.03, lo=300, hi=5000):
    """Sparse wrong/far depth everywhere (what the D415 returns around/behind a too-close object)."""
    m = rng.random(d.shape) < frac
    d[m & (d == 0)] = rng.integers(lo, hi, int((m & (d == 0)).sum()))
    return d


def box(d, x0, y0, x1, y1, z, rng=None, holes=0.0):
    sub = d[y0:y1, x0:x1]; sub[:] = z
    if rng is not None:
        sub[:] = (z + rng.normal(0, 2, sub.shape)).clip(1, 65535)
        if holes: sub[rng.random(sub.shape) < holes] = 0
    return d


# ======================================================================== disparity window math
D415_424 = (305.0, 55.0)       # fx (px) of a 424x240 depth stream, baseline (mm)
D415_1280 = (920.0, 55.0)


def test_window_shift0_has_no_far_limit_and_min_z_scales_with_resolution():
    n424, f424 = rd.depth_window(*D415_424, 0)
    n1280, f1280 = rd.depth_window(*D415_1280, 0)
    assert math.isinf(f424) and math.isinf(f1280)
    assert n424 == pytest.approx(305 * 55 / 126) and 125 < n424 < 145        # ~13-14 cm at 424x240
    assert 380 < n1280 < 420                                                  # ~40 cm at 1280x720 (issue 13)
    assert rd.window_info(*D415_424, 0)["far_mm"] is None


def test_window_with_shift():
    near, far = rd.depth_window(*D415_424, 50)
    assert near == pytest.approx(305 * 55 / 176) and far == pytest.approx(305 * 55 / 50)
    assert near < far
    assert rd.depth_window(*D415_424, -7) == rd.depth_window(*D415_424, 0)    # negative = 0


@pytest.mark.parametrize("near_mm", [130, 100, 80, 60, 45])
def test_shift_for_near_is_the_smallest_shift_reaching_it(near_mm):
    s, clamped = rd.shift_for_near(*D415_424, near_mm)
    assert not clamped and s >= 0
    assert rd.depth_window(*D415_424, s)[0] <= near_mm + 1e-6
    if s > 0: assert rd.depth_window(*D415_424, s - 1)[0] > near_mm


def test_shift_for_near_limits():
    assert rd.shift_for_near(*D415_424, 300) == (0, False)        # farther than Min-Z at shift 0: nothing to do
    assert rd.shift_for_near(*D415_424, None) == (0, False)
    s, clamped = rd.shift_for_near(*D415_424, 5)                  # 5 mm: beyond the hardware
    assert clamped and s == rd.SHIFT_MAX
    assert rd.shift_for_near(*D415_424, 0) == (rd.SHIFT_MAX, True)
    with pytest.raises(ValueError): rd.shift_for_near(0, 55, 100)
    with pytest.raises(ValueError): rd.depth_window(305, 0, 0)


def test_resolve_config_and_initial_shift():
    c = rd.resolve_config(None)
    assert (c["width"], c["height"], c["shift"], c["advanced"]) == (424, 240, None, True)
    assert rd.resolve_config({"width": 1000, "height": 3})["width"] == 424          # unsupported mode -> default
    assert rd.resolve_config({"width": 1280, "height": 720})["height"] == 720
    assert rd.resolve_config({"shift": 9999})["shift"] == rd.SHIFT_MAX
    assert rd.resolve_config({"shift": -3})["shift"] == 0
    assert rd.initial_shift(rd.resolve_config({}), *D415_424)[0] == 0
    assert rd.initial_shift(rd.resolve_config({"near_mm": 80}), *D415_424)[0] == rd.shift_for_near(*D415_424, 80)[0]
    assert rd.initial_shift(rd.resolve_config({"near_mm": 80, "shift": 7}), *D415_424)[0] == 7     # explicit wins


def test_candidate_shifts_and_pick_shift():
    c = rd.candidate_shifts(*D415_424)
    assert c[0] == 0 and c == sorted(set(c)) and len(c) >= 5 and c[-1] <= rd.SHIFT_MAX
    r = lambda s, px, ok=True: {"shift": s, "masked": ok, "object_px": px, "z_mm": 90}
    # plateau 40..100 (within 95 % of the best): the middle one leaves margin on both sides of the window
    res = [r(0, 1000), r(20, 50000), r(40, 98000), r(60, 100000), r(80, 99000), r(100, 97000), r(120, 20000), r(140, 0, False)]
    assert rd.pick_shift(res)["shift"] == 60
    assert rd.pick_shift([r(0, 0, False), r(10, 0, False)]) is None
    assert rd.pick_shift([]) is None


def test_measure_counts_only_real_depth_in_the_cluster(cfg):
    d, rng = scene()
    box(d, 700, 300, 1250, 850, 90, rng)
    d[500:520, 800:1100] = 0                                       # a filled hole: no depth there
    st = rd.measure(d, [600, 200, 1400, 900], prep.inhand_mask)
    assert st["masked"] and abs(st["z_mm"] - 90) <= 3
    assert st["object_px"] == pytest.approx(550 * 550 - 20 * 300, rel=0.01)
    assert 0.5 < st["valid_roi"] < 0.6
    st0 = rd.measure(None, [0, 0, 10, 10], prep.inhand_mask)
    assert not st0["masked"] and st0["valid_roi"] == 0


# ======================================================================== advanced mode with a fake pyrealsense2
class FakeRS:
    """Just enough of pyrealsense2 for rs_depth: context().query_devices(), rs400_advanced_mode, camera_info."""
    camera_info = type("ci", (), {"serial_number": "sn"})

    def __init__(self, enabled=False, back_after=3, never_back=False, toggle_raises=False):
        self.enabled, self.back_after, self.never_back, self.toggle_raises = enabled, back_after, never_back, toggle_raises
        self.present, self.polls, self.toggles = True, 0, 0
        self.table = type("T", (), {"disparityShift": 0})()
        rs = self

        class Dev:
            def get_info(self, _): return "123"

        class Adv:
            def __init__(self, dev): pass
            def is_enabled(self): return rs.enabled
            def toggle_advanced_mode(self, on):
                if rs.toggle_raises: raise RuntimeError("not supported")
                rs.toggles += 1; rs.present = False                      # the camera resets: gone for a while
            def get_depth_table(self): return rs.table
            def set_depth_table(self, t): rs.table = t

        class Ctx:
            def query_devices(self):
                rs.polls += 1
                if not rs.present and not rs.never_back and rs.polls >= rs.back_after:
                    rs.present, rs.enabled = True, True
                return [Dev()] if rs.present else []

        self.context, self.rs400_advanced_mode, self.Dev = Ctx, Adv, Dev


class Clock:
    def __init__(self): self.t = 0.0
    def sleep(self, s): self.t += s
    def now(self): return self.t


def test_enable_advanced_already_on_does_not_reset():
    rs = FakeRS(enabled=True); clk = Clock()
    assert rd.enable_advanced(rs, "123", sleep=clk.sleep, now=clk.now, log=lambda *a: None)[0]
    assert rs.toggles == 0 and clk.t == 0


def test_enable_advanced_waits_for_the_camera_to_come_back():
    rs = FakeRS(enabled=False, back_after=6); clk = Clock()
    ok, msg = rd.enable_advanced(rs, "123", timeout=20, sleep=clk.sleep, now=clk.now, log=lambda *a: None)
    assert ok and "reset" in msg and rs.toggles == 1 and clk.t < 20


def test_enable_advanced_times_out_and_reports():
    rs = FakeRS(enabled=False, never_back=True); clk = Clock()
    ok, msg = rd.enable_advanced(rs, "123", timeout=8, sleep=clk.sleep, now=clk.now, log=lambda *a: None)
    assert not ok and "8 s" in msg and clk.t <= 8 + 3 + 1          # bounded: initial wait + timeout + one poll


def test_enable_advanced_missing_device_or_unsupported():
    clk = Clock()
    assert not rd.enable_advanced(FakeRS(), "999", sleep=clk.sleep, now=clk.now, log=lambda *a: None)[0]
    ok, msg = rd.enable_advanced(FakeRS(toggle_raises=True), "123", sleep=clk.sleep, now=clk.now, log=lambda *a: None)
    assert not ok and "not available" in msg


def test_set_shift_writes_the_depth_table_and_clamps():
    rs = FakeRS(enabled=True); dev = rs.Dev()
    assert rd.set_shift(rs, dev, 77) == 77 and rs.table.disparityShift == 77
    assert rd.set_shift(rs, dev, 5000) == rd.SHIFT_MAX
    assert rd.set_shift(rs, dev, -1) == 0


# ======================================================================== inhand_mask on synthetic depth
def frac_in(mask, x0, y0, x1, y1):
    return float((mask[y0:y1, x0:x1] > 0).mean())


def test_brick_close_with_sparse_speckle_elsewhere(cfg):
    """The fixed station: brick + jaws at ~80 mm with real depth, sparse junk everywhere else."""
    d, rng = scene(1)
    box(d, 650, 250, 1250, 800, 80, rng, holes=0.02)
    speckle(d, rng)
    m, info = prep.inhand_mask(d)
    assert m is not None, info
    assert abs(info["z_mm"] - 80) <= 5
    assert frac_in(m, 650, 250, 1250, 800) > 0.99                  # the whole brick, small depth holes filled
    out = m.copy(); out[230:820, 630:1270] = 0
    assert out.sum() == 0                                          # nothing outside it (dilation margin only)


def test_brick_inside_min_z_returns_none_not_a_blob(cfg):
    """Issue 13 synthetically: the brick has only ~9 % (wrong) depth, a real object at 42 cm sits at the top edge."""
    d, rng = scene(2)
    b = d[440:1010, 1020:1500]; m9 = rng.random(b.shape) < 0.09
    b[m9] = rng.integers(450, 6000, int(m9.sum()))
    box(d, 1000, 40, 1170, 290, 420, rng)                          # white robot column, 42 cm
    box(d, 1476, 736, 1584, 800, 418, rng)                         # a patch of the yellow plate
    speckle(d, rng, 0.05)
    m, why = prep.inhand_mask(d, roi=STATION_FRONT_ROI)             # the station's search area (capture.sync_inhand)
    assert m is None and isinstance(why, str), why
    # without a search area a solid object straddling the image centre is geometrically plausible; the auto-tune's
    # expected distance still refuses it
    m, why = prep.inhand_mask(d, roi=None, expect_z_mm=95)
    assert m is None and "expected 95" in why, why


def test_no_depth_at_all(cfg):
    assert prep.inhand_mask(None) == (None, "no depth")
    m, why = prep.inhand_mask(np.zeros((H, W), np.uint16))
    assert m is None and "no depth" in why
    crop, info = prep.inspect_crop(np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.uint16))
    assert info["masked"] is False and info["why"]


def test_brick_touching_the_image_edge(cfg):
    d, rng = scene(3)
    box(d, 700, 520, 1300, H, 110, rng)                            # runs out of the bottom of the frame
    d[900:H, 950:1050] = 600                                        # notch open to the edge: background, not a hole
    m, info = prep.inhand_mask(d)
    assert m is not None, info
    assert frac_in(m, 700, 520, 950, H) > 0.99 and frac_in(m, 1050, 520, 1300, H) > 0.99
    assert frac_in(m, 960, 920, 1040, H) < 0.05                    # the notch stays out
    crop, ci = prep.inspect_crop(np.full((H, W, 3), 200, np.uint8), d)
    assert ci["masked"] and crop.shape[0] == crop.shape[1]


def test_pen_cross_hole_inside_the_brick_is_filled(cfg):
    d, rng = scene(4)
    box(d, 700, 300, 1260, 860, 90, rng)
    cross = np.zeros((H, W), np.uint8)                              # ~3 mm strokes at 90 mm (~45 px), no depth
    cv2.line(cross, (820, 420), (1140, 740), 255, 40); cv2.line(cross, (1140, 420), (820, 740), 255, 40)
    d[cross > 0] = 0
    m, info = prep.inhand_mask(d)
    assert m is not None, info
    assert (m[cross > 0] > 0).mean() > 0.999                        # the cross stays IN the mask


def jaws_scene(gap_depth):
    """Gripper body (top bar) + two jaws + the brick at the bottom enclose a 200 x 160 px gap."""
    d, rng = scene(5)
    box(d, 640, 250, 1280, 400, 100, rng)                           # gripper body
    box(d, 640, 400, 860, 560, 105, rng); box(d, 1060, 400, 1280, 560, 105, rng)   # jaws
    box(d, 640, 560, 1280, 900, 110, rng)                           # brick
    d[400:560, 860:1060] = gap_depth                                # what is seen between jaws and brick
    return d


def test_background_seen_through_the_jaws_is_not_filled(cfg):
    d = jaws_scene(650)                                             # background with valid depth
    m, info = prep.inhand_mask(d)
    assert m is not None, info
    assert frac_in(m, 880, 420, 1040, 540) == 0                     # the gap stays grey
    assert frac_in(m, 650, 260, 1270, 390) > 0.99 and frac_in(m, 650, 570, 1270, 890) > 0.99
    # the old unguarded fill would have let the background in
    assert frac_in(prep.fill_holes(m), 880, 420, 1040, 540) == 1


def test_background_gap_beyond_the_far_limit_is_not_filled(cfg):
    d = jaws_scene(0)                                               # disparity shift: background has NO depth
    m, info = prep.inhand_mask(d)
    assert m is not None, info
    assert frac_in(m, 880, 420, 1040, 540) == 0                     # 15+ mm wide hole without depth: not a pen stroke


def test_object_mostly_without_depth_is_refused(cfg):
    d, rng = scene(6)
    box(d, 640, 250, 1280, 900, 90, rng)
    d[330:820, 720:1200] = 0                                        # only a rim has depth: too close for the window
    m, why = prep.inhand_mask(d)
    assert m is None and ("has no depth" in why or "without depth" in why), why


def test_tiny_near_speckle_blob_is_not_chosen(cfg):
    d, rng = scene(7)
    box(d, 700, 300, 1200, 700, 150, rng)                           # brick + jaws at 15 cm
    box(d, 1250, 450, 1290, 490, 60, rng)                           # 40 x 40 px speck at 6 cm
    m, info = prep.inhand_mask(d)
    assert m is not None and abs(info["z_mm"] - 150) <= 5, info
    assert frac_in(m, 1250, 450, 1290, 490) == 0
    # a bigger near blob that IS a histogram peak but far too small for an object that close: skipped, brick kept
    box(d, 1250, 300, 1500, 550, 60, rng)
    m, info = prep.inhand_mask(d)
    assert m is not None and abs(info["z_mm"] - 150) <= 5, info
    assert frac_in(m, 1260, 310, 1490, 540) == 0


def test_person_closer_than_the_brick_at_the_frame_edge(cfg):
    d, rng = scene(8)
    box(d, 0, 0, 300, H, 280, rng)                                  # someone leaning in at the left edge, 28 cm
    box(d, 850, 350, 1100, 650, 380, rng)                           # brick + jaws at 38 cm, centre
    m, info = prep.inhand_mask(d)
    assert m is not None and abs(info["z_mm"] - 380) <= 5, info
    assert frac_in(m, 0, 0, 300, H) == 0 and frac_in(m, 850, 350, 1100, 650) > 0.99
    m, info = prep.inhand_mask(d, roi=STATION_FRONT_ROI)            # same with the station's search area
    assert m is not None and abs(info["z_mm"] - 380) <= 5


def test_expected_distance_from_the_autotune_rejects_other_objects(cfg):
    d, rng = scene(9)
    box(d, 850, 350, 1100, 600, 380, rng)                           # something at 38 cm where the brick should be
    assert prep.inhand_mask(d)[0] is not None                       # without a prior it is a plausible object
    prep.configure_inhand(expect_z_mm=95)
    m, why = prep.inhand_mask(d)
    assert m is None and "expected 95" in why
    assert prep.inhand_mask(d, expect_z_mm=None)[0] is not None      # explicit None switches the prior off
    d2, rng = scene(10); box(d2, 650, 250, 1250, 800, 100, rng)
    assert prep.inhand_mask(d2)[0] is not None                      # the brick near the expected distance passes


def test_configured_roi_is_the_search_area(cfg):
    d, rng = scene(11)
    box(d, 100, 300, 600, 800, 120, rng)                            # object far left: outside the centre region
    assert prep.inhand_mask(d)[0] is None
    prep.configure_inhand(roi=[50, 250, 700, 850])
    m, info = prep.inhand_mask(d)
    assert m is not None and info["search"] == "roi"
    assert prep.inhand_mask(d, roi=None)[0] is None


def test_fill_holes_without_depth_is_unchanged():
    m = np.zeros((100, 100), np.uint8); cv2.rectangle(m, (10, 10), (90, 90), 255, 5)
    f = prep.fill_holes(m)
    assert f[50, 50] == 255 and f[5, 5] == 0


def test_small_depth_image_still_works(cfg):
    """sim_test.py and older callers pass small depth maps: thresholds scale with the image."""
    d = np.full((120, 160), 2000, np.uint16); d[20:100, 40:120] = 400
    m, info = prep.inhand_mask(d, bg=None)
    assert m is not None and m.shape == d.shape and info["z_mm"] == 400 and not info["bg"]
    m, info = prep.inhand_mask(d, bg=np.full((120, 160), 2000, np.uint16))
    assert m is not None and m.shape == d.shape and info["z_mm"] == 400 and info["bg"]


# ======================================================================== the real capture (issue 13)
@pytest.mark.skipif(not Path(str(REAL) + "_depth.png").exists(), reason="station capture not present (training_images is not in git)")
def test_real_wp2_capture_at_1280x720_gives_no_mask(cfg):
    """The only real WP2 photo: 1280x720 depth, the brick ~9 cm away has (almost) no depth. The old inhand_mask kept
    a speckle blob at the top of the frame and greyed out the brick; now it must refuse."""
    d = cv2.imread(str(REAL) + "_depth.png", cv2.IMREAD_UNCHANGED)
    img = cv2.imread(str(REAL) + ".png")
    assert d is not None and d.dtype == np.uint16 and d.shape == (H, W)
    for roi in (None, STATION_FRONT_ROI):
        m, why = prep.inhand_mask(d, roi=roi)
        assert m is None, (roi, why)
    prep.configure_inhand(roi=STATION_FRONT_ROI)
    crop, info = prep.inspect_crop(img, d)
    assert info["masked"] is False                                  # -> the sorter's "no depth mask" (never good)


# ======================================================================== mock camera's depth window
def test_mock_wp2_brick_is_inside_min_z_at_1280_but_masked_at_424(cfg):
    mh = _load("mock_hw")
    p = {k: float(v) for k, v in mh.mock_routines()["wp2"]["waypoints"][0]["pos"].items()}
    w = mh.World(); p["gripper"] = 1890.0; w.servo["follower"]["pos"] = p
    k = mh.FK(p); b = w.active; b.R = k.R.copy(); b.pos = k.tip.copy(); b.state = "held"
    fp, bs = w.snapshot(); _, dep = mh.RENDER.render("front", fp, bs, want_depth=True)
    prep.configure_inhand(roi=[840, 240, 1380, 780])
    for width, ok in ((1280, False), (424, True)):
        near, far = rd.depth_window(mh.MOCK_FX_PER_W * width, mh.MOCK_BASELINE_MM, 0)
        dw = np.where((dep >= near) & (dep <= far), dep, 0).astype(np.uint16)
        m, info = prep.inhand_mask(dw)
        assert (m is not None) == ok, (width, info)


# ======================================================================== mock station API
def test_depth_api_on_the_mock(connected):
    srv = connected
    st = srv.get("/depth/status")
    assert st["available"] and (st["width"], st["height"], st["shift"]) == (424, 240, 0)
    assert st["near_mm"] < 150 and st["far_mm"] is None and st["inhand"]["roi"] == srv.get("/status")["roi"]["front"]
    assert srv.get("/status")["settings"]["front"]["depth"]["width"] == 424
    # the auto-tune needs the empty-scene background first (else the table can win the sweep)
    assert not srv.post("/depth/autotune")["ok"] and not st["background"]["usable"]
    srv.goto("home", 0)
    b = srv.post("/depth/background", frames=6)
    assert b["ok"] and b["status"]["usable"] and b["background"]["shifts"][0] == 0 and len(b["background"]["shifts"]) > 3, b
    assert (srv.root / "settings" / "front_depth_bg.png").exists() and srv.get("/depth/status")["inhand"]["bg"]
    srv.goto("wp2", 0)
    st = srv.wait(lambda: (lambda s: s if s["mask_ok"] else None)(srv.get("/depth/status")), 10, "mask at WP2")
    assert 300 < st["z_mm"] < 420 and st["valid_roi"] > 0.5
    r = srv.post("/depth/autotune")
    assert r["ok"], r
    assert srv.get("/depth/status")["background"]["usable"]           # it covers every shift: still valid after the tune
    assert len(r["results"]) >= 5 and r["results"][0]["shift"] == 0
    assert r["best"]["masked"] and (r["best"]["far_mm"] is None or r["best"]["far_mm"] > r["best"]["z_mm"])
    cs = json.loads((srv.root / "settings" / "cam_settings.json").read_text())["front_depth"]
    assert cs["shift"] == r["best"]["shift"] and cs["expect_z_mm"] == r["best"]["z_mm"]
    st = srv.get("/depth/status")
    assert st["shift"] == r["best"]["shift"] and st["inhand"]["expect_z_mm"] == r["best"]["z_mm"] and st["mask_ok"], (st["mask_why"], st["age_s"], st["z_mm"])
    assert srv.get("/status")["settings"]["front"]["depth"]["shift"] == r["best"]["shift"]
    # an explicit sweep that can't see the brick (far limit in front of it) fails and restores the shift
    bad = srv.post("/depth/autotune", shifts="200,300")
    assert not bad["ok"] and srv.get("/depth/status")["shift"] == r["best"]["shift"]
    # a capture at WP2 records the depth configuration with the image
    srv.post("/arm/run", cycles=1, label="good", brick="blue", speed=1.0, cams="front", save=True, routine="wp2")
    srv.wait(lambda: not srv.get("/arm/status")["running"] and srv.get("/arm/status")["msg"].startswith("✓"), 90, "capture run")
    lines = [json.loads(l) for l in (srv.root / "training_images" / "meta.jsonl").read_text().splitlines() if l.strip()]
    inhand = [l for l in lines if l.get("stage") == "inhand" and l["cam"] == "front"]
    assert inhand and all(l["depth_cfg"]["shift"] == r["best"]["shift"] and l["depth_cfg"]["width"] == 424 for l in inhand)
    assert all(l["depth"] or l["depth_mask_why"] for l in inhand)
    rs = srv.post("/depth/reset")
    assert rs["shift"] == 0 and rs["inhand"]["expect_z_mm"] is None
    assert "shift" not in json.loads((srv.root / "settings" / "cam_settings.json").read_text())["front_depth"]


# ======================================================================== review fixes (regressions)
def wall_scene(wall_z, brick_box=(700, 300, 1220, 820), brick_z=0, seed=20):
    """A wall/table/arm link at wall_z behind everything; the brick in front of it has NO depth (inside Min-Z)
    unless brick_z is given."""
    d, rng = scene(seed)
    box(d, 0, 0, W, H, wall_z, rng)
    x0, y0, x1, y1 = brick_box
    d[y0:y1, x0:x1] = brick_z
    return d


def test_wall_behind_a_brick_inside_min_z_is_not_the_held_object(cfg):
    """Review item 1: the brick (100 mm, shift 0 -> no depth) is a hole in a wall at 600 mm."""
    d = wall_scene(600)
    m, why = prep.inhand_mask(d)
    assert m is None, why
    crop, info = prep.inspect_crop(np.full((H, W, 3), 90, np.uint8), d)
    assert info["masked"] is False
    # a full-frame wall nearer than the held-object limit is background-sized
    m, why = prep.inhand_mask(wall_scene(380))
    assert m is None and "background-sized" in why, why
    # an object-sized surface (arm link, plate) at 38 cm behind the brick: the brick-sized hole without depth refuses it
    d, rng = scene(21); box(d, 500, 150, 1420, 970, 380, rng); d[300:820, 700:1220] = 0
    m, why = prep.inhand_mask(d)
    assert m is None and "without depth" in why, why
    # the same with the limit switched off: the hole check alone
    m, why = prep.inhand_mask(wall_scene(600), max_obj_mm=None)
    assert m is None, why
    # brick at the bottom edge (not a hole of the wall): the wall is background-sized
    m, why = prep.inhand_mask(wall_scene(380, (700, 600, 1220, H)))
    assert m is None, why
    m, why = prep.inhand_mask(wall_scene(380, (700, 600, 1220, H)), roi=STATION_FRONT_ROI)
    assert m is None, why


def test_brick_in_front_of_a_wall_is_still_found(cfg):
    d = wall_scene(600, brick_z=95)
    m, info = prep.inhand_mask(d)
    assert m is not None and abs(info["z_mm"] - 95) <= 5, info
    assert frac_in(m, 720, 320, 1200, 800) > 0.99 and frac_in(m, 0, 0, 500, H) == 0


def test_autotune_picks_the_near_brick_not_the_bigger_wall(cfg):
    """Review item 1, auto-tune: emulate the sweep on a 424x240 D415 (brick at 90 mm, wall at 600 mm)."""
    full = wall_scene(600, brick_z=90)
    fx, B = D415_424
    results = []
    for sh in rd.candidate_shifts(fx, B):
        near, far = rd.depth_window(fx, B, sh)
        dw = np.where((full >= near) & (full <= far), full, 0).astype(np.uint16)
        results.append({**rd.measure(dw, STATION_FRONT_ROI, lambda x: prep.inhand_mask(x, expect_z_mm=None)),
                        **rd.window_info(fx, B, sh)})
    best = rd.pick_shift(results)
    assert best is not None and abs(best["z_mm"] - 90) <= 5, results
    assert best["near_mm"] <= 90 <= (best["far_mm"] or 1e9)
    # pick_shift alone: even if a wall cluster were masked with 7x the pixels, the nearest cluster wins
    r = lambda s, z, px: {"shift": s, "masked": True, "z_mm": z, "object_px": px}
    res = [r(0, 600, 1_800_000), r(14, 600, 1_800_000), r(42, 101, 270_000), r(72, 99, 275_000), r(114, 98, 274_000),
           r(154, 97, 120_000)]
    assert rd.pick_shift(res)["shift"] == 72


def test_tuning_flag_refuses_photo_masks(cfg):
    d = wall_scene(600, brick_z=95)
    prep.TUNING = True
    try:
        crop, info = prep.inspect_crop(np.full((H, W, 3), 90, np.uint8), d)
        assert info["masked"] is False and "auto-tune" in info["why"]
    finally:
        prep.TUNING = False
    crop, info = prep.inspect_crop(np.full((H, W, 3), 90, np.uint8), d)
    assert info["masked"] and info["inhand_cfg"]["fx"] == FX and "max_obj_mm" in info["inhand_cfg"]   # provenance


@pytest.mark.parametrize("bad", [{"width": "abc"}, {"width": 1280, "height": None}, {"fps": "x"}, {"shift": "lots"},
                                 {"near_mm": -4}, {"expect_z_mm": "far"}, {"advanced": "yes"}, {"max_obj_mm": float("nan")},
                                 "not-a-dict", ["list"], {"width": True}])
def test_resolve_config_never_raises(bad):
    c = rd.resolve_config(bad)
    assert (c["width"], c["height"]) in rd.MODES and c["fps"] in (6, 15, 30, 60, 90)
    assert c["config_warning"] and isinstance(c["advanced"], bool)
    assert c["shift"] is None or 0 <= c["shift"] <= rd.SHIFT_MAX
    assert rd.resolve_config({"width": 640, "height": 480, "shift": "12"})["config_warning"] is None


def test_missing_baseline_does_not_crash_the_shift_helpers():
    assert rd.candidate_shifts(305, None) == [0]
    assert rd.initial_shift(rd.resolve_config({"near_mm": 80}), 305, None)[0] == 0
    assert rd.initial_shift(rd.resolve_config({"shift": 9}), None, None)[0] == 9


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (10, 10), (15, 200), (16, 16)])
def test_tiny_depth_images_do_not_crash(cfg, shape):
    d = np.full(shape, 300, np.uint16)
    m, info = prep.inhand_mask(d)
    assert m is None or m.shape == d.shape


def test_threshold_zero_does_not_produce_nan_reasons(cfg):
    d = np.zeros((64, 64), np.uint16); d[::9, ::7] = 700                # sparse, min_px tiny -> thr could be 0
    m, why = prep.inhand_mask(d, min_px=1, max_obj_mm=None)
    assert "nan" not in str(why).lower()


# ---- RSCam with a fake pyrealsense2 (start-up paths that can't run here otherwise)
class FakeRSModule:
    """pyrealsense2 stand-in for capture.RSCam: pipeline/config/align/streams/options/advanced mode. Records events."""
    def __init__(self, adv=True, shift=0, baseline=55.0, back_after=2):
        self.adv, self.shift, self.baseline, self.events = adv, shift, baseline, []
        self.present, self.polls, self.back_after = True, 0, back_after
        rs = self
        self.option = type("option", (), {n: n for n in ("stereo_baseline", "enable_auto_exposure",
                                                          "enable_auto_white_balance", "gain", "white_balance", "exposure")})
        self.stream = type("stream", (), {"color": "color", "depth": "depth"})
        self.format = type("format", (), {"bgr8": "bgr8", "z16": "z16"})
        self.camera_info = type("ci", (), {"serial_number": "sn"})

        class Intr:
            def __init__(self, fx): self.fx = fx

        class SP:
            def __init__(self, kind): self.kind = kind
            def as_video_stream_profile(self): return self
            def get_intrinsics(self): return Intr(305.0 if self.kind == "depth" else 1380.0)

        class Sensor:
            def set_option(self, *a): pass
            def get_depth_scale(self): return 0.001
            def get_option(self, o):
                if rs.baseline is None: raise RuntimeError("option not supported")
                return -rs.baseline                                    # librealsense reports it negative

        class Dev:
            def get_info(self, _): return "243622061066"
            def first_depth_sensor(self): return Sensor()
            def first_color_sensor(self): return Sensor()

        class Prof:
            def get_device(self): return Dev()
            def get_stream(self, k): return SP(k)

        class Frames:
            def get_color_frame(self): return self
            def get_data(self): return np.zeros((4, 4, 3), np.uint8)

        class Pipe:
            def __init__(self): rs.events.append("pipeline")
            def start(self, cfg): rs.events.append("start"); return Prof()
            def stop(self): pass
            def wait_for_frames(self, *a): return Frames()

        class Cfg:
            def enable_device(self, s): pass
            def enable_stream(self, *a): rs.events.append(("stream",) + a[:3])

        class Adv:
            def __init__(self, dev): pass
            def is_enabled(self): return rs.adv
            def toggle_advanced_mode(self, on): rs.events.append("toggle"); rs.present = False
            def get_depth_table(self): return type("T", (), {"disparityShift": rs.shift})()
            def set_depth_table(self, t): rs.shift = t.disparityShift; rs.events.append(("shift", rs.shift))

        class Ctx:
            def query_devices(self):
                rs.polls += 1
                if not rs.present and rs.polls >= rs.back_after: rs.present, rs.adv = True, True
                return [Dev()] if rs.present else []

        self.pipeline, self.config, self.context, self.rs400_advanced_mode = Pipe, Cfg, Ctx, Adv
        self.align = lambda s: object()


@pytest.fixture
def cap(tmp_path, monkeypatch):
    """capture.py imported for real (not mock mode) with a fake pyrealsense2 and a temp settings file."""
    monkeypatch.syspath_prepend(str(AC))
    monkeypatch.delenv("JUNCTION_MOCK", raising=False)
    key = "_depthtest_capture"
    if key not in sys.modules:
        monkeypatch.setitem(sys.modules, "pyrealsense2", FakeRSModule())
        spec = importlib.util.spec_from_file_location(key, AC / "capture.py")
        mod = importlib.util.module_from_spec(spec); sys.modules[key] = mod; spec.loader.exec_module(mod)
    mod = sys.modules[key]
    assert not mod.MOCK
    monkeypatch.setattr(mod, "SETTINGS", tmp_path / "cam_settings.json")

    def make(front_depth=None, **kw):
        s = {"front": {"exposure": 300, "gain": 96.0, "white_balance": 4750}}
        if front_depth is not None: s["front_depth"] = front_depth
        (tmp_path / "cam_settings.json").write_text(json.dumps(s))
        rs = FakeRSModule(**kw); monkeypatch.setattr(mod, "rs", rs)
        return mod.RSCam("front"), rs
    return make


def test_rscam_bad_settings_fall_back_to_defaults(cap):
    c, rs = cap({"width": "abc", "shift": "x"})
    d = c.depth_info()
    assert d["enabled"] and (d["width"], d["height"]) == (424, 240) and "width" in d["warning"]
    assert ("stream", "depth", 424, 240) in rs.events


def test_rscam_clears_a_stale_shift_when_advanced_is_off(cap):
    c, rs = cap({"advanced": False}, adv=True, shift=80)               # device still in advanced mode, old shift
    assert rs.shift == 0 and c.depth_info()["shift"] == 0 and not c.depth_info()["advanced"]


def test_rscam_applies_the_configured_shift_and_reports_the_window(cap):
    c, rs = cap({"near_mm": 80})
    d = c.depth_info()
    assert d["advanced"] and rs.shift == d["shift"] == rd.shift_for_near(305, 55, 80)[0]
    assert d["near_mm"] <= 80 and d["baseline_mm"] == 55.0 and d["fx_color"] == 1380.0


def test_rscam_without_baseline_still_starts(cap):
    c, rs = cap({"near_mm": 80}, baseline=None)
    d = c.depth_info()
    assert d["enabled"] and d["shift"] == 0 and d["near_mm"] is None and "baseline" in d["warning"]


def test_rscam_creates_the_pipeline_after_the_advanced_mode_reset(cap):
    c, rs = cap(None, adv=False)                                       # first start: toggle -> reset -> back
    ev = [e for e in rs.events if e in ("toggle", "pipeline", "start")]
    assert ev[:3] == ["toggle", "pipeline", "start"], rs.events
    assert c.depth_info()["advanced"]


# ---- mock server: bad settings, exclusivity, provenance
def test_mock_starts_with_bad_front_depth_settings(tmp_path):
    from conftest import MockServer
    root = tmp_path / "mock_bad"; (root / "settings").mkdir(parents=True)
    (root / "settings" / "cam_settings.json").write_text(json.dumps({"front_depth": {"width": "abc", "fps": "x"}}))
    srv = MockServer(root)
    try:
        st = srv.get("/depth/status")
        assert st["available"] and st["width"] == 424 and "width" in st["warning"]
        assert "front" in srv.get("/status")["live"]
    finally:
        srv.stop()


def test_autotune_is_exclusive_with_arm_jobs(connected):
    import threading
    srv = connected
    srv.goto("wp2", 0)
    # 1. refused while an arm routine runs
    srv.post("/arm/run", cycles=1, label="good", brick="blue", speed=0.05, cams="front", save=False, routine="wp2")
    srv.wait(lambda: srv.get("/arm/status")["running"], 10, "routine running")
    r = srv.post("/depth/autotune")
    assert not r["ok"] and "busy" in r["msg"], r
    assert not srv.post("/depth/reset")["ok"]
    srv.post("/arm/stop")
    srv.wait(lambda: not srv.get("/arm/status")["running"], 30, "routine stopped")
    srv.goto("wp2", 0)
    # 2. while tuning: ROI / reset / arm runs are refused
    out = {}
    t = threading.Thread(target=lambda: out.update(r=srv.post("/depth/autotune"))); t.start()
    srv.wait(lambda: srv.get("/depth/status")["tuning"], 10, "tuning")
    roi0 = srv.get("/status")["roi"]["front"]
    s = srv.post("/roi/front", x0=10, y0=10, x1=900, y1=900)
    assert s["roi"]["front"] == roi0 and "auto-tune" in s["msg"]
    assert not srv.post("/depth/reset")["ok"]
    assert srv.get("/arm/status")["running"]                         # the tune holds the arm's busy flag
    srv.post("/arm/run", cycles=1, label="good", brick="blue", speed=1.0, cams="front", save=True, routine="wp2")
    t.join(60)
    assert out["r"]["ok"], out
    assert not srv.get("/arm/status")["running"] and not srv.get("/depth/status")["tuning"]
    assert srv.post("/depth/reset")["ok"]


def test_capture_meta_records_why_there_is_no_mask(connected):
    srv = connected
    srv.goto("wp2", 0)
    srv.post("/mock/camera", cam="front", depth=0)
    try:
        srv.post("/arm/run", cycles=1, label="good", brick="nodepth", speed=1.0, cams="front", save=True, routine="wp2")
        srv.wait(lambda: not srv.get("/arm/status")["running"] and srv.get("/arm/status")["msg"].startswith("✓"), 90, "capture run")
    finally:
        srv.post("/mock/camera", cam="front", depth=1)
    lines = [json.loads(l) for l in (srv.root / "training_images" / "meta.jsonl").read_text().splitlines() if l.strip()]
    nod = [l for l in lines if l.get("brick") == "nodepth" and l["cam"] == "front"]
    assert nod and all(l["depth"] is None and l["depth_mask_why"] == "no depth" for l in nod)
    assert all(l["inhand_cfg"]["roi"] and l["inhand_cfg"]["max_obj_mm"] for l in nod)


# ======================================================================== depth background (table in front of the camera)
def table_depth(h=150.0, tilt_deg=30.0, f=FX, seed=30, noise=1.0, plate=True):
    """Aligned depth of a table top seen by a camera h mm above it, tilted down tilt_deg (z 180-900 mm in the lower
    part of the frame, as on the station), with a plate rim standing 15 mm off it."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.radians(tilt_deg)
    den = (v - H / 2) / f * np.cos(t) + np.sin(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(den > 1e-3, h / den, 0)
    rng = np.random.default_rng(seed)
    z = np.where((z > 0) & (z < 2500), z + rng.normal(0, noise, z.shape), 0)
    if plate:                                                         # a plate rim: a band 15 mm nearer
        rim = (np.abs(np.hypot(u - 500, (v - 980) * 2.2) - 260) < 14) & (z > 0)
        z[rim] -= 15
    return z.clip(0, 65535).astype(np.uint16)


def held_on_table(table, z0=110, seed=31):
    """Gripper (body at z0+25) + brick (z0) held in front of the table, in the detection area."""
    d = table.copy(); rng = np.random.default_rng(seed)
    box(d, 900, 330, 1450, 560, z0 + 25, rng)                        # gripper body
    box(d, 980, 560, 1380, 900, z0, rng)                              # brick between the jaws
    return d


def test_table_alone_is_refused_without_a_background(cfg):
    """The station finding: nothing held, /depth/status said mask_ok with z 209 (the table)."""
    t = table_depth()
    for roi in (None, STATION_FRONT_ROI):
        m, why = prep.inhand_mask(t, roi=roi, bg=None)
        assert m is None, (roi, why)
    m, why = prep.inhand_mask(t, roi=STATION_FRONT_ROI, bg=None, max_obj_mm=None)
    assert m is None and "flat surface" in why, why                    # the plane rule itself, not the distance limit


def test_table_alone_is_refused_with_a_background(cfg):
    bg = prep.median_depth([prep.small_depth(table_depth(seed=s)) for s in range(5)])
    m, why = prep.inhand_mask(table_depth(seed=99), roi=STATION_FRONT_ROI, bg=bg, max_obj_mm=None)
    assert m is None and "background" in why, why


def test_brick_held_in_front_of_the_table_is_kept_with_a_background(cfg):
    t = table_depth()
    prep.set_background(prep.median_depth([prep.small_depth(table_depth(seed=s)) for s in range(5)]), {"file": "t.png"})
    try:
        d = held_on_table(table_depth(seed=7))
        m, info = prep.inhand_mask(d, roi=STATION_FRONT_ROI)
        assert m is not None and abs(info["z_mm"] - 110) <= 30 and info["bg"], info
        assert frac_in(m, 990, 570, 1370, 890) > 0.99                   # the brick
        tb = (t > 0) & (np.abs(d.astype(int) - t.astype(int)) < 3)       # table pixels (not covered by the object)
        tb[300:930, 880:1470] = False
        assert (m[tb] > 0).mean() < 0.002                               # the table stays grey
        crop, ci = prep.inspect_crop(np.full((H, W, 3), 100, np.uint8), d)
        assert ci["masked"] and ci["inhand_cfg"]["bg"]["file"] == "t.png"
    finally:
        prep.set_background(None)


def test_background_of_another_size_is_ignored(cfg):
    d = held_on_table(table_depth())
    m, info = prep.inhand_mask(d, bg=np.zeros((100, 100), np.uint16), roi=STATION_FRONT_ROI)
    assert m is not None and not info["bg"] and "ignored" in info["bg_note"]


def test_background_status_checks_the_depth_config():
    desc = {"width": 424, "height": 240, "shifts": [0, 14, 42, 72], "fx_color": 1363.9}
    di = {"width": 424, "height": 240, "shift": 42, "fx_color": 1363.9}
    assert rd.background_status(desc, di) == (True, None)
    ok, why = rd.background_status(desc, {**di, "width": 1280, "height": 720}); assert not ok and "424x240" in why
    ok, why = rd.background_status(desc, {**di, "shift": 150}); assert not ok and "shift" in why
    ok, why = rd.background_status({**desc, "shifts": [0]}, {**di, "shift": 42}); assert not ok
    ok, why = rd.background_status(desc, {**di, "fx_color": 1380.0}); assert not ok and "fx" in why
    ok, why = rd.background_status(None, di); assert not ok and "store" in why


def test_median_and_merge_backgrounds():
    a = np.array([[100, 0, 0], [200, 0, 300]], np.uint16); b = np.array([[102, 0, 50], [0, 0, 300]], np.uint16)
    c = np.array([[98, 0, 52], [204, 7, 0]], np.uint16)
    m = prep.median_depth([a, b, c])
    assert m[0, 0] == 100 and m[0, 1] == 0 and m[0, 2] == 51 and m[1, 0] == 202 and m[1, 1] == 0   # 1 of 3 valid -> 0
    near = np.array([[0, 90]], np.uint16); far = np.array([[400, 0]], np.uint16)
    assert prep.merge_backgrounds([near, far]).tolist() == [[400, 90]]


def test_mock_background_with_another_config_is_ignored(connected):
    srv = connected
    srv.goto("home", 0)
    if not srv.get("/depth/status")["background"]["usable"]: assert srv.post("/depth/background", frames=6)["ok"]
    p = srv.root / "settings" / "cam_settings.json"
    cs = json.loads(p.read_text()); good = json.loads(json.dumps(cs))
    cs["front_depth"]["background"].update(width=1280, height=720); p.write_text(json.dumps(cs, indent=2))
    srv.post("/roi/front/reset")                                        # any settings sync re-checks the background
    st = srv.get("/depth/status")["background"]
    assert not st["usable"] and "1280x720" in st["warning"] and srv.get("/depth/status")["inhand"]["bg"] is None
    p.write_text(json.dumps(good, indent=2)); srv.post("/roi/front/reset")
    assert srv.get("/depth/status")["background"]["usable"]
    assert srv.post("/depth/background/clear")["ok"] and not srv.get("/depth/status")["background"]["usable"]


# ======================================================================== reprocess_inhand.py
def test_reprocess_inhand_script(tmp_path, cfg):
    rp = _load("reprocess_inhand")
    data = tmp_path / "training_images"; d = data / "front" / "inhand" / "good"; d.mkdir(parents=True)
    st = tmp_path / "settings"; st.mkdir()
    table = table_depth()
    cv2.imwrite(str(st / "front_depth_bg.png"), prep.median_depth([prep.small_depth(table_depth(seed=s)) for s in range(3)]))
    dcfg = {"width": 424, "height": 240, "shift": 0, "fx_color": FX}
    (st / "cam_settings.json").write_text(json.dumps({"roi": {"front": STATION_FRONT_ROI}, "front_depth": {
        "background": {"file": "front_depth_bg.png", "width": 424, "height": 240, "shifts": [0, 14], "fx_color": FX,
                       "taken": "2026-11-14T20:00:00"}}}))
    img = np.full((H, W, 3), 120, np.uint8)
    meta = []
    for stem, dep, t in (("20261114_190000_000_blue_wp2-1", table_depth(seed=5), "20261114_190000_000"),   # nothing held
                         ("20261114_190100_000_blue_wp2-1", held_on_table(table_depth(seed=6)), "20261114_190100_000"),
                         ("20261114_180000_000_blue_wp2-1", held_on_table(table_depth(seed=8)), "20261114_180000_000")):
        cv2.imwrite(str(d / f"{stem}.png"), img); cv2.imwrite(str(d / f"{stem}_depth.png"), dep)
        cv2.imwrite(str(d / f"{stem}_crop.png"), img[:50, :50]); cv2.imwrite(str(d / f"{stem}_masked.jpg"), img[:50, :50])
        meta.append({"file": f"front/inhand/good/{stem}.png", "stage": "inhand", "cam": "front", "time": t, "depth_cfg": dcfg,
                     "depth": {"z_mm": 209}})
    meta.append({"file": "wrist/inhand/good/x.png", "stage": "release", "cam": "wrist", "time": "20261114_190200_000"})
    (data / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    meta_before = (data / "meta.jsonl").read_text()
    # dry run: nothing written
    r = rp.reprocess(data, st / "cam_settings.json", since="20261114_19", log=lambda *a: None)
    assert r["stats"] == {"masked": 1, "refused": 1, "crop removed (mask now refused)": 1}, r
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before and (data / "meta.jsonl").read_text() == meta_before
    # apply, only the captures since 19:00
    r = rp.reprocess(data, st / "cam_settings.json", apply=True, since="20261114_19", log=lambda *a: None)
    tbl, held, old = "20261114_190000_000_blue_wp2-1", "20261114_190100_000_blue_wp2-1", "20261114_180000_000_blue_wp2-1"
    assert not (d / f"{tbl}_crop.png").exists() and not (d / f"{tbl}_masked.jpg").exists()      # table: no crop left
    assert (d / f"{held}_crop.png").exists() and (d / f"{held}_crop.png").read_bytes() != before[f"{held}_crop.png"]
    assert (d / f"{old}_crop.png").read_bytes() == before[f"{old}_crop.png"]                      # before --since: untouched
    for stem in (tbl, held):                                                                       # raw + depth untouched
        assert (d / f"{stem}.png").read_bytes() == before[f"{stem}.png"]
        assert (d / f"{stem}_depth.png").read_bytes() == before[f"{stem}_depth.png"]
    bk = Path(r["backup"])
    assert (bk / "meta.jsonl").read_text() == meta_before
    assert (bk / "front/inhand/good" / f"{tbl}_crop.png").read_bytes() == before[f"{tbl}_crop.png"]
    assert not list(bk.glob("front/inhand/good/*_crop.png")) == []                                 # backups kept ...
    assert not list((data / "front" / "inhand" / "good").glob("*.bak"))                            # ... outside the trainer's folder
    recs = {json.loads(l)["file"].split("/")[-1][:-4]: json.loads(l) for l in (data / "meta.jsonl").read_text().splitlines()}
    assert recs[tbl]["reprocessed"]["masked"] is False and recs[tbl]["depth"] is None and recs[tbl]["depth_mask_why"]
    assert recs[held]["reprocessed"]["masked"] and recs[held]["reprocessed"]["background"]["file"] == "front_depth_bg.png"
    assert abs(recs[held]["depth"]["z_mm"] - 110) <= 30 and "reprocessed" not in recs[old]
    assert len(recs) == 4 and recs["x"]["stage"] == "release"
