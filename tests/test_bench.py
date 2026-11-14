"""Tests for bench/ (Phase 7 measurement framework). Self-contained: no conftest fixtures, no hardware.

Tiny OpenVINO models are built with the openvino Python API; sensors run against a fake sysfs tree in tmp_path;
the sorter cycle log is synthetic but in sorter.py's real format (runs/<session>/log.jsonl + settings.json).

    .venv/bin/python -m pytest tests/test_bench.py -q
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ov = pytest.importorskip("openvino")
cv2 = pytest.importorskip("cv2")

from bench import bench_model, cycle_stats, devices, quantize, report, run_all, sensors  # noqa: E402


# ---------------------------------------------------------------------- helpers
def tiny_anomaly_model(path, dynamic=True):
    """conv -> relu -> mean = 'anomaly_map', max = 'pred_score' (the names Anomalib's export uses)."""
    from openvino import opset13 as ops
    p = ops.parameter(ov.PartialShape([-1, 3, -1, -1] if dynamic else [1, 3, 32, 32]), ov.Type.f32, name="input")
    w = np.random.default_rng(0).standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.2
    c = ops.relu(ops.convolution(p, ops.constant(w), [1, 1], [1, 1], [1, 1], [1, 1]))
    amap = ops.reduce_mean(c, ops.constant(np.array([1])), True)
    score = ops.reduce_max(amap, ops.constant(np.array([1, 2, 3])), False)
    score.output(0).get_tensor().set_names({"pred_score"}); amap.output(0).get_tensor().set_names({"anomaly_map"})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov.Model([score, amap], [p], "tiny"), str(path))
    return Path(path)


def tiny_policy_model(path):
    """Two inputs like an ACT export (image + state), static shapes."""
    from openvino import opset13 as ops
    img = ops.parameter([1, 3, 24, 32], ov.Type.f32, name="observation.images.top")
    st = ops.parameter([1, 6], ov.Type.f32, name="observation.state")
    f = ops.reduce_mean(img, ops.constant(np.array([2, 3])), False)
    out = ops.matmul(ops.concat([f, st], 1), ops.constant(np.ones((9, 6), np.float32)), False, False)
    out.output(0).get_tensor().set_names({"action"})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov.Model([out], [img, st], "act"), str(path))
    return Path(path)


def write_images(d, n=5, size=40, prefix="img"):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"{prefix}{i}_crop.png"), (np.random.default_rng(i).random((size, size, 3)) * 255).astype(np.uint8))
    return sorted(d.glob("*.png"))


def w(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(str(v) + "\n")


def fake_sysfs(root):
    r = Path(root)
    pc = r / "sys/class/powercap"
    for z, name, e in (("intel-rapl:0", "package-0", 1_000_000), ("intel-rapl:0:0", "core", 500_000), ("intel-rapl:0:1", "uncore", 100_000),
                       ("intel-rapl:1", "psys", 2_000_000)):
        w(pc / z / "name", name); w(pc / z / "energy_uj", e); w(pc / z / "max_energy_range_uj", 262143328850)
    w(pc / "intel-rapl-mmio:0/name", "package-0"); w(pc / "intel-rapl-mmio:0/energy_uj", 5)       # must be ignored
    tz = r / "sys/class/thermal"
    w(tz / "thermal_zone0/type", "acpitz"); w(tz / "thermal_zone0/temp", 40000)
    w(tz / "thermal_zone0/trip_point_0_type", "critical"); w(tz / "thermal_zone0/trip_point_0_temp", 110000)
    w(tz / "thermal_zone1/type", "x86_pkg_temp"); w(tz / "thermal_zone1/temp", 45000)
    w(tz / "thermal_zone2/type", "broken")                                           # no temp file: unreadable, skipped
    hw = r / "sys/class/hwmon/hwmon0"
    w(hw / "name", "coretemp"); w(hw / "temp1_input", 46000); w(hw / "temp1_label", "Package id 0"); w(hw / "temp1_crit", 100000)
    w(hw / "temp2_input", 44000); w(hw / "temp2_label", "Core 0"); w(hw / "temp2_crit", 100000)
    for c in range(2):
        w(r / f"sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq", 2_000_000 + c * 1_000_000)
        w(r / f"sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq", 5_100_000)
    w(r / "proc/stat", "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 1 1 1")
    g = r / "sys/class/drm/card0/device"
    w(g / "uevent", "DRIVER=xe\nPCI_ID=8086:B080")
    w(g / "tile0/gt0/freq0/act_freq", 300); w(g / "tile0/gt0/freq0/rp0_freq", 2900)
    w(g / "tile0/gt0/gtidle/name", "gt0-rc"); w(g / "tile0/gt0/gtidle/idle_residency_ms", 10_000)
    w(r / "sys/class/drm/card0-eDP-1/status", "connected")                            # connector: ignored
    n = r / "sys/class/accel/accel0/device"
    w(n / "uevent", "DRIVER=intel_vpu"); w(n / "npu_busy_time_us", 0); w(n / "freq/current_freq", 0); w(n / "freq/hw_max_freq", 1950)
    w(n / "npu_memory_utilization", 0); w(n / "power/runtime_status", "suspended")
    w(r / "sys/class/power_supply/BAT0/type", "Battery"); w(r / "sys/class/power_supply/BAT0/status", "Discharging")
    w(r / "sys/class/power_supply/BAT0/power_now", 12_500_000)
    w(r / "sys/class/power_supply/AC/type", "Mains"); w(r / "sys/class/power_supply/AC/online", 0)
    return r


class FakePower(sensors.Sensors):
    """No sysfs: a package energy counter that integrates `self.watts`, and a temperature that follows the load."""

    def __init__(self):
        super().__init__(root="/nonexistent-sysfs-root")
        self.watts, self.e, self.t = 2.0, 0.0, time.monotonic()
        self._add("rapl.package-0", "energy_uj", "W", "fake package", max_range=1e12)
        self._add("hw.coretemp.Package_id_0", "temp_c", "C", "fake pkg temp", crit=100.0, hwmon="coretemp")
        self.readers = [self._rd]

    def _rd(self):
        now = time.monotonic(); self.e += self.watts * (now - self.t) * 1e6; self.t = now
        return {"rapl.package-0": self.e, "hw.coretemp.Package_id_0": 40 + 2 * self.watts}


# ---------------------------------------------------------------------- sensors
def test_sensors_fake_sysfs_discovery(tmp_path):
    s = sensors.Sensors(fake_sysfs(tmp_path))
    m = s.meta
    for k in ("rapl.package-0", "rapl.core", "rapl.uncore", "rapl.psys", "tz.acpitz", "tz.x86_pkg_temp", "hw.coretemp.Package_id_0",
              "cpu.freq_mean_mhz", "cpu.busy_j", "gpu0.tile0.gt0.act_mhz", "gpu0.tile0.gt0.idle_ms", "npu.busy_us", "npu.freq_mhz",
              "npu.mem_bytes", "bat.BAT0.w"):
        assert k in m, k
    assert not any("mmio" in v.get("zone", "") for v in m.values())
    assert m["tz.acpitz"]["crit"] == 110.0 and m["hw.coretemp.Package_id_0"]["crit"] == 100.0
    assert s.info["npu_max_mhz"] == 1950 and s.info["npu_runtime_status"] == "suspended"
    assert s.unavailable == {}
    r = s.read()
    assert r["hw.coretemp.Package_id_0"] == 46.0 and r["cpu.freq_mean_mhz"] == 2500.0 and r["bat.BAT0.w"] == 12.5
    assert s.battery_status() == {"AC": "offline", "BAT0": "Discharging"}


def test_sensors_summary_power_busy_temp(tmp_path):
    root = fake_sysfs(tmp_path); s = sensors.Sensors(root)
    a = s.read()
    pc = root / "sys/class/powercap"
    w(pc / "intel-rapl:0/energy_uj", 1_000_000 + 20_000_000)                # +20 J in 2 s = 10 W
    w(pc / "intel-rapl:1/energy_uj", 2_000_000 + 30_000_000)                # psys 15 W
    w(root / "sys/class/accel/accel0/device/npu_busy_time_us", 1_500_000)  # 1.5 s busy of 2 s = 75 %
    w(root / "sys/class/drm/card0/device/tile0/gt0/gtidle/idle_residency_ms", 11_500)   # idle 1.5 of 2 s -> 25 % busy
    w(root / "sys/class/hwmon/hwmon0/temp1_input", 71000)
    w(root / "proc/stat", "cpu  200 0 200 900 0 0 0 0 0 0")                 # +200 busy of +300 total
    b = s.read()
    summ = sensors.summarize([(0.0, a), (2.0, b)], s.meta)
    assert summ["power_w"]["rapl.package-0"] == pytest.approx(10.0) and summ["power_w"]["rapl.psys"] == pytest.approx(15.0)
    assert summ["power"]["source"] == "rapl.package-0" and summ["power"]["energy_j"] == pytest.approx(20.0)
    assert summ["busy_pct"]["npu"] == pytest.approx(75.0) and summ["busy_pct"]["gpu0.tile0.gt0"] == pytest.approx(25.0)
    assert summ["busy_pct"]["cpu"] == pytest.approx(66.67, abs=0.01)
    ct = summ["cpu_temp"]
    assert ct["source"] == "hw.coretemp.Package_id_0" and ct["peak"] == 71.0 and ct["rise"] == 25.0
    assert ct["limit"] == 100.0 and ct["headroom"] == 29.0
    idle = sensors.summarize([(0.0, a), (2.0, a)], s.meta)                  # 0 W idle (no counter change)
    s2 = sensors.summarize([(0.0, a), (2.0, b)], s.meta, idle)
    assert s2["power"]["w_above_idle"] == pytest.approx(10.0) and s2["cpu_temp"]["above_idle"] == pytest.approx(25.0)
    assert "W" in sensors.headline(s2) and "headroom" in sensors.headline(s2)


def test_rapl_wraparound():
    mr = 262143328850
    assert sensors.counter_delta(mr - 1_000_000, 1_000_000, mr) == 2_000_000
    assert sensors.counter_delta(5, 3, None) is None                         # unknown range: counter reset, dropped
    meta = {"rapl.package-0": {"kind": "energy_uj", "max_range": mr, "unit": "W", "label": ""}}
    samples = [(0.0, {"rapl.package-0": mr - 1_000_000}), (1.0, {"rapl.package-0": 1_000_000}), (2.0, {"rapl.package-0": 3_000_000})]
    assert sensors.summarize(samples, meta)["power_w"]["rapl.package-0"] == pytest.approx(2.0)


def test_rapl_permission_denied(tmp_path, monkeypatch):
    root = fake_sysfs(tmp_path)
    real = sensors._read

    def deny(p):
        if str(p).endswith("energy_uj"): raise PermissionError(13, "Permission denied", str(p))
        return real(p)
    monkeypatch.setattr(sensors, "_read", deny)
    s = sensors.Sensors(root)
    assert not any(k.startswith("rapl.") for k in s.meta)
    assert "PLATYPUS" in s.unavailable["rapl"] and sensors.RAPL_FIX in s.fixes
    assert "hw.coretemp.Package_id_0" in s.meta                             # the rest still works


def test_missing_sources_never_crash(tmp_path):
    s = sensors.Sensors(tmp_path / "empty")
    assert s.meta == {} and s.read() == {}
    for g in ("rapl", "thermal", "hwmon", "gpu", "npu", "battery"): assert g in s.unavailable
    smp = sensors.Sampler(s, 0.01).start(); time.sleep(0.05); smp.stop()
    summ = smp.summary()
    assert summ["n_samples"] >= 2 and "power" not in summ and "cpu_temp" not in summ
    assert sensors.headline(summ) == "no power/temperature sources"
    assert sensors.wait_cool(s, summ, 1.0) == 0.0
    s.readers.append(lambda: 1 / 0)                                          # a broken reader is skipped
    assert s.read() == {}


def test_sampler_thread_with_fake_power():
    s = FakePower()
    idle = sensors.measure_idle(s, 0.3, 0.02)
    s.watts = 7.0
    smp = sensors.Sampler(s, 0.02).start(); time.sleep(0.3); smp.stop()
    summ = smp.summary(idle)
    assert idle["power"]["w"] == pytest.approx(2.0, rel=0.05)
    assert summ["power"]["w_above_idle"] == pytest.approx(5.0, rel=0.05)
    assert summ["series"]["power_w"] and summ["series"]["cpu_temp_c"]


# ---------------------------------------------------------------------- devices
def test_devices_report(tmp_path):
    r = devices.report(root=fake_sysfs(tmp_path))
    assert "CPU" in r["openvino"]["available"] and r["openvino"]["devices"]["CPU"]["FULL_DEVICE_NAME"]
    assert isinstance(r["openvino"]["devices"]["CPU"].get("INFERENCE_PRECISION_HINT", "f32"), str)
    n = r["npu"]
    if not n["ok"]:
        assert n["fix"][:3] == devices.NPU_FIX and "dpkg -i ~/Downloads/npu-driver-v1.38.0/*.deb" in n["fix"][1]
    txt = devices.text(r)
    assert "OpenVINO" in txt and "NPU" in txt
    json.dumps(r, default=str)


# ---------------------------------------------------------------------- bench_model
def test_static_shapes_and_precision(tmp_path):
    m = ov.Core().read_model(str(tiny_anomaly_model(tmp_path / "m.xml")))
    shapes, notes = bench_model.static_shapes(m, None, 64)
    assert shapes == {0: [1, 3, 64, 64]}
    assert bench_model.static_shapes(m, None, (48, 40))[0] == {0: [1, 3, 48, 40]}
    assert bench_model.static_shapes(m, [2, 3, 16, 16])[0] == {0: [2, 3, 16, 16]}
    assert bench_model.precision_of(m) == (None, "FP16")                    # save_model compresses weights to FP16
    assert bench_model.parse_shape("a=1,3,8,8;b=1,6") == {"a": [1, 3, 8, 8], "b": [1, 6]}


def test_make_feeds_real_images(tmp_path):
    m = ov.Core().read_model(str(tiny_anomaly_model(tmp_path / "m.xml")))
    imgs = write_images(tmp_path / "imgs", 3, 50)
    feeds, desc = bench_model.make_feeds(m, {0: [1, 3, 32, 32]}, imgs, k=8)
    assert len(feeds) == 3 and "real images" in desc
    a = feeds[0][0]
    assert a.shape == (1, 3, 32, 32) and a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1
    feeds, desc = bench_model.make_feeds(m, {0: [1, 3, 32, 32]}, [], k=4)
    assert len(feeds) == 4 and desc.startswith("random")


def test_bench_models_cpu_end_to_end(tmp_path):
    mp = tiny_anomaly_model(tmp_path / "models/front/red/weights/openvino/model.xml")
    imgs = write_images(tmp_path / "imgs", 4, 36)
    pol = tiny_policy_model(tmp_path / "act.xml")
    sens = FakePower()
    sess = bench_model.bench_models(
        [{"path": str(mp), "tag": "front/red", "images": str(tmp_path / "imgs")}, {"path": str(pol), "tag": "policy/act", "role": "policy"},
         {"path": str(tmp_path / "missing.xml"), "tag": "missing"}],
        devices=("CPU", "NPU", "AUTO"), hints=("LATENCY", "THROUGHPUT"), precisions=("default", "f32"), n=20, min_s=0.2, warmup=2,
        async_s=0.2, idle_s=0.3, cooldown_s=0, interval=0.02, out_dir=tmp_path / "res", sensors=sens, log=lambda *a: None)
    d = json.loads(sess.path.read_text())
    assert d["kind"] == "bench_session" and d["idle"]["power"]["w"] == pytest.approx(2.0, rel=0.1)
    assert "NPU" in d["skipped_devices"] or "NPU" in ov.Core().available_devices
    ok = [r for r in d["results"] if not r.get("error")]
    assert any(r.get("error", "").startswith("read_model") for r in d["results"])
    r = next(r for r in ok if r["model"]["tag"] == "front/red" and r["device"] == "CPU" and r["hint"] == "LATENCY" and r["precision_requested"] == "default")
    assert r["sync"]["n"] >= 20 and r["sync"]["p50_ms"] > 0 and r["sync"]["p99_ms"] >= r["sync"]["p50_ms"]
    assert r["async"]["fps"] > 0 and r["compile_ms"] > 0 and r["first_infer_ms"] > 0
    assert r["model"]["static_shapes"]["input"] == [1, 3, 36, 36]           # dynamic -> the real image size
    assert any("reshaped" in n for n in r["shape_notes"]) and "real images" in r["model"]["data"]
    assert len(r["model"]["sha256_16"]) == 16 and r["openvino"] == ov.get_version()
    assert r["exec_devices"] == ["CPU"] and r["precision"] in ("FP32", "BF16", "FP16")
    assert r["energy"]["j_per_inf_above_idle"] is not None and r["energy"]["w"] > 0
    assert r["thermal"]["headroom"] is not None
    assert any(x["device"] == "AUTO" for x in ok) and any(x["model"]["tag"] == "policy/act" for x in ok)
    # f32 only where it differs from the device default (and never on AUTO)
    cpu_default = ov.Core().get_property("CPU", "INFERENCE_PRECISION_HINT").to_string()
    f32 = [x for x in ok if x["precision_requested"] == "f32"]
    assert (not f32) if cpu_default == "f32" else all(x["device"] == "CPU" and x["precision"] == "FP32" for x in f32)


# ---------------------------------------------------------------------- quantize / accuracy guard
def test_score_drift_guard_without_nncf(tmp_path):
    a = tiny_anomaly_model(tmp_path / "a/model.xml")
    m = ov.Core().read_model(str(a))
    ov.save_model(m, str(tmp_path / "b/model.xml"), compress_to_fp16=False)
    imgs = write_images(tmp_path / "imgs", 4, 32)
    w(tmp_path / "a/thresholds.json", json.dumps({"t_low": 0.05, "t_high": 0.5}))
    thr, f = quantize.find_thresholds(a)
    assert thr["t_low"] == 0.05 and f.endswith("thresholds.json")
    g = quantize.score_drift(a, tmp_path / "b/model.xml", [str(p) for p in imgs], thresholds=thr, hw=32)
    assert g["n"] == 4 and g["max_abs"] < 1e-2 and g["decision_flips"] == 0 and "spearman" in g
    assert "accuracy guard" in quantize.guard_text(g)
    lst = tmp_path / "list.txt"; lst.write_text("\n".join(map(str, imgs[:2])) + "\n/does/not/exist.png\n")
    assert len(quantize.image_files(lst)) == 2


def test_quantize_int8(tmp_path):
    pytest.importorskip("nncf")
    a = tiny_anomaly_model(tmp_path / "model.xml")
    imgs = write_images(tmp_path / "imgs", 6, 32)
    rep = quantize.quantize(a, tmp_path / "imgs", hw=32, log=lambda *x: None)
    q = Path(rep["dst"])
    assert q.name == "model_int8.xml" and q.exists() and q.with_suffix(".quant.json").exists()
    assert bench_model.precision_of(ov.Core().read_model(str(q)))[0] == "INT8"
    assert rep["guard"]["n"] == len(imgs) and rep["guard"]["max_abs"] < 0.05   # random images: tiny score range, so only an absolute bound


def test_quantize_missing_nncf_message(monkeypatch):
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (_ for _ in ()).throw(ImportError("x")) if n == "nncf" else real(n, *a, **k))
    with pytest.raises(SystemExit, match="nncf is not installed"):
        quantize._nncf()


# ---------------------------------------------------------------------- cycle_stats (sorter.py's log format)
def shot(stages, decided_by, photo_ms=600.0):
    return {"file": "01_g1_wp2_0.png", "photo_ms": photo_ms, "motion": 0.3, "stages": stages,
            "score": stages[-1]["score"], "decided_by": decided_by}


def st(stage, dev, score, ms, dec, load=None):
    return {"stage": stage, "device": dev, "score": score, "infer_ms": ms, "load_ms": load, "decision": dec}


def write_runs(root):
    """Two sessions like Sorter.run()/cycle() write them; the second has ACT picks (policy.py's act_info)."""
    s1 = root / "runs/sort_20261115_100000"; s1.mkdir(parents=True)
    (s1 / "settings.json").write_text(json.dumps({"judge": "model", "device": "GPU", "cascade": True, "fast_device": "NPU",
                                                  "fast_variant": "_efficientad", "brick": "red", "cam": "front", "pick": "handfeed", "plan": {}}))
    cyc = []
    for i, (verdict, shots_) in enumerate([
            ("good", [shot([st("fast", "NPU", 0.1, 3.0, "clean", 900.0)], "fast"), shot([st("fast", "NPU", 0.12, 3.2, "clean")], "fast")]),
            ("defect", [shot([st("fast", "NPU", 0.5, 3.1, "uncertain"), st("main", "GPU", 0.9, 20.0, "defect", 2500.0)], "main")]),
            ("aborted", [])], 1):
        steps = [{"kind": "inhand", "grip": 1, "view": "wp2_front", "cam": "front", "shows": ["front"], "shots": shots_,
                  "decision": "clean", "score": 0.1}] if shots_ else []
        inf = [s["infer_ms"] for sh in shots_ for s in sh["stages"]]
        cyc.append({"cycle": i, "time": "2026-11-15T10:00:00", "brick": "red", "judge": "model", "device": "GPU", "cascade": True,
                    "pick": "handfeed", "verdict": verdict, "reason": "x", "steps": steps, "regions": {}, "seen_pct": 100.0, "regrasps": 1,
                    "photos": len(shots_), "t_pick_s": 5.0 + i, "t_inspect_s": 10.0, "t_sort_s": 4.0, "t_cycle_s": 19.0 + i, "t_auto_s": 14.0,
                    "infer_ms_mean": float(np.mean(inf)) if inf else None, "infer_ms_max": max(inf) if inf else None,
                    "infer_ms_total": sum(inf) if inf else None, **({"lost": True} if verdict == "aborted" else {})})
    (s1 / "log.jsonl").write_text("\n".join(json.dumps(c) for c in cyc) + "\n{broken line\n")
    s2 = root / "runs/sort_20261115_110000"; s2.mkdir(parents=True)
    (s2 / "settings.json").write_text(json.dumps({"judge": "model", "device": "CPU", "cascade": False, "pick": "act", "act_device": "GPU"}))
    act = {"ok": True, "error": None, "model": "/x/exports/openvino", "backend": "openvino", "device": "GPU", "load_ms": 4000.0,
           "attempts": [{"ok": True, "t_s": 6.5, "inferences": 12, "infer_ms_p50": 14.0}], "n_attempts": 1, "inferences": 12,
           "infer_ms_p50": 14.0, "infer_ms_p95": 18.0, "infer_ms_max": 25.0, "obs_ms_p50": 3.0, "obs_ms_p95": 4.0}
    rec = {"cycle": 1, "verdict": "good", "truth": "good", "steps": [{"shots": [shot([st("main", "CPU", 0.1, 40.0, "clean")], "main")]}],
           "photos": 1, "t_pick_s": 7.0, "t_inspect_s": 9.0, "t_sort_s": 4.0, "t_cycle_s": 20.0, "t_auto_s": 13.0,
           "act": act, "act_infer_ms_p50": 14.0, "act_infer_ms_p95": 18.0, "act_attempts": 1, "new_future_thing_ms": 1.5}
    (s2 / "log.jsonl").write_text(json.dumps(rec) + "\n")
    (root / "truth.csv").write_text("session,cycle,truth\nsort_20261115_100000,1,good\nsort_20261115_100000,2,good\nsort_20261115_100000,3,defect\n")
    return root / "runs"


def test_cycle_stats_real_format(tmp_path):
    runs = write_runs(tmp_path)
    r = cycle_stats.collect([runs], tmp_path / "truth.csv")
    assert r["n_cycles"] == 4 and r["bad_lines"] == 1
    assert r["verdicts"] == {"good": 2, "defect": 1, "aborted": 1} and r["success_rate"] == 0.75
    assert r["phases_s"]["t_cycle_s"]["mean"] == pytest.approx((20 + 21 + 22 + 20) / 4)
    assert r["inference_ms"]["fast@NPU"]["n"] == 3 and r["inference_ms"]["main@GPU"]["mean"] == 20.0
    assert r["load_ms"]["main@GPU"]["mean"] == 2500.0
    c = r["cascade"]
    assert c["photos"] == 3 and c["fast_settled"] == 2 and c["escalated"] == 1 and c["fast_settled_share"] == pytest.approx(0.6667, abs=1e-3)
    a = r["accuracy"]                       # truth: 1 good->good (TN), 2 good->defect (FP), 3 defect->aborted, s2 good->good (TN)
    assert a["labelled"] == 4 and a["tn"] == 2 and a["fp"] == 1 and a["aborted"] == 1 and a["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert "act.infer_ms_p50" in r["policy_timings"] and "act_infer_ms_p95" in r["policy_timings"]
    assert "act.attempts[].infer_ms_p50" in r["other_timings"] and "new_future_thing_ms" in r["other_timings"]
    assert r["settings"]["sort_20261115_110000"]["act_device"] == "GPU"
    assert r["calls_per_cycle"] == {"fast": 0.75, "main": 0.5, "policy": 12.0}
    t = cycle_stats.text(r)
    assert "cascade: fast model settled 2/3" in t and "fast@NPU" in t
    assert cycle_stats.collect([tmp_path / "nothing"])["n_cycles"] == 0


# ---------------------------------------------------------------------- report + run_all
def _session_with_results(tmp_path):
    mp = tiny_anomaly_model(tmp_path / "m/front/red/weights/openvino/model.xml")
    return bench_model.bench_models([{"path": str(mp), "tag": "front/red"}], devices=("CPU",), hints=("LATENCY",), n=10, min_s=0.1,
                                    warmup=1, async_s=0.1, idle_s=0.2, cooldown_s=0, interval=0.02, hw=16,
                                    out_dir=tmp_path / "res", sensors=FakePower(), log=lambda *a: None)


def test_report_build(tmp_path):
    _session_with_results(tmp_path)
    runs = write_runs(tmp_path)
    out = report.build(tmp_path / "res", "all", [runs], None, log=lambda *a: None)
    assert out["n_results"] == 1 and out["cycles"] == 4
    rows = (tmp_path / "res/section9_rows.md").read_text().splitlines()
    assert rows[0] == report.HEADER.splitlines()[0] and len(rows) == 4      # header, sep, model row, cycle row
    assert all(r.count("|") == 9 for r in rows[2:])
    assert "front/red" in rows[2] and "mJ/inf" in rows[2] and "headroom" in rows[2]
    assert "Sorting cycle" in rows[3] and "fast model settled 67%" in rows[3]
    md = (tmp_path / "res/REPORT.md").read_text()
    for h in ("## Environment", "## Device placement recommendation", "## Every configuration", "## Real sorting cycles", "## Provenance"):
        assert h in md
    try:
        import matplotlib  # noqa: F401
        assert "latency_per_device.png" in out["charts"] and (tmp_path / "res/latency_per_device.png").stat().st_size > 1000
        assert "temperature_over_time.png" in out["charts"]
    except ImportError:
        assert "charts skipped" in md


def test_report_without_matplotlib(tmp_path, monkeypatch):
    _session_with_results(tmp_path)
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (_ for _ in ()).throw(ImportError(n)) if n.startswith("matplotlib") else real(n, *a, **k))
    out = report.build(tmp_path / "res", "latest", [tmp_path / "noruns"], None, log=lambda *a: None)
    assert out["charts"] == [] and "charts skipped: matplotlib not importable" in (tmp_path / "res/REPORT.md").read_text()


def test_recommendation_plan():
    def r(tag, dev, p50, mj, role):
        return {"model": {"tag": tag, "role": role}, "device": dev, "hint": "LATENCY", "precision_requested": "default",
                "sync": {"p50_ms": p50}, "energy": {"j_per_inf_above_idle": mj / 1000}}
    res = [r("front/red", "CPU", 30, 300, "main"), r("front/red", "GPU", 8, 90, "main"), r("front/red", "NPU", 12, 40, "main"),
           r("front/red_efficientad", "CPU", 6, 50, "fast"), r("front/red_efficientad", "GPU", 3, 30, "fast"),
           r("front/red_efficientad", "NPU", 2.5, 8, "fast"), r("policy/act", "CPU", 40, 500, "policy"), r("policy/act", "GPU", 10, 120, "policy"),
           r("policy/act", "NPU", 25, 100, "policy")]
    txt = "\n".join(report.recommend(res))
    assert "**front/red** (main): fastest on **GPU**, most energy-efficient on **NPU**" in txt
    # equal weights: main NPU 12 + fast CPU 6 + policy GPU 10 = 28 ms is the cheapest distinct assignment
    assert "main -> **NPU**" in txt and "fast -> **CPU**" in txt and "policy -> **GPU**" in txt and "one call per role" in txt
    cyc = {"n_cycles": 10, "calls_per_cycle": {"fast": 10, "main": 1, "policy": 100}}
    txt = "\n".join(report.recommend(res, cyc))     # 10 fast photos per cycle: fast NPU 25 + main CPU 30 < fast CPU 60 + main NPU 12
    assert "policy -> **GPU** (10.00 ms x 100)" in txt and "fast -> **NPU**" in txt and "main -> **CPU**" in txt and "10 real cycles" in txt


def test_run_all_discovery_and_one_command(tmp_path):
    root = tmp_path / "proj"
    good = write_images(root / "training_images/front/inhand/good", 4, 30, "20261115_r1_c1_red_v")
    bad = write_images(root / "training_images/front/inhand/defect", 2, 30, "20261115_r1_c2_red_v")
    tiny_anomaly_model(root / "models/front/red/weights/openvino/model.xml")
    (root / "models/front/red/work").mkdir()
    tiny_anomaly_model(root / "models/front/red/work/other/model.xml")       # Anomalib's work dir: ignored
    (root / "models/front/red/split.json").write_text(json.dumps({"train": [good[0].name, good[1].name], "holdout": [good[2].name],
                                                                 "defect": [bad[0].name]}))
    tiny_anomaly_model(root / "models/front/red_efficientad/weights/openvino/model.xml")
    studio = tmp_path / "studio"
    tiny_policy_model(studio / "1234abcd-ffff/exports/openvino/act.xml")
    (studio / "1234abcd-ffff/exports/openvino/manifest.json").write_text(json.dumps({"policy": {"name": "act"}}))
    (studio / "1234abcd-ffff/exports/onnx").mkdir(parents=True)
    (studio / "1234abcd-ffff/exports/onnx/manifest.json").write_text("{}")
    write_runs(root)
    am = run_all.anomaly_models(root)
    assert [m["tag"] for m in am] == ["front/red", "front/red_efficientad"] and [m["role"] for m in am] == ["main", "fast"]
    assert len(am[0]["images"]) == 2 and all(Path(f).exists() for f in am[0]["images"])     # holdout + defect from split.json
    assert len(am[1]["images"]) == 4                                                          # no split.json: the good folder
    pm = run_all.policy_models([studio])
    assert len(pm) == 1 and pm[0]["tag"] == "policy/act-1234abcd" and pm[0]["path"].endswith("act.xml")
    cal, guard = run_all.calib_lists(am[0], Path(am[0]["path"]).with_name("model_int8.xml"), root)
    assert len(Path(cal).read_text().split()) == 2 and len(Path(guard).read_text().split()) == 2
    rc = run_all.main(["--root", str(root), "--studio-models", str(studio), "--devices", "CPU", "--hints", "LATENCY",
                       "--precisions", "default", "--n", "10", "--min-seconds", "0.1", "--async-seconds", "0.1", "--idle-seconds", "0.2",
                       "--cooldown", "0", "--quantize", "no", "--out", str(tmp_path / "res")])
    assert rc == 0
    d = json.loads(next((tmp_path / "res").glob("bench_*.json")).read_text())
    tags = sorted(r["model"]["tag"] for r in d["results"] if not r.get("error"))
    assert tags == ["front/red", "front/red_efficientad", "policy/act-1234abcd"]
    assert d["cycle_stats"]["n_cycles"] == 4
    md = (tmp_path / "res/REPORT.md").read_text()
    assert "policy/act-1234abcd" in md and "Sorting cycle" in (tmp_path / "res/section9_rows.md").read_text()
