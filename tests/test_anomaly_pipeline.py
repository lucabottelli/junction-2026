"""Tests of the anomaly-model pipeline (synth_data.py, train_anomaly.py split, eval_anomaly.py) without the station.

    .venv/bin/python -m pytest tests/test_anomaly_pipeline.py -q

Self-contained (no conftest fixtures). Scoring goes through the REAL scorer.py + score_worker.py with a FAKE
`anomalib.deploy` package on PYTHONPATH (same trick as sim_test.py), so the run-time protocol is exercised without
Anomalib. The last test trains a real (tiny) PaDiM and is skipped unless an Anomalib python is found
(env ANOMALIB_PY, .venv-anomalib/ or ~/anomalib/.venv).
"""
import json, os, subprocess, sys, types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
AC = ROOT / "anomaly_capture"
sys.path.insert(0, str(AC))

import eval_anomaly as E  # noqa: E402
import synth_data  # noqa: E402
import train_anomaly as T  # noqa: E402

META_KEYS = {"file", "stage", "cam", "group", "label", "brick", "view", "cycle", "run", "routine", "joints", "motion",
             "frames_averaged", "time", "depth", "cam_settings"}

FAKE_DEPLOY = '''
import json, os, types
from pathlib import Path
import numpy as np
TRUTH = {Path(json.loads(l)["file"]).name: json.loads(l) for l in open(os.environ["FAKE_TRUTH"]) if l.strip()}
class OpenVINOInferencer:
    """Fake: knows the synthetic truth. Main model: visible cross -> high score. Fast model: unsure about small ones."""
    def __init__(self, path, device):
        self.fast = "efficientad" in str(path)
        if device == "NPU": raise RuntimeError("no NPU here")
    def predict(self, image):
        name = Path(image).name; t = TRUTH.get(name, {})
        h = (sum(map(ord, name)) % 97) / 97
        v = 0.15 + 0.2 * h
        if t.get("visible"): v = 0.7 + 0.02 * min(10, t["cross_mm"]) if t["cross_mm"] >= 2 else 0.42 + 0.1 * h
        if self.fast and 0.4 < v < 0.8: v = 0.5
        return types.SimpleNamespace(pred_score=np.array([v]), pred_label=np.array([v > .5]), anomaly_map=np.random.rand(1, 16, 16))
TorchInferencer = OpenVINOInferencer
'''


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth") / "training_images"
    synth_data.generate(d, bricks=("red", "blue"), n_good=12, n_defect=8, cams=("front", "wrist"), seed=3, raw=False)
    return d


@pytest.fixture()
def fake_anomalib(tmp_path, ds, monkeypatch):
    (tmp_path / "fake" / "anomalib" / "deploy").mkdir(parents=True)
    (tmp_path / "fake" / "anomalib" / "__init__.py").write_text("")
    (tmp_path / "fake" / "anomalib" / "deploy" / "__init__.py").write_text(FAKE_DEPLOY)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake"))
    monkeypatch.setenv("FAKE_TRUTH", str(ds / "synth_truth.jsonl"))
    monkeypatch.setattr(E.scorer_mod, "ANOMALIB_PY", Path(sys.executable))
    return tmp_path


def fake_model(models, data, cam, name, model="patchcore"):
    """A model dir as train_anomaly.py writes it (split from the real split code), with a dummy OpenVINO file."""
    r = T.main(["--data", str(data), "--models", str(models), "--brick", name, "--cam", cam, "--model", model, "--dry-run"])
    out = r["out"]; (out / "weights" / "openvino").mkdir(parents=True)
    (out / "weights" / "openvino" / "model.xml").write_text("<fake/>")
    (out / "split.json").write_text(json.dumps(r["split"]))
    (out / "holdout.json").write_text(json.dumps(r["split"]["holdout"]))
    (out / "model_card.json").write_text(json.dumps({"brick": name, "cam": cam, "model": model, "data_dir": str(data),
                                                     "image_size": 256, "data": r["counts"]}))
    return out


# ---------------------------------------------------------------- synthetic data format
def test_synth_format_matches_station(ds):
    lines = [json.loads(l) for l in open(ds / "meta.jsonl")]
    assert lines and all(set(j) == META_KEYS for j in lines)
    for j in lines:
        raw = ds / j["file"]
        assert j["file"] == f"{j['cam']}/inhand/{j['label']}/{j['group']}_{j['brick']}_{j['view']}.png"
        crop = raw.with_name(raw.stem + "_crop.png")
        assert crop.exists() and T.brick_of(crop) == j["brick"]
        assert j["stage"] == ("release" if j["cam"] == "wrist" else "inhand")
        assert (j["depth"] is None) == (j["cam"] == "wrist")
    assert len({j["group"] for j in lines}) == len(lines)                  # capture groups are unique
    truth = [json.loads(l) for l in open(ds / "synth_truth.jsonl")]
    assert {Path(t["file"]).name for t in truth} == {p.name for p in ds.glob("*/inhand/*/*_crop.png")}
    d = [t for t in truth if t["label"] == "defect"]
    assert d and all("cross_mm" in t and "visible" in t for t in d) and any(t["visible"] for t in d)
    assert json.loads((ds / "synth_info.json").read_text())["synthetic"] is True


def test_synth_raw_files_and_determinism(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b): synth_data.generate(d, bricks=("green",), n_good=1, n_defect=1, cams=("front", "wrist"), seed=5, raw=True)
    assert (a / "meta.jsonl").read_text() == (b / "meta.jsonl").read_text()
    front = sorted((a / "front" / "inhand" / "good").iterdir())
    assert {p.name.split("wp2-1")[-1] for p in front if "wp2-1" in p.name} == {".png", "_depth.png", "_masked.jpg", "_crop.png"}
    for p in sorted(a.rglob("*_crop.png")):
        assert p.read_bytes() == (b / p.relative_to(a)).read_bytes()
    import cv2
    dep = cv2.imread(str(next((a / "front").rglob("*_depth.png"))), cv2.IMREAD_UNCHANGED)
    assert dep.dtype == np.uint16 and dep.shape == (1080, 1920)


def test_real_loaders_accept_synth(ds):
    good = T.list_crops(ds, "front", "good", "red"); bad = T.list_crops(ds, "front", "defect", "red")
    assert len(good) == 12 * 2 and len(bad) == 8 * 2
    assert all(T.brick_of(f) == "red" for f in good + bad)
    r = T.main(["--data", str(ds), "--brick", "red", "--dry-run"])
    assert r["counts"]["train"] + r["counts"]["holdout"] == 24 and r["counts"]["defect"] == 16
    # sorter.calibrate's own episode lookup must map every crop to the same episode as the trainer's split
    try:
        import lerobot.motors.feetech.feetech  # noqa: F401
    except Exception:
        for m in ["lerobot", "lerobot.motors", "lerobot.motors.feetech", "lerobot.motors.feetech.feetech", "lerobot.motors.motors_bus"]:
            sys.modules.setdefault(m, types.ModuleType(m))
        sys.modules["lerobot.motors.feetech.feetech"].FeetechMotorsBus = object
        sys.modules["lerobot.motors.motors_bus"].Motor = object
        sys.modules["lerobot.motors.motors_bus"].MotorNormMode = types.SimpleNamespace(RANGE_M100_100=0)
    try: import sorter
    except Exception as e: pytest.skip(f"sorter.py not importable here: {e}")
    ep = sorter.Sorter._episode_of(types.SimpleNamespace(DATA=ds))
    meta = T.load_meta(ds)
    for f in good + bad:
        run, cyc = ep[f.stem[:-len("_crop")]]
        assert T.episode_of(f, meta) == f"{run}/{cyc}"
    rng = np.random.default_rng(0)                                        # eval uses the sorter's own thresholds_from
    g, d = rng.uniform(0, .5, 30), rng.uniform(.3, 1, 8)
    assert E.thresholds_from(g, d) == sorter.thresholds_from(g, d)


# ---------------------------------------------------------------- split
def test_split_by_episode_no_leakage(ds):
    meta = T.load_meta(ds)
    files = T.list_crops(ds, "front", "good", "")
    for seed in range(5):
        tr, ho, eps = T.split_by_episode(files, meta, 0.25, seed)
        assert tr and ho and len(tr) + len(ho) == len(files)
        assert not {eps[f.name] for f in tr} & {eps[f.name] for f in ho}
        # all views of a held-out episode are held out together
        for f in ho: assert sum(eps[g.name] == eps[f.name] for g in ho) == 2
    assert T.split_by_episode(files, meta, 0.25, 1)[1] == T.split_by_episode(files, meta, 0.25, 1)[1]
    tr, ho, eps = T.split_by_episode(files, {}, 0.25, 0)          # no meta: falls back to the capture group
    assert all(e.startswith("group:") for e in eps.values()) and ho
    assert len(T.thin_by_episode(tr, eps, 5)) == 5


# ---------------------------------------------------------------- metrics on known inputs
def test_metric_functions():
    assert E.auroc([0, 1], [2, 3]) == 1.0 and E.auroc([2, 3], [0, 1]) == 0.0 and E.auroc([1], [1]) == 0.5
    assert E.recall_at_zero_fa([0.1, 0.3], [0.2, 0.4, 0.5]) == pytest.approx(2 / 3)
    assert E.fa_at_full_recall([0.1, 0.3, 0.25], [0.2, 0.4]) == pytest.approx(2 / 3)
    assert E.by_episode({"a": 1, "b": 3, "c": 2}, {"a": "x", "b": "x", "c": "y"}) == {"x": 3, "y": 2}
    assert E.object_verdict(["clean", "clean"]) == ("good", "clean")
    assert E.object_verdict(["clean", "uncertain"]) == ("defect", "uncertain")
    assert E.object_verdict(["uncertain", "defect"]) == ("defect", "defect")
    st = [{"t_low": 0.3, "t_high": 0.7}, {"t_low": 0.4, "t_high": 0.6}]
    assert E.cascade_photo([0.1, 0.9], st) == ("clean", 1)          # the fast stage is sure: main never asked
    assert E.cascade_photo([0.5, 0.65], st) == ("defect", 2)
    assert E.cascade_photo([0.5, 0.5], st) == ("uncertain", 2)
    c = E.size_curve({"a": .9, "b": .9, "c": .9, "d": .1, "e": .9, "f": .9}, {"a": 10, "b": 10, "c": 10, "d": 1.5, "e": 5, "f": 5}, 0.5)
    assert c["smallest_reliable"] == ">=10" and [r["bin"] for r in c["bins"]] == ["<=1.5", "5", ">=10"]
    c = E.size_curve({k: .9 for k in "abcdef"}, dict(zip("abcdef", [5, 5, 5, 10, 10, 10])), 0.5)
    assert c["smallest_reliable"] == "5"
    g = {f"g{i}": 0.1 + 0.01 * i for i in range(12)}; ge = {k: f"G{i // 2}" for i, k in enumerate(g)}
    d = {f"d{i}": 0.8 + 0.01 * i for i in range(10)}; de = {k: f"D{i // 2}" for i, k in enumerate(d)}
    cc, ts = E.cross_calibrated(g, ge, d, de)
    assert cc["accuracy"] == 1.0 and cc["defect_missed"] == 0 and len(ts) == 2
    o = E.outcome({"a": ("good", "clean"), "b": ("defect", "uncertain")}, {"c": ("good", "clean")})
    assert o["false_reject_rate"] == 0.5 and o["miss_rate"] == 1.0 and o["good_rejected_uncertain"] == 1


# ---------------------------------------------------------------- eval through the run-time path (fake worker)
def test_eval_through_fake_worker(ds, fake_anomalib):
    models = fake_anomalib / "models"
    main = fake_model(models, ds, "front", "red")
    fast = fake_model(models, ds, "front", "red", "efficientad")
    r = E.evaluate(main, ds, "CPU", ("CPU", "NPU"), fast=fast, write_thresholds=True)
    assert r["synthetic_data"] and r["sorter_functions"] in ("import", "source")
    assert r["n_good_photos"] > 0 and r["n_defect_photos"] == 16
    assert r["object"]["auroc"] > 0.9 and r["photo"]["auroc_visible"] > 0.9
    th = json.loads((main / "thresholds.json").read_text())
    assert th["t_low"] <= th["t_high"] and {"n_good", "defect_bricks", "auroc", "overlap", "calibrated"} <= set(th)
    assert th["synthetic_data"] is True
    assert (fast / "thresholds.json").exists()
    cz = r["cascade"]
    assert 0 < cz["share_settled_by_fast"] <= 1 and "object" in cz and cz["mean_ms_per_photo"] > 0
    lat = {x["device"]: x for x in r["latency"]}
    assert "infer_ms" in lat["CPU"] and "error" in lat["NPU"]           # a missing device is reported, not fatal
    assert r["size"]["photo"]["bins"] and "by_hard_case" in r["size"]
    for f in ("eval.json", "report.md", "hist.png", "worst.png", "size_curve.png"):
        assert (main / "eval" / f).exists(), f
    md = (main / "eval" / "report.md").read_text()
    assert "SYNTHETIC DATA" in md and "Cascade" in md and "cross-calibrated" in md
    assert json.loads((main / "model_card.json").read_text())["eval"]["synthetic_data"] is True
    # the run-time path really was used: the scorer finds these models the way the sorter does
    from scorer import Scorer
    s = Scorer(models, fake_anomalib / "w.log")
    assert Path(s.find("front", "red")[1]) == main and Path(s.find("front", "red", "_efficientad")[1]) == fast
    assert Scorer.thresholds(main)["t_high"] == th["t_high"]


def test_eval_refuses_leakage(ds, fake_anomalib):
    main = fake_model(fake_anomalib / "models", ds, "front", "blue")
    s = json.loads((main / "split.json").read_text()); s["train"].append(s["holdout"][0])
    (main / "split.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="trained on"):
        E.evaluate(main, ds, "CPU", ("CPU",))


def test_compare_variants(ds, fake_anomalib):
    models = fake_anomalib / "models"
    a = fake_model(models, ds, "wrist", "red"); b = fake_model(models, ds, "wrist", "red", "padim")
    out = fake_anomalib / "cmp"
    m = E.main(["--brick", "red", "--cam", "wrist", "--models", str(models), "--compare", "--out", str(out), "--data", str(ds)])
    assert [x["variant"] for x in m["rows"]] == ["red", "red_padim"] and m["rows"][0]["earns_its_place"] == "baseline"
    md = next(out.glob("compare_wrist_red_*.md")).read_text()
    assert "SYNTHETIC" in md and "red_padim" in md


# ---------------------------------------------------------------- optional: real Anomalib end to end (slow)
def _anomalib_python():
    for p in [os.environ.get("ANOMALIB_PY"), ROOT / ".venv-anomalib/bin/python", Path.home() / "anomalib/.venv/bin/python"]:
        if p and Path(p).exists() and subprocess.run([str(p), "-c", "import anomalib"], capture_output=True).returncode == 0:
            return Path(p)
    return None


def test_real_anomalib_end_to_end(tmp_path, monkeypatch):
    """SLOW (~1 min): skipped without an Anomalib python, or with ANOMALY_SKIP_SLOW=1."""
    if os.environ.get("ANOMALY_SKIP_SLOW"): pytest.skip("ANOMALY_SKIP_SLOW set")
    py = _anomalib_python()
    if not py: pytest.skip("no python with anomalib (set ANOMALIB_PY)")
    data = tmp_path / "training_images"
    synth_data.generate(data, bricks=("red",), n_good=14, n_defect=6, cams=("front",), seed=11, raw=False)
    r = subprocess.run([str(py), str(AC / "train_anomaly.py"), "--data", str(data), "--models", str(tmp_path / "models"),
                        "--brick", "red", "--model", "padim", "--no-pretrained", "--image-size", "96", "--device", "cpu"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    mdir = tmp_path / "models" / "front" / "red_padim"
    assert (mdir / "weights" / "openvino" / "model.xml").exists() and (mdir / "model_card.json").exists()
    monkeypatch.setattr(E.scorer_mod, "ANOMALIB_PY", py)
    res = E.evaluate(mdir, data, "CPU", ("CPU",))
    assert res["n_good_photos"] > 0 and res["photo"]["auroc_all_defect_photos"] is not None
    assert (mdir / "eval" / "report.md").exists()
