"""The grasp tuner learns a better grasp from success / failure only."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "anomaly_capture"))
from grasp_tuner import Tuner  # noqa: E402


def station(p, rng):
    """Hidden truth: best grasp = 3 mm deeper and aimed +4 mm in x; the rest barely matters."""
    q = 0.35
    q += 0.25 if p["dz_mm"] == -3 else (0.1 if p["dz_mm"] == -6 else 0.0)
    q += 0.3 if p["dxy_mm"] == 3 else (-0.1 if p["dxy_mm"] in (-3, -6) else 0.0)
    return rng.random() < min(max(q, 0.02), 0.97)


def test_learns_hidden_best(tmp_path):
    rng = random.Random(1); t = Tuner(tmp_path / "t.json", seed=2); t.enabled = True
    wins = []
    for _ in range(150):
        p = t.suggest("pink"); ok = station(p, rng); t.update(p, ok); wins.append(ok)
    best = t.suggest("pink", explore=False)
    assert best["dz_mm"] == -3 and best["dxy_mm"] == 3
    assert sum(wins[-50:]) / 50 > sum(wins[:30]) / 30 + 0.1       # gets better while it runs


def test_persists_and_neutral_start(tmp_path):
    t = Tuner(tmp_path / "t.json", seed=0)
    for _ in range(20):   # disabled (default): always the plain scripted grasp
        assert all(t.suggest("blue")[k] == v[0] for k, v in t.space.items())
    t.enabled = True
    assert all(t.suggest("blue", explore=False)[k] == v[0] for k, v in t.space.items())   # fresh = neutral
    p = t.suggest("blue"); t.update(p, True, {"episode": "ep0001"})
    t2 = Tuner.load(tmp_path / "t.json")
    assert t2.history[-1]["episode"] == "ep0001" and t2.summary()["tries"] == 1
