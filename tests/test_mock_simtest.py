"""anomaly_capture/sim_test.py (hardware-free inspection-logic test) as a pytest test.

It runs on a COPY of anomaly_capture/ in a temp dir: sim_test.py's API section builds a Sorter whose settings path is
anomaly_capture/sort_settings.json, so running it in place overwrites the station's sort settings (reported bug).

    .venv/bin/python -m pytest tests/test_mock_simtest.py        (~1 min)
"""
import shutil, subprocess, sys

from conftest import AC


def test_sim_test_passes(tmp_path):
    work = tmp_path / "anomaly_capture"
    shutil.copytree(AC, work, ignore=shutil.ignore_patterns("__pycache__", "*.log", "web"))
    r = subprocess.run([sys.executable, str(work / "sim_test.py")], capture_output=True, text=True, timeout=900, cwd=str(tmp_path))
    assert r.returncode == 0 and "ALL CHECKS PASSED" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
