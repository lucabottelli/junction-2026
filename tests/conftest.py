"""Shared pytest helpers for the mock-station tests (tests/test_mock_*.py, test_api_*.py, test_ui_*.py).

What: a `MockServer` that runs the REAL control server (anomaly_capture/capture.py) in mock mode as a subprocess on a
free port with its own temporary data root, plus polling helpers. Why a subprocess: the server starts daemon threads
(cameras, 50 Hz arm loop) and module globals; a separate process per test module keeps tests independent and is
exactly what `make mock` runs.

Rules (other agents' tests share this directory): no autouse fixtures, no chdir, no sys.path / os.environ changes at
import time. Markers: `slow` (end-to-end flows on the mock, minutes), `ui` (browser smoke via Node + Playwright).

    .venv/bin/python -m pytest tests -m "not slow and not ui"     # fast  (make test)
    .venv/bin/python -m pytest tests                               # everything (make test-all)
"""
import json, os, socket, subprocess, sys, time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
AC = REPO / "anomaly_capture"
PY = sys.executable


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end flow on the mock station (takes minutes)")
    config.addinivalue_line("markers", "ui: browser smoke test (Node + Playwright + Chromium)")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


class MockServer:
    """capture.py --mock in a subprocess. get/post return parsed JSON (or the httpx response with raw=True)."""

    def __init__(self, root, env=None, ready_timeout=60):
        import httpx
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.port = free_port(); self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = self.root / "server.log"
        e = {**os.environ, "JUNCTION_MOCK": "1", "JUNCTION_PORT": str(self.port), "JUNCTION_DATA_ROOT": str(self.root),
             "PYTHONUNBUFFERED": "1", **(env or {})}
        self.proc = subprocess.Popen([PY, "-u", str(AC / "capture.py")], cwd=str(REPO), env=e,
                                     stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT)
        self.http = httpx.Client(base_url=self.url, timeout=30)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None: raise RuntimeError(f"mock server died:\n{self.log()}")
            try:
                if self.http.get("/sort/status").status_code == 200 and self.http.get("/mock/state").status_code == 200: break
            except Exception: pass
            if time.time() - t0 > ready_timeout: self.stop(); raise RuntimeError(f"mock server not ready:\n{self.log()}")
            time.sleep(0.3)

    def log(self, n=60):
        try: return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except Exception: return ""

    def get(self, path, raw=False, **params):
        r = self.http.get(path, params=params)
        if raw: return r
        assert r.status_code == 200, f"GET {path} -> {r.status_code}: {r.text[:300]}\n{self.log()}"
        return r.json()

    def post(self, path, raw=False, **params):
        r = self.http.post(path, params=params)
        if raw: return r
        assert r.status_code == 200, f"POST {path} -> {r.status_code}: {r.text[:300]}\n{self.log()}"
        return r.json()

    def wait(self, fn, timeout=60, what="condition", every=0.15):
        """Poll fn() until it returns something truthy; returns it. Fails with the last value + server log."""
        t0 = time.time(); last = None
        while time.time() - t0 < timeout:
            try: last = fn()
            except AssertionError: raise
            except Exception as e: last = e
            if last and not isinstance(last, Exception): return last
            time.sleep(every)
        pytest.fail(f"timeout ({timeout} s) waiting for {what}; last: {str(last)[:600]}\n--- server log ---\n{self.log()}")

    # ---- common steps
    def connect(self):
        s = self.post("/arm/connect"); assert s["connected"], s; return s

    def arm_idle(self, timeout=30):
        return self.wait(lambda: (lambda a: a if not a["running"] else None)(self.get("/arm/status")), timeout, "arm idle")

    def goto(self, routine, i=0, timeout=30):
        """Drive to a taught waypoint (the UI's Go button) and wait until the arm is there."""
        tgt = self.get("/arm/status")["all"][routine]["waypoints"][i]["pos"]
        self.post(f"/arm/wp/{i}/goto", routine=routine)
        return self.wait(lambda: self.near(tgt), timeout, f"arm at {routine}[{i}]")

    def near(self, tgt, tol=45):
        p = self.get("/arm/status")["pos"]
        return p if all(abs(p[k] - tgt[k]) <= tol for k in tgt if k != "gripper") else None

    def stop(self):
        try: self.http.close()
        except Exception: pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(10)
            except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait(5)


@pytest.fixture(scope="module")
def mock_server(tmp_path_factory):
    """One fresh mock station per test module (own port, own data root)."""
    srv = MockServer(tmp_path_factory.mktemp("mock_root"))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def connected(mock_server):
    mock_server.connect(); return mock_server


def station_files():
    """Station-specific files the mock must NEVER modify (content snapshot)."""
    out = {}
    for f in ["routines.json", "cam_settings.json", "collect_settings.json", "sort_settings.json", "inspect_settings.json"]:
        p = AC / f
        out[f] = p.read_bytes() if p.exists() else None
    for d in ["training_images", "episodes", "runs", "models"]:
        p = REPO / d
        out[d] = sorted(str(x) for x in p.rglob("*")) if p.exists() else None
    return out


def judge_by_truth(srv, s):
    """Act as the human judge of /sort: answer from the mock world's ground truth (is the pen cross visible?)."""
    if not s.get("judge"): return
    tr = srv.get("/mock/state")["truth"]
    cam = "wrist" if "tape_top" in (s.get("judge_caption") or "") else "front"
    srv.post("/sort/judge/" + ("defect" if tr[cam] else "clean"))


def dump(obj):
    return json.dumps(obj, indent=1)[:2000]
