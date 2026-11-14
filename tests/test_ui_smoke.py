"""Browser smoke test: tests/ui_smoke.mjs (Node + global Playwright + Chromium) against a fresh mock server.
Screenshots of every page go to tests/screenshots/ (gitignored). Skipped when node/playwright are missing.

    .venv/bin/python -m pytest tests/test_ui_smoke.py -s        (make test-ui)
"""
import os, shutil, subprocess

import pytest

from conftest import REPO

pytestmark = pytest.mark.ui


def _has_playwright():
    if not shutil.which("node"): return False
    r = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _has_playwright(), reason="node + playwright not installed")
def test_ui_smoke(mock_server):
    out = REPO / "tests" / "screenshots"
    r = subprocess.run(["node", str(REPO / "tests" / "ui_smoke.mjs"), mock_server.url, str(out)], capture_output=True, text=True,
                       timeout=600, env={**os.environ})
    print(r.stdout[-6000:])
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-3000:] + "\n--- server log ---\n" + mock_server.log()
    assert len(list(out.glob("*.png"))) >= 12
