"""Every route of the control server answers sanely on the mock station (capture.py, arm.py, collect.py, sorter.py,
ui.py, mock_hw.py). Routes are read from the live server's OpenAPI schema, so routes added later are covered too.

    .venv/bin/python -m pytest tests/test_api_sweep.py
"""
import re

import cv2
import numpy as np
import pytest

from conftest import AC

# sample values for parameters, by name (anything else: a neutral value of its type)
SAMPLE = {"cam": "top", "on": True, "x0": 100, "y0": 100, "x1": 900, "y1": 700, "label": "good", "force": False,
          "brick": "red", "m": "hold", "name": "sweep", "i": 0, "op": "rename", "v": "renamed", "routine": "wp1",
          "flag": "next", "session": "none", "episode": "none", "answer": "unsure", "n": 1, "kind": "wp2", "sides": "front",
          "view": "wp2-1", "dry": True, "full": False, "state": "ok", "x": 0.2, "y": 0.0, "z": 0.1, "mode": "auto",
          "cycles": 1, "speed": 0.5, "save": False, "defect": "", "reset": True}
FLOATS = {"x0": 10.0, "y0": 10.0, "x1": 90.0, "y1": 90.0}           # /sort/release_roi takes percent floats
NEVER = {"/mock/fault"}                                               # sweep would e.g. inject faults: tested elsewhere
STREAMS = ("/stream/", "/zoom/")


def params_for(op):
    out, path = {}, {}
    for p in op.get("parameters", []):
        n, t = p["name"], p["schema"].get("type") or (p["schema"].get("anyOf") or [{}])[0].get("type")
        v = SAMPLE.get(n, {"string": "", "integer": 0, "number": 0.0, "boolean": False}.get(t, ""))
        if t == "number" and n in FLOATS: v = FLOATS[n]
        if not p.get("required") and n not in SAMPLE: continue
        (path if p["in"] == "path" else out)[n] = v
    return out, path


@pytest.fixture(scope="module")
def schema(mock_server):
    return mock_server.get("/openapi.json")


def fill(route, pp):
    return re.sub(r"\{(\w+)\}", lambda m_: str(pp[m_.group(1)]), route)


def test_every_get_route(mock_server, schema):
    srv = mock_server; n = 0
    for route, ops in schema["paths"].items():
        if "get" not in ops: continue
        q, pp = params_for(ops["get"])
        cams = ["top", "front", "wrist", "nosuchcam"] if "{cam}" in route else [None]
        for cam in cams:
            if cam: pp["cam"] = cam
            url = fill(route, pp)
            if route.startswith(STREAMS):
                with srv.http.stream("GET", url) as r:
                    assert r.status_code == 200 and "multipart" in r.headers["content-type"]
                    chunk = next(r.iter_bytes()); assert b"image/jpeg" in chunk or chunk
                n += 1; continue
            r = srv.http.get(url, params=q, follow_redirects=False)
            assert r.status_code in (200, 204, 307), f"GET {url} -> {r.status_code} {r.text[:200]}"
            if r.status_code == 200 and r.headers["content-type"].startswith("image/jpeg"):
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                assert img is not None and img.size > 0, url
            n += 1
    assert n > 25


def test_every_ui_page_and_static(mock_server):
    srv = mock_server
    paths = [re.search(r'data-path="([^"]+)"', f.read_text()).group(1) for f in sorted((AC / "web" / "pages").glob("*.html"))]
    assert "/" in paths and "/sort" in paths
    n_sections = len(list((AC / "web" / "pages").glob("*.html")))
    for p in paths:
        r = srv.http.get(p)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), p
        assert r.text.count("<section class=page") >= n_sections, p        # a page may mention the tag in text
        assert "id=mockbadge" in r.text                                   # mock mode says so on every page
    assert srv.http.get("/arm", follow_redirects=False).status_code == 307
    assert srv.http.get("/collect", follow_redirects=False).status_code == 307
    fonts = list((AC / "web" / "static" / "fonts").glob("*.woff2"))
    assert fonts and srv.http.get(f"/static/fonts/{fonts[0].name}").status_code == 200
    assert srv.http.get("/static/../ui.py").status_code == 404
    assert srv.http.get("/static/nope.css").status_code == 404
    assert srv.http.get("/manual").status_code == 200 and srv.http.get("/arm/routines").status_code == 200


def test_every_post_route_answers_json(mock_server, schema):
    """Every POST with plausible parameters returns 200 + a JSON object (never a 500), even with the arm offline
    first, then connected. Long-running starters are stopped right after."""
    srv = mock_server
    posts = [(r, ops["post"]) for r, ops in schema["paths"].items() if "post" in ops and r not in NEVER]
    assert len(posts) > 30
    for phase in ("offline", "connected"):
        if phase == "connected": srv.connect()
        for route, op in posts:
            q, pp = params_for(op)
            url = fill(route, pp)
            r = srv.http.post(url, params=q)
            assert r.status_code == 200, f"[{phase}] POST {url} {q} -> {r.status_code}: {r.text[:300]}\n{srv.log()}"
            assert isinstance(r.json(), dict), url
            for stop in ("/collect/stop", "/sort/stop", "/arm/stop"):
                if route.endswith(("/start", "/run", "/test_wp2", "/goto")): srv.post(stop)
    srv.post("/sort/stop"); srv.post("/collect/stop"); srv.post("/arm/stop")
    srv.wait(lambda: not srv.get("/collect/status")["active"] and not srv.get("/sort/status")["active"], 20, "jobs stopped")
    assert srv.get("/status")["live"] == ["top", "front", "wrist"]
    assert srv.proc.poll() is None


def test_missing_parameters_are_422_not_500(mock_server):
    for url in ("/roi/top", "/arm/mode", "/arm/routine/new", "/collect/keep", "/sort/release_roi", "/sort/faces"):
        assert mock_server.http.post(url).status_code == 422, url
    assert mock_server.http.get("/recent/notanumber").status_code == 422
