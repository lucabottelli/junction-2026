"""The control UI for the whole project: http://127.0.0.1:7870/  (user page: web/operator.html at / ; dev interface: /dev)

A single-page app assembled from the files in web/ on EVERY request (edit a file, reload the browser; no restart):
  web/shell.html        frame: head, icon sprite, sidebar, top bar, toasts, modal + {{CSS}} {{PAGES}} {{SCRIPTS}}
  web/style.css         design system (tokens + shared components)
  web/css/<slug>.css    optional page-specific CSS (injected after style.css, in page order)
  web/pages/NN-slug.html one <section class=page data-path=...> per page, sorted by file name (= nav order)
  web/js/core.js        shared helpers, router, polling, top bar, sidebar, keyboard, PAGE registry
  web/js/<slug>.js      optional page logic (each in its own <script>, so one broken page can't kill the others)
  web/static/...        fonts etc., served at /static/...
See web/DESIGN.md for the contract. Every page path (data-path) is served the same HTML; core.js routes client-side.
It only calls the existing APIs of arm.py (/arm/*), collect.py (/collect/*) and capture.py (/snap, /status, ...).
"""
import os, re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

WEB = Path(__file__).resolve().parent / "web"
STATIC = WEB / "static"
NO_STORE = {"Cache-Control": "no-store"}
router = APIRouter()


def _pages():
    """[(slug, path, file)] for web/pages/NN-slug.html, sorted by file name."""
    out = []
    for f in sorted((WEB / "pages").glob("*.html")):
        slug = re.sub(r"^\d+-", "", f.stem)
        m = re.search(r'data-path="([^"]+)"', f.read_text(encoding="utf-8"))
        out.append((slug, m.group(1) if m else None, f))
    return out


def _read(p):
    return p.read_text(encoding="utf-8") if p.exists() else ""


def render():
    pages = _pages()
    css = _read(WEB / "style.css") + "".join(
        f"\n/* ===== css/{s}.css ===== */\n" + _read(WEB / "css" / f"{s}.css") for s, _, _ in pages if (WEB / "css" / f"{s}.css").exists())
    html = "\n".join(f"<!-- ===== pages/{f.name} ===== -->\n" + _read(f) for _, _, f in pages)
    js = [("core.js", _read(WEB / "js" / "core.js"))] + [
        (f"{s}.js", _read(WEB / "js" / f"{s}.js")) for s, _, _ in pages if (WEB / "js" / f"{s}.js").exists()]
    scripts = "".join("<script>/* js/%s */\n%s\n</script>\n" % (n, src.replace("</script", "<\\/script")) for n, src in js)
    scripts += "<script>boot()</script>"
    shell = _read(WEB / "shell.html")
    out = shell.replace("{{CSS}}", css).replace("{{PAGES}}", html).replace("{{SCRIPTS}}", scripts)
    if os.environ.get("JUNCTION_MOCK") == "1":     # mock station (mock_hw.py): say so on every page
        i = out.rfind("</body>"); out = out[:i] + MOCK_BADGE + out[i:] if i >= 0 else out + MOCK_BADGE
    return out


MOCK_BADGE = ('<div id=mockbadge title="capture.py runs with simulated cameras and arms (mock_hw.py, docs/DEV.md); '
              'nothing here moves real hardware" style="position:fixed;left:50%;transform:translateX(-50%);bottom:10px;z-index:50;padding:4px 10px;'
              'border:1px solid var(--warn);border-radius:var(--r-xs);background:var(--surface);color:var(--warn);'
              'font:600 11px var(--sans);letter-spacing:.06em;text-transform:uppercase;pointer-events:auto">Mock station</div>')


def serve():
    return HTMLResponse(render(), headers=NO_STORE)


def operator():
    """The user-facing page (web/operator.html, standalone): cameras, run the sort task, results. Served at /.
    Everything above is the dev interface; its overview moved from / to /dev (Dev menu on the user page)."""
    f = WEB / "operator.html"
    return HTMLResponse(_read(f), headers=NO_STORE) if f.exists() else serve()


# One route per page path, read from the page files at start-up (a NEW page file needs a server restart;
# editing an existing one doesn't). Explicit paths, never a catch-all: routers included after this one
# (collect.py) must keep their GET routes.
PATHS = sorted({p for _, p, _ in _pages() if p} | {"/", "/dev"})
for _p in PATHS:
    router.add_api_route(_p, operator if _p == "/" else serve, methods=["GET"], include_in_schema=False)


@router.get("/arm", include_in_schema=False)
def old_arm():
    return RedirectResponse("/")          # the browser keeps the #hash; core.js maps old #views to the new paths


@router.get("/static/{path:path}", include_in_schema=False)
def static(path: str):
    f = (STATIC / path).resolve()
    if not f.is_file() or not f.is_relative_to(STATIC):
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})
