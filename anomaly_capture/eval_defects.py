"""Synthetic-defect evaluation of the whole in-hand pipeline (colour mask -> crop -> PatchCore -> thresholds).

Paints one synthetic defect (tape, pen line/cross, dot) onto each HELD-OUT clean raw photo, runs it through
prep.inspect_crop exactly like live inspection, scores it and reports how many are caught. Also measures how much
of the painted defect survives prep.brick_mask ("kept"), the diagnostic for defects masked away before the model.
Only calls prep's public functions (configure_inhand, brick_mask, inspect_crop); never writes models/ or training_images/.

  python eval_defects.py --brick all --n 12 [--seed 0] [--out DIR]
"""
import argparse, json, pathlib, sys, tempfile, time, traceback
import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep, scorer  # noqa: E402

ROOT = HERE.parent
GOOD = ROOT / "training_images/front/inhand/good"
MODELS = ROOT / "models"
VIEWS = ("wp2-1", "wp2-2", "wp2-3")


# ------------------------------------------------------------------ photo selection and block region
def pick_photos(brick, n, seed):
    """Held-out raw photos of this brick (existing files only), at most n, round-robin over the three views."""
    split = json.loads((MODELS / "front" / brick / "split.json").read_text())
    paths = [GOOD / c.replace("_crop.png", ".png") for c in split["holdout"]]
    paths = [p for p in paths if p.exists()]
    rng = np.random.default_rng(seed)
    groups = [[p for p in paths if v in p.name] for v in VIEWS] + [[p for p in paths if not any(v in p.name for v in VIEWS)]]
    for g in groups: rng.shuffle(g)
    out = []
    while len(out) < n and any(groups):
        for g in groups:
            if g and len(out) < n: out.append(g.pop(0))
    return out, len(paths)


def block_region(mask, brick):
    """-> (region, inner, edge_pts, bw): block pixels, safe interior for placing defects, outer-edge points, block width."""
    m = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if brick == "pink":   # mask includes the red jaws: open with a big disc so thin jaw parts drop, keep the blob at the DT max
        r = max(3, int(0.6 * dist.max()))
        reg = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        _, lab = cv2.connectedComponents(reg)
        y, x = np.unravel_index(np.argmax(dist * (reg > 0)), dist.shape)
        reg = (lab == lab[y, x]).astype(np.uint8) if lab[y, x] else m
    else:
        reg = m
    bw = float(np.sqrt(reg.sum()))
    e = max(2, int((0.15 if brick == "pink" else 0.06) * bw))
    inner = cv2.erode(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2))
    if brick == "pink":   # place only near the DT maximum (the brick body), never on the jaw seam
        y, x = np.unravel_index(np.argmax(dist * (reg > 0)), dist.shape)
        inner = inner & cv2.circle(np.zeros_like(inner), (int(x), int(y)), int(0.7 * dist[y, x]), 1, -1)
    if inner.sum() < 50: inner = reg
    cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack(cs).reshape(-1, 2) if cs else np.zeros((0, 2), int)
    outer = pts[dist[pts[:, 1], pts[:, 0]] <= 3] if len(pts) else pts      # on the TRUE mask outline (not a jaw seam)
    return reg, inner, (outer if len(outer) > 20 else pts), bw


# ------------------------------------------------------------------ defect painters (each draws into img and a 0/255 layer)
def _rand_pt(inner, rng):
    """Random pixel (x, y) inside the region."""
    ys, xs = np.nonzero(inner); i = rng.integers(len(xs)); return int(xs[i]), int(ys[i])


def _rect(img, lay, c, size, ang, colour, rng, noise=6):
    """Filled rotated rectangle with slight noise (tape)."""
    box = cv2.boxPoints((c, size, ang)).astype(np.int32)
    m = np.zeros(lay.shape, np.uint8); cv2.fillConvexPoly(m, box, 255)
    tex = np.clip(np.array(colour, float) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    img[m > 0] = tex[m > 0]; lay |= m


def _line(img, lay, c, length, ang, thick, colour):
    """Straight line of given length/angle centred at c (pen)."""
    dx, dy = 0.5 * length * np.cos(ang), 0.5 * length * np.sin(ang)
    p0, p1 = (int(c[0] - dx), int(c[1] - dy)), (int(c[0] + dx), int(c[1] + dy))
    cv2.line(img, p0, p1, colour, thick, cv2.LINE_AA); cv2.line(lay, p0, p1, 255, thick, cv2.LINE_8)


def _tape(colour):
    """Tape painter factory: 0.5*bw x 0.25*bw rectangle of the given colour inside the block."""
    def f(img, lay, g, rng):
        _rect(img, lay, _rand_pt(g["inner"], rng), (0.5 * g["bw"], 0.25 * g["bw"]), rng.uniform(0, 180), colour(rng), rng)
    return f


def _pen(length, thick, colour):
    """Pen-line painter factory: line of length*bw, thick px, random angle."""
    def f(img, lay, g, rng):
        _line(img, lay, _rand_pt(g["inner"], rng), length * g["bw"], rng.uniform(0, np.pi), thick, colour)
    return f


def tape_edge(img, lay, g, rng):
    """Grey tape centred ON the block outline (half sticks out)."""
    x, y = g["edge"][rng.integers(len(g["edge"]))]
    _rect(img, lay, (int(x), int(y)), (0.5 * g["bw"], 0.25 * g["bw"]), rng.uniform(0, 180), [rng.uniform(150, 190)] * 3, rng)


def tape_face(img, lay, g, rng):
    """Grey tape over ~80 % of the block (rectangle 0.9 x 0.9 of the block bbox at its centre)."""
    x, y, w, h = cv2.boundingRect(g["region"])
    _rect(img, lay, (x + w / 2, y + h / 2), (0.9 * w, 0.9 * h), 0, [rng.uniform(150, 190)] * 3, rng)


def pen_cross_black(img, lay, g, rng):
    """Two crossing black lines 0.25*bw, 5 px."""
    c, a = _rand_pt(g["inner"], rng), rng.uniform(0, np.pi)
    for da in (0, np.pi / 2): _line(img, lay, c, 0.25 * g["bw"], a + da, 5, (20, 20, 20))


def dot_black(img, lay, g, rng):
    """Filled black circle radius 0.04*bw."""
    c, r = _rand_pt(g["inner"], rng), max(2, int(0.04 * g["bw"]))
    cv2.circle(img, c, r, (20, 20, 20), -1, cv2.LINE_AA); cv2.circle(lay, c, r, 255, -1)


DEFECTS = {
    "tape_grey": _tape(lambda r: [r.uniform(150, 190)] * 3),
    "tape_edge": tape_edge,
    "tape_face": tape_face,
    "tape_white": _tape(lambda r: [r.uniform(230, 250)] * 3),
    "tape_black": _tape(lambda r: [r.uniform(20, 40)] * 3),
    "tape_yellow": _tape(lambda r: (30, 210, 230)),
    "pen_line_black": _pen(0.3, 4, (20, 20, 20)),
    "pen_line_thin": _pen(0.2, 2, (20, 20, 20)),
    "pen_cross_black": pen_cross_black,
    "pen_line_red": _pen(0.3, 4, (40, 30, 200)),
    "pen_line_blue": _pen(0.3, 4, (180, 60, 20)),
    "dot_black": dot_black,
}


def paint(raw, g, kind, rng):
    """-> (defect image, painted-pixel mask): paint one defect, lightly blur the painted area (sigma 1)."""
    img, lay = raw.copy(), np.zeros(raw.shape[:2], np.uint8)
    DEFECTS[kind](img, lay, g, rng)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    near = cv2.dilate(lay, np.ones((5, 5), np.uint8)) > 0
    img[near] = blur[near]
    return img, lay


# ------------------------------------------------------------------ scoring
def verdict(s, t):
    """Score -> anomalous / uncertain / clean by the model's thresholds."""
    return "anomalous" if s >= t["t_high"] else "clean" if s <= t["t_low"] else "uncertain"


def run_one(img, brick, lay, sc, model, t, tmp):
    """Mask/crop/score one image -> result dict (+ crop)."""
    m, _ = prep.brick_mask(img, brick)
    kept = None if lay is None else (0.0 if m is None else float(((m > 0) & (lay > 0)).sum() / max(1, (lay > 0).sum())))
    prep.configure_inhand(brick=brick)
    crop, info = prep.inspect_crop(img)
    if not info.get("masked"): return {"kept": kept, "verdict": "no_mask", "score": None, "why": str(info.get("why"))}, None
    cv2.imwrite(str(tmp), crop)
    s = float(sc.score(model, str(tmp), "GPU")["score"])
    return {"kept": kept, "score": s, "verdict": verdict(s, t)}, crop


def sheet(rows, t, path):
    """Contact sheet: one row per type, label + up to 6 crops at 160x160 with score."""
    S, out = 160, []
    for name, items in rows.items():
        lab = np.full((S, S, 3), 30, np.uint8)
        cv2.putText(lab, name, (4, S // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cells = [lab]
        for crop, r in items[:6]:
            c = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA) if crop is not None else np.full((S, S, 3), 60, np.uint8)
            col = {"anomalous": (60, 60, 255), "uncertain": (0, 200, 255), "clean": (80, 220, 80)}.get(r["verdict"], (200, 200, 200))
            txt = f"{r['score']:.2f}" if r["score"] is not None else r["verdict"]
            cv2.rectangle(c, (0, 0), (S - 1, 22), (0, 0, 0), -1)
            cv2.putText(c, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cells.append(c)
        cells += [np.full((S, S, 3), 20, np.uint8)] * (7 - len(cells))
        out.append(np.hstack(cells))
    cv2.imwrite(str(path), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 90])


def table(brick, t, res, n_photos, n_have, errors):
    """Print the per-type summary table."""
    print(f"\n=== {brick}: {n_photos} held-out photos used (of {n_have} found), t_low={t['t_low']} t_high={t['t_high']}, errors={errors}")
    print(f"{'type':<16}{'n':>3}{'kept':>7}{'min':>7}{'med':>7}{'max':>7}{'anom':>6}{'unc':>5}{'clean':>6}{'nomask':>7}{'>cmax':>7}")
    cmax = max([r["score"] for r in res if r["type"] == "clean" and r["score"] is not None] or [float("inf")])
    for kind in ["clean"] + list(DEFECTS):
        rs = [r for r in res if r["type"] == kind]
        if not rs: continue
        sc_ = [r["score"] for r in rs if r["score"] is not None]
        kp = [r["kept"] for r in rs if r["kept"] is not None]
        cnt = {v: sum(r["verdict"] == v for r in rs) for v in ("anomalous", "uncertain", "clean", "no_mask")}
        f = lambda v: f"{v:7.3f}" if v is not None else f"{'-':>7}"
        print(f"{kind:<16}{len(rs):>3}{f(np.mean(kp) if kp else None)}{f(min(sc_) if sc_ else None)}"
              f"{f(float(np.median(sc_)) if sc_ else None)}{f(max(sc_) if sc_ else None)}"
              f"{cnt['anomalous']:>6}{cnt['uncertain']:>5}{cnt['clean']:>6}{cnt['no_mask']:>7}{sum(x > cmax for x in sc_):>7}")


def eval_brick(brick, args, sc, out):
    """Evaluate one brick: paint, crop, score, print, write json + sheet."""
    model, root = sc.find("front", brick, args.variant)
    t = sc.thresholds(root) or {"t_low": float("nan"), "t_high": float("nan")}
    photos, n_have = pick_photos(brick, args.n, args.seed)
    tmp = pathlib.Path(tempfile.mkdtemp()) / "crop.png"
    res, rows, errors = [], {k: [] for k in ["clean"] + list(DEFECTS)}, 0
    for pi, p in enumerate(photos):
        try:
            raw = cv2.imread(str(p))
            m, why = prep.brick_mask(raw, brick)
            if m is None: raise RuntimeError(f"clean mask failed: {why}")
            reg, inner, edge, bw = block_region(m, brick)
            g = {"region": reg, "inner": inner, "edge": edge, "bw": bw}
        except Exception as e:
            errors += 1; print(f"  skip {p.name}: {e}"); continue
        for ki, kind in enumerate(["clean"] + list(DEFECTS)):
            try:
                if kind == "clean": img, lay = raw, None
                else: img, lay = paint(raw, g, kind, np.random.default_rng([args.seed, pi, ki]))
                r, crop = run_one(img, brick, lay, sc, model, t, tmp)
                res.append({"type": kind, "photo": p.name, "bw": round(bw, 1), **r})
                rows[kind].append((crop, r))
            except Exception as e:
                errors += 1; print(f"  error {p.name} {kind}: {e!r}")
                if isinstance(e, (AttributeError, SyntaxError, NameError)): traceback.print_exc()
    table(brick, t, res, len(photos), n_have, errors)
    (out / f"results_{brick}.json").write_text(json.dumps({"brick": brick, "model": model, "thresholds": t,
        "photos": [p.name for p in photos], "errors": errors, "results": res}, indent=1))
    sheet(rows, t, out / f"sheet_{brick}.jpg")


def main():
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--brick", required=True, choices=["blue", "pink", "green", "all"])
    ap.add_argument("--variant", default="", help="model variant suffix, e.g. _s384 (models/front/<brick>_s384)")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs" / time.strftime("eval_defects_%Y%m%d_%H%M%S")))
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sc = scorer.Scorer(MODELS, pathlib.Path(tempfile.mkdtemp()) / "s.log")
    try:
        for b in (["blue", "pink", "green"] if args.brick == "all" else [args.brick]): eval_brick(b, args, sc, out)
    finally:
        sc.close()
    print(f"\noutput: {out}")


if __name__ == "__main__":
    main()
