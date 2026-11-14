"""Evaluate exported anomaly models THROUGH THE RUN-TIME PATH (scorer.py -> score_worker.py -> OpenVINO), so training/
run-time preprocessing skew is impossible, on the held-out good + defect crops. Writes thresholds where the sorter reads them.

    python3 anomaly_capture/eval_anomaly.py --brick red                      # models/front/red (+ red_efficientad = cascade)
    python3 .../eval_anomaly.py --model-dir models/front/red --device GPU --latency-devices CPU,GPU,NPU --write-thresholds
    python3 .../eval_anomaly.py --brick red --compare                        # every models/front/red* variant, one table
    python3 .../eval_anomaly.py --model-dir A --model-dir B --compare        # explicit variants (A = baseline)
Any python with numpy + opencv works (Studio venv, the Anomalib venv, the repo .venv); the model itself always runs in
the Anomalib venv: env ANOMALIB_PY (default ~/anomalib/.venv/bin/python; if missing, this python is used).

What it measures (decision 9: "every change must earn its place on a held-out test set"):
  * photo level: image AUROC, recall at zero false alarms (defects above the highest good), false-alarm rate at 100 %
    recall, score histograms, per-view breakdown
  * object level = the sorter's rule: an object (episode = run/cycle) is a defect if ANY of its photos is; score =
    max over its views; verdicts with the sorter's own decide() + thresholds_from() (imported from sorter.py; where
    sorter.py can't be imported (no lerobot/fastapi) its functions are loaded from its source, never copied)
  * thresholds t_low/t_high from all held-out data (what --write-thresholds saves = what the page's calibrate does),
    plus an HONEST estimate: 2-fold cross-calibration by episode (fit on one half, apply to the other)
  * cascade fast -> main (sorter.look logic): share settled by the fast model, verdict accuracy, mean latency
  * latency per image per device (worker inference ms + end-to-end ms with IPC), model load ms
  * with size labels (synth_truth.jsonl from synth_data.py, or a hand-made defect_sizes.jsonl with {"episode"|"file",
    "cross_mm"} lines from measured crosses): detection rate vs cross size and the smallest size detected reliably
  * heat-map panels of the worst misses and false alarms
Output: <model-dir>/eval/{eval.json, report.md, hist.png, size_curve.png, worst.png}; --compare writes each variant's
details to <model-dir>/eval_compare/ (common test set) and the table to runs/eval/compare_<cam>_<brick>_<time>.{json,md}. A model card's "eval" entry is updated. SYNTHETIC data (a
synth_info.json in the data dir) is labelled as such in every output.
"""
import argparse, ast, json, os, sys, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import scorer as scorer_mod  # noqa: E402
import train_anomaly as T  # noqa: E402


# ---------------------------------------------------------------- the sorter's own decision functions (reused)
def _sorter_functions():
    try:
        import sorter
        return sorter.decide, sorter.auroc, sorter.thresholds_from, "import"
    except Exception:          # sorter imports arm (lerobot) + fastapi: load only these pure functions from its source
        src = (HERE / "sorter.py").read_text()
        keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("decide", "auroc", "thresholds_from")]
        ns = {"np": np}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(HERE / "sorter.py"), "exec"), ns)
        return ns["decide"], ns["auroc"], ns["thresholds_from"], "source"


decide, auroc, thresholds_from, SORTER_HOW = _sorter_functions()


# ---------------------------------------------------------------- metrics (pure, tested)
def recall_at_zero_fa(good, defect):
    """Share of defects scoring above the highest good score (no false alarm at all)."""
    if not len(good) or not len(defect): return None
    return float((np.asarray(defect) > np.max(good)).mean())


def fa_at_full_recall(good, defect):
    """Share of goods scoring at or above the lowest defect (false alarms needed to catch every defect)."""
    if not len(good) or not len(defect): return None
    return float((np.asarray(good) >= np.min(defect)).mean())


def by_episode(scores, eps):
    """{file: score} -> {episode: max score} (max over views = the sorter's 'any view anomalous')."""
    out = {}
    for f, s in scores.items(): out[eps[f]] = max(out.get(eps[f], -np.inf), s)
    return out


def object_verdict(decisions):
    """Sorter rule for one object: any 'defect' -> defect; any 'uncertain' (after retakes) -> defect plate (fail-safe);
    all 'clean' -> good."""
    if "defect" in decisions: return "defect", "defect"
    if "uncertain" in decisions: return "defect", "uncertain"
    return "good", "clean"


def cascade_photo(scores, stages):
    """One photo through the stages as in sorter.Sorter.look (no retakes): a stage that is sure decides, else the next.
    scores[i] = score of stage i. -> (decision, number of stages used)."""
    dec = "uncertain"
    for i, (s, st) in enumerate(zip(scores, stages)):
        dec, _ = decide([s], st["t_low"], st["t_high"])
        if dec != "uncertain": return dec, i + 1
    return dec, len(stages)


def verdicts(photo_scores, eps, stages_scores=None, t=None, stages=None):
    """Per object (episode): verdict + reason from per-photo decisions. Single model: t = thresholds dict;
    cascade: stages_scores = [{file: score} per stage], stages = [thresholds per stage]."""
    per = {}
    for f in photo_scores:
        if stages:
            d, _ = cascade_photo([ss[f] for ss in stages_scores], stages)
        else:
            d, _ = decide([photo_scores[f]], t["t_low"], t["t_high"])
        per.setdefault(eps[f], []).append(d)
    return {e: object_verdict(d) for e, d in per.items()}


def outcome(vg, vd):
    """vg/vd: verdicts of good/defect objects -> rates the sorter would produce."""
    n_g, n_d = len(vg), len(vd)
    r = {"good_objects": n_g, "defect_objects": n_d,
         "good_to_good": sum(v[0] == "good" for v in vg.values()), "good_rejected_uncertain": sum(v[1] == "uncertain" for v in vg.values()),
         "good_rejected_defect": sum(v[1] == "defect" for v in vg.values()),
         "defect_caught": sum(v[0] == "defect" for v in vd.values()), "defect_caught_uncertain": sum(v[1] == "uncertain" for v in vd.values()),
         "defect_missed": sum(v[0] == "good" for v in vd.values())}
    r["accuracy"] = round((r["good_to_good"] + r["defect_caught"]) / max(1, n_g + n_d), 4)
    r["false_reject_rate"] = round(1 - r["good_to_good"] / n_g, 4) if n_g else None
    r["miss_rate"] = round(r["defect_missed"] / n_d, 4) if n_d else None
    return r


def folds(eps_list, k=2, seed=0):
    import hashlib
    rank = sorted(set(eps_list), key=lambda e: hashlib.sha1(f"fold{seed}:{e}".encode()).hexdigest())
    return [set(rank[i::k]) for i in range(k)]


def cross_calibrated(good_scores, good_eps, defect_scores, defect_eps, seed=0):
    """2-fold by episode: thresholds fitted on one fold (sorter.thresholds_from), verdicts on the other. Honest estimate
    of what fresh objects would get. -> (pooled outcome, per-fold thresholds) or (None, reason)."""
    gf, df = folds(good_eps.values(), 2, seed), folds(defect_eps.values(), 2, seed)
    vg, vd, ts = {}, {}, []
    for i in range(2):
        j = 1 - i
        fit_g = [s for f, s in good_scores.items() if good_eps[f] in gf[i]]
        fit_d = [m for e, m in by_episode({f: s for f, s in defect_scores.items() if defect_eps[f] in df[i]}, defect_eps).items()]
        if len(fit_g) < 2 or len(fit_d) < 2: return None, "too few episodes for 2-fold cross-calibration (need >= 4 defect + 4 good)"
        try: t = thresholds_from(fit_g, fit_d)
        except RuntimeError as e: return None, f"fold {i}: {e}"
        ts.append({k: t[k] for k in ("t_low", "t_high", "overlap")})
        vg.update(verdicts({f: s for f, s in good_scores.items() if good_eps[f] in gf[j]}, good_eps, t=t))
        vd.update(verdicts({f: s for f, s in defect_scores.items() if defect_eps[f] in df[j]}, defect_eps, t=t))
    return outcome(vg, vd), ts


SIZE_BINS = [(0, 1.75, "<=1.5"), (1.75, 2.5, "2"), (2.5, 4.0, "3"), (4.0, 6.5, "5"), (6.5, 9.0, "8"), (9.0, 1e9, ">=10")]


def size_curve(scores, sizes, thr_zero_fa, t_high=None, min_n=3, reliable=0.9):
    """scores/sizes: {item: value} for defects whose cross is visible. Detection rate per size bin at the zero-false-alarm
    threshold (score > highest good) and at t_high. smallest_reliable = smallest bin from which every bin up has
    rate >= reliable with >= min_n samples."""
    rows = []
    for lo, hi, name in SIZE_BINS:
        v = [scores[k] for k in scores if k in sizes and lo <= sizes[k] < hi]
        if not v: continue
        rows.append({"bin": name, "lo": lo, "n": len(v), "rate_zero_fa": round(float((np.asarray(v) > thr_zero_fa).mean()), 3),
                     "rate_t_high": None if t_high is None else round(float((np.asarray(v) >= t_high).mean()), 3),
                     "score_median": round(float(np.median(v)), 4)})
    small = None
    for i in range(len(rows)):
        if all(r["n"] >= min_n and r["rate_zero_fa"] >= reliable for r in rows[i:]): small = rows[i]["bin"]; break
    return {"bins": rows, "smallest_reliable": small, "rule": f"every bin from here up: >= {reliable:.0%} detected at zero false alarms, n >= {min_n}"}


def latency_stats(ms):
    if not ms: return None
    a = np.asarray(ms, float)
    return {"n": len(a), "mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2)}


# ---------------------------------------------------------------- data
def load_truth(data, path=None):
    """Size labels for evaluation: synth_truth.jsonl (synthetic) or defect_sizes.jsonl (hand-measured crosses).
    -> ({crop name: rec}, {episode: cross_mm}, source)"""
    cands = [Path(path)] if path else [Path(data) / "synth_truth.jsonl", Path(data) / "defect_sizes.jsonl"]
    for p in cands:
        if not p.exists(): continue
        files, eps = {}, {}
        for line in open(p):
            line = line.strip()
            if not line: continue
            j = json.loads(line)
            if j.get("file"): files[Path(j["file"]).name] = j
            if j.get("episode") and j.get("cross_mm") is not None: eps[j["episode"]] = float(j["cross_mm"])
        return files, eps, str(p)
    return {}, {}, None


def test_set(mdir, data, cam, brick):
    """Held-out good + all defect crops (defects are never trained on). -> (good, defect, train names, note)."""
    sp = mdir / "split.json"; note = None
    if sp.exists():
        s = json.loads(sp.read_text()); hold, train = set(s["holdout"]), set(s["train"])
    else:
        try: hold = set(json.loads((mdir / "holdout.json").read_text()))
        except Exception: hold = set()
        train = set()
        note = "no split.json (old trainer): held-out goods from holdout.json, train set unknown"
    allg = T.list_crops(data, cam, "good", brick)
    good = [f for f in allg if f.name in hold]
    if not good:
        good = [f for f in allg if f.name not in train]
        note = "no held-out goods: scoring goods that may have been trained on (optimistic)"
    return good, T.list_crops(data, cam, "defect", brick), train, note


def model_xml(mdir):
    m = scorer_mod.Scorer._model_file(Path(mdir))
    if not m: raise SystemExit(f"no model.xml / model.pt under {mdir}")
    return Path(m)


# ---------------------------------------------------------------- scoring through the live path
class Runner:
    def __init__(self, log_dir):
        if not scorer_mod.ANOMALIB_PY.exists():
            print(f"note: {scorer_mod.ANOMALIB_PY} not found, using this python for the score worker")
            scorer_mod.ANOMALIB_PY = Path(sys.executable)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.s = scorer_mod.Scorer(log_dir, log_dir / "score_worker.log")

    def score(self, model, files, device, cache=None, heat_dir=None):
        """-> {name: {"score", "ms", "wall_ms"}} ; cache = dict persisted by the caller."""
        out, key = {}, f"{model}|{device}|{Path(model).stat().st_mtime}"
        c = (cache or {}).setdefault(key, {}) if cache is not None else {}
        for f in files:
            if f.name in c: out[f.name] = c[f.name]; continue
            t0 = time.perf_counter()
            r = self.s.score(model, f, device)
            out[f.name] = c[f.name] = {"score": float(r["score"]), "ms": r.get("ms"), "wall_ms": round((time.perf_counter() - t0) * 1000, 2)}
        return out

    def usable(self, model, f, device):
        """device if the model runs there, else CPU (e.g. the NPU driver is missing): the metrics don't depend on it."""
        try: self.s.score(model, f, device); return device
        except Exception as e:
            if device == "CPU": raise
            print(f"WARNING: {Path(model).parent.parent.parent.name} does not run on {device} ({str(e)[:120]}): scoring on CPU")
            return "CPU"

    def heat(self, model, f, device, out):
        self.s.score(model, f, device, heatmap=out)
        return out

    def latency(self, model, files, device, n=20):
        try:
            r0 = self.s.score(model, files[0], device)
            ms, wall = [], []
            for f in (files * 3)[1:n + 1]:
                t0 = time.perf_counter(); r = self.s.score(model, f, device)
                ms.append(r["ms"]); wall.append((time.perf_counter() - t0) * 1000)
            return {"device": device, "load_ms": r0.get("load_ms"), "infer_ms": latency_stats(ms), "end_to_end_ms": latency_stats(wall)}
        except Exception as e:
            return {"device": device, "error": str(e)[:300]}

    def close(self): self.s.close()


# ---------------------------------------------------------------- plots (opencv only)
def _stamp(img, synthetic):
    if synthetic: cv2.putText(img, "SYNTHETIC DATA", (img.shape[1] - 230, 24), 0, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
    return img


def plot_hist(path, good, defect, t, title, synthetic):
    Wd, Hd, m = 760, 360, 50
    img = np.full((Hd, Wd, 3), 255, np.uint8)
    allv = np.asarray(list(good) + list(defect), float)
    lo, hi = float(allv.min()), float(allv.max()); hi = hi if hi > lo else lo + 1
    bins = np.linspace(lo, hi, 31)
    hg, hd = np.histogram(good, bins)[0], np.histogram(defect, bins)[0]
    top = max(1, hg.max(), hd.max())
    X = lambda v: int(m + (v - lo) / (hi - lo) * (Wd - 2 * m)); Y = lambda c: int(Hd - m - c / top * (Hd - 2 * m))
    for h, col, off in ((hg, (200, 120, 30), 0), (hd, (30, 110, 230), 3)):
        for i, c in enumerate(h):
            if c: cv2.rectangle(img, (X(bins[i]) + off, Y(c)), (X(bins[i + 1]) - 3 + off, Hd - m), col, -1 if off == 0 else 2)
    if t:
        for k, col in (("t_low", (0, 160, 0)), ("t_high", (0, 0, 200))):
            if lo <= t[k] <= hi: cv2.line(img, (X(t[k]), m - 10), (X(t[k]), Hd - m), col, 2); cv2.putText(img, k, (X(t[k]) + 3, m), 0, 0.45, col, 1)
    cv2.line(img, (m, Hd - m), (Wd - m, Hd - m), (0, 0, 0), 1)
    for v in np.linspace(lo, hi, 5): cv2.putText(img, f"{v:.3f}", (X(v) - 18, Hd - m + 18), 0, 0.4, (0, 0, 0), 1)
    cv2.putText(img, title, (10, 22), 0, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"good (filled, n={len(good)})   defect (outline, n={len(defect)})   count max {top}", (10, Hd - 10), 0, 0.45, (60, 60, 60), 1)
    cv2.imwrite(str(path), _stamp(img, synthetic))


def plot_size(path, curve, title, synthetic):
    rows = curve["bins"]
    Wd, Hd, m = 640, 340, 55
    img = np.full((Hd, Wd, 3), 255, np.uint8)
    if rows:
        X = lambda i: int(m + (i + 0.5) / len(rows) * (Wd - 2 * m)); Y = lambda r: int(Hd - m - r * (Hd - 2 * m))
        for key, col in (("rate_zero_fa", (30, 110, 230)), ("rate_t_high", (0, 150, 0))):
            pts = [(X(i), Y(r[key])) for i, r in enumerate(rows) if r[key] is not None]
            for p, q in zip(pts, pts[1:]): cv2.line(img, p, q, col, 2, cv2.LINE_AA)
            for p in pts: cv2.circle(img, p, 4, col, -1)
        for i, r in enumerate(rows):
            cv2.putText(img, f"{r['bin']} mm", (X(i) - 25, Hd - m + 18), 0, 0.45, (0, 0, 0), 1)
            cv2.putText(img, f"n={r['n']}", (X(i) - 18, Hd - m + 34), 0, 0.4, (90, 90, 90), 1)
    for r in (0, 0.5, 0.9, 1.0):
        y = int(Hd - m - r * (Hd - 2 * m)); cv2.line(img, (m, y), (Wd - m, y), (220, 220, 220), 1)
        cv2.putText(img, f"{r:.0%}", (8, y + 4), 0, 0.4, (0, 0, 0), 1)
    cv2.putText(img, title, (10, 20), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "detected @ zero false alarm (orange) / @ t_high (green)", (10, 40), 0, 0.42, (60, 60, 60), 1)
    cv2.imwrite(str(path), _stamp(img, synthetic))


def panel(path, items, title, synthetic, side=200):
    """items: [(crop path, heat path or None, caption)] -> one PNG, a row per item (crop | heat-map)."""
    if not items: return None
    rows = []
    for crop, heat, cap in items:
        a = cv2.resize(cv2.imread(str(crop)), (side, side))
        b = cv2.resize(cv2.imread(str(heat)), (side, side)) if heat and Path(heat).exists() else np.full_like(a, 230)
        r = np.hstack([a, b, np.full((side, 330, 3), 255, np.uint8)])
        for i, line in enumerate(cap.split("\n")): cv2.putText(r, line, (2 * side + 8, 24 + 20 * i), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(r)
    head = np.full((34, rows[0].shape[1], 3), 255, np.uint8); cv2.putText(head, title, (8, 22), 0, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), _stamp(np.vstack([head] + rows), synthetic))
    return path


# ---------------------------------------------------------------- one model
def evaluate(mdir, data=None, device="CPU", lat_devices=("CPU",), fast=None, fast_device=None, truth_path=None,
             write_thresholds=False, files_override=None, runner=None, n_worst=6, use_cache=True, seed=0, eval_name="eval"):
    mdir = Path(mdir).resolve()
    card = {}
    try: card = json.loads((mdir / "model_card.json").read_text())
    except Exception: pass
    data = Path(data or card.get("data_dir") or ROOT / "training_images")
    cam = card.get("cam") or mdir.parent.name
    brick = card.get("brick", mdir.name.split("_")[0])
    brick = "" if brick == "all" else brick
    synthetic = (data / "synth_info.json").exists()
    ev = mdir / eval_name; ev.mkdir(exist_ok=True)
    good, defect, train, note = test_set(mdir, data, cam, brick)
    if files_override: good, defect = files_override
    leak = sorted({f.name for f in good + defect} & train)
    if leak: raise SystemExit(f"{len(leak)} test files were trained on (e.g. {leak[0]}): refusing to evaluate")
    meta = T.load_meta(data)
    eps = {f.name: T.episode_of(f, meta) for f in good + defect}
    if set(eps[f.name] for f in good) & set(eps[f.name] for f in defect):
        print("WARNING: some episodes have both good and defect photos")
    tf, tep, tsrc = load_truth(data, truth_path)
    xml = model_xml(mdir)
    own = runner is None
    runner = runner or Runner(ev)
    cache_p = ev / "scores_cache.json"
    cache = {}
    if use_cache:
        try: cache = json.loads(cache_p.read_text())
        except Exception: cache = {}
    t0 = time.time()
    try:
        device = runner.usable(xml, (good + defect)[0], device)
        sg = runner.score(xml, good, device, cache); sd = runner.score(xml, defect, device, cache)
        res = {"model_dir": str(mdir), "model": str(xml), "created": datetime.now().isoformat(timespec="seconds"),
               "synthetic_data": synthetic, "data_dir": str(data), "cam": cam, "brick": brick or "all", "device": device,
               "sorter_functions": SORTER_HOW, "card": {k: card.get(k) for k in ("model", "tag", "image_size", "params", "compress", "data", "train_s")},
               "n_good_photos": len(good), "n_defect_photos": len(defect), "note": note, "truth": tsrc}
        G = {k: v["score"] for k, v in sg.items()}; D = {k: v["score"] for k, v in sd.items()}
        g, d = list(G.values()), list(D.values())
        vis = {k: v for k, v in D.items() if k in tf and tf[k].get("visible")} if tf else None
        vis = vis or None                      # no per-photo visibility labels (e.g. only measured sizes per episode)
        res["photo"] = {"auroc_all_defect_photos": auroc(g, d), "recall_at_zero_fa": recall_at_zero_fa(g, d),
                        "fa_at_full_recall": fa_at_full_recall(g, d), "good_max": max(g) if g else None,
                        "good_p95": float(np.percentile(g, 95)) if g else None}
        if vis is not None:
            res["photo"].update({"n_defect_visible": len(vis), "auroc_visible": auroc(g, list(vis.values())),
                                 "recall_at_zero_fa_visible": recall_at_zero_fa(g, list(vis.values())),
                                 "fa_at_full_recall_visible": fa_at_full_recall(g, list(vis.values()))})
        geps = {k: eps[k] for k in G}; deps = {k: eps[k] for k in D}
        Og, Od = by_episode(G, geps), by_episode(D, deps)
        res["object"] = {"n_good_objects": len(Og), "n_defect_objects": len(Od), "auroc": auroc(list(Og.values()), list(Od.values())),
                         "recall_at_zero_fa": recall_at_zero_fa(list(Og.values()), list(Od.values())),
                         "fa_at_full_recall": fa_at_full_recall(list(Og.values()), list(Od.values()))}
        # thresholds exactly as the page's calibrate: held-out good photos + the max photo of every defect object
        try:
            t = thresholds_from(g, list(Od.values()))
            res["thresholds"] = t
            res["object"]["in_sample"] = outcome(verdicts(G, geps, t=t), verdicts(D, deps, t=t))
        except Exception as e:
            t = None; res["thresholds"] = {"error": str(e)}
        cc, info = cross_calibrated(G, geps, D, deps, seed)
        res["object"]["cross_calibrated"] = cc if cc else {"skipped": info}
        if cc: res["object"]["cross_calibrated_thresholds"] = info
        # per view
        view = lambda k: T.crop_stem(k).split("_", 4)[-1]
        res["per_view"] = {}
        for v in sorted({view(k) for k in list(G) + list(D)}):
            gv = [s for k, s in G.items() if view(k) == v]
            dv = [s for k, s in (vis if vis is not None else D).items() if view(k) == v]
            res["per_view"][v] = {"n_good": len(gv), "n_defect": len(dv), "auroc": auroc(gv, dv), "recall_at_zero_fa": recall_at_zero_fa(gv, dv)}
        # size curve (photo level on visible crosses; object level on episode sizes)
        if tf or tep:
            gmax = max(g)
            psz = {k: tf[k]["cross_mm"] for k in (vis or {}) if tf[k].get("cross_mm") is not None}
            osz = {e: tep[e] for e in Od if e in tep}
            res["size"] = {"photo": size_curve({k: D[k] for k in psz}, psz, gmax, t and t.get("t_high")),
                           "object": size_curve({e: Od[e] for e in osz}, osz, max(Og.values()), t and t.get("t_high")),
                           "source": tsrc}
            if vis is not None:
                hard = {}
                for k, s in vis.items():
                    for h in tf[k].get("hard") or ["none"]:
                        hard.setdefault(h, []).append(s > gmax)
                res["size"]["by_hard_case"] = {h: {"n": len(v), "rate_zero_fa": round(float(np.mean(v)), 3)} for h, v in hard.items()}
                res["size"]["not_visible_objects"] = sum(1 for e in Od if not any(tf.get(k, {}).get("visible") for k in D if deps[k] == e))
        # cascade
        if fast:
            fxml = model_xml(fast); fdev = runner.usable(fxml, (good + defect)[0], fast_device or device)
            fg = runner.score(fxml, good, fdev, cache); fd = runner.score(fxml, defect, fdev, cache)
            FG = {k: v["score"] for k, v in fg.items()}; FD = {k: v["score"] for k, v in fd.items()}
            try:
                ft = thresholds_from(list(FG.values()), list(by_episode(FD, deps).values()))
            except Exception as e:
                ft = None; res["cascade"] = {"fast_model": str(fxml), "error": f"fast model not calibratable: {e}"}
            if ft and t:
                stages = [ft, t]
                used = [cascade_photo([FG.get(k, FD.get(k)), {**G, **D}[k]], stages)[1] for k in list(G) + list(D)]
                ms_f = [v["ms"] for v in list(fg.values()) + list(fd.values())]; ms_m = [v["ms"] for v in list(sg.values()) + list(sd.values())]
                per_photo = [ms_f[i] + (ms_m[i] if u == 2 else 0) for i, u in enumerate(used)]
                res["cascade"] = {"fast_model": str(fxml), "fast_device": fdev, "fast_thresholds": ft,
                                  "fast_alone": {"auroc_photo": auroc(list(FG.values()), list(FD.values())),
                                                 "object": outcome(verdicts(FG, geps, t=ft), verdicts(FD, deps, t=ft))},
                                  "share_settled_by_fast": round(float(np.mean([u == 1 for u in used])), 3),
                                  "object": outcome(verdicts(G, geps, [FG, G], stages=stages), verdicts(D, deps, [FD, D], stages=stages)),
                                  "main_alone_object": res["object"].get("in_sample"),
                                  "mean_ms_per_photo": round(float(np.mean(per_photo)), 2),
                                  "main_alone_mean_ms": round(float(np.mean(ms_m)), 2),
                                  "note": "in-sample thresholds for both stages; latency = worker inference ms on the eval device(s)"}
                if write_thresholds:
                    scorer_mod.Scorer.save_thresholds(Path(fast), _thr_record(ft, len(FG), fdev, synthetic))
        # latency per device
        res["latency"] = [runner.latency(xml, good + defect, dv) for dv in lat_devices]
        # worst cases with heat-maps
        heat = ev / "heat"; heat.mkdir(exist_ok=True)
        fa = sorted(G.items(), key=lambda kv: -kv[1])[:n_worst]
        pool = vis if vis else {k: s for k, s in D.items() if s == Od[deps[k]]}
        miss = sorted(pool.items(), key=lambda kv: kv[1])[:n_worst]
        paths = {f.name: f for f in good + defect}
        items = []
        for kind, lst in (("FALSE ALARM?", fa), ("MISS?", miss)):
            for k, s in lst:
                hp = runner.heat(xml, paths[k], device, heat / (k[:-4] + "_heat.jpg"))
                tr = tf.get(k, {})
                extra = f"cross {tr.get('cross_mm')} mm {','.join(tr.get('hard') or [])}" if tr.get("cross_mm") else ""
                items.append((paths[k], hp, f"{kind} score {s:.4f}\n{k[:40]}\nepisode {eps[k][:34]}\n{extra}"))
        res["worst_png"] = str(panel(ev / "worst.png", items, f"highest good / lowest defect scores - {mdir.name}", synthetic) or "")
        if g and d: plot_hist(ev / "hist.png", g, d, t, f"{mdir.name}: photo scores (held-out good vs all defect photos)", synthetic)
        if res.get("size", {}).get("photo", {}).get("bins"):
            plot_size(ev / "size_curve.png", res["size"]["photo"], f"{mdir.name}: detection rate vs cross size (photo, visible)", synthetic)
        res["eval_s"] = round(time.time() - t0, 1)
        if write_thresholds and t:
            scorer_mod.Scorer.save_thresholds(mdir, _thr_record(t, len(g), device, synthetic))
            res["thresholds_written"] = str(mdir / "thresholds.json")
    finally:
        if use_cache: cache_p.write_text(json.dumps(cache))
        if own: runner.close()
    (ev / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    (ev / "report.md").write_text(report_md(res))
    if card and eval_name == "eval":
        card["eval"] = {"date": res["created"], "dir": str(ev), "synthetic_data": synthetic, "photo_auroc": res["photo"]["auroc_all_defect_photos"],
                        "object_auroc": res["object"]["auroc"], "object_recall_at_zero_fa": res["object"]["recall_at_zero_fa"],
                        "thresholds": {k: res["thresholds"].get(k) for k in ("t_low", "t_high")} if "t_low" in res["thresholds"] else None}
        (mdir / "model_card.json").write_text(json.dumps(card, indent=1))
    return res


def _thr_record(t, n_good, device, synthetic):
    """Same keys as sorter.calibrate() writes (+ where it came from)."""
    return {"t_low": t["t_low"], "t_high": t["t_high"], "calibrated": datetime.now().isoformat(timespec="seconds"),
            "n_good": n_good, "defect_bricks": t["defect_bricks"], "auroc": t["auroc"], "overlap": t["overlap"],
            "source": "eval_anomaly.py", "device": device, "synthetic_data": synthetic}


# ---------------------------------------------------------------- reports
def _f(v, p=3):
    return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def report_md(r):
    L = []
    if r["synthetic_data"]: L += ["> **SYNTHETIC DATA** (anomaly_capture/synth_data.py): these numbers test the pipeline, they are NOT station results.", ""]
    c = r["card"]
    L += [f"# Anomaly model evaluation: {r['cam']}/{r['brick']} ({c.get('model') or '?'}{'_' + c['tag'] if c.get('tag') else ''})", "",
          f"- model: `{r['model']}`  ·  scored through scorer.py -> score_worker.py on **{r['device']}**  ·  {r['created']}",
          f"- data: `{r['data_dir']}`  ·  held-out good photos {r['n_good_photos']} ({r['object']['n_good_objects']} objects), "
          f"defect photos {r['n_defect_photos']} ({r['object']['n_defect_objects']} objects)",
          f"- trained on: {_f((c.get('data') or {}).get('train'))} good crops, image size {c.get('image_size')}, params `{json.dumps(c.get('params'))}`"]
    if r.get("note"): L.append(f"- note: {r['note']}")
    p, o = r["photo"], r["object"]
    L += ["", "## Separation", "", "| level | AUROC | recall @ 0 false alarms | false alarms @ 100 % recall |", "|---|---|---|---|",
          f"| photo (all defect photos, incl. cross facing away) | {_f(p['auroc_all_defect_photos'])} | {_f(p['recall_at_zero_fa'])} | {_f(p['fa_at_full_recall'])} |"]
    if "auroc_visible" in p:
        L.append(f"| photo (cross visible, n={p['n_defect_visible']}) | {_f(p['auroc_visible'])} | {_f(p['recall_at_zero_fa_visible'])} | {_f(p['fa_at_full_recall_visible'])} |")
    L += [f"| **object** (max over views, sorter rule) | **{_f(o['auroc'])}** | **{_f(o['recall_at_zero_fa'])}** | **{_f(o['fa_at_full_recall'])}** |", ""]
    t = r["thresholds"]
    if "t_low" in t:
        L += ["## Thresholds (sorter.thresholds_from, all held-out data)", "",
              f"t_low **{t['t_low']}**, t_high **{t['t_high']}** (good max {t['good_max']}, defect min {t['defect_min']}, overlap {t['overlap']}, "
              f"defect objects not visible {t['defect_bricks_not_visible']}/{t['defect_bricks']}, good photos in the uncertain band {t['good_uncertain_share']})"
              + (f"  ->  written to `{r['thresholds_written']}`" if r.get("thresholds_written") else ""), ""]
    else:
        L += ["## Thresholds", "", f"not computable: {t.get('error')}", ""]
    L += ["## What the sorter would do (object level)", "", "| thresholds | good -> good plate | good rejected (defect / uncertain) | defects caught (of which uncertain) | defects MISSED | accuracy |", "|---|---|---|---|---|---|"]
    for name, k in (("in-sample (optimistic)", "in_sample"), ("2-fold cross-calibrated (honest)", "cross_calibrated")):
        x = o.get(k)
        if not x or "skipped" in x:
            L.append(f"| {name} | {(x or {}).get('skipped', 'n/a')} | | | | |"); continue
        L.append(f"| {name} | {x['good_to_good']}/{x['good_objects']} | {x['good_rejected_defect']} / {x['good_rejected_uncertain']} | "
                 f"{x['defect_caught']}/{x['defect_objects']} ({x['defect_caught_uncertain']}) | {x['defect_missed']} | {x['accuracy']} |")
    L += ["", "## Per view (photo level)", "", "| view | good | defect | AUROC | recall @ 0 FA |", "|---|---|---|---|---|"]
    for v, x in r["per_view"].items(): L.append(f"| {v} | {x['n_good']} | {x['n_defect']} | {_f(x['auroc'])} | {_f(x['recall_at_zero_fa'])} |")
    if r.get("size"):
        s = r["size"]
        L += ["", f"## Detection rate vs cross size (labels: `{s['source']}`)", "", "| cross | n | detected @ 0 FA | detected @ t_high | median score |", "|---|---|---|---|---|"]
        for b in s["photo"]["bins"]: L.append(f"| {b['bin']} mm | {b['n']} | {_f(b['rate_zero_fa'], 2)} | {_f(b['rate_t_high'], 2)} | {_f(b['score_median'])} |")
        L += ["", f"Smallest cross detected reliably (photo): **{s['photo']['smallest_reliable'] or 'none'}** mm; object level: "
              f"**{s['object']['smallest_reliable'] or 'none'}** mm ({s['photo']['rule']})."]
        if s.get("by_hard_case"):
            L += ["", "Hard cases (visible crosses, detected @ 0 FA): " + ", ".join(f"{h} {x['rate_zero_fa']:.2f} (n={x['n']})" for h, x in s["by_hard_case"].items())]
        L += ["", "![size](size_curve.png)"]
    if r.get("cascade"):
        cz = r["cascade"]
        L += ["", "## Cascade fast -> main (sorter.look logic)", ""]
        if "error" in cz: L.append(cz["error"])
        else:
            L += [f"- fast model `{cz['fast_model']}` on {cz['fast_device']}, thresholds {cz['fast_thresholds']['t_low']} / {cz['fast_thresholds']['t_high']}",
                  f"- photos settled by the fast model: **{cz['share_settled_by_fast']:.0%}**",
                  f"- mean inference per photo: cascade **{cz['mean_ms_per_photo']} ms** vs main alone {cz['main_alone_mean_ms']} ms",
                  f"- object accuracy: cascade {cz['object']['accuracy']} (missed {cz['object']['defect_missed']}) vs main alone "
                  f"{(cz['main_alone_object'] or {}).get('accuracy')} vs fast alone {cz['fast_alone']['object']['accuracy']}", f"- {cz['note']}"]
    L += ["", "## Latency per photo", "", "| device | load ms | inference mean / p50 / p95 ms | end-to-end (with IPC) mean ms |", "|---|---|---|---|"]
    for x in r["latency"]:
        if "error" in x: L.append(f"| {x['device']} | error: {x['error'][:80]} | | |")
        else: L.append(f"| {x['device']} | {_f(x['load_ms'], 0)} | {x['infer_ms']['mean']} / {x['infer_ms']['p50']} / {x['infer_ms']['p95']} | {x['end_to_end_ms']['mean']} |")
    L += ["", "![hist](hist.png)", "", "![worst](worst.png)", "", f"_eval {r.get('eval_s')} s; decision functions from sorter.py ({r['sorter_functions']})_", ""]
    return "\n".join(L)


def compare(dirs, data=None, device="CPU", lat_devices=("CPU",), truth=None, out_dir=None):
    """Evaluate every variant on the SAME test set (intersection of their held-out goods, minus anything any variant
    trained on; all defects) and write one table. dirs[0] = baseline."""
    dirs = [Path(d).resolve() for d in dirs]
    cards = []
    for d in dirs:
        try: cards.append(json.loads((d / "model_card.json").read_text()))
        except Exception: cards.append({})
    cams = {c.get("cam") or d.parent.name for c, d in zip(cards, dirs)}
    if len(cams) > 1: raise SystemExit(f"--compare needs one camera, got {cams}")
    sets = [test_set(d, Path(data or c.get("data_dir") or ROOT / "training_images"), c.get("cam") or d.parent.name,
                     "" if c.get("brick", "all") == "all" else c.get("brick", d.name.split("_")[0])) for d, c in zip(dirs, cards)]
    trained = set().union(*[s[2] for s in sets])
    common = set.intersection(*[{f.name for f in s[0]} for s in sets]) - trained
    good = [f for f in sets[0][0] if f.name in common]
    defect = [f for f in sets[0][1] if f.name not in trained]
    if not good: raise SystemExit("the variants share no held-out good crops (different splits?): retrain with the same --seed/--holdout-frac")
    runner = Runner((Path(out_dir) if out_dir else ROOT / "runs" / "eval") / "worker")
    rows = []
    try:
        for d in dirs:
            r = evaluate(d, data, device, lat_devices, truth_path=truth, files_override=(good, defect), runner=runner,
                         eval_name="eval_compare")
            lat = next((x for x in r["latency"] if "infer_ms" in x), None)
            cc = r["object"].get("cross_calibrated") or {}
            rows.append({"variant": d.name, "dir": str(d), "model": r["card"].get("model"), "image_size": r["card"].get("image_size"),
                         "params": r["card"].get("params"), "photo_auroc": r["photo"]["auroc_all_defect_photos"],
                         "object_auroc": r["object"]["auroc"], "object_recall_zero_fa": r["object"]["recall_at_zero_fa"],
                         "object_fa_full_recall": r["object"]["fa_at_full_recall"], "cc_accuracy": cc.get("accuracy"),
                         "cc_missed": cc.get("defect_missed"), "smallest_mm": (r.get("size") or {}).get("photo", {}).get("smallest_reliable"),
                         "ms": lat and lat["infer_ms"]["mean"], "ms_device": lat and lat["device"], "synthetic": r["synthetic_data"]})
    finally:
        runner.close()
    base = rows[0]
    for x in rows:
        better = ((x["object_recall_zero_fa"] or 0) > (base["object_recall_zero_fa"] or 0)
                  or ((x["object_recall_zero_fa"] or 0) == (base["object_recall_zero_fa"] or 0) and (x["object_auroc"] or 0) > (base["object_auroc"] or 0) + 0.005))
        x["earns_its_place"] = "baseline" if x is base else ("yes" if better else "no")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    od = Path(out_dir) if out_dir else ROOT / "runs" / "eval"; od.mkdir(parents=True, exist_ok=True)
    name = f"compare_{cams.pop()}_{dirs[0].name}_{stamp}"
    meta = {"created": stamp, "device": device, "n_good_photos": len(good), "n_defect_photos": len(defect), "rows": rows}
    (od / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    L = (["> **SYNTHETIC DATA**: pipeline test, not station results.", ""] if any(x["synthetic"] for x in rows) else [])
    L += [f"# Variant comparison ({len(good)} held-out good + {len(defect)} defect photos, same for every row; device {device})", "",
          "| variant | model | size | object AUROC | object recall @ 0 FA | object FA @ 100 % R | honest accuracy (missed) | photo AUROC | smallest mm | ms | earns its place? |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        L.append(f"| {x['variant']} | {x['model']} | {x['image_size']} | {_f(x['object_auroc'])} | {_f(x['object_recall_zero_fa'])} | {_f(x['object_fa_full_recall'])} | "
                 f"{_f(x['cc_accuracy'])} ({_f(x['cc_missed'])}) | {_f(x['photo_auroc'])} | {x['smallest_mm'] or '-'} | {_f(x['ms'], 1)} | {x['earns_its_place']} |")
    L += ["", "Rule: a variant earns its place over the baseline (first row) if it catches more defect objects at zero false alarms, "
          "or the same with a higher object AUROC (+0.005). Small test sets: check n before believing a difference.", ""]
    (od / f"{name}.md").write_text("\n".join(L))
    print("\n".join(L)); print("->", od / f"{name}.md")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="evaluate anomaly models through the run-time path (see the module docstring)")
    ap.add_argument("--brick", default="", help="models/<cam>/<brick> (+ <brick>_efficientad as the cascade's fast stage)")
    ap.add_argument("--cam", default="front")
    ap.add_argument("--model-dir", action="append", default=[], help="model dir(s); repeat for --compare")
    ap.add_argument("--fast", default="", help="fast-stage model dir for the cascade ('' = <brick>_efficientad if it exists, 'none' = off)")
    ap.add_argument("--device", default="CPU", help="OpenVINO device used for the metrics (CPU, GPU, NPU, AUTO)")
    ap.add_argument("--fast-device", default="", help="device of the fast stage (default: --device)")
    ap.add_argument("--latency-devices", default="", help="comma list, e.g. CPU,GPU,NPU (default: --device)")
    ap.add_argument("--data", default="", help="training_images dir (default: the one in the model card)")
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--truth", default="", help="size labels (default: <data>/synth_truth.jsonl or <data>/defect_sizes.jsonl)")
    ap.add_argument("--write-thresholds", action="store_true", help="write thresholds.json next to the model(s) (what the sorter reads)")
    ap.add_argument("--compare", action="store_true", help="all variants on one common test set -> runs/eval/compare_*.md")
    ap.add_argument("--out", default="", help="--compare output dir (default runs/eval)")
    ap.add_argument("--anomalib-py", default="", help="python of the Anomalib venv (default: env ANOMALIB_PY)")
    a = ap.parse_args(argv)
    if a.anomalib_py: scorer_mod.ANOMALIB_PY = Path(a.anomalib_py)
    lat = [x for x in (a.latency_devices or a.device).split(",") if x]
    dirs = [Path(d) for d in a.model_dir]
    cam_dir = Path(a.models) / a.cam
    if not dirs and a.brick:
        if a.compare:
            dirs = sorted((d for d in cam_dir.glob(f"{a.brick}*") if d.is_dir() and (d.name == a.brick or d.name.startswith(a.brick + "_"))
                           and scorer_mod.Scorer._model_file(d)), key=lambda d: (d.name != a.brick, d.name))
        else:
            dirs = [cam_dir / a.brick]
    if not dirs: raise SystemExit("give --brick or --model-dir")
    if a.compare:
        return compare(dirs, a.data or None, a.device, lat, a.truth or None, a.out or None)
    fast = None
    if a.fast and a.fast != "none": fast = Path(a.fast)
    elif not a.fast and a.brick and (cam_dir / f"{a.brick}_efficientad").is_dir() and dirs[0].name == a.brick:
        fast = cam_dir / f"{a.brick}_efficientad"
    out = []
    for d in dirs:
        r = evaluate(d, a.data or None, a.device, lat, fast, a.fast_device or None, a.truth or None, a.write_thresholds)
        o = r["object"]; t = r["thresholds"]
        print(f"{'[SYNTHETIC] ' if r['synthetic_data'] else ''}{d}: photo AUROC {_f(r['photo']['auroc_all_defect_photos'])}, object AUROC "
              f"{_f(o['auroc'])}, object recall@0FA {_f(o['recall_at_zero_fa'])}, thresholds {t.get('t_low')}/{t.get('t_high')}"
              + (f", cascade settles {r['cascade'].get('share_settled_by_fast')}" if r.get("cascade") else "") + f"  -> {d / 'eval' / 'report.md'}")
        out.append(r)
    return out


if __name__ == "__main__":
    main()
