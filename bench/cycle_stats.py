"""Statistics of the sorter's real cycles, from the logs sorter.py writes: runs/<session>/log.jsonl (one JSON object
per cycle) + runs/<session>/settings.json.

Per cycle the sorter logs: verdict (good/defect/aborted), reason, photos, regrasps, seen_pct, t_pick_s / t_inspect_s /
t_sort_s / t_cycle_s / t_auto_s, infer_ms_mean/max/total and steps[].shots[] with photo_ms and stages[] = {stage
("fast"/"main"), device, score, infer_ms, load_ms, decision}; shot.decided_by names the stage that settled the photo.
Unknown keys are welcome (e.g. the policy runner's per-step latencies): every numeric field whose name ends in _ms or
_s anywhere in the record is aggregated under its path (lists flattened as []), so new timings show up without code
changes here; keys with "policy" in the path or starting with act_/act. (sorter.py's ACT pick: act.infer_ms_p50,
act_infer_ms_p95, act.attempts[]...) are also listed separately as policy timings.

Accuracy needs ground truth: a "truth"/"expected"/"label" field in the cycle record, or --truth file.csv with
columns session,cycle,truth (good/defect).

    python -m bench.cycle_stats                          # all sessions under ROOT/runs
    python -m bench.cycle_stats runs/sort_20261115_101500 --truth truth.csv --json out.json
"""
import argparse, csv, json, re
from pathlib import Path

import numpy as np

from . import ROOT

TRUTH_KEYS = ("truth", "expected", "label", "gt", "ground_truth")
PHASES = ("t_pick_s", "t_inspect_s", "t_sort_s", "t_auto_s", "t_cycle_s")


def stats(v):
    a = np.asarray([x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)], float)
    if not a.size: return None
    return {"n": int(a.size), "mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "sum": round(float(a.sum()), 3)}


def find_logs(paths=None):
    """Session dirs / log files / a runs dir -> sorted list of log.jsonl paths."""
    paths = [Path(p) for p in (paths or [ROOT / "runs"])]
    out = []
    for p in paths:
        if p.is_file(): out.append(p)
        elif (p / "log.jsonl").exists(): out.append(p / "log.jsonl")
        elif p.is_dir(): out += sorted(p.glob("*/log.jsonl"))
    return sorted(set(out))


def read_cycles(log):
    """-> (cycles, n_bad_lines). Tolerant: skips broken lines (a cycle being written while we read)."""
    cyc, bad = [], 0
    for line in Path(log).read_text(errors="replace").splitlines():
        if not line.strip(): continue
        try:
            j = json.loads(line)
            if isinstance(j, dict): j.setdefault("_session", Path(log).parent.name); cyc.append(j)
            else: bad += 1
        except json.JSONDecodeError: bad += 1
    return cyc, bad


def read_truth(path):
    if not path: return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try: out[(r["session"].strip(), int(r["cycle"]))] = r["truth"].strip().lower()
            except (KeyError, ValueError): pass
    return out


def walk_timings(obj, path="", out=None):
    """Every numeric *_ms / *_s value, keyed by its path with list indices dropped."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"_ms(_|$)|_s$", str(k)): out.setdefault(p, []).append(v)
            elif isinstance(v, (dict, list)): walk_timings(v, p, out)
    elif isinstance(obj, list):
        for v in obj: walk_timings(v, path + "[]", out)
    return out


def shots_of(c):
    for st in c.get("steps") or []:
        if not isinstance(st, dict): continue
        for sh in st.get("shots") or []:
            if isinstance(sh, dict): yield st, sh


def analyse(cycles, truth=None):
    truth = truth or {}
    verdicts = {}
    for c in cycles: verdicts[c.get("verdict", "?")] = verdicts.get(c.get("verdict", "?"), 0) + 1
    n = len(cycles); done = [c for c in cycles if c.get("verdict") in ("good", "defect")]
    out = {"n_cycles": n, "sessions": sorted({c["_session"] for c in cycles}), "verdicts": verdicts,
           "success_rate": round(len(done) / n, 3) if n else None,
           "phases_s": {k: stats([c.get(k) for c in cycles]) for k in PHASES if any(k in c for c in cycles)},
           "phases_s_completed": {k: stats([c.get(k) for c in done]) for k in PHASES if any(k in c for c in done)},
           "photos_per_cycle": stats([c.get("photos") for c in cycles]), "regrasps": stats([c.get("regrasps") for c in cycles]),
           "seen_pct": stats([c.get("seen_pct") for c in cycles])}
    for k in list(out["phases_s"]):
        if out["phases_s"][k] is None: del out["phases_s"][k]
    # ---- ground truth -> accuracy / confusion (defect = positive)
    conf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "aborted": 0}; labelled = 0
    for c in cycles:
        t = next((str(c[k]).lower() for k in TRUTH_KEYS if c.get(k) is not None), None) or truth.get((c["_session"], c.get("cycle")))
        if t not in ("good", "defect"): continue
        labelled += 1; v = c.get("verdict")
        if v not in ("good", "defect"): conf["aborted"] += 1; continue
        conf[("t" if v == t else "f") + ("p" if v == "defect" else "n")] += 1
    if labelled:
        dec = labelled - conf["aborted"]
        out["accuracy"] = {"labelled": labelled, **conf, "accuracy": round((conf["tp"] + conf["tn"]) / dec, 4) if dec else None,
                           "defect_recall": round(conf["tp"] / (conf["tp"] + conf["fn"]), 4) if conf["tp"] + conf["fn"] else None,
                           "false_alarm_rate": round(conf["fp"] / (conf["fp"] + conf["tn"]), 4) if conf["fp"] + conf["tn"] else None}
    # ---- inference per stage and device, cascade
    per, load, photo_ms = {}, {}, []
    shots = 0; fast_settled = 0; escalated = 0; casc_shots = 0; fast_ms, main_ms = [], []
    manual = 0
    for c in cycles:
        for st, sh in shots_of(c):
            shots += 1
            if sh.get("photo_ms") is not None: photo_ms.append(sh["photo_ms"])
            if sh.get("judge") == "manual": manual += 1
            stg = [s for s in sh.get("stages") or [] if isinstance(s, dict)]
            for s in stg:
                key = f"{s.get('stage', '?')}@{s.get('device', '?')}"
                if s.get("infer_ms") is not None: per.setdefault(key, []).append(s["infer_ms"])
                if s.get("load_ms") is not None: load.setdefault(key, []).append(s["load_ms"])
                (fast_ms if s.get("stage") == "fast" else main_ms).append(s.get("infer_ms") or 0)
            names = [s.get("stage") for s in stg]
            if "fast" in names:
                casc_shots += 1
                if sh.get("decided_by") == "fast" or names == ["fast"]: fast_settled += 1
                else: escalated += 1
    out["photos"] = {"n": shots, "manual": manual, "photo_ms": stats(photo_ms)}
    if n:                                     # how often each role's model runs per cycle (weights for the placement plan)
        cnt = {"fast": 0, "main": 0}
        for c in cycles:
            for _, sh in shots_of(c):
                for s in sh.get("stages") or []:
                    if isinstance(s, dict) and s.get("stage") in cnt: cnt[s["stage"]] += 1
        act = [c["act"].get("inferences") for c in cycles if isinstance(c.get("act"), dict) and c["act"].get("inferences")]
        out["calls_per_cycle"] = {"fast": round(cnt["fast"] / n, 2), "main": round(cnt["main"] / n, 2),
                                  "policy": round(sum(act) / len(act), 1) if act else 0}
    out["inference_ms"] = {k: stats(v) for k, v in sorted(per.items())}
    out["load_ms"] = {k: stats(v) for k, v in sorted(load.items())}
    if casc_shots:
        main_mean = float(np.mean([v for k, vs in per.items() if k.startswith("main@") for v in vs])) if any(k.startswith("main@") for k in per) else None
        spent = float(np.sum(fast_ms) + np.sum(main_ms))
        cas = {"photos": casc_shots, "fast_settled": fast_settled, "escalated": escalated,
               "fast_settled_share": round(fast_settled / casc_shots, 4), "infer_ms_per_photo": round(spent / max(1, shots), 3)}
        if main_mean is not None:
            cas["main_only_ms_per_photo_est"] = round(main_mean, 3)
            cas["saving_est"] = round(1 - (spent / max(1, shots)) / main_mean, 4) if main_mean else None
        out["cascade"] = cas
    # ---- everything else that looks like a timing (policy runner etc.)
    known = set(PHASES) | {"infer_ms_mean", "infer_ms_max", "infer_ms_total", "steps[].shots[].photo_ms",
                           "steps[].shots[].stages[].infer_ms", "steps[].shots[].stages[].load_ms"}
    tim = {}
    for c in cycles:
        for k, v in walk_timings({kk: vv for kk, vv in c.items() if not kk.startswith("_")}).items(): tim.setdefault(k, []).extend(v)
    out["other_timings"] = {k: stats(v) for k, v in sorted(tim.items()) if k not in known}
    out["policy_timings"] = {k: v for k, v in out["other_timings"].items() if "policy" in k.lower() or re.match(r"act[._]", k)}
    return out


def load_settings(logs):
    s = {}
    for l in logs:
        try: s[l.parent.name] = json.loads((l.parent / "settings.json").read_text())
        except Exception: pass
    return s


def collect(paths=None, truth_csv=None):
    logs = find_logs(paths); cycles, bad = [], 0
    for l in logs:
        c, b = read_cycles(l); cycles += c; bad += b
    res = analyse(cycles, read_truth(truth_csv))
    res.update(logs=[str(l) for l in logs], bad_lines=bad)
    st = load_settings(logs)
    res["settings"] = {k: {kk: v.get(kk) for kk in ("judge", "device", "cascade", "fast_device", "fast_variant", "brick", "cam", "pick",
                                                                 "act_device", "act_model", "act_mode")}
                       for k, v in st.items()}
    by = {}
    for c in cycles: by.setdefault(c["_session"], []).append(c)
    res["by_session"] = {k: {"n": len(v), "verdicts": analyse(v)["verdicts"], "t_cycle_s": stats([c.get("t_cycle_s") for c in v]),
                             "t_auto_s": stats([c.get("t_auto_s") for c in v]), **{kk: (st.get(k) or {}).get(kk) for kk in ("judge", "device", "cascade", "fast_device", "pick", "act_device")}}
                         for k, v in sorted(by.items())}
    return res


def text(r):
    if not r["n_cycles"]: return f"no sorter cycles found ({', '.join(r['logs']) or 'no runs/*/log.jsonl'})"
    L = [f"{r['n_cycles']} cycles in {len(r['sessions'])} session(s): " + ", ".join(f"{k} {v}" for k, v in r["verdicts"].items())
         + f" · success (not aborted) {100 * r['success_rate']:.0f}%" + (f" · {r['bad_lines']} unreadable line(s)" if r["bad_lines"] else "")]
    for k, s in r["phases_s"].items(): L.append(f"  {k:<12} mean {s['mean']:.2f} s  p50 {s['p50']:.2f}  p95 {s['p95']:.2f}  max {s['max']:.2f}  (n={s['n']})")
    if r.get("accuracy"):
        a = r["accuracy"]; L.append(f"  accuracy {a['accuracy']} on {a['labelled']} labelled (TP {a['tp']} FP {a['fp']} TN {a['tn']} FN {a['fn']}, aborted {a['aborted']})"
                                   f" · defect recall {a['defect_recall']} · false alarms {a['false_alarm_rate']}")
    L.append(f"  photos {r['photos']['n']}" + (f" (manual {r['photos']['manual']})" if r["photos"]["manual"] else "")
             + (f", photo+crop {r['photos']['photo_ms']['mean']:.0f} ms mean" if r["photos"]["photo_ms"] else ""))
    for k, s in r["inference_ms"].items():
        if s: L.append(f"  infer {k:<14} mean {s['mean']:.2f} ms  p50 {s['p50']:.2f}  p95 {s['p95']:.2f}  max {s['max']:.2f}  (n={s['n']})")
    for k, s in r["load_ms"].items():
        if s: L.append(f"  first load {k:<9} {s['mean']:.0f} ms (n={s['n']})")
    if r.get("cascade"):
        c = r["cascade"]; L.append(f"  cascade: fast model settled {c['fast_settled']}/{c['photos']} photos ({100 * c['fast_settled_share']:.0f}%), escalated {c['escalated']}"
                                   + (f"; {c['infer_ms_per_photo']:.1f} ms/photo vs ~{c['main_only_ms_per_photo_est']:.1f} main-only (saving ~{100 * (c['saving_est'] or 0):.0f}%)" if c.get("main_only_ms_per_photo_est") else ""))
    for k, s in r["other_timings"].items():
        if s: L.append(f"  {k:<40} mean {s['mean']:.3f}  p95 {s['p95']:.3f}  max {s['max']:.3f}  (n={s['n']})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", nargs="*", help="runs dir, session dirs or log.jsonl files (default ROOT/runs)")
    ap.add_argument("--truth", help="CSV session,cycle,truth")
    ap.add_argument("--json", help="write the statistics here")
    a = ap.parse_args()
    r = collect(a.paths or None, a.truth)
    print(text(r))
    if a.json: Path(a.json).write_text(json.dumps(r, indent=1)); print("wrote", a.json)


if __name__ == "__main__":
    main()
