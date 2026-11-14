"""bench/results/bench_*.json + the sorter's cycle logs -> the evidence for the judges:
  (a) markdown rows for SO101_SETUP_STATUS.md section 9   (bench/results/section9_rows.md, also printed)
      | Model | Device | Precision | Latency per step | Cycle time | Power | Temperature | Notes |
  (b) bench/results/REPORT.md: environment, idle baseline, every configuration, failures, cycle statistics, the
      device-placement recommendation derived from the numbers, methodology and provenance
  (c) PNG charts if matplotlib is importable (latency per device, energy per inference, temperature/power over time)

    python -m bench.report                          # all result files (latest run per model/device/hint/precision)
    python -m bench.report --session latest --runs runs/sort_20261115_101500
"""
import argparse, itertools, json
from pathlib import Path

from . import RESULTS, ROOT
from .cycle_stats import collect as collect_cycles, text as cycle_text

DEV_ORDER = ("CPU", "GPU", "NPU", "AUTO")


def load_sessions(res_dir=RESULTS, session="all"):
    files = sorted(Path(res_dir).glob("bench_*.json"))
    if session == "latest": files = files[-1:]
    elif session != "all": files = [f for f in files if session in f.name]
    out = []
    for f in files:
        try: d = json.loads(f.read_text()); d["_file"] = str(f); out.append(d)
        except Exception: pass
    return out


def latest_results(sessions):
    """Newest result per (model sha, device, hint, precision); failures kept separately."""
    best, fails = {}, []
    for s in sessions:
        for r in s.get("results", []):
            r = {**r, "_session": s.get("session")}
            if r.get("error"): fails.append(r); continue
            k = (r["model"].get("sha256_16"), r["device"], r["hint"], r.get("precision_requested"))
            if k not in best or (r.get("date") or "") >= (best[k].get("date") or ""): best[k] = r
    ok_keys = {(r["model"].get("sha256_16"), r["device"]) for r in best.values()}
    fails = [f for f in fails if (f["model"].get("sha256_16"), f.get("device")) not in ok_keys]
    return sorted(best.values(), key=_order), fails


def _order(r):
    d = r["device"].split(":")[0]
    return (r["model"].get("tag") or "", DEV_ORDER.index(d) if d in DEV_ORDER else 9, r["hint"], r.get("precision") or "")


def role_of(r):
    m = r["model"]; t = (m.get("role") or "").lower()
    if t: return t
    tag = (m.get("tag") or m.get("path") or "").lower()
    return "policy" if "policy" in tag or "act" in Path(m.get("path", "")).stem.lower() else "fast" if "efficientad" in tag else "main"


def _f(x, fmt="{:.2f}", none="-"):
    return none if x is None else fmt.format(x)


def dev_label(r):
    ex = ",".join(r.get("exec_devices") or [])
    d = r["device"]
    return d if not ex or ex == d else f"{d} -> {ex}"


def power_cell(r):
    e = r.get("energy")
    if not e: return "n/a (no RAPL)"
    if "w_above_idle" in e: return f"+{e['w_above_idle']:.2f} W above idle ({e['w']:.1f} W {short_src(e['source'])}), {1000 * e['j_per_inf_above_idle']:.2f} mJ/inf"
    return f"{e['w']:.2f} W {short_src(e['source'])}, {1000 * e['j_per_inf']:.2f} mJ/inf (no idle baseline)"


def short_src(s): return s.replace("rapl.", "").replace("package-0", "pkg")


def temp_cell(r):
    t = r.get("thermal")
    if not t: return "n/a"
    c = f"{t['peak']:.0f} C peak (+{t['rise']:.1f})"
    if "headroom" in t: c += f", headroom {t['headroom']:.0f} C"
    return c


def lat_cell(r):
    s = r["sync"]; return f"{s['p50_ms']:.2f} ms p50 (p95 {s['p95_ms']:.2f}, max {s['max_ms']:.2f})"


def busy_note(r):
    """GPU/NPU/CPU utilisation during the timed run (above the idle baseline when there is one)."""
    sens = r["sync"].get("sensors") or {}
    above = sens.get("busy_above_idle_pct"); b = above or r.get("busy_pct") or {}
    parts = [f"{k} {'+' if above else ''}{v:.0f}%" for k, v in b.items() if (k.startswith(("npu", "gpu")) and abs(v) >= 1) or k == "cpu"]
    return (("busy above idle " if above else "busy ") + ", ".join(parts)) if parts else ""


def cycle_cell(r, cyc):
    """Mean cycle time of the real sorter sessions that ran this model's role on this device."""
    if not cyc or not cyc.get("n_cycles"): return "-"
    role = role_of(r); key = {"main": "device", "fast": "fast_device", "policy": "act_device"}.get(role, "device")
    t = []
    for name, s in cyc.get("by_session", {}).items():
        st = cyc.get("settings", {}).get(name) or {}
        if not s.get("t_cycle_s") or st.get(key) != r["device"]: continue
        if (role == "fast" and not st.get("cascade")) or (role == "policy" and st.get("pick") != "act"): continue
        if role == "main" and st.get("judge") != "model": continue
        tag = r["model"].get("tag") or ""                    # "<cam>/<brick>[_efficientad][-int8]": the sorter loads model.xml
        if role != "policy" and "/" in tag:
            cam, name = tag.split("/", 1)
            if "-" in name or (st.get("cam") and st["cam"] != cam): continue
            if "brick" in st and name.replace("_efficientad", "") not in ((st.get("brick") or "all"), "all"): continue
        t.append(s["t_cycle_s"]["mean"])
    return f"{sum(t) / len(t):.1f} s" if t else "-"


def section9_rows(results, cyc=None, hints=("LATENCY",)):
    rows = []
    for r in results:
        if r["hint"] not in hints: continue
        notes = [f"{r['hint']}", f"{r['sync']['fps']:.0f}/s sync"]
        if r.get("async"): notes.append(f"{r['async']['fps']:.0f}/s async x{r['async']['jobs']}")
        notes += [f"compile {r['compile_ms']:.0f} ms", f"first {r['first_infer_ms']:.1f} ms"]
        if busy_note(r): notes.append(busy_note(r))
        notes.append(f"sha {r['model'].get('sha256_16')}, OV {str(r.get('openvino', '')).split('-')[0]}, {r.get('date', '')[:16]}")
        rows.append(f"| {r['model'].get('tag')} | {dev_label(r)} | {r.get('precision')} | {lat_cell(r)} | {cycle_cell(r, cyc)} | "
                    f"{power_cell(r)} | {temp_cell(r)} | {'; '.join(notes)} |")
    if cyc and cyc.get("n_cycles"):
        ph = cyc["phases_s"]; inf = cyc.get("inference_ms") or {}
        lat = "; ".join(f"{k} {v['p50']:.1f} ms p50" for k, v in inf.items() if v) or "-"
        cas = cyc.get("cascade")
        devs = sorted({f"{k}: {v.get('device')}" + (f", fast {v.get('fast_device')}" if v.get("cascade") else "") for k, v in cyc.get("settings", {}).items()})
        n = [f"{cyc['n_cycles']} cycles, {', '.join(f'{k} {v}' for k, v in cyc['verdicts'].items())}"]
        if cyc.get("accuracy"): n.append(f"accuracy {cyc['accuracy']['accuracy']}, defect recall {cyc['accuracy']['defect_recall']}")
        if cas: n.append(f"cascade: fast model settled {100 * cas['fast_settled_share']:.0f}% of photos")
        if "t_pick_s" in ph: n.append(f"pick {ph['t_pick_s']['mean']:.1f} s, inspect {ph.get('t_inspect_s', {}).get('mean', 0):.1f} s, sort {ph.get('t_sort_s', {}).get('mean', 0):.1f} s")
        act = (cyc.get("other_timings") or {}).get("act_infer_ms_p50")
        if act: n.append(f"ACT step {act['mean']:.1f} ms p50")
        ct = ph.get("t_cycle_s")
        rows.append(f"| Sorting cycle (real, runs/) | {'; '.join(devs) or '-'} | - | {lat} | "
                    f"{_f(ct and ct['mean'], '{:.1f} s')} mean (p95 {_f(ct and ct['p95'], '{:.1f} s')}) | - | - | {'; '.join(n)} |")
    return rows


HEADER = "| Model | Device | Precision | Latency per step | Cycle time | Power | Temperature | Notes |\n|---|---|---|---|---|---|---|---|"


# ---------------------------------------------------------------------- recommendation
def _speed(k): return f"{k:.2f}x faster" if k >= 1 else f"{1 / k:.2f}x SLOWER"


def recommend(results, cyc=None):
    """From the LATENCY runs: per model the fastest and the most energy-efficient device, then one plan that puts the
    roles (policy / fast anomaly / main anomaly) on distinct devices with the least total slowdown."""
    lat = [r for r in results if r["hint"] == "LATENCY" and r["device"] in ("CPU", "GPU", "NPU") and r.get("precision_requested", "default") == "default"]
    if not lat: return ["No LATENCY results on CPU/GPU/NPU yet: nothing to recommend."]
    L = []
    by_model = {}
    for r in lat: by_model.setdefault(r["model"]["tag"], []).append(r)
    for tag, rs in by_model.items():
        fast = min(rs, key=lambda r: r["sync"]["p50_ms"])
        en = [r for r in rs if (r.get("energy") or {}).get("j_per_inf_above_idle") is not None]
        eff = min(en, key=lambda r: r["energy"]["j_per_inf_above_idle"]) if en else None
        cmp = ", ".join(f"{r['device']} {r['sync']['p50_ms']:.2f} ms" + (f" / {1000 * r['energy']['j_per_inf_above_idle']:.2f} mJ" if r in en else "")
                        for r in sorted(rs, key=lambda r: r["sync"]["p50_ms"]))
        s = f"- **{tag}** ({role_of(fast)}): fastest on **{fast['device']}**"
        if eff and eff is not fast: s += f", most energy-efficient on **{eff['device']}**"
        elif eff: s += " (also the most energy-efficient)"
        L.append(s + f". p50 / energy above idle per inference: {cmp}.")
    # ---- roles -> distinct devices, minimising the inference time per sorting cycle
    roles = {}
    for r in lat: roles.setdefault(role_of(r), {}).setdefault(r["device"], []).append(r["sync"]["p50_ms"])
    devs = sorted({d for v in roles.values() for d in v}, key=lambda d: DEV_ORDER.index(d))
    cost = {ro: {d: sum(v) / len(v) for d, v in dv.items()} for ro, dv in roles.items()}
    names = sorted(cost)
    calls = {n: ((cyc or {}).get("calls_per_cycle") or {}).get(n) or 0 for n in names}
    src = f"calls per cycle from {cyc['n_cycles']} real cycles" if cyc and cyc.get("n_cycles") and any(calls.values()) else "one call per role (no cycle logs yet)"
    if not any(calls.values()): calls = {n: 1 for n in names}
    if len(names) <= len(devs) and len(names) > 1:
        best = None
        for perm in itertools.permutations(devs, len(names)):
            if any(perm[i] not in cost[n] for i, n in enumerate(names)): continue
            c = sum(calls[n] * cost[n][perm[i]] for i, n in enumerate(names))
            if best is None or c < best[0]: best = (c, perm)
        if best:
            own = sum(calls[n] * min(cost[n].values()) for n in names)
            plan = ", ".join(f"{n} -> **{best[1][i]}** ({cost[n][best[1][i]]:.2f} ms x {calls[n]:g})" for i, n in enumerate(names))
            L.append(f"- **Plan with one device per role** (weights: {src}): {plan} = {best[0]:.1f} ms of inference per cycle "
                     f"(every role on its own fastest device, sharing allowed: {own:.1f} ms). Separate devices let the stages overlap "
                     "without contending, keep every compiled model resident, and leave the CPU free for the cameras, cropping and "
                     "the arm loop; the phases of one cycle run one after another, so sharing a device also works.")
    # ---- INT8 vs its FP model, with the accuracy guard written by quantize.py
    for tag, rs in by_model.items():
        if not tag.endswith("-int8") or tag[:-5] not in by_model: continue
        base = {r["device"]: r for r in by_model[tag[:-5]]}
        cmp = [f"{r['device']} " + _speed(base[r["device"]]["sync"]["p50_ms"] / r["sync"]["p50_ms"])
               + (f", {base[r['device']]['energy']['j_per_inf_above_idle'] / r['energy']['j_per_inf_above_idle']:.2f}x less energy"
                  if (r.get("energy") or {}).get("j_per_inf_above_idle") and (base[r["device"]].get("energy") or {}).get("j_per_inf_above_idle") else "")
               for r in rs if r["device"] in base]
        g = None
        try: g = json.loads(Path(rs[0]["model"]["path"]).with_suffix(".quant.json").read_text()).get("guard")
        except Exception: pass
        acc = (f" Accuracy guard on {g['n']} images: max |score change| {g['max_abs']:.4g}, spearman {g['spearman']}"
               + (f", {g['decision_flips']} decision flip(s) at the calibrated thresholds" if "decision_flips" in g else "") + ".") if g else " (no accuracy guard file)"
        if cmp: L.append(f"- **INT8** {tag}: " + "; ".join(cmp) + "." + acc)
    L.append("- Check that the NPU/GPU rows show busy % > 0 in the notes: that is the evidence the work ran there "
             "(AUTO rows also list where AUTO actually executed).")
    return L


# ---------------------------------------------------------------------- charts
def charts(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return [], f"charts skipped: matplotlib not importable ({type(e).__name__})"
    lat = [r for r in results if r["hint"] == "LATENCY"]
    if not lat: return [], "charts skipped: no LATENCY results"
    files = []
    tags = sorted({r["model"]["tag"] for r in lat}); devs = [d for d in DEV_ORDER if any(r["device"] == d for r in lat)]
    devs += sorted({r["device"] for r in lat} - set(devs))
    cols = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#72B7B2"]

    def bars(metric, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(tags) + 2), 4.2)); w = 0.8 / max(1, len(devs)); any_ = False
        for j, d in enumerate(devs):
            ys, es = [], []
            for t in tags:
                r = next((r for r in lat if r["model"]["tag"] == t and r["device"] == d and r.get("precision_requested", "default") == "default"), None)
                v = metric(r) if r else None
                ys.append(v[0] if v else 0); es.append(v[1] if v else 0); any_ |= bool(v)
            ax.bar([i + j * w for i in range(len(tags))], ys, w, yerr=es if any(es) else None, label=d, color=cols[j % len(cols)], capsize=3)
        if not any_: plt.close(fig); return
        ax.set_xticks([i + w * (len(devs) - 1) / 2 for i in range(len(tags))]); ax.set_xticklabels(tags, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); p = Path(out_dir) / fname; fig.savefig(p, dpi=120); plt.close(fig); files.append(p.name)

    bars(lambda r: (r["sync"]["p50_ms"], r["sync"]["p95_ms"] - r["sync"]["p50_ms"]), "latency per inference (ms)", "latency_per_device.png",
         "Sync latency per device (bar p50, whisker to p95), LATENCY hint")
    bars(lambda r: (1000 * r["energy"]["j_per_inf_above_idle"], 0) if (r.get("energy") or {}).get("j_per_inf_above_idle") is not None else None,
         "energy per inference above idle (mJ)", "energy_per_inference.png", "Energy per inference above idle (RAPL package)")
    for key, ylabel, fname in (("cpu_temp_c", "CPU temperature (C)", "temperature_over_time.png"), ("power_w", "package power (W)", "power_over_time.png")):
        series = [(r, r["sync"]["sensors"].get("series", {}).get(key)) for r in lat]
        series = [(r, s) for r, s in series if s]
        if not series: continue
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for i, (r, s) in enumerate(series):
            ax.plot([p[0] for p in s], [p[1] for p in s], label=f"{r['model']['tag']} @ {r['device']}", lw=1.2, color=cols[i % len(cols)] if len(series) <= 6 else None)
        ax.set_xlabel("seconds into the timed run"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_title(ylabel + " during each timed sync run", fontsize=10)
        fig.tight_layout(); p = Path(out_dir) / fname; fig.savefig(p, dpi=120); plt.close(fig); files.append(p.name)
    return files, None


# ---------------------------------------------------------------------- REPORT.md
def env_lines(sessions):
    s = sessions[-1]; e = s.get("env") or {}; ov = e.get("openvino") or {}
    L = [f"- Host `{e.get('host')}`, {e.get('os')}, kernel {e.get('kernel')}, CPU {e.get('cpu_model')}, RAM {e.get('mem_gb')} GB, repo {e.get('repo_rev')}",
         f"- OpenVINO {ov.get('version')}; devices: " + ", ".join(f"{d} = {p.get('FULL_DEVICE_NAME')}" for d, p in (ov.get("devices") or {}).items())]
    n = e.get("npu") or {}
    if n and not n.get("ok"): L.append("- **NPU not usable** in this session: " + "; ".join(f"{x['step']}: {x['detail']}" for x in n.get("steps", []) if not x["ok"]))
    sk = sorted({d for s in sessions for d in s.get("skipped_devices", [])})
    if sk: L.append(f"- Requested but not available: {', '.join(sk)}")
    sen = e.get("sensors") or {}
    if sen:
        L.append("- Sensors read: " + (", ".join(sorted({k.split('.')[0] for k in sen.get('sources', {})})) or "none"))
        for k, why in (sen.get("unavailable") or {}).items(): L.append(f"  - {k}: {why}")
        for f in sen.get("fixes") or []: L.append(f"  - fix: `{f}`")
    idle = s.get("idle")
    if idle:
        from .sensors import headline
        L.append(f"- Idle baseline ({idle.get('duration_s', 0):.0f} s): {headline(idle)}")
    return L


def full_table(results):
    h = ("| Model | Device (executed) | Hint | Precision | Compile ms | First ms | p50 | p95 | p99 | max | sync /s | async /s | W | W above idle | "
         "mJ/inf above idle | CPU peak C | rise | headroom | busy |\n|" + "---|" * 19)
    rows = []
    for r in results:
        s = r["sync"]; e = r.get("energy") or {}; t = r.get("thermal") or {}; a = r.get("async") or {}
        rows.append(f"| {r['model']['tag']} | {dev_label(r)} | {r['hint']} | {r.get('precision')}{'' if r.get('precision_requested', 'default') == 'default' else ' (forced)'} | "
                    f"{r['compile_ms']:.0f} | {r['first_infer_ms']:.1f} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['p99_ms']:.2f} | {s['max_ms']:.2f} | "
                    f"{s['fps']:.0f} | {_f(a.get('fps'), '{:.0f}')} | {_f(e.get('w'))} | {_f(e.get('w_above_idle'))} | "
                    f"{_f(e.get('j_per_inf_above_idle') and 1000 * e['j_per_inf_above_idle'])} | {_f(t.get('peak'), '{:.1f}')} | {_f(t.get('rise'), '{:.1f}')} | "
                    f"{_f(t.get('headroom'), '{:.0f}')} | {busy_note(r).replace('busy ', '')} |")
    return [h] + rows


def build(res_dir=RESULTS, session="all", runs=None, truth=None, out=None, log=print):
    out = Path(out or res_dir); out.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(res_dir, session)
    results, fails = latest_results(sessions)
    try: cyc = collect_cycles(runs or None, truth)
    except Exception as e: cyc = {"n_cycles": 0, "logs": [], "error": str(e)}
    rows = section9_rows(results, cyc)
    (out / "section9_rows.md").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    files, chart_note = charts(results, out)
    M = ["# Phase 7 measurements: latency, power, temperature on the Core Ultra", "",
         "Generated by `python -m bench.report` from " + (", ".join(f"`{Path(s['_file']).name}`" for s in sessions) or "no result files") + ".", ""]
    if sessions: M += ["## Environment", ""] + env_lines(sessions) + [""]
    M += ["## Rows for SO101_SETUP_STATUS.md section 9", "", HEADER] + (rows or ["| (no results yet) | | | | | | | |"]) + [""]
    M += ["## Device placement recommendation", ""] + recommend(results, cyc) + [""]
    M += ["## Every configuration", ""] + (full_table(results) if results else ["(none)"]) + [""]
    if fails:
        M += ["## Configurations that failed", ""] + [f"- {f['model'].get('tag')} on {f.get('device')} {f.get('hint', '')}: `{f['error'][:300]}`" for f in fails] + [""]
    M += ["## Real sorting cycles (runs/*/log.jsonl)", "", "```", cycle_text(cyc) if cyc.get("logs") is not None else str(cyc), "```", ""]
    if files: M += ["## Charts", ""] + [f"![{f}]({f})" for f in files] + [""]
    elif chart_note: M += [f"_{chart_note}_", ""]
    M += ["## Method", "",
          "- Latency = wall time of one synchronous `InferRequest.infer()` from Python (input upload and output included), after a warm-up; "
          "at least N runs and at least a few seconds per configuration so the power average is meaningful. Async = `AsyncInferQueue` with the "
          "plugin's optimal number of requests.",
          "- Power = RAPL energy counter difference over the timed window (package-0; psys/battery listed when present), wrap-around handled. "
          "'Above idle' subtracts the idle baseline measured at the start of the session, so it is the cost of the inference itself.",
          "- Temperature = peak of the CPU package sensor during the run, rise from the start of the run, headroom to TjMax (coretemp temp*_crit) "
          "or the critical trip point. Between runs the bench waits until the package is back within 3 C of idle.",
          "- Dynamic inputs are reshaped to one static shape (the NPU requires static shapes) and the same shape is used on every device.",
          "- GPU/NPU busy % come from the xe gtidle residency and intel_vpu npu_busy_time_us counters.", "",
          "## Provenance", ""]
    seen = set()
    for r in results + fails:
        m = r["model"]
        if m.get("sha256_16") in seen: continue
        seen.add(m.get("sha256_16"))
        M.append(f"- `{m.get('tag')}`: `{m.get('rel') or m.get('path')}` sha256 {m.get('sha256_16')}, {m.get('size_mb')} MB, "
                 f"{m.get('quantized') or 'IR ' + str(m.get('ir_precision'))}, inputs {', '.join(i['name'] + ' ' + i['shape'] for i in m.get('inputs', []))}, data: {m.get('data')}")
    (out / "REPORT.md").write_text("\n".join(M) + "\n")
    from .bench_model import _chown
    for f in ["REPORT.md", "section9_rows.md"] + files: _chown(out / f)            # run under sudo: files stay the user's
    log(HEADER); [log(r) for r in rows]
    log(f"\nwrote {out / 'REPORT.md'}, {out / 'section9_rows.md'}" + (f", charts: {', '.join(files)}" if files else f" ({chart_note})"))
    return {"rows": rows, "report": str(out / "REPORT.md"), "charts": files, "n_results": len(results), "n_failed": len(fails), "cycles": cyc.get("n_cycles", 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--session", default="all", help="all | latest | part of a session id")
    ap.add_argument("--runs", nargs="*", help="sorter runs dir / session dirs (default ROOT/runs)")
    ap.add_argument("--truth", help="CSV session,cycle,truth for the accuracy")
    ap.add_argument("--out", help="output dir (default = --results)")
    a = ap.parse_args()
    build(a.results, a.session, a.runs, a.truth, a.out)


if __name__ == "__main__":
    main()
