"""Benchmark OpenVINO IR / ONNX models on every device (CPU, GPU, NPU, AUTO) while sampling power and temperature.

Per (model, device, performance hint, precision):
  compile time, first inference, warm-up, then >= N timed SYNC inferences (one at a time, like the robot loop)
  -> mean/p50/p90/p95/p99/max latency; then an ASYNC phase (AsyncInferQueue with the optimal number of requests)
  -> throughput. Both phases run inside the sensor sampler -> average W (and W above the idle baseline), energy per
  inference (J), peak CPU temperature, rise, headroom to TjMax, GPU/NPU busy % (proof the work ran there).
Inputs: real images when --images DIR is given (anomaly crops: RGB, resized to the model input, 0..1 float NCHW =
what Anomalib's OpenVINOInferencer feeds), else random data of the right shape. Dynamic shapes are reshaped to
static (NPU needs static shapes; the same static shape is used on every device so the numbers compare).
Results: bench/results/bench_<date>.json, rewritten after every run (a crash keeps what was measured), with full
provenance (model sha256, device names, OpenVINO version, kernel, repo revision, arguments).

    python -m bench.bench_model models/front/red/weights/openvino/model.xml --images training_images/front/inhand/good
    python -m bench.bench_model a.xml b_int8.xml --devices CPU,GPU,NPU --hints LATENCY --n 300 --shape 1,3,256,256
    python -m bench.bench_model policy.xml --devices GPU,NPU --precisions default,f32     # f32 vs default (GPU: f16)
"""
import argparse, glob, hashlib, json, os, platform, socket, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import RESULTS, ROOT
from .sensors import Sampler, Sensors, headline, measure_idle, wait_cool

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
HINTS = ("LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT")


# ---------------------------------------------------------------------- model inspection
def file_hash(path):
    """sha256 over the model file(s): .xml + .bin for IR, the file itself otherwise (first 16 hex)."""
    p = Path(path); h = hashlib.sha256(); size = 0
    for f in [p] + ([p.with_suffix(".bin")] if p.suffix == ".xml" and p.with_suffix(".bin").exists() else []):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk); size += len(chunk)
    return h.hexdigest()[:16], round(size / 1e6, 3)


def precision_of(model):
    """-> (label, ir weight precision). INT8 = FakeQuantize/FakeConvert ops (NNCF PTQ); 'INT8-W' = weights only."""
    types, consts = set(), {}
    for op in model.get_ordered_ops():
        t = op.get_type_name(); types.add(t)
        if t == "Constant":
            et = op.get_output_element_type(0).to_string(); consts[et] = consts.get(et, 0) + int(np.prod(list(op.get_output_shape(0))) or 1)
    fl = {k: v for k, v in consts.items() if k in ("f32", "f16", "bf16")}
    ir = max(fl, key=fl.get).upper().replace("F", "FP") if fl else "?"
    if types & {"FakeQuantize", "FakeConvert"}: return "INT8", ir
    if any(k in consts for k in ("i8", "u8", "i4", "u4")) and "Convert" in types and consts.get("i8", 0) + consts.get("u8", 0) > sum(fl.values()):
        return "INT8-W", ir
    return None, ir


def _name(port, i):
    try: return port.any_name
    except Exception: return f"input{i}"


def static_shapes(model, override=None, hw=256):
    """-> ({input index: static shape list}, notes). override: one shape for the first input, or {name: shape}.
    hw: side (int) or (h, w) used for dynamic image dims."""
    shapes, notes = {}, []
    hh, ww = (hw, hw) if isinstance(hw, int) else hw
    for i, port in enumerate(model.inputs):
        name = _name(port, i); ps = port.get_partial_shape()
        o = override.get(name) if isinstance(override, dict) else (override if i == 0 else None)
        if o: shapes[i] = list(o); continue
        if ps.rank.is_dynamic:
            shapes[i] = [1, 3, hh, ww]; notes.append(f"{name}: dynamic rank, assumed {shapes[i]} (pass --shape)"); continue
        dims = [d.get_length() if d.is_static else None for d in ps]
        if None not in dims: shapes[i] = dims; continue
        r = len(dims); out = []
        for j, d in enumerate(dims):
            if d is not None: out.append(d)
            elif j == 0: out.append(1)
            elif r == 4 and j in (2, 3) and dims[1] in (1, 3): out.append(hh if j == 2 else ww)          # NCHW image
            elif r == 4 and j in (1, 2) and dims[3] in (1, 3): out.append(hh if j == 1 else ww)          # NHWC image
            elif r == 4 and j == 1 and dims[1] is None: out.append(3)
            else: out.append(1)
        shapes[i] = out
    return shapes, notes


def describe_ports(ports):
    out = []
    for i, p in enumerate(ports):
        try: nm = p.any_name
        except Exception: nm = f"#{i}"
        out.append({"name": nm, "shape": str(p.get_partial_shape()), "type": p.get_element_type().to_string()})
    return out


# ---------------------------------------------------------------------- inputs
def list_images(folder, limit=64):
    if not folder: return []
    fs = sorted(f for f in glob.glob(os.path.join(str(folder), "**", "*"), recursive=True) if f.lower().endswith(IMG_EXT))
    fs = [f for f in fs if "_heat" not in Path(f).stem]
    if len(fs) > limit: fs = fs[:: max(1, len(fs) // limit)][:limit]          # spread over the folder
    return fs


def image_layout(shape):
    if len(shape) == 4 and shape[1] in (1, 3): return "NCHW"
    if len(shape) == 4 and shape[3] in (1, 3): return "NHWC"
    return None


def load_image(f, shape, dtype, scale="unit", bgr=False):
    import cv2
    lay = image_layout(shape)
    n, c, h, w = (shape if lay == "NCHW" else [shape[0], shape[3], shape[1], shape[2]])
    img = cv2.imread(str(f), cv2.IMREAD_COLOR if c == 3 else cv2.IMREAD_GRAYSCALE)
    if img is None: raise RuntimeError(f"cannot read image {f}")
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if c == 3 and not bgr: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    a = img.reshape(h, w, c).astype(np.float32)
    if np.issubdtype(dtype, np.floating):
        if scale == "unit": a /= 255.0
        elif scale == "imagenet": a = (a / 255.0 - np.array([0.485, 0.456, 0.406][:c], np.float32)) / np.array([0.229, 0.224, 0.225][:c], np.float32)
    a = a.transpose(2, 0, 1) if lay == "NCHW" else a
    return np.ascontiguousarray(np.repeat(a[None], n, 0).astype(dtype))


def make_feeds(model, shapes, images=(), k=16, scale="unit", bgr=False, seed=0):
    """-> (list of k feed dicts {input index: array}, data description). Image-like inputs get the real images."""
    rng = np.random.default_rng(seed)
    feeds = [dict() for _ in range(max(1, min(k, len(images)) if images else k))]
    used = False
    for i, port in enumerate(model.inputs):
        dt = port.get_element_type().to_dtype(); shp = shapes[i]
        if images and image_layout(shp):
            for j, fd in enumerate(feeds): fd[i] = load_image(images[j % len(images)], shp, dt, scale, bgr)
            used = True
        else:
            for fd in feeds:
                fd[i] = (rng.random(shp, dtype=np.float32).astype(dt) if np.issubdtype(dt, np.floating) else np.zeros(shp, dt))
    return feeds, (f"{len(feeds)} real images ({scale} scale)" if used else f"random ({len(feeds)} feeds)")


# ---------------------------------------------------------------------- one run
def lat_stats(lat_s):
    a = np.asarray(lat_s) * 1000
    q = lambda p: round(float(np.percentile(a, p)), 4)
    return {"n": int(a.size), "mean_ms": round(float(a.mean()), 4), "std_ms": round(float(a.std()), 4), "min_ms": round(float(a.min()), 4),
            "p50_ms": q(50), "p90_ms": q(90), "p95_ms": q(95), "p99_ms": q(99), "max_ms": round(float(a.max()), 4),
            "fps": round(1000 / float(a.mean()), 2)}


def _prop(obj, k):
    try:
        from .devices import _s
        return _s(obj.get_property(k))
    except Exception: return None


def _energy(summ, n):
    p = (summ or {}).get("power")
    if not p or not n: return None
    e = {"source": p["source"], "w": p["w"], "j_per_inf": round(p["w"] * summ["duration_s"] / n, 6)}
    if "w_above_idle" in p:
        e.update(w_above_idle=p["w_above_idle"], j_per_inf_above_idle=round(p["w_above_idle"] * summ["duration_s"] / n, 6))
    return e


def bench_one(core, model, device, hint="LATENCY", precision="default", feeds=(), n=200, min_s=5.0, max_s=60.0, warmup=20,
              async_s=5.0, sensors=None, idle=None, interval=0.25):
    """Compile + measure one configuration -> result dict (raises on compile/infer errors)."""
    import openvino as ov
    cfg = {"PERFORMANCE_HINT": hint}
    notes = []
    if precision != "default":
        dev0 = device.split(":")[0].split(".")[0]
        sup = core.get_property(dev0, "SUPPORTED_PROPERTIES") if dev0 not in ("AUTO", "MULTI", "HETERO") else {}
        if "INFERENCE_PRECISION_HINT" in sup: cfg["INFERENCE_PRECISION_HINT"] = precision
        else: raise RuntimeError(f"{device} has no INFERENCE_PRECISION_HINT: precision {precision} not selectable")
    t0 = time.perf_counter(); cm = core.compile_model(model, device, cfg); compile_ms = (time.perf_counter() - t0) * 1000
    req = cm.create_infer_request()
    t0 = time.perf_counter(); req.infer(feeds[0], share_inputs=True, share_outputs=True); first_ms = (time.perf_counter() - t0) * 1000
    exec_first = _prop(cm, "EXECUTION_DEVICES")
    for i in range(warmup): req.infer(feeds[i % len(feeds)], share_inputs=True, share_outputs=True)
    sensors = sensors or Sensors()
    lat = []; smp = Sampler(sensors, interval).start(); t_start = time.perf_counter(); i = 0
    while (i < n or time.perf_counter() - t_start < min_s) and time.perf_counter() - t_start < max_s:
        t0 = time.perf_counter(); req.infer(feeds[i % len(feeds)], share_inputs=True, share_outputs=True); lat.append(time.perf_counter() - t0); i += 1
    smp.stop(); s_sync = smp.summary(idle)
    sync = {**lat_stats(lat), "seconds": round(time.perf_counter() - t_start, 3), "sensors": s_sync}
    res = {"device": device, "hint": hint, "precision_requested": precision, "config": cfg,
           "exec_devices": _prop(cm, "EXECUTION_DEVICES"), "exec_devices_first_infer": exec_first,
           "inference_precision": _prop(cm, "INFERENCE_PRECISION_HINT"), "num_streams": _prop(cm, "NUM_STREAMS"),
           "optimal_requests": _prop(cm, "OPTIMAL_NUMBER_OF_INFER_REQUESTS"),
           "compile_ms": round(compile_ms, 2), "first_infer_ms": round(first_ms, 3), "warmup": warmup, "sync": sync,
           "energy": _energy(s_sync, len(lat)), "thermal": s_sync.get("cpu_temp"), "busy_pct": s_sync.get("busy_pct"), "notes": notes}
    if exec_first != res["exec_devices"]:
        notes.append(f"AUTO started on {exec_first} while the accelerator compiled, then {res['exec_devices']}")
    if async_s > 0:
        q = ov.AsyncInferQueue(cm, 0); smp = Sampler(sensors, interval).start(); t0 = time.perf_counter(); k = 0
        while time.perf_counter() - t0 < async_s:
            q.start_async(feeds[k % len(feeds)]); k += 1
        q.wait_all(); dt = time.perf_counter() - t0; smp.stop(); s_async = smp.summary(idle)
        res["async"] = {"jobs": len(q), "n": k, "seconds": round(dt, 3), "fps": round(k / dt, 2), "sensors": s_async,
                        "energy": _energy(s_async, k)}
    del req, cm
    return res


# ---------------------------------------------------------------------- session / provenance
def _chown(p):
    """Under sudo, give the result files back to the real user."""
    try:
        if os.geteuid() == 0 and os.environ.get("SUDO_UID"): os.chown(p, int(os.environ["SUDO_UID"]), int(os.environ.get("SUDO_GID", os.environ["SUDO_UID"])))
    except Exception: pass


class Session:
    """One results file bench/results/bench_<date>.json, rewritten after every added result."""

    def __init__(self, out_dir=RESULTS, args=None, sensors=None, idle=None, env=None):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True); _chown(self.dir)
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"bench_{self.id}.json"
        if env is None:
            from .devices import report
            try: env = report()
            except Exception as e: env = {"error": str(e)}
        self.data = {"kind": "bench_session", "session": self.id, "date": datetime.now().isoformat(timespec="seconds"),
                     "host": socket.gethostname(), "args": args or {}, "env": env, "idle": idle, "results": [], "cycle_stats": None}
        self.save()

    def add(self, r):
        self.data["results"].append(r); self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(self.data, indent=1, default=str)); tmp.replace(self.path); _chown(self.path)


def parse_shape(s):
    if not s: return None
    if "=" in s:                          # name=1,3,256,256;other=1,6
        return {kv.split("=")[0]: [int(x) for x in kv.split("=")[1].split(",")] for kv in s.split(";") if kv}
    return [int(x) for x in s.split(",")]


def bench_models(models, devices=("CPU", "GPU", "NPU", "AUTO"), hints=("LATENCY", "THROUGHPUT"), precisions=("default",),
                 n=200, min_s=5.0, warmup=20, async_s=5.0, idle_s=10.0, cooldown_s=20.0, interval=0.25, shape=None, hw=256,
                 scale="unit", bgr=False, out_dir=RESULTS, session=None, sensors=None, log=print, args=None):
    """models: list of dicts {path, tag?, role?, images?}. -> Session. Every configuration that fails is recorded with
    its error and the loop goes on (e.g. an op the NPU compiler does not support)."""
    import openvino as ov
    core = ov.Core(); avail = list(core.available_devices)
    sensors = sensors or Sensors()
    if session is None:
        idle = None
        if idle_s > 0:
            log(f"idle baseline: {idle_s:.0f} s, keep the machine quiet ..."); idle = measure_idle(sensors, idle_s, interval)
            log("  idle: " + headline(idle))
        session = Session(out_dir, args, sensors, idle)
    idle = session.data.get("idle")
    want = [d for d in devices if d.split(":")[0] in avail or d.startswith(("AUTO", "MULTI", "HETERO"))]
    skipped = [d for d in devices if d not in want]
    if skipped: log(f"not available here (skipped): {', '.join(skipped)}; available: {', '.join(avail)}")
    session.data.setdefault("skipped_devices", skipped); session.save()
    for m in models:
        path = Path(m["path"]); tag = m.get("tag") or path.parent.name
        try:
            model = core.read_model(str(path))
        except Exception as e:
            log(f"{tag}: cannot read {path}: {e}")
            session.add({"kind": "bench", "model": {"path": str(path), "tag": tag}, "error": f"read_model: {e}"}); continue
        sha, size = file_hash(path); qlabel, ir = precision_of(model)
        dyn = any(p.get_partial_shape().is_dynamic for p in model.inputs)
        imgs = list(m["images"]) if isinstance(m.get("images"), (list, tuple)) else list_images(m.get("images"))
        hw_m = hw
        if dyn and imgs and not (m.get("shape") or shape):          # dynamic H,W (Anomalib resizes inside the graph):
            import cv2                                               # run at the real crop size
            im = cv2.imread(str(imgs[0]))
            if im is not None: hw_m = im.shape[:2]
        shapes, snotes = static_shapes(model, m.get("shape") or shape, hw_m)
        if dyn:
            model.reshape({model.inputs[i]: ov.PartialShape(s) for i, s in shapes.items()})
            snotes.append("reshaped dynamic input(s) to static " + "; ".join(f"{_name(model.inputs[i], i)}={s}" for i, s in shapes.items())
                          + " (NPU needs static shapes; same shape on every device)")
        feeds, data = make_feeds(model, shapes, imgs, scale=scale, bgr=bgr)
        minfo = {"path": str(path), "rel": _rel(path), "tag": tag, "role": m.get("role"), "format": path.suffix.lstrip("."), "sha256_16": sha,
                 "size_mb": size, "quantized": qlabel, "ir_precision": ir, "inputs": describe_ports(model.inputs),
                 "outputs": describe_ports(model.outputs), "static_shapes": {_name(model.inputs[i], i): s for i, s in shapes.items()},
                 "data": data, "images": (str(m["images"]) if not isinstance(m.get("images"), (list, tuple)) else f"{len(imgs)} listed files") if m.get("images") else None}
        log(f"\n== {tag}  ({_rel(path)}, {size} MB, sha {sha}, {qlabel or 'IR ' + ir}, {data})")
        for note in snotes: log("   " + note)
        for dev in want:
            for hint in hints:
                for prec in precisions_for(core, dev, precisions, qlabel):
                    if session.data.get("idle") and cooldown_s > 0:
                        w = wait_cool(sensors, idle, cooldown_s)
                        if w >= 1: log(f"   (cooled down {w:.0f} s)")
                    rec = {"kind": "bench", "date": datetime.now().isoformat(timespec="seconds"), "model": minfo,
                           "openvino": ov.get_version(), "shape_notes": snotes}
                    try:
                        r = bench_one(core, model, dev, hint, prec, feeds, n, min_s, max(min_s * 4, 60), warmup, async_s, sensors, idle, interval)
                        dname = None
                        try: dname = core.get_property((r["exec_devices"] or [dev])[0].split(".")[0], "FULL_DEVICE_NAME")
                        except Exception: pass
                        r["device_name"] = dname
                        if r["inference_precision"] is None and r["exec_devices"]:          # AUTO: ask the device it picked
                            try: r["inference_precision"] = core.get_property(r["exec_devices"][-1].split(".")[0], "INFERENCE_PRECISION_HINT").to_string()
                            except Exception: pass
                        r["precision"] = qlabel or _plabel(r["inference_precision"], dev, r["exec_devices"])
                        rec.update(r)
                        log("   " + line(rec))
                    except Exception as e:
                        rec.update(device=dev, hint=hint, precision_requested=prec, error=f"{type(e).__name__}: {str(e)[:600]}")
                        log(f"   {dev:<5} {hint:<10} {prec:<7} FAILED: {rec['error'][:200]}")
                    session.add(rec)
    return session


def precisions_for(core, dev, precisions, quantized=None):
    """'default' always; a forced INFERENCE_PRECISION_HINT only where the device has the property, it differs from the
    device's default (CPU on Core Ultra is already f32, GPU defaults to f16), and the model is not INT8."""
    out = []
    for p in precisions:
        if p == "default": out.append(p); continue
        if quantized or dev.split(":")[0] in ("AUTO", "MULTI", "HETERO", "NPU"): continue        # NPU: FP16/INT8 only
        try:
            if "INFERENCE_PRECISION_HINT" not in core.get_property(dev, "SUPPORTED_PROPERTIES"): continue
            if core.get_property(dev, "INFERENCE_PRECISION_HINT").to_string() == p: continue
        except Exception: continue
        out.append(p)
    return out or ["default"]


def _plabel(ip, dev, execd):
    ed = " ".join(execd or [dev])
    if "NPU" in ed: return "FP16"                # the NPU computes in FP16 (or INT8 for quantized models)
    return {"f32": "FP32", "f16": "FP16", "bf16": "BF16"}.get(str(ip), str(ip).upper() if ip else "?")


def _rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except ValueError: return str(p).replace(str(Path.home()), "~")


def line(r):
    s = r["sync"]; t = f"{r['device']:<5} {r['hint']:<10} {r['precision']:<6} compile {r['compile_ms']:.0f} ms, first {r['first_infer_ms']:.1f} ms, " \
                      f"p50 {s['p50_ms']:.2f} / p95 {s['p95_ms']:.2f} / max {s['max_ms']:.2f} ms ({s['fps']:.0f}/s sync"
    if r.get("async"): t += f", {r['async']['fps']:.0f}/s async x{r['async']['jobs']}"
    t += f") on {','.join(r.get('exec_devices') or [])}"
    h = headline(s["sensors"])
    e = r.get("energy")
    if e: h += f"; {e.get('j_per_inf_above_idle', e['j_per_inf']) * 1000:.2f} mJ/inf" + (" above idle" if "j_per_inf_above_idle" in e else "")
    return t + " | " + h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("models", nargs="+", help=".xml (OpenVINO IR) or .onnx")
    ap.add_argument("--devices", default="CPU,GPU,NPU,AUTO")
    ap.add_argument("--hints", default="LATENCY,THROUGHPUT")
    ap.add_argument("--precisions", default="default,f32", help="default,f32,f16,bf16 = INFERENCE_PRECISION_HINT; skipped where it is "
                    "the device default or not selectable (NPU)")
    ap.add_argument("--images", help="folder of real input images (e.g. anomaly crops)")
    ap.add_argument("--shape", help="static input shape, e.g. 1,3,256,256 or 'name=1,3,224,224;state=1,6'")
    ap.add_argument("--hw", type=int, default=256, help="H=W used for dynamic image dims without --shape")
    ap.add_argument("--scale", default="unit", choices=["unit", "raw", "imagenet"], help="image values: 0..1 / 0..255 / ImageNet-normalised")
    ap.add_argument("--bgr", action="store_true", help="feed BGR instead of RGB")
    ap.add_argument("--n", type=int, default=200, help="min timed sync inferences")
    ap.add_argument("--min-seconds", type=float, default=5.0, help="min sync time (power needs a few seconds)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--async-seconds", type=float, default=5.0, help="0 = skip the throughput phase")
    ap.add_argument("--idle-seconds", type=float, default=10.0, help="idle baseline before the runs (0 = none)")
    ap.add_argument("--cooldown", type=float, default=20.0, help="max s to wait for the CPU to cool to idle+3 C between runs")
    ap.add_argument("--interval", type=float, default=0.25, help="sensor sampling period (s)")
    ap.add_argument("--tag", help="label for the (single) model in the report")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    ms = [{"path": p, "tag": a.tag if a.tag and len(a.models) == 1 else None, "images": a.images} for p in a.models]
    s = bench_models(ms, a.devices.split(","), a.hints.split(","), a.precisions.split(","), a.n, a.min_seconds, a.warmup, a.async_seconds,
                     a.idle_seconds, a.cooldown, a.interval, parse_shape(a.shape), a.hw, a.scale, a.bgr, a.out, args=vars(a))
    print("\nresults:", s.path, "\nreport :  python -m bench.report")


if __name__ == "__main__":
    main()
