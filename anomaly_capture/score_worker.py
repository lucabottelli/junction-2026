"""Anomaly scoring worker. Runs in the ANOMALIB venv (~/anomalib/.venv), started by scorer.py.

Why a separate process: Anomalib's torch must not clash with Studio's torch (the capture tool runs in Studio's
venv). Protocol: one JSON request per line on stdin, one JSON reply per line on stdout.

  request : {"id": 1, "model": "/path/model.xml" (or .pt), "image": "/path/crop.png", "device": "GPU",
             "heatmap": "/path/out_heat.jpg" (optional)}
  reply   : {"id": 1, "score": 0.83, "label": true, "ms": 12.4, "load_ms": 900.0 (first use only)}
            or {"id": 1, "error": "..."}

Uses Anomalib's own inferencers, so pre-processing (resize/normalise) and post-processing (score normalisation,
threshold 0.5) are exactly what Anomalib exported with the model.
"""
import json, sys, time, traceback

import cv2
import numpy as np

CACHE = {}
STATIC = {}                 # (model, device) -> (h, w) the input is resized to (NPU: static shapes only)


def npu_model(model):
    """The NPU compiles static shapes only, and the Anomalib export has a dynamic [1,3,?,?] input (its graph resizes
    any crop to the training size itself). -> path of a sibling IR fixed at that size (written once, rewritten when
    the export is newer) and its (h, w). The caller resizes each crop to (h, w), so the graph's own resize is a no-op;
    CPU / GPU keep the dynamic export untouched."""
    import openvino as ov
    from pathlib import Path
    src = Path(model)
    m = ov.Core().read_model(str(src))
    amap = next((o for o in m.outputs if "anomaly_map" in o.get_names()), m.outputs[-1])
    h, w = (int(d) for d in amap.get_partial_shape().to_shape()[-2:])
    dst = src.with_name(f"{src.stem}_static{h}x{w}.xml")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        m.reshape({m.inputs[0].any_name: [1, 3, h, w]})
        ov.save_model(m, str(dst), compress_to_fp16=False)      # weights are already FP16 in the source IR
    return str(dst), (h, w)


def _get(p, k):
    v = getattr(p, k, None)
    if v is None and isinstance(p, dict): v = p.get(k)
    return v


def load(model, device):
    key = (model, device)
    if key in CACHE: return CACHE[key], None
    t0 = time.perf_counter()
    if model.endswith(".xml") or model.endswith(".onnx"):
        from anomalib.deploy import OpenVINOInferencer
        path = model
        if (device or "").upper() == "NPU" and model.endswith(".xml"):
            path, STATIC[key] = npu_model(model)
        inf = OpenVINOInferencer(path=path, device=device or "AUTO")
    else:
        from anomalib.deploy import TorchInferencer
        dev = {"GPU": "xpu", "NPU": "cpu", "CPU": "cpu", "AUTO": "auto"}.get(device or "AUTO", device)
        inf = TorchInferencer(path=model, device=dev)
    CACHE[key] = inf
    return inf, (time.perf_counter() - t0) * 1000


def heatmap(img_path, amap, out):
    img = cv2.imread(img_path)
    a = np.asarray(amap, np.float32).squeeze()
    if a.ndim != 2: return
    a = cv2.resize(a, (img.shape[1], img.shape[0]))
    a = np.clip((a - a.min()) / max(1e-6, a.max() - a.min()), 0, 1)
    col = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(out, cv2.addWeighted(img, 0.55, col, 0.45, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


def handle(req):
    inf, load_ms = load(req["model"], req.get("device"))
    hw = STATIC.get((req["model"], req.get("device")))
    img = req["image"]
    if hw:                                       # static-shape model: resize the crop like the graph would
        img = cv2.cvtColor(cv2.imread(req["image"]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA if img.shape[0] > hw[0] else cv2.INTER_LINEAR)
    t0 = time.perf_counter()
    pred = inf.predict(image=img)
    ms = (time.perf_counter() - t0) * 1000
    score = float(np.asarray(_get(pred, "pred_score")).reshape(-1)[0])
    lab = _get(pred, "pred_label")
    out = {"id": req.get("id"), "score": score, "ms": round(ms, 2),
           "label": None if lab is None else bool(np.asarray(lab).reshape(-1)[0])}
    if load_ms is not None: out["load_ms"] = round(load_ms, 1)
    amap = _get(pred, "anomaly_map")
    if req.get("heatmap") and amap is not None:
        try: heatmap(req["image"], amap, req["heatmap"])
        except Exception: pass
    return out


def main():
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        req = None
        try:
            req = json.loads(line)
            out = handle(req)
        except Exception as e:
            out = {"id": req.get("id") if isinstance(req, dict) else None,
                   "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
