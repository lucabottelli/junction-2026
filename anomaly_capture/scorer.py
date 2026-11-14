"""Anomaly scores for the sorter (runs in the capture tool's process = Studio's venv).

Models live in ~/junction26/models/<cam>/<brick>/... (any depth): the first `model.xml` (OpenVINO export from
Anomalib) or `model.pt` found. `<cam>/all/` is the fallback model for every brick. <cam> = front for the in-hand
photos, wrist for the top photo of the released brick. `<brick>_efficientad/` = optional fast first-stage model.
Optional `thresholds.json` next to the model folder root: {"t_low": .., "t_high": ..} overrides the global ones
(written by the calibration on the page).

The model itself runs in score_worker.py inside the Anomalib venv (separate process, so the two torch installs
never meet); one worker is kept alive and caches loaded models, so only the first score pays the load time.
"""
import json, os, queue, subprocess, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANOMALIB_PY = Path(os.environ.get("ANOMALIB_PY", Path.home() / "anomalib/.venv/bin/python"))


class Scorer:
    def __init__(self, models_dir: Path, log_path: Path):
        self.dir = Path(models_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.proc = None; self.q = queue.Queue(); self.lock = threading.Lock(); self.n = 0

    # ------------------------------------------------------------------ model lookup
    def models(self):
        out = []
        for cam_dir in sorted(p for p in self.dir.glob("*") if p.is_dir()):
            for b in sorted(p for p in cam_dir.glob("*") if p.is_dir()):
                m = self._model_file(b)
                if m: out.append({"cam": cam_dir.name, "brick": b.name, "path": str(m), "thresholds": self.thresholds(b)})
        return out

    @staticmethod
    def _model_file(d: Path):
        for pat in ("model.xml", "model.pt"):
            hits = sorted(d.rglob(pat))
            if hits: return hits[0]
        return None

    def find(self, cam, brick, variant=""):
        """-> (model path, model root dir) for this camera/brick, falling back to <cam>/all.
        variant: "" = main model, "_efficientad" etc. = a second model trained by train_anomaly.py --model ..."""
        for b in ([brick] if brick else []) + ["all"]:
            d = self.dir / cam / (b + variant)
            m = self._model_file(d) if d.is_dir() else None
            if m: return str(m), d
        have = [f"{m['cam']}/{m['brick']}" for m in self.models()]
        raise RuntimeError(f"no anomaly model for {cam}/{(brick or 'all') + variant} in {self.dir} (have: {', '.join(have) or 'none'}) - "
                           "train + export one, or use the MANUAL judge to test the motions")

    @staticmethod
    def thresholds(model_root: Path):
        try: return json.loads((Path(model_root) / "thresholds.json").read_text())
        except Exception: return None

    @staticmethod
    def save_thresholds(model_root: Path, t):
        (Path(model_root) / "thresholds.json").write_text(json.dumps(t, indent=1))

    # ------------------------------------------------------------------ worker process
    def _ensure(self):
        if self.proc is not None and self.proc.poll() is None: return
        if not ANOMALIB_PY.exists(): raise RuntimeError(f"Anomalib python not found at {ANOMALIB_PY} (set ANOMALIB_PY)")
        self.q = queue.Queue()
        self.proc = subprocess.Popen([str(ANOMALIB_PY), "-u", str(HERE / "score_worker.py")], cwd=str(HERE),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(self.log_path, "a"), text=True, bufsize=1)

        def pump(p=self.proc, q=self.q):
            for line in p.stdout:
                try: q.put(json.loads(line))
                except Exception: pass
            q.put({"error": "worker exited (see " + str(self.log_path) + ")"})
        threading.Thread(target=pump, daemon=True).start()
        r = self.q.get(timeout=120)               # anomalib import can take a while the first time
        if not r.get("ready"): raise RuntimeError(f"score worker failed to start: {r.get('error')}")

    def score(self, model, image, device="GPU", heatmap=None, timeout=180):
        """-> {"score", "label", "ms", ["load_ms"]}; raises on error."""
        with self.lock:
            self._ensure()
            self.n += 1
            req = {"id": self.n, "model": str(model), "image": str(image), "device": device}
            if heatmap: req["heatmap"] = str(heatmap)
            self.proc.stdin.write(json.dumps(req) + "\n"); self.proc.stdin.flush()
            t0 = time.time()
            while True:
                r = self.q.get(timeout=max(1, timeout - (time.time() - t0)))
                if r.get("id") == self.n or "id" not in r: break
            if "error" in r: raise RuntimeError(f"anomaly model: {r['error']}")
            return r

    def close(self):
        if self.proc and self.proc.poll() is None:
            try: self.proc.stdin.close(); self.proc.wait(3)
            except Exception: self.proc.kill()
