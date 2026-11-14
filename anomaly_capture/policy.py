"""Phase 6, the autonomous pick: run a trained ACT export from Physical AI Studio on the live arm (no human).

Used by sorter.py as pick = "act" (page /sort, step 8). One pick = start pose -> the policy drives the arm at the
dataset rate until the fingers stall on the brick (the collector's hand-off rule) -> the same scripted rise + carry
to WP1 as in collection -> the brick must still be held. Timeout / missed brick -> let go, back to the start pose,
try again (N retries) -> PickFailed (the sorter asks the human for help; the cycle never crashes).

Observation = EXACTLY what export_act.py wrote into the Studio dataset (training/serving skew is the #1 risk, so
the code is shared, not copied) and what Studio's own model worker sends (runtime_session/model.py):
    state         [1, 6] float32   follower ticks A.pos -> export_act.normalize()  (body -100..100, gripper 0..100)
    images.<cam>  [1, 480, 640, 3] uint8 RGB = prep.act_policy_image(full frame, collector act_roi[cam])
                  (= prep.act_jpeg() as recorded -> JPEG decode -> BGR2RGB, as export_act.py reads it)
    task          ["pick up the <brick> brick"]  only if the export takes a language input (ACT doesn't)
Action: the export's chunk (n, 6) in the same normalized units -> export_act.denormalize() -> ticks -> A.goal.

Export layout (read from physical-ai-studio@junction2026 library/src/physicalai/export/mixin_policy.py +
policies/act/policy.py, and physicalai@8381d43, the revision Studio pins):
    ~/.local/share/physicalai/models/<model id>/exports/<openvino|onnx|torch|executorch>/manifest.json + act.xml|...
    manifest.model.input_features: [{class_path ...InferenceFeature, init_args {name: "state" | "images.<cam>",
    ftype: STATE | VISUAL, shape, dtype}}]; preprocessors to_float_tensor + resize (letterbox to ACT image_size);
    postprocessor action_chunk_trimmer when n_action_steps < chunk_size. Dataset / ImageNet normalisation is inside
    the network. Output "action" [1, n, 6].
Loading: physicalai.inference.InferenceModel(export_dir, backend=, device=) - the call Studio's model worker makes
(present in Studio's Python, which runs capture.py). Without physicalai a small direct OpenVINO / ONNX Runtime
path applies the same three manifest components and refuses an export that needs anything else.

Execution (what Studio's "Run model" does by default: SyncExecution(0.5) + ChunkedActionQueue(LerpSmoother(5))):
    chunk     take one action per tick; when <= 50 % of the chunk is left, infer again (in the tick) and blend
              the rest of the old chunk into the new one over 5 frames
    async     same, but inference runs beside the loop while the queue keeps playing (the new chunk is aligned
              to the frame it was computed from)
    ensemble  ACT temporal ensembling: infer every tick, exponentially weighted average (coeff 0.01)
Safety: targets go only through A.goal in "hold" mode, so the arm's 50 Hz loop clamps them to the joint limits
and rate-limits them (MIRROR_VMAX x speed, as when the demos were teleoperated); body targets are also kept
within MAX_LEAD_DEG of where the arm actually is. STOP (/sort/stop, A.stop) is checked every tick and right
before every write, under the arm's lock, so no target is written after STOP. An arm that stays far behind its
target for BLOCK_S (something in the way) ends the pick at once, without retries.
Latency: inference ms per call, observation ms per call, loop rate -> p50 / p95 in the cycle log (efficiency).

Standalone check of an export (no arm, no cameras: random frames):
    python anomaly_capture/policy.py --list
    python anomaly_capture/policy.py --model ~/.local/share/physicalai/models/<id> --device GPU --bench 100
"""
import json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import prep
from arm import BLOCKED, BODY, JOINTS, MIRROR_VMAX, TICK_PER_DEG, Stopped
from collect import FPS, HOLD_GAP, REGRIP_SQUEEZE, TAIL_S, GraspDetector
from export_act import POLICY_CAMS, denormalize, normalize

HERE = Path(__file__).resolve().parent
STUDIO_MODELS = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "physicalai" / "models"
STUDIO_API = "http://127.0.0.1:7860/api"
BACKENDS = ("openvino", "onnx", "torch")     # preference when a model dir has several exports
DEVICES = ("CPU", "GPU", "NPU", "AUTO")
MODES = ("chunk", "async", "ensemble")
REQUEST_THRESHOLD = 0.5    # re-infer when this fraction of the chunk is left (Studio's default)
LERP_FRAMES = 5            # blend old/new chunk over this many frames (Studio's LerpSmoother default)
ENSEMBLE_COEFF = 0.01      # ACT paper / LeRobot value
MAX_LEAD_DEG = 25          # body targets are kept within this of the actual joint position
BLOCK_S = 1.0              # arm > arm.BLOCKED ticks behind its target this long = something in the way
CAM_MAX_AGE_S = 0.5        # older frame = camera stalled (export_act rejects such training episodes too)
SOFT_GAP = HOLD_GAP // 2   # second hand-off rule: fingers stalled with the target only this far past them ...
SOFT_HOLD_S = 0.6          # ... for this long (a policy can squeeze less than the human in the demos). 0 = off
SINGLE_CAM = "top"         # camera for an export with one unnamed "images" input


class PickFailed(RuntimeError):
    """The policy did not get the brick to WP1 (after the retries). info = the attempt log."""
    def __init__(self, msg, info=None, retry=True):
        super().__init__(msg); self.info = info or {}; self.retry = retry


def pct(v, q):
    return round(float(np.percentile(v, q)), 2) if len(v) else None


# ====================================================================== finding exports
def _features(manifest):
    out = []
    for f in (manifest.get("model") or {}).get("input_features") or []:
        a = f.get("init_args") or f
        if "name" in a: out.append({"name": a["name"], "ftype": str(a.get("ftype", "")).upper(), "shape": a.get("shape")})
    return out


def describe(export_dir):
    """Summary of one export directory (None if it is not a Studio/physicalai policy package)."""
    d = Path(export_dir)
    try: m = json.loads((d / "manifest.json").read_text())
    except Exception: return None
    if m.get("format", "policy_package") != "policy_package" or "model" not in m: return None
    arts = m["model"].get("artifacts") or {}
    feats = _features(m)
    vis = [f["name"] for f in feats if f["ftype"] == "VISUAL"]
    outs = [(f.get("init_args") or f) for f in m["model"].get("output_features") or []]
    chunk = next((o.get("shape", [None])[0] for o in outs if o.get("name") == "action"), None)
    for p in m["model"].get("postprocessors") or []:
        if "trimmer" in (p.get("type") or p.get("class_path", "")).lower():
            chunk = (p.get("init_args") or p).get("n_action_steps", chunk)
    exp = d.parent
    return {"path": str(d), "backend": next(iter(arts), d.name), "policy": (m.get("policy") or {}).get("name", ""),
            "id": exp.parent.name if exp.name == "exports" else d.name,
            "cams": [n.split(".", 1)[1] if "." in n else SINGLE_CAM for n in vis] or list(POLICY_CAMS),
            "chunk": chunk, "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime((d / "manifest.json").stat().st_mtime))}


def _studio_name(model_id, cache={}):
    """Best effort: the model's name in Studio (its backend must be running); '' otherwise."""
    if model_id not in cache:
        try: cache[model_id] = json.load(urllib.request.urlopen(f"{STUDIO_API}/models/{model_id}", timeout=0.4)).get("name", "")
        except Exception: cache[model_id] = ""
    return cache[model_id]


def find_exports(roots=None, names=True):
    """Every policy export under Studio's models dir and the given roots (e.g. ROOT/models), newest first."""
    roots = [STUDIO_MODELS] + [Path(r) for r in (roots or [])]
    seen, out = set(), []
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir(): continue
        for pat in ("*/exports/*/manifest.json", "exports/*/manifest.json", "*/*/manifest.json", "*/manifest.json"):
            for mf in root.glob(pat):
                d = mf.parent.resolve()
                if d in seen: continue
                info = describe(d)
                if info:
                    seen.add(d); info["source"] = "studio" if root == STUDIO_MODELS else str(root)
                    info["name"] = _studio_name(info["id"]) if names and info["source"] == "studio" else ""
                    out.append(info)
    return sorted(out, key=lambda e: e["mtime"], reverse=True)


def resolve(path, backend="auto"):
    """A model dir (<id>), its exports/ dir or one exports/<backend>/ dir -> (export dir, backend)."""
    p = Path(path).expanduser()
    if (p / "manifest.json").exists():
        return p, (describe(p) or {}).get("backend", p.name) if backend == "auto" else backend
    base = p / "exports" if (p / "exports").is_dir() else p
    order = BACKENDS if backend == "auto" else (backend,)
    for b in order:
        if (base / b / "manifest.json").exists(): return base / b, b
    have = sorted(x.name for x in base.iterdir()) if base.is_dir() else []
    raise FileNotFoundError(f"no {'/'.join(order)} policy export in {p} (found: {', '.join(have) or 'nothing'})")


# ====================================================================== loading + running one export
def _spec(s):
    name = (s.get("type") or s.get("class_path", "").rsplit(".", 1)[-1]).lower()
    return name, {**{k: v for k, v in s.items() if k not in ("type", "class_path", "init_args")}, **(s.get("init_args") or {})}


def _to_float_chw(img):
    """physicalai ToFloatTensorPreprocessor: uint8 -> float32 / 255, BHWC -> BCHW."""
    x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    return np.transpose(x, (0, 3, 1, 2)) if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3 else x


def _letterbox(img, res, mode="letterbox", pad=0.0):
    """physicalai ResizePreprocessor (letterbox / stretch, bilinear, symmetric padding) -> float32 BCHW."""
    x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    if x.shape[1] == 3 and x.shape[-1] != 3: x = np.transpose(x, (0, 2, 3, 1))
    th, tw = res; h, w = x.shape[1:3]
    if mode == "letterbox":
        r = max(w / tw, h / th); rh, rw = max(1, min(int(h / r), th)), max(1, min(int(w / r), tw))
    else: rh, rw = th, tw
    if (rh, rw) != (h, w): x = np.stack([cv2.resize(i, (rw, rh), interpolation=cv2.INTER_LINEAR) for i in x])
    if mode == "letterbox" and (rh < th or rw < tw):
        pt, pl = (th - rh) // 2, (tw - rw) // 2
        x = np.pad(x, ((0, 0), (pt, th - rh - pt), (pl, tw - rw - pl), (0, 0)), constant_values=pad)
    return np.transpose(x, (0, 3, 1, 2))


class DirectModel:
    """Fallback when physicalai isn't installed: OpenVINO / ONNX Runtime + the manifest components ACT uses.
    Same interface as physicalai's InferenceModel (predict_action_chunk, reset, close)."""
    def __init__(self, d, manifest, backend, device):
        arts = manifest["model"].get("artifacts") or {}
        f = d / (arts.get(backend) or next(iter(arts.values()), ""))
        if not f.is_file():
            f = next(iter(sorted(d.glob("*.xml" if backend == "openvino" else "*.onnx"))), None)
            if f is None: raise FileNotFoundError(f"no {backend} model file in {d}")
        self.pre = [_spec(s) for s in manifest["model"].get("preprocessors") or []]
        self.post = [_spec(s) for s in manifest["model"].get("postprocessors") or []]
        for n, _ in self.pre + self.post:
            if not any(k in n for k in ("to_float", "tofloat", "resize", "trimmer")):
                raise RuntimeError(f"this export needs the '{n}' component: run capture.py with Studio's Python (physicalai installed)")
        if backend == "openvino":
            import openvino as ov
            self.cm = ov.Core().compile_model(str(f), device)
            self.inputs = [i.any_name for i in self.cm.inputs]; self.outputs = [o.any_name for o in self.cm.outputs]
            self._run = lambda feed: [np.asarray(v) for v in self.cm(feed).values()]
        elif backend == "onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
            self.inputs = [i.name for i in self.sess.get_inputs()]; self.outputs = [o.name for o in self.sess.get_outputs()]
            self._run = lambda feed: self.sess.run(None, feed)
        else:
            raise RuntimeError(f"the {backend} export needs physicalai (Studio's Python); choose the openvino export")

    def predict_action_chunk(self, obs):
        x = dict(obs)
        for n, a in self.pre:
            for k in [k for k in x if k.startswith("images") and "is_pad" not in k]:
                x[k] = _to_float_chw(x[k]) if "float" in n else _letterbox(x[k], a["image_resolution"], a.get("mode", "letterbox"), a.get("pad_value", 0))
        missing = [k for k in self.inputs if k not in x]
        if missing: raise KeyError(f"export expects inputs {missing}; observation has {list(x)}")
        out = dict(zip(self.outputs, self._run({k: x[k] for k in self.inputs})))
        act = out.get("action", next(iter(out.values())))
        for n, a in self.post:
            if act.ndim == 3 and act.shape[1] > a["n_action_steps"]: act = act[:, :a["n_action_steps"]]
        return np.atleast_2d(act[0] if act.ndim == 3 else act)

    def reset(self): pass
    def close(self): pass


class Policy:
    """One loaded export. chunk(obs) -> (n, 6) float32 actions in Studio normalized units."""
    def __init__(self, path, device="GPU", backend="auto", runtime="auto"):
        self.export_dir, self.backend = resolve(path, backend)
        self.manifest = json.loads((self.export_dir / "manifest.json").read_text())
        d = describe(self.export_dir) or {}
        self.cams = d.get("cams") or list(POLICY_CAMS)
        feats = _features(self.manifest)
        vis = [f["name"] for f in feats if f["ftype"] == "VISUAL"]
        self.image_keys = vis or [f"images.{c}" for c in self.cams]
        self.takes_task = any(f["ftype"] == "LANGUAGE" or f["name"] == "task" for f in feats)
        dev = device.upper()
        if self.backend == "torch": dev = {"CPU": "cpu", "GPU": "xpu", "AUTO": "auto"}.get(dev) or "cpu"
        elif self.backend == "onnx": dev = "cpu"
        self.device = dev
        t0 = time.perf_counter()
        IM = None
        if runtime in ("auto", "physicalai"):
            try: from physicalai.inference import InferenceModel as IM
            except ImportError:
                if runtime == "physicalai": raise
        if IM is not None:
            self.m, self.runtime = IM(self.export_dir, backend=self.backend, device=dev), "physicalai"
        else:
            self.m, self.runtime = DirectModel(self.export_dir, self.manifest, self.backend, dev), "direct"
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.label = f"{d.get('id', self.export_dir.name)} · {self.backend} · {self.device}"

    def chunk(self, obs):
        return np.asarray(self.m.predict_action_chunk(obs), np.float32).reshape(-1, 6)

    def reset(self): self.m.reset()

    def close(self):
        try: self.m.close()
        except Exception: pass


def load(path, device="GPU", backend="auto"):
    """sorter.py calls this (tests replace it with a fake policy)."""
    return Policy(path, device, backend)


def observation(frames, pos, limits, image_keys, cams, rois, task=None, jpeg=True):
    """The policy input from live data, through the same functions as the training data (see the header)."""
    obs = {"state": normalize(np.array([[pos[n] for n in JOINTS]], np.float64), limits)}
    for key, cam in zip(image_keys, cams):
        f = frames.get(cam)
        if f is None: raise RuntimeError(f"camera {cam} gives no frames: the policy needs {', '.join(cams)}")
        obs[key] = prep.act_policy_image(f, rois[cam], jpeg)[np.newaxis]
    if task: obs["task"] = [task]
    return obs


# ====================================================================== chunk execution
class Executor:
    """Turns chunks into one action per tick (modes: see the header). step(get_obs) -> (6,) normalized action."""
    def __init__(self, policy, mode="chunk", threshold=REQUEST_THRESHOLD, lerp=LERP_FRAMES, coeff=ENSEMBLE_COEFF):
        self.p, self.mode, self.thr, self.lerp, self.coeff = policy, mode if mode in MODES else "chunk", threshold, lerp, coeff
        self.pool = ThreadPoolExecutor(1) if self.mode == "async" else None
        self.reset()

    def reset(self):
        self.q = np.zeros((0, 6), np.float32); self.ens = self.cnt = None; self.pending = None
        self.n = 0; self.chunk_len = 0; self.infer_ms, self.obs_ms = [], []

    def _infer(self, obs):
        t0 = time.perf_counter(); c = self.p.chunk(obs); self.infer_ms.append((time.perf_counter() - t0) * 1000)
        self.chunk_len = max(self.chunk_len, len(c)); return c

    def _obs(self, get_obs):
        t0 = time.perf_counter(); o = get_obs(); self.obs_ms.append((time.perf_counter() - t0) * 1000); return o

    def _merge(self, inc):
        """physicalai LerpSmoother: fade the remaining old actions into the new chunk, then the new tail."""
        r = self.q; k = min(len(r), len(inc))
        if not k: self.q = inc.astype(np.float32); return
        w = np.maximum(1.0 - np.arange(len(r)) / max(min(len(r), self.lerp), 1), 0.0)[:k, None]
        self.q = np.concatenate([w * r[:k] + (1 - w) * inc[:k], inc[k:]]).astype(np.float32)

    def _ensemble(self, c):
        """LeRobot ACTTemporalEnsembler for batch 1."""
        n = len(c); w = np.exp(-self.coeff * np.arange(n)); cw = np.cumsum(w)
        if self.ens is None:
            self.ens, self.cnt = c.astype(np.float64), np.ones(n, int)
        else:
            m = min(len(self.ens), n); i = np.minimum(self.cnt[:m], n - 1)
            e = (self.ens[:m] * cw[i - 1][:, None] + c[:m] * w[i][:, None]) / cw[i][:, None]
            self.ens = np.concatenate([e, c[m:]]); self.cnt = np.r_[np.minimum(self.cnt[:m] + 1, n), np.ones(n - m, int)]
        a = self.ens[0]; self.ens, self.cnt = self.ens[1:], self.cnt[1:]
        return a.astype(np.float32)

    def step(self, get_obs):
        if self.mode == "ensemble":
            self.n += 1; return self._ensemble(self._infer(self._obs(get_obs)))
        low = len(self.q) <= int(self.thr * self.chunk_len)
        if self.mode == "async":
            if self.pending is not None and (self.pending[0].done() or not len(self.q)):
                fut, at = self.pending; self.pending = None
                inc = fut.result()[self.n - at:]                      # drop what was played while it computed
                if len(inc): self._merge(inc)
                low = len(self.q) <= int(self.thr * self.chunk_len)
            if (low or not len(self.q)) and self.pending is None:
                self.pending = (self.pool.submit(self._infer, self._obs(get_obs)), self.n)
                if not len(self.q):                                   # nothing to play yet: wait for it
                    fut, at = self.pending; self.pending = None; self._merge(fut.result())
        elif low:
            self._merge(self._infer(self._obs(get_obs)))
        a = self.q[0]; self.q = self.q[1:]; self.n += 1
        return a

    def stats(self):
        return {"mode": self.mode, "inferences": len(self.infer_ms), "chunk": self.chunk_len,
                "infer_ms_p50": pct(self.infer_ms, 50), "infer_ms_p95": pct(self.infer_ms, 95),
                "infer_ms_max": round(max(self.infer_ms), 2) if self.infer_ms else None,
                "obs_ms_p50": pct(self.obs_ms, 50), "obs_ms_p95": pct(self.obs_ms, 95)}

    def close(self):
        if self.pool: self.pool.shutdown(wait=True, cancel_futures=True)


class RealClock:
    now = staticmethod(time.monotonic)

    @staticmethod
    def sleep_until(t):
        d = t - time.monotonic()
        if d > 0: time.sleep(d)


# ====================================================================== the pick
class ActPick:
    """One autonomous pick with retries. run() -> (clamp command g_lock, info) or raises PickFailed / Stopped.
    A = arm.Arm (pos, cmd, goal, fl, io, vmax, mode), C = collect.Collector (goto, rise, hold_here, holding, wps,
    cfg act_roi), S = capture state (frames, frame_t). check() raises Stopped; grasped() = the human pressed G."""
    def __init__(self, A, C, S, policy, max_s=20, retries=2, mode="chunk", speed=1.0, task=None, jpeg=True,
                 check=None, grasped=None, say=None, clock=None):
        self.A, self.C, self.S, self.p = A, C, S, policy
        self.max_s, self.retries, self.speed, self.task, self.jpeg = float(max_s), int(retries), float(speed), task, jpeg
        self.ex = Executor(policy, mode)
        self.check = check or (lambda: None); self.grasped = grasped or (lambda: False)
        self.say = say or (lambda m, ok=True: None); self.clock = clock or RealClock()
        self.live = {}; self.attempts = []; self.all_infer, self.all_obs = [], []

    def _check(self):
        self.check()
        if self.A.stop.is_set(): raise Stopped()

    def _obs(self):
        S, now = self.S, time.time()
        for c in self.p.cams:
            t = getattr(S, "frame_t", {}).get(c)
            if t is not None and now - t > CAM_MAX_AGE_S:
                raise PickFailed(f"camera {c} stalled ({now - t:.1f} s old frame): the policy would act blind", retry=False)
        return observation(S.frames, self.A.pos, self.A.fl, self.p.image_keys, self.p.cams, self.C.cfg["act_roi"],
                           self.task if self.p.takes_task else None, self.jpeg)

    def _target(self, act):
        A = self.A; t = denormalize(act[None], A.fl)[0]; lead = MAX_LEAD_DEG * TICK_PER_DEG
        tgt = {n: float(t[j]) for j, n in enumerate(JOINTS)}
        for n in BODY: tgt[n] = min(A.pos[n] + lead, max(A.pos[n] - lead, tgt[n]))
        for n in JOINTS: tgt[n] = min(A.fl[n][1], max(A.fl[n][0], tgt[n]))
        return tgt

    def _write(self, tgt):
        with self.A.io:                 # STOP sets the flag first, then freezes the goal under this lock
            if self.A.stop.is_set(): raise Stopped()
            self.A.mode = "hold"; self.A.goal = tgt

    def attempt(self, k):
        """Drive with the policy from the start pose until the hand-off.
        -> 'grasp' (the demos' stall rule) | 'grasp_soft' (stalled on something, but squeezing only lightly, for
        SOFT_HOLD_S) | 'manual' (G) | 'timeout'."""
        A, clk = self.A, self.clock
        self.p.reset(); self.ex.reset()
        det = GraspDetector(A.fl["gripper"][0]); t0 = clk.now(); i = 0; done_at = None; why = "timeout"
        soft = GraspDetector(A.fl["gripper"][0], gap=SOFT_GAP, hold=SOFT_HOLD_S) if SOFT_HOLD_S else None
        lag_since = None; late = 0; dts = []; last = t0
        A.vmax = MIRROR_VMAX * max(0.1, min(1.0, self.speed))
        while True:
            self._check()
            now = clk.now(); dts.append(now - last); last = now
            tgt = self._target(self.ex.step(self._obs))
            self._check(); self._write(tgt)
            now = clk.now(); t = now - t0
            lag = max(abs(A.cmd[n] - A.pos[n]) for n in BODY)
            lag_since = (lag_since or now) if lag > BLOCKED else None
            if lag_since and now - lag_since > BLOCK_S:
                raise PickFailed(f"BLOCKED: the arm stays {lag / TICK_PER_DEG:.0f}° behind the policy's target: something is in the way", retry=False)
            if done_at is None:
                if self.grasped(): done_at, why = now, "manual"
                elif det.update(now, A.pos["gripper"], A.goal["gripper"]): done_at, why = now, "grasp"
                elif soft and soft.update(now, A.pos["gripper"], A.goal["gripper"]): done_at, why = now, "grasp_soft"
            self.live = {"attempt": k + 1, "of": self.retries + 1, "t": round(t, 1), "steps": i + 1, **self.ex.stats()}
            if done_at is not None and now - done_at >= TAIL_S: break     # keep closing a little longer, as in the demos
            if done_at is None and t > self.max_s: break
            i += 1; nxt = t0 + i / FPS
            if clk.now() > nxt + 0.5 / FPS: late += 1
            clk.sleep_until(nxt)
        self.attempts.append({"attempt": k + 1, "result": why, "t_s": round(clk.now() - t0, 2), "steps": i + 1,
                              "late_ticks": late, "hz": round(len(dts[1:]) / max(1e-6, sum(dts[1:])), 1) if len(dts) > 1 else None,
                              **self.ex.stats()})
        self.all_infer += self.ex.infer_ms; self.all_obs += self.ex.obs_ms
        return why

    def run(self):
        A, C = self.A, self.C
        home = C.wps("home")[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]
        gmin, gmax = A.fl["gripper"]; g_open = max(home["gripper"], gmin + 0.35 * (gmax - gmin))
        reasons = []
        try:
            for k in range(self.retries + 1):
                self._check()
                self.say(f"ACT pick {k + 1}/{self.retries + 1}: robot to the start pose")
                C.goto(home, gripper=g_open)
                self.say(f"ACT pick {k + 1}/{self.retries + 1}: the policy is picking ({self.p.label}) · G if it holds the object")
                why = self.attempt(k)
                if why == "timeout":
                    reasons.append(f"no grasp within {self.max_s:.0f} s")
                else:
                    C.hold_here(); g_lock = A.cmd["gripper"]
                    if why == "grasp_soft":                 # light squeeze: clamp as firmly as the tape re-grip does
                        g_lock = max(gmin, min(g_lock, A.pos["gripper"] - REGRIP_SQUEEZE))
                        C.goto(dict(A.cmd), 0.5, gripper=g_lock)
                    self.say("hand-off: the robot lifts the object to waypoint 1")
                    C.rise(wp1); C.goto(wp1, gripper=g_lock); self.clock.sleep_until(self.clock.now() + 0.3)
                    if C.holding():
                        self.attempts[-1]["held_at_wp1"] = True
                        return g_lock, self.info(True)
                    self.attempts[-1]["held_at_wp1"] = False
                    reasons.append("closed on nothing (not held at waypoint 1)")
                if k < self.retries:                       # let go of whatever, go straight up, try again
                    self.say(f"ACT pick {k + 1} failed ({reasons[-1]}): letting go and trying again", False)
                    C.hold_here(); C.goto(dict(A.cmd), 0.4, gripper=g_open); C.rise(wp1)
            raise PickFailed(f"ACT pick failed {len(reasons)}x: {'; '.join(reasons)}", self.info(False))
        except PickFailed as e:
            if not e.info: e.info = self.info(False, str(e))
            raise
        except Stopped:
            raise
        except Exception as e:                             # BLOCKED move, camera gone, model error: no retry
            raise PickFailed(f"ACT pick stopped: {e}", self.info(False, str(e)), retry=False) from e
        finally:
            try: C.hold_here()
            except Exception: pass

    def info(self, ok, error=None):
        """For the cycle log: model, device, attempts, and inference latency over all attempts (efficiency)."""
        return {"ok": ok, "error": error, "model": str(self.p.export_dir), "backend": self.p.backend, "device": self.p.device,
                "runtime": self.p.runtime, "load_ms": self.p.load_ms, "mode": self.ex.mode, "speed": self.speed,
                "attempts": self.attempts, "n_attempts": len(self.attempts), "inferences": len(self.all_infer),
                "infer_ms_p50": pct(self.all_infer, 50), "infer_ms_p95": pct(self.all_infer, 95),
                "infer_ms_max": round(max(self.all_infer), 2) if self.all_infer else None,
                "obs_ms_p50": pct(self.all_obs, 50), "obs_ms_p95": pct(self.all_obs, 95)}


def roi_warning(C, episodes_dir):
    """The policy crop must be the one the training demos were recorded with: compare with the newest episode."""
    try:
        m = max(Path(episodes_dir).glob("*/*/meta.json"), key=lambda p: p.stat().st_mtime)
        rec = json.loads(m.read_text()).get("act_roi") or {}
    except Exception:
        return None
    bad = [c for c in POLICY_CAMS if c in rec and list(rec[c]) != list(C.cfg["act_roi"].get(c, []))]
    return (f"the policy crop of {', '.join(bad)} differs from the one the newest demos were recorded with "
            f"({m.parent.parent.name}/{m.parent.name}): the policy will see a different picture") if bad else None


# ====================================================================== standalone check
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true"); ap.add_argument("--model", default="")
    ap.add_argument("--device", default="CPU"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--bench", type=int, default=30, help="inferences on random frames")
    a = ap.parse_args()
    if a.list or not a.model:
        for e in find_exports([HERE.parent / "models"]):
            print(f"{e['mtime']}  {e['backend']:8s} {e['policy']:6s} chunk {e['chunk']}  cams {','.join(e['cams'])}  {e['name']}  {e['path']}")
        return
    p = load(a.model, a.device, a.backend)
    print(f"loaded {p.label} via {p.runtime} in {p.load_ms} ms; inputs {p.image_keys} + state")
    rng = np.random.default_rng(0); lim = {n: (0, 4095) for n in JOINTS}
    ex = Executor(p, "ensemble")
    for _ in range(a.bench):
        frames = {c: rng.integers(0, 255, (1080, 1920, 3), np.uint8) for c in p.cams}
        ex.step(lambda: observation(frames, {n: 2048 for n in JOINTS}, lim, p.image_keys, p.cams,
                                    {c: prep.default_act_roi(c) for c in p.cams}))
    print(json.dumps(ex.stats()))


if __name__ == "__main__":
    main()
