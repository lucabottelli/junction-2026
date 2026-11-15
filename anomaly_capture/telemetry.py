"""Live power, temperature and device load for the user page: GET /telemetry.

One background thread samples the sysfs counters found by bench/sensors.py (the same sources the Phase 7 bench
uses, so the page and the report agree) once per second and turns them into rates:

  pkg_w      RAPL package-0 power (CPU + iGPU + NPU + uncore: the SoC)       W
  core_w     RAPL core (CPU cores only)                                        W
  psys_w     RAPL psys (whole platform, when the firmware exposes it)          W
  temp_c     CPU package temperature (coretemp "Package id 0")                 C
  tjmax_c    its thermal limit (coretemp temp*_crit); headroom_c = tjmax - temp
  cpu_pct    CPU utilisation (/proc/stat)                                       %
  gpu_pct    iGPU render engine busy (xe gt0-rc, 100 - idle residency)          %
  npu_pct    NPU busy (intel_vpu npu_busy_time_us)                              %
  energy_j   package energy since the sampler started (for energy per block)    J

A missing source (e.g. RAPL not readable without the chmod in bench/sensors.RAPL_FIX) gives null for its fields
and a note in `missing`; nothing else depends on it.
"""
import sys, threading, time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # repo root, for bench/
from bench.sensors import Sensors, counter_delta, _primary_temp, _tjmax   # noqa: E402

INTERVAL = 1.0
HISTORY = 120                       # samples kept for the sparkline (2 min)


class Telemetry:
    def __init__(self):
        self.s = Sensors(); m = self.s.meta
        self.pkg = "rapl.package-0" if "rapl.package-0" in m else next((k for k in m if k.startswith("rapl.package")), None)
        self.core = "rapl.core" if "rapl.core" in m else None
        self.psys = "rapl.psys" if "rapl.psys" in m else None
        self.temp = _primary_temp(m); self.tjmax = _tjmax(m, self.temp)[0] if self.temp else None
        gpus = [k for k, v in m.items() if v["kind"] == "idle_ms"]
        self.gpu = next((k for k in gpus if "rc" in str(m[k].get("engine", ""))), gpus[0] if gpus else None)
        self.npu = "npu.busy_us" if "npu.busy_us" in m else None
        self.missing = dict(self.s.unavailable)
        self.lock = threading.Lock(); self.cur = {}; self.hist = deque(maxlen=HISTORY); self.energy = 0.0
        self.t0 = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _rate(self, k, a, b, dt):
        if not k: return None
        d = counter_delta(a.get(k), b.get(k), self.s.meta[k].get("max_range"))
        return None if d is None else d / dt

    def _run(self):
        prev, tp = self.s.read(), time.time()
        while True:
            time.sleep(INTERVAL)
            now, t = self.s.read(), time.time(); dt = t - tp
            if dt <= 0: prev, tp = now, t; continue
            uj = lambda k: (lambda r: None if r is None else round(r * 1e-6, 2))(self._rate(k, prev, now, dt))
            pkg = uj(self.pkg)
            g = self._rate(self.gpu, prev, now, dt)                       # ms idle per s
            n = self._rate(self.npu, prev, now, dt)                       # us busy per s
            cb = counter_delta(prev.get("cpu.busy_j"), now.get("cpu.busy_j")); ct = counter_delta(prev.get("cpu.total_j"), now.get("cpu.total_j"))
            temp = now.get(self.temp) if self.temp else None
            clip = lambda v: None if v is None else round(min(100.0, max(0.0, v)), 1)
            cur = {"t": round(t, 1), "pkg_w": pkg, "core_w": uj(self.core), "psys_w": uj(self.psys),
                   "temp_c": temp, "tjmax_c": self.tjmax,
                   "headroom_c": round(self.tjmax - temp, 1) if temp is not None and self.tjmax else None,
                   "cpu_pct": clip(100.0 * cb / ct) if cb is not None and ct else None,
                   "gpu_pct": clip(100.0 - g / 10.0) if g is not None else None,
                   "npu_pct": clip(n / 1e4) if n is not None else None}
            with self.lock:
                if pkg is not None: self.energy += pkg * dt
                cur["energy_j"] = round(self.energy, 1)
                self.cur = cur; self.hist.append((cur["t"], pkg, temp))
            prev, tp = now, t

    def energy_j(self):
        """Package energy (J) since start: read it before and after a block for J per block. None without RAPL."""
        with self.lock: return round(self.energy, 1) if self.pkg else None

    def status(self, history=False):
        with self.lock:
            out = {"ok": bool(self.cur), **self.cur, "missing": self.missing}
            if history: out["history"] = {"t": [h[0] for h in self.hist], "pkg_w": [h[1] for h in self.hist], "temp_c": [h[2] for h in self.hist]}
        return out


_T = None
_LOCK = threading.Lock()


def get():
    """The one sampler of this process (started on first use)."""
    global _T
    with _LOCK:
        if _T is None: _T = Telemetry()
        return _T


def energy_j():
    try: return get().energy_j()
    except Exception: return None


def make_router():
    from fastapi import APIRouter
    router = APIRouter()

    @router.get("/telemetry")
    def telemetry(history: int = 0):
        try: return get().status(bool(history))
        except Exception as e: return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    get()                                                       # start sampling at server start, so the first read has rates
    return router
