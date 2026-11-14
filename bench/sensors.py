"""Power / temperature / frequency / busy sampler for Linux on Intel (Core Ultra), read straight from sysfs.

Why: the judges want power, thermal headroom and "it really ran on the NPU" as evidence. Everything here is a plain
file read (no psutil, no root daemon), so it runs in any venv. Every source is optional: what is missing is listed
under `unavailable` with the reason (and the fix, for the root-only RAPL counters); nothing ever raises.

Sources (sysfs root injectable for tests: Sensors(root="/tmp/fake")):
  rapl     /sys/class/powercap/intel-rapl:*  energy_uj counters (package-0, core, uncore = iGPU on client parts,
           psys = whole platform when the firmware exposes it), wrap-around via max_energy_range_uj.
           energy_uj is root-only since the 2020 PLATYPUS fix: run with sudo, or `sudo chmod a+r ...` (RAPL_FIX).
  thermal  /sys/class/thermal/thermal_zone*/{type,temp,trip_point_*}   (x86_pkg_temp, acpitz, TCPU ...)
  hwmon    /sys/class/hwmon/hwmon*/{name,temp*_input,temp*_label,temp*_crit,power*_input,energy*_input}
           (coretemp "Package id 0" + temp*_crit = TjMax)
  cpu      /proc/stat utilisation, /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
  gpu      xe:   /sys/class/drm/cardN/device/tile*/gt*/freq0/{act_freq,cur_freq,rp0_freq}, gtidle/idle_residency_ms
           i915: /sys/class/drm/cardN/gt_act_freq_mhz, power/rc6_residency_ms     (busy % = 100 - idle residency %)
  npu      intel_vpu (drivers/accel/ivpu/ivpu_sysfs.c) on the PCI device (/sys/class/accel/accel*/device,
           /sys/bus/pci/devices/0000:00:0b.0): npu_busy_time_us, freq/current_freq (legacy npu_current_frequency_mhz),
           freq/hw_max_freq, npu_memory_utilization, power/runtime_status
  battery  /sys/class/power_supply/BAT*/power_now (or current_now*voltage_now): whole-system cross-check, only
           meaningful while discharging (unplug the charger for that run)

    python -m bench.sensors                 # what is available here + 5 s idle reading
    python -m bench.sensors --idle 30 --json bench/results/idle.json
"""
import argparse, glob, json, os, re, threading, time
from pathlib import Path

RAPL_FIX = "sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj"
RAPL_FIX_PERSIST = ('echo \'ACTION=="add", SUBSYSTEM=="powercap", RUN+="/bin/chmod a+r /sys%p/energy_uj"\' | '
                    'sudo tee /etc/udev/rules.d/90-rapl-read.rules   # (untested here; or re-run the chmod after a reboot)')


def _read(p):
    with open(p) as f: return f.read().strip()


def _num(p, scale=1.0):
    try: return float(_read(p)) * scale
    except (OSError, ValueError): return None


def _uevent(dev):
    try:
        for line in _read(Path(dev) / "uevent").splitlines():
            if line.startswith("DRIVER="): return line.split("=", 1)[1]
    except OSError: pass
    try: return os.path.basename(os.readlink(Path(dev) / "driver"))
    except OSError: return None


def _clean(s): return re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip()).strip("_") or "x"


class Sensors:
    """Discovers the sources once; read() -> {key: float} for one sample. meta[key] = {kind, unit, label, ...}.
    kinds: energy_uj (counter, W = d/dt), busy_us (counter, % = d/dt), idle_ms (counter, busy % = 100 - d/dt),
    jiffies (cpu.busy_j / cpu.total_j counters), temp_c, freq_mhz, power_w, bytes (gauges)."""

    def __init__(self, root="/"):
        self.root = Path(root)
        self.meta, self.readers, self.unavailable, self.fixes, self.info = {}, [], {}, [], {}
        for grp in ("rapl", "thermal", "hwmon", "cpu", "gpu", "npu", "battery"):
            try: getattr(self, "_find_" + grp)()
            except Exception as e: self.unavailable[grp] = f"discovery failed: {type(e).__name__}: {e}"

    def p(self, rel): return self.root / rel.lstrip("/")

    def _add(self, key, kind, unit, label, **kw):
        self.meta[key] = {"kind": kind, "unit": unit, "label": label, **kw}

    # ------------------------------------------------------------------ discovery
    def _find_rapl(self):
        zones = sorted(glob.glob(str(self.p("sys/class/powercap/intel-rapl:*"))))
        if not zones:
            self.unavailable["rapl"] = "no /sys/class/powercap/intel-rapl:* (not Intel, VM/container, or intel_rapl_msr not loaded)"
            return
        denied, seen = [], set()
        for z in zones:
            z = Path(z)
            try: name = _read(z / "name")
            except OSError: continue
            zid = z.name.split(":", 1)[1]
            real = os.path.realpath(z)
            if real in seen: continue
            seen.add(real)
            key = f"rapl.{name}" if f"rapl.{name}" not in self.meta else f"rapl.{name}.{zid}"
            try: _read(z / "energy_uj")
            except PermissionError: denied.append(z.name); continue
            except OSError: continue
            self._add(key, "energy_uj", "W", f"RAPL {name}", path=str(z / "energy_uj"),
                      max_range=_num(z / "max_energy_range_uj"), zone=z.name)
            self.readers.append(lambda k=key, f=z / "energy_uj": {k: _num(f)})
        if denied:
            root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
            self.unavailable["rapl"] = (f"energy_uj not readable for {', '.join(denied)} (root-only since the 2020 PLATYPUS fix"
                                        + (")" if root else "; run with sudo or make it readable)"))
            self.fixes += [RAPL_FIX, RAPL_FIX_PERSIST]

    def _find_thermal(self):
        zones = sorted(glob.glob(str(self.p("sys/class/thermal/thermal_zone*"))), key=lambda s: int(re.sub(r"\D", "", s.rsplit("zone", 1)[1]) or 0))
        bad = []
        for z in map(Path, zones):
            try: typ = _read(z / "type")
            except OSError: continue
            if _num(z / "temp") is None: bad.append(typ); continue
            trips = {}
            for t in sorted(z.glob("trip_point_*_type")):
                try: tt = _read(t); tv = _num(str(t).replace("_type", "_temp"), 1e-3)
                except OSError: continue
                if tv is not None and tv > 0: trips[tt] = min(tv, trips.get(tt, tv))
            key = f"tz.{_clean(typ)}"
            if key in self.meta: key += f".{z.name.replace('thermal_zone', '')}"
            self._add(key, "temp_c", "C", f"thermal zone {typ}", crit=trips.get("critical"), hot=trips.get("hot"),
                      passive=trips.get("passive"), zone=z.name)
            self.readers.append(lambda k=key, f=z / "temp": {k: _num(f, 1e-3)})
        if not self.meta or not any(m["kind"] == "temp_c" for m in self.meta.values()):
            self.unavailable["thermal"] = "no readable /sys/class/thermal/thermal_zone*" + (f" (unreadable: {', '.join(bad)})" if bad else "")

    def _find_hwmon(self):
        mons = sorted(glob.glob(str(self.p("sys/class/hwmon/hwmon*"))))
        if not mons: self.unavailable["hwmon"] = "no /sys/class/hwmon"; return
        names = {}
        for h in map(Path, mons):
            try: name = _read(h / "name")
            except OSError: continue
            names[name] = names.get(name, 0) + 1
            base = f"hw.{_clean(name)}" + (f"{names[name] - 1}" if names[name] > 1 else "")
            for inp in sorted(h.glob("temp*_input")):
                pre = inp.name[:-len("_input")]
                if _num(inp) is None: continue
                try: lab = _read(h / f"{pre}_label")
                except OSError: lab = pre
                key = f"{base}.{_clean(lab)}"
                self._add(key, "temp_c", "C", f"{name} {lab}", crit=_num(h / f"{pre}_crit", 1e-3), max=_num(h / f"{pre}_max", 1e-3),
                          hwmon=name)
                self.readers.append(lambda k=key, f=inp: {k: _num(f, 1e-3)})
            for inp in sorted(h.glob("power*_input")):
                key = f"{base}.{inp.name[:-len('_input')]}"
                if _num(inp) is None: continue
                self._add(key, "power_w", "W", f"{name} {inp.name}", hwmon=name)
                self.readers.append(lambda k=key, f=inp: {k: _num(f, 1e-6)})
            for inp in sorted(h.glob("energy*_input")):
                key = f"{base}.{inp.name[:-len('_input')]}"
                if _num(inp) is None: continue
                self._add(key, "energy_uj", "W", f"{name} {inp.name}", hwmon=name, max_range=None)
                self.readers.append(lambda k=key, f=inp: {k: _num(f)})

    def _find_cpu(self):
        stat = self.p("proc/stat")
        if _cpu_jiffies(stat):
            self._add("cpu.busy_j", "jiffies", "%", "CPU utilisation (/proc/stat)")
            self.readers.append(lambda f=stat: _cpu_jiffies(f) or {})
        files = sorted(glob.glob(str(self.p("sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))))
        if files:
            self._add("cpu.freq_mean_mhz", "freq_mhz", "MHz", f"CPU frequency, mean of {len(files)} cpus")
            self._add("cpu.freq_max_mhz", "freq_mhz", "MHz", "CPU frequency, fastest cpu")
            mx = [_num(Path(f).with_name("cpuinfo_max_freq"), 1e-3) for f in files]
            self.info["cpu_max_freq_mhz"] = max([m for m in mx if m] or [0]) or None

            def rd(fs=files):
                v = [x for x in (_num(f, 1e-3) for f in fs) if x]
                return {"cpu.freq_mean_mhz": sum(v) / len(v), "cpu.freq_max_mhz": max(v)} if v else {}
            self.readers.append(rd)
        else:
            self.unavailable["cpufreq"] = "no cpufreq in sysfs"

    def _find_gpu(self):
        cards = [c for c in sorted(glob.glob(str(self.p("sys/class/drm/card[0-9]*")))) if re.fullmatch(r"card\d+", Path(c).name)]
        found = False
        for c in map(Path, cards):
            dev = c / "device"; drv = _uevent(dev); n = c.name.replace("card", "")
            if drv == "xe" or list(dev.glob("tile*/gt*/freq0")):
                for gt in sorted(dev.glob("tile*/gt*")):
                    tag = f"gpu{n}.{gt.parent.name}.{gt.name}"
                    try: gname = _read(gt / "gtidle" / "name")
                    except OSError: gname = gt.name
                    f = gt / "freq0" / "act_freq"
                    if _num(f) is not None:
                        self._add(f"{tag}.act_mhz", "freq_mhz", "MHz", f"xe {gname} actual frequency", max=_num(gt / "freq0" / "rp0_freq"))
                        self.readers.append(lambda k=f"{tag}.act_mhz", f=f: {k: _num(f)}); found = True
                    f = gt / "gtidle" / "idle_residency_ms"
                    if _num(f) is not None:
                        self._add(f"{tag}.idle_ms", "idle_ms", "%", f"xe {gname} busy (100 - idle residency)", engine=gname)
                        self.readers.append(lambda k=f"{tag}.idle_ms", f=f: {k: _num(f)}); found = True
            elif drv == "i915" or (c / "gt_act_freq_mhz").exists():
                f = c / "gt_act_freq_mhz"
                if _num(f) is not None:
                    self._add(f"gpu{n}.act_mhz", "freq_mhz", "MHz", "i915 actual frequency", max=_num(c / "gt_RP0_freq_mhz"))
                    self.readers.append(lambda k=f"gpu{n}.act_mhz", f=f: {k: _num(f)}); found = True
                for f in (c / "power" / "rc6_residency_ms", c / "gt" / "gt0" / "rc6_residency_ms"):
                    if _num(f) is not None:
                        self._add(f"gpu{n}.idle_ms", "idle_ms", "%", "i915 busy (100 - RC6 residency)")
                        self.readers.append(lambda k=f"gpu{n}.idle_ms", f=f: {k: _num(f)}); found = True; break
        if not found:
            self.unavailable["gpu"] = "no Intel GPU frequency/idle counters in /sys/class/drm (no xe/i915 GPU)"

    def _find_npu(self):
        cands = sorted(glob.glob(str(self.p("sys/class/accel/accel*/device")))) + [str(self.p("sys/bus/pci/devices/0000:00:0b.0"))]
        seen = set()
        for d in map(Path, cands):
            real = os.path.realpath(d)
            if real in seen or not d.exists(): continue
            seen.add(real)
            drv = _uevent(d)
            if drv != "intel_vpu" and not (d / "npu_busy_time_us").exists(): continue
            self.info["npu_pci"] = Path(real).name; self.info["npu_driver"] = drv
            self.info["npu_max_mhz"] = _num(d / "freq" / "hw_max_freq") or _num(d / "npu_max_frequency_mhz")
            try: self.info["npu_runtime_status"] = _read(d / "power" / "runtime_status")
            except OSError: pass
            if _num(d / "npu_busy_time_us") is not None:
                self._add("npu.busy_us", "busy_us", "%", "NPU busy (npu_busy_time_us)")
                self.readers.append(lambda f=d / "npu_busy_time_us": {"npu.busy_us": _num(f)})
            for f in (d / "freq" / "current_freq", d / "npu_current_frequency_mhz"):
                if _num(f) is not None:
                    self._add("npu.freq_mhz", "freq_mhz", "MHz", "NPU current frequency (0 when idle)", max=self.info["npu_max_mhz"])
                    self.readers.append(lambda f=f: {"npu.freq_mhz": _num(f)}); break
            if _num(d / "npu_memory_utilization") is not None:
                self._add("npu.mem_bytes", "bytes", "B", "NPU memory in use")
                self.readers.append(lambda f=d / "npu_memory_utilization": {"npu.mem_bytes": _num(f)})
            if "npu.busy_us" not in self.meta:
                self.unavailable["npu"] = f"intel_vpu device {Path(real).name} found but no npu_busy_time_us (kernel < 6.11?)"
            return
        self.unavailable["npu"] = "no intel_vpu device in sysfs (no NPU, or the intel_vpu kernel driver is not loaded)"

    def _find_battery(self):
        bats = sorted(glob.glob(str(self.p("sys/class/power_supply/BAT*"))))
        for b in map(Path, bats):
            def rd(b=b):
                w = _num(b / "power_now", 1e-6)
                if w is None:
                    i, v = _num(b / "current_now"), _num(b / "voltage_now")
                    w = i * v * 1e-12 if i is not None and v is not None else None
                return {} if w is None else {f"bat.{b.name}.w": w}
            if rd():
                self._add(f"bat.{b.name}.w", "power_w", "W", f"battery {b.name} power (whole system while discharging)")
                self.readers.append(rd)
        if not any(k.startswith("bat.") for k in self.meta):
            self.unavailable["battery"] = "no /sys/class/power_supply/BAT*/power_now"

    def battery_status(self):
        out = {}
        for b in sorted(glob.glob(str(self.p("sys/class/power_supply/*")))):
            try:
                t = _read(Path(b) / "type")
                if t == "Battery": out[Path(b).name] = _read(Path(b) / "status")
                elif t == "Mains": out[Path(b).name] = "online" if _read(Path(b) / "online") == "1" else "offline"
            except OSError: pass
        return out

    # ------------------------------------------------------------------ reading
    def read(self):
        out = {}
        for r in self.readers:
            try:
                for k, v in r().items():
                    if v is not None: out[k] = v
            except Exception: pass
        return out

    def describe(self):
        return {"root": str(self.root), "sources": self.meta, "unavailable": self.unavailable, "fixes": self.fixes,
                "info": self.info, "power_status": self.battery_status()}

    def text(self):
        groups = {}
        for k, m in self.meta.items(): groups.setdefault(k.split(".")[0], []).append(k)
        lines = [f"  {g:<6} {len(ks):>3} source(s): {', '.join(ks[:6])}{' ...' if len(ks) > 6 else ''}" for g, ks in groups.items()]
        lines += [f"  -- {g}: unavailable: {why}" for g, why in self.unavailable.items()]
        lines += [f"  FIX: {f}" for f in self.fixes]
        return "\n".join(lines)


def _cpu_jiffies(stat):
    try: f = _read(stat).splitlines()[0].split()
    except (OSError, IndexError): return None
    if not f or f[0] != "cpu": return None
    v = [float(x) for x in f[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    tot = sum(v[:8])
    return {"cpu.busy_j": tot - idle, "cpu.total_j": tot}


# ---------------------------------------------------------------------- sampling
class Sampler:
    """Background thread: sample every `interval` s until stop(). with Sampler(s) as smp: ...; smp.summary(idle)"""

    def __init__(self, sensors, interval=0.25):
        self.s, self.interval = sensors, interval
        self.samples, self._stop, self._t = [], threading.Event(), None

    def start(self):
        self.samples = [(time.monotonic(), self.s.read())]
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self.samples.append((time.monotonic(), self.s.read()))

    def stop(self):
        self._stop.set()
        if self._t: self._t.join(2)
        self.samples.append((time.monotonic(), self.s.read()))
        return self.samples

    def __enter__(self): return self.start()
    def __exit__(self, *a): self.stop()

    def summary(self, idle=None): return summarize(self.samples, self.s.meta, idle, self.s.info)


def counter_delta(v0, v1, max_range=None):
    """Delta of a monotonically increasing counter; handles one wrap-around when max_range is known
    (RAPL energy_uj wraps at max_energy_range_uj). Unknown range + negative delta = counter reset -> None."""
    if v0 is None or v1 is None: return None
    d = v1 - v0
    if d < 0:
        if not max_range: return None
        d += max_range
    return d


def _counter(samples, key, max_range=None):
    """-> (total delta, seconds covered, per-interval list of (t_mid, rate))."""
    tot, secs, series = 0.0, 0.0, []
    for (t0, a), (t1, b) in zip(samples, samples[1:]):
        d = counter_delta(a.get(key), b.get(key), max_range)
        if d is None or t1 <= t0: continue
        tot += d; secs += t1 - t0; series.append(((t0 + t1) / 2, d / (t1 - t0)))
    return tot, secs, series


def _primary_power(meta):
    pk = sorted(k for k, m in meta.items() if m["kind"] == "energy_uj" and k.startswith("rapl.package"))
    if pk: return pk
    for k in ("rapl.psys",):
        if k in meta: return [k]
    return [k for k, m in meta.items() if m["kind"] == "energy_uj"][:1] or [k for k in meta if k.startswith("bat.")][:1]


def _primary_temp(meta):
    for pat in (r"hw\.coretemp\d*\.Package_id_0$", r"hw\.coretemp\d*\.Package", r"tz\.x86_pkg_temp", r"tz\.TCPU", r"hw\.k10temp",
                r"tz\.acpitz", r"tz\.cpu"):
        for k in meta:
            if re.match(pat, k) and meta[k]["kind"] == "temp_c": return k
    t = [k for k, m in meta.items() if m["kind"] == "temp_c"]
    return t[0] if t else None


def _tjmax(meta, key):
    """Thermal limit for the primary temperature: coretemp temp*_crit (= TjMax) or the zone's critical trip."""
    if key and meta[key].get("crit"): return meta[key]["crit"], key + " crit"
    for k, m in meta.items():
        if m.get("hwmon") == "coretemp" and m.get("crit"): return m["crit"], k + " crit"
    for k, m in meta.items():
        if k.startswith("tz.x86_pkg_temp") and (m.get("crit") or m.get("hot")): return m.get("crit") or m.get("hot"), k + " trip"
    for k, m in meta.items():
        if m.get("crit"): return m["crit"], k + " crit"
    return None, None


def _thin(series, n=600):
    if len(series) <= n: return series
    step = len(series) / n
    return [series[int(i * step)] for i in range(n)]


def summarize(samples, meta, idle=None, info=None):
    """Samples -> averages over the window. idle = summary of an idle window (then *_above_idle fields)."""
    if not samples: return {"duration_s": 0, "n_samples": 0}
    t0 = samples[0][0]; dur = samples[-1][0] - t0
    out = {"duration_s": round(dur, 3), "n_samples": len(samples), "power_w": {}, "temp_c": {}, "freq_mhz": {}, "busy_pct": {},
           "series": {}}
    for k, m in meta.items():
        kind = m["kind"]
        if kind == "energy_uj":
            tot, secs, ser = _counter(samples, k, m.get("max_range"))
            if secs > 0:
                out["power_w"][k] = round(tot * 1e-6 / secs, 3)
                if k in _primary_power(meta): out["series"].setdefault("power_w", [(round(t - t0, 3), round(w * 1e-6, 3)) for t, w in ser])
        elif kind == "busy_us":
            tot, secs, _ = _counter(samples, k)
            if secs > 0: out["busy_pct"][k.rsplit(".", 1)[0]] = round(min(100.0, 100 * tot * 1e-6 / secs), 2)
        elif kind == "idle_ms":
            tot, secs, _ = _counter(samples, k)
            if secs > 0: out["busy_pct"][k.rsplit(".", 1)[0]] = round(max(0.0, min(100.0, 100 - 100 * tot * 1e-3 / secs)), 2)
        elif kind == "jiffies":
            b, _, _ = _counter(samples, "cpu.busy_j"); t, _, _ = _counter(samples, "cpu.total_j")
            if t > 0: out["busy_pct"]["cpu"] = round(100 * b / t, 2)
        else:
            v = [(t, s[k]) for t, s in samples if k in s]
            if not v: continue
            vals = [x for _, x in v]
            st = {"start": round(vals[0], 2), "mean": round(sum(vals) / len(vals), 2), "peak": round(max(vals), 2), "end": round(vals[-1], 2)}
            if kind == "temp_c": out["temp_c"][k] = st
            elif kind == "freq_mhz": out["freq_mhz"][k] = {"mean": st["mean"], "max": st["peak"]}
            elif kind == "power_w": out["power_w"][k] = st["mean"]
            elif kind == "bytes": out.setdefault("bytes", {})[k] = st["peak"]
    # ---- headline numbers
    pk = [k for k in _primary_power(meta) if k in out["power_w"]]
    if pk:
        w = sum(out["power_w"][k] for k in pk)
        main = {"source": "+".join(pk), "w": round(w, 3)}
        if idle and idle.get("power", {}).get("source") == main["source"]:
            main["idle_w"] = idle["power"]["w"]; main["w_above_idle"] = round(w - idle["power"]["w"], 3)
        main["energy_j"] = round(w * dur, 3)
        out["power"] = main
    ct = _primary_temp(meta)
    if ct and ct in out["temp_c"]:
        st = out["temp_c"][ct]; lim, lsrc = _tjmax(meta, ct)
        cpu = {"source": ct, **st, "rise": round(st["peak"] - st["start"], 2)}
        if idle and idle.get("cpu_temp", {}).get("source") == ct:
            cpu["idle"] = idle["cpu_temp"]["mean"]; cpu["above_idle"] = round(st["peak"] - cpu["idle"], 2)
        if lim: cpu.update(limit=lim, limit_source=lsrc, headroom=round(lim - st["peak"], 2))
        out["cpu_temp"] = cpu
        out["series"]["cpu_temp_c"] = [(round(t - t0, 3), round(s[ct], 2)) for t, s in samples if ct in s]
    if out["temp_c"]:
        hk = max(out["temp_c"], key=lambda k: out["temp_c"][k]["peak"])
        out["hottest"] = {"source": hk, "peak": out["temp_c"][hk]["peak"]}
    if idle and idle.get("busy_pct"):
        out["busy_above_idle_pct"] = {k: round(v - idle["busy_pct"].get(k, 0), 2) for k, v in out["busy_pct"].items()}
    for k in list(out["series"]): out["series"][k] = _thin(out["series"][k])
    return out


def measure_idle(sensors, seconds=10.0, interval=0.25):
    """Idle baseline: sample `seconds` with nothing running (close the browser / Studio streams first)."""
    smp = Sampler(sensors, interval).start()
    time.sleep(seconds)
    smp.stop()
    s = smp.summary(); s["label"] = "idle"
    return s


def wait_cool(sensors, idle, max_s=60.0, margin_c=3.0, interval=1.0):
    """Wait until the primary temperature is within margin_c of the idle mean (or max_s) -> seconds waited."""
    ct = (idle or {}).get("cpu_temp", {})
    if not ct.get("source") or max_s <= 0: return 0.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_s:
        v = sensors.read().get(ct["source"])
        if v is None or v <= ct["mean"] + margin_c: break
        time.sleep(interval)
    return round(time.monotonic() - t0, 1)


def headline(s):
    """One line for humans from a summary."""
    parts = []
    if s.get("power"):
        p = s["power"]; parts.append(f"{p['w']:.2f} W ({p['source']})" + (f", +{p['w_above_idle']:.2f} W above idle" if "w_above_idle" in p else ""))
    if s.get("cpu_temp"):
        c = s["cpu_temp"]; parts.append(f"CPU {c['peak']:.1f} C peak (+{c['rise']:.1f})" + (f", headroom {c['headroom']:.1f} C to {c['limit']:.0f}" if "headroom" in c else ""))
    for k, v in s.get("busy_pct", {}).items(): parts.append(f"{k} busy {v:.1f}%")
    return "; ".join(parts) or "no power/temperature sources"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/", help="sysfs root (tests)")
    ap.add_argument("--idle", type=float, default=5.0, help="seconds of idle sampling (0 = only list sources)")
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--json", help="write {sources, idle summary} here")
    a = ap.parse_args()
    s = Sensors(a.root)
    print(f"sensors (root {a.root}):\n{s.text()}")
    out = {"sensors": s.describe()}
    if a.idle > 0:
        print(f"sampling {a.idle:g} s idle ...")
        out["idle"] = measure_idle(s, a.idle, a.interval)
        print("idle:", headline(out["idle"]))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(out, indent=1))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
