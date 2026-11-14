"""Environment report for the measurements: OpenVINO version, every OpenVINO device with its full name and key
properties, kernel / OS / CPU / RAM, the NPU driver chain (kernel driver -> /dev/accel -> firmware -> Level Zero
user-space driver -> OpenVINO lists NPU) with the exact fix commands from SO101_SETUP_STATUS.md section 6, and
which power/thermal sensors are readable (bench/sensors.py).

    python -m bench.devices                      # readable
    python -m bench.devices --json bench/results/devices.json
"""
import argparse, glob, json, os, platform, socket, subprocess, sys
from datetime import datetime
from pathlib import Path

NPU_FIX = ["sudo apt install -y libtbb12",
           "sudo dpkg -i ~/Downloads/npu-driver-v1.38.0/*.deb",
           "sudo modprobe -r intel_vpu && sudo modprobe intel_vpu   # or reboot"]
NPU_VERIFY = 'python -c "import openvino as ov; print(ov.Core().available_devices)"   # should list NPU'
KEY_PROPS = ["FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "DEVICE_TYPE", "OPTIMIZATION_CAPABILITIES", "INFERENCE_PRECISION_HINT",
             "RANGE_FOR_ASYNC_INFER_REQUESTS", "RANGE_FOR_STREAMS", "DEVICE_GOPS", "GPU_DEVICE_TOTAL_MEM_SIZE",
             "GPU_EXECUTION_UNITS_COUNT", "GPU_UARCH_VERSION", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION",
             "NPU_DEVICE_TOTAL_MEM_SIZE", "NPU_DEVICE_ALLOC_MEM_SIZE", "DEVICE_ID", "DEVICE_UUID", "DEVICE_LUID",
             "DEVICE_PCI_INFO", "DEVICE_THERMAL"]
LIB_DIRS = ["/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/local/lib", "/lib/x86_64-linux-gnu", "/opt/intel"]


def _s(v):
    """OpenVINO property value -> JSON-able."""
    if isinstance(v, (str, int, float, bool)) or v is None: return v
    if isinstance(v, (list, tuple)): return [_s(x) for x in v]
    if isinstance(v, dict): return {str(k): _s(x) for k, x in v.items()}
    if hasattr(v, "to_string"): return v.to_string()          # ov.Type -> 'f32' / 'bf16' / 'f16'
    return str(v)


def ov_devices():
    try:
        import openvino as ov
    except Exception as e:
        return {"error": f"openvino not importable: {e}"}
    core = ov.Core(); out = {"version": ov.get_version(), "available": list(core.available_devices), "devices": {}}
    for d in core.available_devices:
        props = {}
        try: supported = core.get_property(d, "SUPPORTED_PROPERTIES")
        except Exception: supported = {}
        for k in KEY_PROPS + [p for p in supported if p not in KEY_PROPS]:
            if supported and k not in supported: continue
            if str(supported.get(k, "RO")) not in ("RO", "RW"): continue
            try: props[k] = _s(core.get_property(d, k))
            except Exception: pass
        out["devices"][d] = props
    return out


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except OSError: pass
    return platform.processor() or None


def _os_name():
    try:
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME="): return line.split("=", 1)[1].strip().strip('"')
    except OSError: pass
    return platform.platform()


def _mem_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"): return round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError: pass
    return None


def _libs(patterns):
    hits = []
    for d in LIB_DIRS:
        for p in patterns: hits += glob.glob(os.path.join(d, "**", p) if d == "/opt/intel" else os.path.join(d, p), recursive=True)
    return sorted(set(hits))


def _dpkg(names):
    try:
        r = subprocess.run(["dpkg-query", "-W", "-f", "${Package} ${Version} ${Status}\n", *names], capture_output=True, text=True, timeout=10)
        return {l.split()[0]: l.split()[1] for l in r.stdout.splitlines() if l.strip().endswith("installed") and " " in l}
    except Exception: return {}


def _git_rev(root):
    try: return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception: return None


def npu_check(ov_info, root="/"):
    """Walk the NPU chain; -> {ok, steps: [(name, ok, detail)], fix: [...]}."""
    r = Path(root)
    steps = []
    pci = [p for p in glob.glob(str(r / "sys/bus/pci/devices/*")) if _drv(p) == "intel_vpu"]
    mod = (r / "sys/module/intel_vpu").exists()
    steps.append(("kernel driver intel_vpu", bool(pci) or mod, (f"bound to {', '.join(Path(p).name for p in pci)}" if pci else
                                                                "module loaded, no device bound" if mod else "not loaded / no NPU")))
    acc = sorted(glob.glob(str(r / "dev/accel/accel*")))
    steps.append(("/dev/accel node", bool(acc), ", ".join(f"{a} ({'rw' if os.access(a, os.R_OK | os.W_OK) else 'NO rw access: add the user to the render group'})" for a in acc) or "missing"))
    fw = sorted(glob.glob(str(r / "lib/firmware/updates/intel/vpu/*")) + glob.glob(str(r / "lib/firmware/intel/vpu/*")))
    steps.append(("NPU firmware", bool(fw), f"{len(fw)} file(s), e.g. {Path(fw[-1]).name}" if fw else "no /lib/firmware/(updates/)intel/vpu/*"))
    ze = _libs(["libze_intel_npu.so*", "libze_intel_vpu.so*"])
    steps.append(("Level Zero NPU user-space driver (libze_intel_npu)", bool(ze), ", ".join(ze[:2]) or "missing (this is the section 6 blocker)"))
    loader = _libs(["libze_loader.so*"])
    steps.append(("Level Zero loader (libze1)", bool(loader), ", ".join(loader[:1]) or "missing"))
    tbb = _libs(["libtbb.so.12*"])
    steps.append(("libtbb12", bool(tbb), ", ".join(tbb[:1]) or "missing"))
    pk = _dpkg(["intel-driver-compiler-npu", "intel-fw-npu", "intel-level-zero-npu", "libze1", "libtbb12"])
    listed = "NPU" in (ov_info.get("available") or [])
    steps.append(("OpenVINO lists NPU", listed, ov_info.get("devices", {}).get("NPU", {}).get("FULL_DEVICE_NAME", "") if listed else "not listed"))
    return {"ok": listed, "steps": [{"step": a, "ok": b, "detail": c} for a, b, c in steps], "packages": pk,
            "fix": [] if listed else NPU_FIX + [NPU_VERIFY]}


def gpu_check(ov_info, root="/"):
    r = Path(root)
    rd = sorted(glob.glob(str(r / "dev/dri/renderD*")))
    cl = _libs(["libze_intel_gpu.so*", "intel-opencl/libigdrcl.so", "libigdrcl.so"])
    listed = "GPU" in (ov_info.get("available") or [])
    out = {"ok": listed, "render_nodes": rd, "compute_runtime": cl,
           "drivers": sorted({d for d in (_drv(Path(c) / "device") for c in glob.glob(str(r / "sys/class/drm/card[0-9]"))) if d})}
    if not listed:
        out["fix"] = ["sudo apt install -y intel-opencl-icd libze-intel-gpu1   # Intel compute runtime (GPU plugin needs OpenCL/L0)",
                      "sudo usermod -aG render $USER   # then log in again"]
    return out


def _drv(dev):
    try:
        for line in open(Path(dev) / "uevent"):
            if line.startswith("DRIVER="): return line.strip().split("=", 1)[1]
    except OSError: pass
    try: return os.path.basename(os.readlink(Path(dev) / "driver"))
    except OSError: return None


def report(root="/"):
    from . import ROOT
    from .sensors import Sensors
    ov_info = ov_devices()
    s = Sensors(root)
    return {"date": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
            "kernel": platform.release(), "os": _os_name(), "cpu_model": _cpu_model(), "cpu_count": os.cpu_count(),
            "mem_gb": _mem_gb(), "python": sys.version.split()[0], "user_is_root": os.geteuid() == 0 if hasattr(os, "geteuid") else None,
            "repo_rev": _git_rev(ROOT), "openvino": ov_info, "npu": npu_check(ov_info, root), "gpu": gpu_check(ov_info, root),
            "sensors": s.describe()}


def text(r):
    ov = r["openvino"]
    L = [f"host {r['host']} · {r['os']} · kernel {r['kernel']} · python {r['python']} · repo {r['repo_rev']}",
         f"CPU  {r['cpu_model']} ({r['cpu_count']} threads), RAM {r['mem_gb']} GB",
         f"OpenVINO {ov.get('version', ov.get('error'))}: devices {', '.join(ov.get('available', [])) or 'none'}"]
    for d, p in ov.get("devices", {}).items():
        L.append(f"  {d:<6} {p.get('FULL_DEVICE_NAME', '?')}")
        for k in KEY_PROPS[1:]:
            if k in p and p[k] not in ("", [], None): L.append(f"         {k:<32} {p[k]}")
    n = r["npu"]
    L.append(f"NPU: {'OK' if n['ok'] else 'NOT usable from OpenVINO'}")
    for st in n["steps"]: L.append(f"  [{'x' if st['ok'] else ' '}] {st['step']}: {st['detail']}")
    if n["packages"]: L.append("  packages: " + ", ".join(f"{k} {v}" for k, v in n["packages"].items()))
    if n["fix"]: L += ["  fix (SO101_SETUP_STATUS.md section 6):"] + [f"    {c}" for c in n["fix"]]
    g = r["gpu"]
    L.append(f"GPU: {'OK' if g['ok'] else 'not listed by OpenVINO'} · kernel driver {', '.join(g['drivers']) or '-'} · render nodes {len(g['render_nodes'])}")
    for c in g.get("fix", []): L.append(f"    {c}")
    s = r["sensors"]
    L.append("Sensors: " + (", ".join(sorted({k.split('.')[0] for k in s['sources']})) or "none readable"))
    for k, why in s["unavailable"].items(): L.append(f"  -- {k}: {why}")
    for f in s["fixes"]: L.append(f"  FIX: {f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the report here")
    ap.add_argument("--root", default="/", help="sysfs root (tests)")
    a = ap.parse_args()
    r = report(a.root)
    print(text(r))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(r, indent=1, default=str))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
