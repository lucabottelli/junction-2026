#!/usr/bin/env bash
# The one command for the station (Phase 7): device report, idle baseline, every model on CPU/GPU/NPU/AUTO with
# power + temperature, cycle statistics, bench/results/REPORT.md. Extra arguments go to `python -m bench`
# (e.g. --quick, --models a.xml, --report-only). PY = the Python to run with (default: Studio's venv).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/physical-ai-studio/application/backend/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
cd "$HERE/.."
# RAPL energy counters are root-only by default: say so up front (the run still works, without power numbers)
if ls /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1 && ! cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1; then
  echo "NOTE: RAPL power counters are not readable. For power numbers run first:"
  echo "  sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj"
  echo
fi
exec "$PY" -m bench "$@"
