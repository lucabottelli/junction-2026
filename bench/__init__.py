"""Phase 7 measurement framework: device report, power/thermal sampler, OpenVINO model benchmark, INT8
quantization, sorter cycle statistics and the report for SO101_SETUP_STATUS.md section 9.

    python -m bench                 # the one command on the station (= bench/run_all.sh), see bench/README.md
    python -m bench.devices         # environment / NPU driver check only
"""
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent                       # ~/junction26
RESULTS = BENCH / "results"
