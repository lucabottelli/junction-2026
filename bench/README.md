# bench: Phase 7 measurements (latency, power, temperature on the Core Ultra)

One command turns the trained models and the sorter's real cycles into the evidence the judges ask for
("latency per inference step, thermal headroom, power consumption", "offloading to NPU/iGPU", "evidence, not claims"):
`bench/results/REPORT.md`, the rows for `SO101_SETUP_STATUS.md` section 9 (`bench/results/section9_rows.md`) and charts.

| File | What |
|---|---|
| `devices.py` | OpenVINO version, every device with its full name and key properties, kernel/CPU/RAM, the NPU driver chain with the section 6 fix commands, which sensors are readable |
| `sensors.py` | Background sampler on sysfs: RAPL (package/core/uncore/psys, wrap-around), thermal zones + coretemp (TjMax), CPU freq/utilisation, xe/i915 GPU frequency and busy, intel_vpu NPU busy/frequency, battery power. Idle baseline, "above idle" |
| `bench_model.py` | One IR/ONNX model on CPU/GPU/NPU/AUTO x LATENCY/THROUGHPUT x precision: compile, first inference, p50..p99/max, sync + async throughput, W above idle, mJ per inference, peak temperature, headroom, busy % |
| `quantize.py` | NNCF INT8 post-training quantization from a folder of crops + the accuracy guard (score drift FP vs INT8, decision flips at the calibrated thresholds) |
| `cycle_stats.py` | `runs/*/log.jsonl` -> cycle time and phases, verdicts/accuracy, inference per stage@device, cascade exit share, ACT step latency (and any new `*_ms` field) |
| `report.py` | results + cycle stats -> section 9 rows, REPORT.md with the device-placement recommendation, PNG charts (matplotlib optional) |
| `run_all.py`, `run_all.sh`, `__main__.py` | The one command: discover models, (INT8), bench all, cycle stats, report |

## Station runbook (tomorrow)

All commands from `~/junction26`. `PY` = Studio's venv (has openvino, numpy, opencv):

```bash
cd ~/junction26
PY=~/physical-ai-studio/application/backend/.venv/bin/python
```

1. **NPU driver** (once; SO101_SETUP_STATUS.md section 6), then verify:
   ```bash
   sudo apt install -y libtbb12
   sudo dpkg -i ~/Downloads/npu-driver-v1.38.0/*.deb
   sudo modprobe -r intel_vpu && sudo modprobe intel_vpu   # or reboot
   $PY -m bench.devices                                      # every [x] ticked, "NPU: OK", NPU listed with its full name
   ```
2. **Power counters readable** (after every reboot; RAPL `energy_uj` is root-only since the 2020 PLATYPUS fix):
   ```bash
   sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj
   $PY -m bench.sensors --idle 10           # should list rapl.package-0 (+ core, uncore, psys) and show W at idle
   ```
   Alternative: run the whole bench with `sudo -E $PY -m bench` (results are chowned back to you).
3. **Quiet machine for the idle baseline:** close Studio's camera previews and the control page's browser tab, stop
   `capture.py` if no cycles are running, plug the charger in (or unplug it for one run: then battery power is a
   whole-system cross-check). Leave the lid open and the fan unobstructed: always the same conditions.
4. **INT8 models** (optional but this is what the NPU wants; nncf is in the Anomalib venv):
   ```bash
   ~/anomalib/.venv/bin/python -m bench.quantize models/front/red/weights/openvino/model.xml \
       --calib training_images/front/inhand/good --max-drift 0.05
   # PatchCore: if the guard drifts, keep the memory-bank distance in FP:  --ignore-types MatMul
   ```
   `python -m bench` does this itself for every anomaly model without a `model_int8.xml` (`--quantize no` to skip):
   with nncf if the running venv has it, else through `~/anomalib/.venv/bin/python`. Calibration = the model's
   training crops (split.json), guard = its held-out + defect crops. The sorter keeps loading `model.xml`; the INT8
   file is only benchmarked until you decide to deploy it.
5. **The one command:**
   ```bash
   bench/run_all.sh            # ~15-25 min for a few models: CPU, GPU, NPU, AUTO x LATENCY, THROUGHPUT x default, f32
   bench/run_all.sh --quick    # ~5 min: LATENCY only, shorter windows
   ```
   It finds `models/<cam>/<name>/**/model.xml` (train_anomaly.py) and Studio's policy exports
   `~/.local/share/physicalai/models/<id>/exports/openvino/*.xml`. Single models:
   `$PY -m bench.bench_model path/model.xml --images DIR --devices NPU,GPU`.
6. **Real cycles:** run the sorter (judge = model, cascade on/off, pick = ACT) for 10+ bricks per configuration.
   Then `bench/run_all.sh --report-only` (or `$PY -m bench.cycle_stats`) adds cycle time, phases, accuracy and the
   cascade exit share. For accuracy, write `truth.csv` (`session,cycle,truth` with good/defect) and pass `--truth truth.csv`.
7. Copy `bench/results/section9_rows.md` into section 9 of the notes; `REPORT.md` + the PNGs go into the report.
   `bench/results/` is gitignored except `.gitkeep`: copy what you want to keep into `report_images/`.

## What the numbers mean

- **Latency per step** = one synchronous `infer()` (what the robot loop does), p50 with p95/max; async throughput is
  the parallel capacity. Compile time and first inference are listed separately (they are paid once at start).
- **Power** = RAPL package energy over the timed window / time; **above idle** subtracts the idle baseline, so it is
  what the inference costs. **mJ/inf** = that energy per inference. On client parts `uncore` is the iGPU.
- **Temperature** = peak CPU package temperature during the run, rise from its start, **headroom** to TjMax
  (coretemp `temp*_crit`). The bench waits for the package to cool to idle + 3 C between runs.
- **busy %** for GPU (xe gtidle) and NPU (`npu_busy_time_us`): the proof that a row really ran on that device.
- Dynamic input shapes are reshaped to one static shape (the NPU needs static shapes); anomaly models run at the
  real crop size (their graph resizes inside), so it is the same work the sorter does.

## Tests

`.venv/bin/python -m pytest tests/test_bench.py -q` builds tiny OpenVINO models and a fake sysfs tree and runs the
whole flow on CPU (the INT8 test is skipped when nncf is not importable).
