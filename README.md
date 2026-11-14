# Junction 2026: SO-101 brick sorting station (Intel Physical AI Challenge)

An SO-101 arm picks Duplo bricks from an orange plate, shows them to a front camera and decides with
anomaly detection (Anomalib) whether a pen-drawn cross is on the brick.

**Start here:** [`SO101_SETUP_STATUS.md`](SO101_SETUP_STATUS.md): every decision, issue and measurement,
the hardware, the plan and the next steps.

## Layout
| Path | What |
|---|---|
| `anomaly_capture/` | The control tool. Control site at `http://127.0.0.1:7870/`: one page per step (connect, waypoints, crop, record, episodes, train, anomaly, inspect & sort) plus the future phases. Front end in `anomaly_capture/web/` (see `web/DESIGN.md`) |
| `anomaly_capture/capture.py` | Server + cameras (2× RealSense D415 + USB wrist cam), locked exposure/white balance |
| `anomaly_capture/arm.py` | Follower/leader control, teach & replay waypoints, safety (STOP, rate limits, blocked detection) |
| `anomaly_capture/collect.py` | Phase 1 (teleop pick + auto carry to WP1, recorded for ACT) and phase 2 (WP2 front-camera images) |
| `anomaly_capture/export_act.py` | Cleans the raw demos (rejects fumbles, trims, de-pauses, speeds up, smooths) and writes a Studio/LeRobot dataset |
| `anomaly_capture/sorter.py` | Phases 3-5: inspect the whole surface of the held object (any shape), stop at the first defect, sort to the good/defect plate; manual judge, Anomalib model or fast→accurate cascade |
| `anomaly_capture/coverage.py` | Shape-agnostic bookkeeping of which surface directions have been seen, and the photo / re-grip plan |
| `anomaly_capture/scorer.py`, `score_worker.py` | Anomaly scoring: the model runs in the Anomalib venv as a persistent worker process |
| `anomaly_capture/train_anomaly.py` | Train + export one anomaly model per brick and camera from the `*_crop.png` images (run with the Anomalib venv) |
| `anomaly_capture/sim_test.py` | Hardware-free test of the inspection logic: `python3 anomaly_capture/sim_test.py` |
| `anomaly_capture/prep.py` | Image preprocessing shared by collection, export and the future runtime (crops, depth background mask) |
| `anomaly_capture/ui.py` | The single web page |
| `anomaly_capture/cam_settings.json`, `routines.json`, `collect_settings.json` | Locked camera settings, taught waypoints, collection settings (station-specific) |
| `scripts/` | Small hardware helper scripts |
| `report_images/` | Images for the report |

## Run (on the station PC)
Needs Intel Physical AI Studio installed at `~/physical-ai-studio` (its Python environment provides lerobot,
pyrealsense2, OpenCV, FastAPI). Close Studio's Environment view (the cameras/arms are exclusive), keep its backend running.

    ~/junction26/anomaly_capture/restart.sh      # then open http://127.0.0.1:7870/

Station-specific values (arm serials, camera serials, wrist camera device) are constants at the top of
`capture.py` and `arm.py`.

## Data (not in git)
`training_images/` (anomaly images) and `episodes/` (raw pick demonstrations) are created by the tool and are
too large for git. Copy them separately (e.g. `rsync -a station:~/junction26/episodes/ ~/junction26/episodes/`)
or publish them as a Hugging Face dataset.
