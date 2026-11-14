# Junction 2026: SO-101 brick sorting station (Intel Physical AI Challenge)

An SO-101 arm picks Duplo bricks from an orange plate, shows them to a front camera and decides with
anomaly detection (Anomalib) whether a pen-drawn cross is on the brick.

**Start here:** [`SO101_SETUP_STATUS.md`](SO101_SETUP_STATUS.md): every decision, issue and measurement,
the hardware, the plan and the next steps.

## Layout
| Path | What |
|---|---|
| `anomaly_capture/` | The control tool. One page at `http://127.0.0.1:7870/arm`: setup, teach waypoints, phase 1+2 data collection, export to Studio, future phases |
| `anomaly_capture/capture.py` | Server + cameras (2× RealSense D415 + USB wrist cam), locked exposure/white balance |
| `anomaly_capture/arm.py` | Follower/leader control, teach & replay waypoints, safety (STOP, rate limits, blocked detection) |
| `anomaly_capture/collect.py` | Phase 1 (teleop pick + auto carry to WP1, recorded for ACT) and phase 2 (WP2 front-camera images) |
| `anomaly_capture/export_act.py` | Cleans the raw demos (rejects fumbles, trims, de-pauses, speeds up, smooths) and writes a Studio/LeRobot dataset |
| `anomaly_capture/prep.py` | Image preprocessing shared by collection, export and the future runtime (crops, depth background mask) |
| `anomaly_capture/ui.py` | The single web page |
| `anomaly_capture/cam_settings.json`, `routines.json`, `collect_settings.json` | Locked camera settings, taught waypoints, collection settings (station-specific) |
| `scripts/` | Small hardware helper scripts |
| `report_images/` | Images for the report |

## Run (on the station PC)
Needs Intel Physical AI Studio installed at `~/physical-ai-studio` (its Python environment provides lerobot,
pyrealsense2, OpenCV, FastAPI). Close Studio's Environment view (the cameras/arms are exclusive), keep its backend running.

    ~/junction26/anomaly_capture/restart.sh      # then open http://127.0.0.1:7870/arm

Station-specific values (arm serials, camera serials, wrist camera device) are constants at the top of
`capture.py` and `arm.py`.

## Data (not in git)
`training_images/` (anomaly images) and `episodes/` (raw pick demonstrations) are created by the tool and are
too large for git. Copy them separately (e.g. `rsync -a station:~/junction26/episodes/ ~/junction26/episodes/`)
or publish them as a Hugging Face dataset.
