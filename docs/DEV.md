# Developing without the station: mock mode and tests

The control tool (`anomaly_capture/capture.py`, port 7870) normally needs 2 RealSense D415, the USB wrist camera
and both Feetech arms. **Mock mode** runs the same server with simulated hardware, so you can click through the
whole site, develop UI and logic, and run the test suite on any laptop or in CI.

## Quick start

```bash
make venv              # .venv from requirements-dev.txt (no pyrealsense2 / lerobot / Studio needed)
make mock              # http://127.0.0.1:7871/   (PORT=7880 make mock for another port)
make test              # fast tests, ~1-2 min
make test-flows        # end-to-end flows on the mock (collect, sort, STOP/recovery), ~10 min
make test-ui           # browser smoke test, screenshots in tests/screenshots/
```

Without make: `JUNCTION_MOCK=1 JUNCTION_PORT=7871 .venv/bin/python anomaly_capture/capture.py` (or `--mock`).
Every page shows a "Mock station" badge.

| env var | meaning |
|---|---|
| `JUNCTION_MOCK=1` | simulated hardware (also `--mock`) |
| `JUNCTION_PORT` | server port (default 7870, also on the station) |
| `JUNCTION_DATA_ROOT` | where the mock writes everything (default `<repo>/mock_data/`; the repo itself is refused) |
| `JUNCTION_MOCK_CAMS=top,front` | start with the wrist camera "unplugged" |
| `JUNCTION_MOCK_ANOMALIB=real` | keep a real `ANOMALIB_PY` instead of the fake scorer |

## Data isolation

In mock mode nothing touches the station's files. Everything goes to the data root: `training_images/`,
`episodes/`, `runs/`, `models/` and `settings/` (`routines.json`, `cam_settings.json`, `collect_settings.json`,
`sort_settings.json`, `inspect_settings.json`). `GET /mock/paths` lists every write location. The root is
seeded once with **pre-taught waypoints** for every slot (home, wp1, drops, wp2_above, wp2 with front/back/down
poses, tape_above, tape_place, sort_good(_above), sort_defect(_above)), the WP2 face ticks and mock models for
`front` and `wrist`, so every flow runs immediately. Delete the folder to start over.

## What the mock simulates (`anomaly_capture/mock_hw.py`)

- **Servo bus**: a drop-in for lerobot's `FeetechMotorsBus` (arm.py loads it in mock mode; the real `Arm` class
  runs unchanged). Each servo moves with a speed limit and a small lag, so moves take time and STOP mid-move
  works. Torque on/off (limp), EEPROM limits, a **gripper that stalls on a brick** (so grasp detection and
  `holding()` work), an optional obstacle (BLOCKED detection), injected bus errors.
- **Scene and cameras**: a small 3D scene (table, orange input plate, tape spot, good and defect plates, the arm
  from simple kinematics, bricks with an optional black pen cross on one face) rendered at 1920x1080, ~30 fps for
  top, front and wrist. The front camera also gives aligned depth in mm, so at WP2 `prep.inhand_mask` isolates
  the gripper and brick. Bricks are gripped, carried, released and land on the plate, tape or a sort plate.
- **The human**: in `auto` mode the leader follows the follower while not mirroring (so MATCH succeeds), picks
  the brick off the input plate during a recorded pick (and keeps squeezing), and hand-feeds a brick into a wide
  open gripper held high (the sorter's hand-feed at WP1). You still press G / judge C-D-U / N yourself.
- **Anomaly models**: a fake `anomalib.deploy` package that the real `score_worker.py` imports. Its score is
  the amount of pen ink in the crop: about 0.05 when clean, about 0.95 when a cross is visible.

### `/mock/*` API (drive the fake world from curl, tests or a browser console)

| route | does |
|---|---|
| `GET /mock/state` | bricks, ground truth (`truth.front/wrist/top`: is the cross visible now?), joints, faults, events |
| `POST /mock/scene?defect=%2Bx&size=2x2&color=red` | new brick on the input plate (`+x -x +y -y +z -z`, empty = clean); `queue=+x,,-z` = defects of the next bricks; `empty=1` = no brick (for the B background) |
| `POST /mock/leader?x=0.18&y=0&z=0.1` or `?shoulder_pan=...` | move the leader yourself (human mode becomes `manual`) |
| `POST /mock/human?mode=auto\|manual\|off&feed=0\|1` | the scripted human |
| `POST /mock/fault?obstacle=1&drop=1&regrip_fail=2&bus_errors=20&unplug=leader` | failures |
| `POST /mock/camera?cam=wrist&state=frozen\|offline\|ok&depth=0` | camera failures, depth off |
| `POST /mock/reset`, `GET /mock/ik?x=&y=&z=`, `GET /mock/paths` | reset, inverse kinematics, write locations |

Brick faces are in the gripper frame of `coverage.py` for a hand-fed brick: `+x` shows at wp2-1 (front), `-x`
at wp2-2 (back), `-z` at wp2-3 (down), `±y` lie under the fingers (seen after the tape re-grip), `+z` is the top
(the release photo).

### Not simulated

Real optics, noise, exposure and lighting; real servo load, sag, temperature and calibration; collisions and
physics beyond grip / carry / release / fall (no tipping, no slipping, apart from the injected faults); people in
the background; the real geometry of the station (the mock layout is its own, so mock waypoints are not station
waypoints); Studio / LeRobot (`EXPORT to Studio` fails cleanly, the dry-run preview works); real Anomalib
models and training (`train_anomaly.py`); the ACT policy. Timing is real-time (moves take as long as on the arm).

## Tests (`tests/`)

| file | what | marker |
|---|---|---|
| `test_mock_hw.py` | kinematics, waypoints, rendering, depth mask, fake scores, servo dynamics, data-root guard, the real `score_worker.py` with the fake anomalib | fast |
| `test_api_sweep.py` | every route (from the live OpenAPI schema) answers sanely, every page and static file, 422 not 500 | fast |
| `test_api_mock.py` | write locations, cameras, stills (B / G / U), ROIs and configs, arm modes, routine editing, Go + STOP mid-move, BLOCKED, bus errors, teaching with the leader, camera freeze and reconnect | fast |
| `test_mock_simtest.py` | `anomaly_capture/sim_test.py` (on a copy, see below) | fast |
| `test_mock_flows_collect.py` | record episodes, keep/untick, export preview, phase 2 photos + tape re-grip, test buttons, discard + NEXT, STOP mid-pick and recovery | slow |
| `test_mock_flows_sort.py` | inspect & sort: manual and model judge, clean / front / finger-side / top defects, STOP mid-cycle, dropped brick, failed re-grip | slow |
| `test_regrip.py` | tape re-grip: computed vs taught `tape_regrip` pose, roll limits, a pose not turned ~90° refused, retry + reasons, contact detection, STOP at every move, step-by-step test (NEXT, timeout, STOP) and its `runs/regrip_test_*` files, coverage with a taught pose either way; the same on the mock station | fast + slow |
| `test_ui_smoke.py` + `ui_smoke.mjs` | Chromium loads every page, fails on console errors, exceptions, failed requests or HTTP errors, clicks the main buttons of each step, saves screenshots | ui |

`tests/conftest.py` starts one mock server per test module (free port, temp data root). The UI smoke test needs
Node with the global `playwright` package and Chromium. CI (`.github/workflows/ci.yml`) runs sim_test, the fast
tests and the flows on every push / PR, with the UI smoke test as a separate job that may fail.
