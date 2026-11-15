# Junction 2026: SO-101 Sorting Station Project Notes

> **Project folder: `~/junction26/`** (everything for the submission is here):
> - `SO101_SETUP_STATUS.md`: these notes
> - `anomaly_capture/`: the capture tool and the locked camera settings
> - `training_images/`: all anomaly training and evaluation images
> - `scripts/`: helper scripts (frame grab, scale measurement, camera probes, follower torque-off)
> - `report_images/`: camera views and measurements for the report
>
> Not moved, because moving them would break their venvs: `~/physical-ai-studio` (Studio) and `~/anomalib`. Studio's data stays in `~/.local/share/physicalai/`.

Running log of the setup, the decisions we made and the problems we fixed. It's written so it can feed straight into the final report.

Last updated: 2026-11-14 night (overnight build, see §7.0 and decisions 23–26)
Challenge guide: https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md

---

## 1. Status at a glance

| Guide step | Status | Notes |
|---|---|---|
| 1. Set up the station | 🟡 Mostly done | Studio runs, arms are calibrated, teleop works. Still open: mount the overhead camera and add the Hugging Face token. |
| 2. Modal for GPU training | ❌ Dropped | Modal requires a payment method for GPU use and declined every card we tried. We train locally on the Intel GPU instead (decision 5). |
| 3. Configure Studio | ✅ Done | Project, 2 robots, 3 cameras, and a "test" environment with teleop |
| 4. Anomaly detection (Anomalib) | 🟡 Capturing | Capture tool built (decisions 12–13); backgrounds set; the next job is collecting about 150 good and 20–30 defect images. |
| 5. Record demos and train a policy | ⏳ Next | 0 datasets so far. Plan: ACT trained locally on the Intel GPU. |
| 6. Close the loop | 🟡 Built, untested on the arm | ACT pick → WP1 → inspect → sort runs end to end on the mock station (decision 23). Needs a trained ACT export and a first slow run on the real arm. |
| 7. Optimize on Intel hardware | 🟡 Harness ready, NPU driver missing | `bench/run_all.sh` measures latency, power and temperature per device and writes the §9 rows (decision 24). Run on CPU only so far. NPU driver still to install (§6). |

---

## 2. Hardware

**AI PC**
- CPU: Intel Core Ultra X9 388H (Panther Lake)
- iGPU: Intel Graphics [0xb080]. The XPU reports 57.2 GiB of shared memory available.
- NPU: Panther Lake NPU [8086:b03e] at PCI `00:0b.0`, kernel driver `intel_vpu`, `/dev/accel0`
- RAM: 61 GB
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic

**Arms: SO-101 leader/follower pair** (Feetech STS3215 servos, CH343 USB serial)

| Role | Serial | Studio robot | Notes |
|---|---|---|---|
| Leader | 5B14111392 | "SO101 Leader" `705fe5df…` | Teleop-verified 2026-11-13 |
| Follower | 5B3D040988 | "SO101 Follower" `9070e8b8…` | Studio drives this arm; its torque is on during teleop |

Both arms are calibrated in Studio (per-joint range_min/max and homing offsets are stored).

**Cameras** (all 640×480 at 30 fps, color only)

| Studio name | Hardware | Identity | Placement / status |
|---|---|---|---|
| `wrist` | USB2.0_CAM1 (Sonix), `usb_camera` driver | Matched by bus/serial (was `/dev/video4`, now `/dev/video16`; Studio still finds it) | On the follower's gripper; looks along the gripper. OK. |
| `front` | Intel RealSense D415, `realsense` driver | serial `243622061066` | On a mini tripod at the table edge opposite the robot, facing it head-on, about 20° down (re-verified 2026-11-14 ≈01:20). Sees the brick's top and studs, both plates and the follower's base; people in the background. A corner position, 30–40 cm up and 30–40° down, would be better but needs a taller mount. Accepted as usable. |
| `top` | Intel RealSense D415, `realsense` driver | serial `243622063627` | **Adjusted 2026-11-14 ≈01:00: nearly top-down; clutter cleared by ≈01:20** (the right laptop is gone; the left laptop is at the top-left corner; the front camera's tripod is at the bottom edge). The inspection spot is near the center, both plates are in view, and the gripper is at the top. **Measured 2026-11-14 at 1920×1080:** the 2×2 Duplo top (31.8 mm) is about 66 px, so **≈0.48 mm/px**. Intrinsics fx≈1367, so the lens is ≈66 cm above the brick (≈70 cm above the table). At this scale a 10 mm cross is ≈20 px, 5 mm ≈10 px, 3 mm ≈6 px and 2 mm ≈4 px; pen lines are only 1–2 px, and the 1080p image is somewhat soft. The overhead alone is good to about 3–5 mm crosses, so **the smallest crosses must come from the wrist close-up (cascade stage 3)**. The 1080p (16:9) view is wider than 640×480 and shows the laptops at the edges; that doesn't matter for crops. Exposure and white balance still need locking before capturing training images. The follower's base is just above the frame. |
| (unused) | HP 5MP laptop webcam | `/dev/video0` | Not useful for manipulation |

The user swapped the `front`/`top` names in the UI after the cameras were first added; the table shows the current mapping.

---

## 3. Software environment

| Component | Version / location |
|---|---|
| Physical AI Studio | `~/physical-ai-studio`, synced to the **`junction2026`** branch (commit 17714488) on 2026-11-14 00:10. Not a git checkout. |
| Studio backend venv | `application/backend/.venv`: Python 3.13, torch 2.11.0+xpu, OpenVINO 2026.4.1, modal 1.6.3, installed with `uv sync --extra xpu` |
| Studio UI | Node v24.22.3 (nvm), rsbuild dev server |
| Anomalib | `~/anomalib/.venv`: anomalib 2.7.1.dev0, XPU OK, OpenVINO 2026.4.1. A separate venv, so its torch can't clash with Studio's. |
| Studio data | `~/.local/share/physicalai/` holds the DB, datasets, models and logs, and survives code upgrades |
| Modal CLI | Login in `~/.modal.toml` (chmod 600), workspace `lucabottelli` |
| Backup of the pre-upgrade Studio | `~/physical-ai-studio-backup-20261114.tgz` |
| NPU driver packages (not installed yet) | `~/Downloads/npu-driver-v1.38.0/` |

**How to run Studio**
```bash
# Backend: API on http://127.0.0.1:7860
cd ~/physical-ai-studio/application/backend
uv run --no-sync physicalai-studio serve >> ~/.local/share/physicalai/logs/studio-backend.log 2>&1

# UI: http://localhost:3000
cd ~/physical-ai-studio/application/ui
npx rsbuild dev >> ~/.local/share/physicalai/logs/studio-ui.log 2>&1
```
- Project: **SO101 Leader-Follower**, id `a0877272-139c-4c5b-925f-9dc7dd2de05e`
- Environment: **test**, id `8e819fb6-e82c-4e8d-a96d-3c490a292c6a`: follower driven by teleop from the leader, plus cameras front, top and wrist
- Useful API calls: `GET /api/hardware/cameras`, `GET|POST /api/projects/<id>/cameras`, `GET /api/projects/<id>/robots`, `GET|PUT /api/projects/<id>/environments/<env_id>`, `GET /api/remote-trainers/<id>/health`, `GET /api/settings`
- After upgrading the code, regenerate the UI's API types: `npm run build:api:download && npm run build:api`.

---

## 4. Decisions and why we made them

1. **We run Studio natively, not in Docker.** The guide recommends it, and running natively gives direct access to the USB serial ports, the RealSense cameras and the Intel XPU.
2. **We upgraded Studio to the `junction2026` branch.** The copy installed on the laptop was older and had no Modal training option. Before upgrading we compared the two versions: the DB migrations were identical, so the project, robot and calibration data carried over without needing a migration.
3. **We use three cameras: wrist, overhead and front.** That's what the guide asks for. The wrist camera gives a close-up for grasping. The overhead camera gives a top-down view of the workspace for inspection and for robustness to shifted objects. The front camera adds depth cues.
   - *Planned:* use the overhead D415 as the Anomalib inspection camera.
4. **We use color only on the D415s for now.** Depth makes recording heavier. Turning it on (`output_type: both`) is optional later.
5. **All training runs locally on the Intel GPU (XPU). Modal is dropped** (2026-11-14), because Modal declined every card we tried.
   - Studio's training devices are `cpu` and `xpu` (Intel Graphics, about 57 GiB shared). Inference devices are torch CPU/XPU and OpenVINO CPU/GPU.
   - Minimum memory per policy, according to Studio: ACT 2 GiB, SmolVLA 3 GiB, Pi0.5 40 GiB, RLDX-1 40 GiB, xr0 60 GiB, MolmoAct2 80 GiB.
   - **Primary policy: ACT.** It's fast to train, which makes it a reliable baseline.
   - **Stretch policy: SmolVLA, trained locally.** It fits in memory but trains slower. It takes a **language task**, so the anomaly result can steer it directly (for example, "put the defective brick in the reject bin"). It needs the Hugging Face token to download its weights.
   - Pi0.5 and the larger models aren't practical on this laptop.
   - *For the report:* training happens entirely on Intel hardware, which counts toward "Intel Tech Integration" (25 pts).
6. **Anomaly models train locally on the XPU** (PaDiM or PatchCore first). The guide says Modal isn't needed for these.
7. **We match devices by serial number, not by `/dev` name.** `ttyACM*` and `video*` numbers change when you replug, while serial numbers don't.

8. **Anomaly model: PatchCore is the main model, with EfficientAd as the second one for the NPU** (2026-11-14). The defect is a **pen-drawn cross** on Duplo bricks, and **smaller detected defects score more**.
   - Installed Anomalib 2.7.1 has every candidate model: Patchcore, Padim, EfficientAd, Dinomaly, FoundAD, and more.
   - **Why PatchCore:** it compares each small image patch with a memory of "good" patches, so it's very good at small local texture changes like a pen cross. It doesn't depend on position, so shifted bricks are fine. It trains in about 3–5 minutes from about 50–200 good images, with no gradient training.
   - **Why EfficientAd as well:** it's very good with fine detail and has the fastest inference. It's a plain convolutional network, so it suits the NPU. That gives a GPU-vs-NPU comparison for Step 7.
   - **Ruled out:** PaDiM, because it models what's normal at each position, so it breaks when bricks move, and it's coarse for small defects. Dinomaly, because it's too heavy. FoundAD stays an optional few-shot experiment.
   - **What matters more than the model choice:**
     - Pixels on the brick: crop each brick and resize the crop to 256×256, raise the overhead camera's resolution, and use the wrist camera for close-ups.
     - Fixed lighting with exposure locked.
     - A complete set of "good" images: all brick colors, orientations and positions.
     - A test ladder of cross sizes (10, 5, 3 and 2 mm) to find the smallest size detected reliably. That result goes in the report.

9. **Anomaly workflow: accuracy first, speed second. Every change must earn its place on a held-out test set** (2026-11-14).
   - **The metric that matters:** the smallest cross size detected reliably at a fixed false-alarm rate. Report image AUROC too.
   - **Accuracy levers, in this order:**
     1. Pixels on the brick: fixed inspection spot, fixed lighting, per-brick crop.
     2. PatchCore at a crop size of 256, then 384 or 512.
     3. PatchCore backbone and layer choice.
     4. **Tiled ensemble** (Anomalib `pipelines/tiled_ensemble`; Rolih et al., CVPRW 2024): one model per tile, which helps most when defects are small.
     5. Model ensemble (PatchCore + EfficientAd ± Dinomaly), with each model's scores normalized on validation data before fusing.
     6. Supervised option: we can draw crosses ourselves to create labeled defect examples, which makes SuperSimpleNet (supervised mode) worth trying.
   - **Why the model ensemble isn't first:**
     - It only helps when the models make *different* mistakes.
     - Latency multiplies by the number of models.
     - Fusing scores needs per-model normalization, tuned on a validation set.
     - The literature's evidence for small defects favors **resolution and tiling**.
   - **Then optimize:** export the winner to OpenVINO, quantize to INT8 with NNCF, and spread it across devices. Check whether a single model comes close to the ensemble's accuracy, and measure everything (§9).
   - **Sources:** tiled ensemble https://arxiv.org/abs/2403.04932 · EfficientAD https://arxiv.org/abs/2303.14535 · PatchCore https://arxiv.org/abs/2106.08265 · PatchCore few/many-shot and resolution https://arxiv.org/abs/2307.10792

10. **Adaptive cascade: "spend compute only on hard bricks"** (user's idea, 2026-11-14). This is our main *Creativity* point, and it also serves *System Efficiency*.
    - **Stage 1, fast:** EfficientAd INT8 on the **NPU**, scoring the overhead crop.
      - Score below `T1_low`: **GOOD**, exit.
      - Score above `T1_high`: **DEFECT**, exit.
      - Anything in between goes to stage 2.
    - **Stage 2, accurate:** PatchCore at a high-resolution crop (or the tiled ensemble) on the **iGPU**, with its own two thresholds.
    - **Stage 3, active perception:** if stage 2 is still unsure, the arm moves the **wrist camera** close to the brick and re-scores the close-up. That gives more pixels on the defect, not just more compute. If it's still unsure, reject the brick (fail safe).
    - **Calibration is the key risk.**
      - A GOOD exit at stage 1 must never miss a defect, so set `T1_low` from the validation defect ladder (including the smallest crosses) to give about 100% defect recall. Stage 1 then only clears bricks it's sure are clean.
      - Set `T1_high` for about 0% false alarms.
      - Each stage gets its own thresholds; scores aren't compared across models.
    - **What to report:**
      - Exit rate per stage, mean and p95 latency per brick, and energy per brick.
      - Accuracy on the ladder, compared with running the heavy model on every brick.
      - The target is the same accuracy at a fraction of the average latency.
    - **Background:** classic cascade classifiers (Viola–Jones) and early-exit / adaptive inference. Coarse-to-fine inspection.

11. **Resolution policy** (2026-11-14).
    - **Anomaly data:** always capture at full resolution (overhead 1920×1080, wrist at its maximum) and keep the raw crops. Images can be shrunk later, but lost detail can't be recovered.
    - **Train each anomaly model at exactly the scale it runs at, with one model per cascade stage:**
      - Stage 1: EfficientAd on overhead crops shrunk to 128–256 px.
      - Stage 2: PatchCore on full-detail overhead crops (≈66 px of brick, padded and enlarged to 256).
      - Stage 3: PatchCore on wrist close-ups, with its own set of "good" images.
    - **Don't mix resolutions in one model.** It widens what counts as "normal" and makes small crosses harder to catch. Up to about ±10% random rescaling during training is fine.
    - **Policy (ACT):** record in Studio at 640×480, 30 fps, on all cameras, and keep exactly that at deployment. ACT shrinks images internally anyway, so higher resolution only costs storage, speed and dropped frames.
    - **Closed loop:** one 1080p overhead stream feeds both. Crop at full resolution for the anomaly models, and shrink the frame to 640×480 for the policy.

12. **Anomaly capture tool and dataset format** (2026-11-14 ≈01:20). Tool: `~/junction26/anomaly_capture/capture.py`, a browser page on `http://127.0.0.1:7870`.
    - **Why a web page:** OpenCV has no window support in either venv (`GUI: NONE`), so the preview and buttons run in a FastAPI page served from Studio's venv.
    - **Locked camera settings:** the overhead D415 has auto exposure and auto white balance turned off. A one-time calibration matches the auto image's brightness and colour, then freezes the values in `~/junction26/anomaly_capture/cam_settings.json`. That file is reused in every session and **must be reused in the closed loop too**, so inference sees the same image statistics as training. The wrist camera locks its auto-settled values (best effort, through V4L2).
      - The calibration landed at **exposure 330 (33 ms, the maximum at 30 fps), gain 128 (maximum), white balance 3250 K**, because the room is dim. Maximum gain means more sensor noise, which hurts tiny-cross detection.
      - **The current light is fixed for the whole event** (team decision, 2026-11-14): no extra lamps, so noise has to be handled in software. That's why the overhead averages **16 frames** per image (≈0.5 s still time, ≈4× less noise) and the wrist averages 8. The closed loop must use the same averaging. The wrist close-up (stage 3) does the most for the smallest crosses.
    - **Frame averaging per saved image** (16 on the overhead, 8 on the wrist): the brick is still, so averaging N frames cuts sensor noise by about √N. The closed loop must average the same number of frames.
    - **Brick detection:** the difference from a saved photo of the empty workspace, so it works for any brick colour; saturation-based detection is the fallback. Only blobs inside a workspace rectangle count (`ROI`, full-res `(340,220)–(1540,960)` on top), with area and side limits (30–200 px on top).
    - **Saving:**
      - Crops are lossless PNG: a square around the brick with 25% padding, about 114 px on the overhead camera. JPEG would smear 1–2 px pen lines.
      - The full raw frame is saved too, so crops can be redone later.
      - Every crop gets a line in `meta.jsonl` with its bbox, camera settings and time.
      - Defect shots are refused unless exactly one brick is in view, so labels can't be wrong.
    - **v2 of the tool** (2026-11-14 ≈01:30, after the first test run):
      - **The detection area is now a tight box around the inspection spot,** full-res `(712,278)–(1132,638)`. It used to cover most of the table and caught cables and clutter. You can redraw it by dragging on the preview. It's saved per camera in `cam_settings.json` under `roi`, so the closed loop uses the same area.
      - **Live full-res zoom of the detected brick,** so you can see the cross before saving.
      - **Motion check:** a shot is refused if the brick moved while the 16 frames were being averaged, because averaging would blur it.
      - **Optional brick id** stored with each shot, so the same defect brick can be tracked across positions. This avoids leakage between training and test sets when we split.
      - **Per-crop sharpness** in `meta.jsonl`.
      - **Progress bars** against the target counts (150 good and 20 per defect size, per camera).
      - **Thumbnails of the last 6 saves.**
      - A single detection pass at about 10 Hz is shared by the preview and the zoom.
    - **v3: multi-camera capture** (2026-11-14 ≈01:45):
      - All three cameras (top, front, wrist) stream side by side.
      - **One key press saves from every ticked camera in the same averaging window.** The crops share a `group` id in `meta.jsonl`, so we have paired multi-view samples (useful for multi-view fusion and for comparing cameras in the report).
      - Each camera has its own detection area (drag to change), empty background and live zoom.
      - A camera with no valid brick (none found, several during a defect shot, or motion) is skipped and reported. The other cameras still save.
      - **The front D415 is now locked too:** exposure 313, gain 96, white balance 4750 K, saved in `cam_settings.json`. It averages 16 frames, like the overhead.
      - **Detection fixes:**
        - The no-background fallback is now "colour differs from the table's median colour inside the area". The old saturation rule failed because the warm-lit wood counts as saturated on the front and wrist cameras.
        - Blobs touching the area's edge are rejected, because they were plates cut off by the box.
        - The blob closest to the area's centre is the primary one.
        - Detection is restricted to the area, so a brick can't merge with something outside it.
      - **Default areas** (full-res):
        - top `(712,278)–(1132,638)`
        - front `(840,240)–(1380,780)`
        - wrist `(300,150)–(1620,950)`
      - **The wrist background is only valid for one arm pose.** Capture wrist images with the follower in a fixed "inspection pose" above the spot, and press B for that pose. The closed loop must use the same pose.
    - **v4: guided auto-capture** (2026-11-14 ≈01:55). The user asked for a set number of shots, with a clear signal for when to move the brick.
      - Set N and press "start GOOD" or "start DEFECT".
      - A big banner shows: PLACE THE BRICK → HOLD STILL → (beep) ✓ SHOT TAKEN: MOVE / ROTATE → … → DONE (N/N).
      - **A shot fires only when** the overhead view has been still for 0.8 s and the brick looks different from the previous shot: ≥15% of the brick's pixels changed, so a brick that hasn't moved never gets duplicated.
      - It uses the same save path as manual G and D, with every check (the defect one-brick rule, the motion check, and skipping a camera without a brick).
      - Esc stops it. Manual G and D still work.
      - **B is refused if a brick is in the area** (issue 9). Shift+B forces it.
{top,front,wrist}/{good,defect}/` plus `raw/` and `background.png`. This is the folder format Anomalib expects: train on `good`, and validate and calibrate on the `defect_*` ladder.

13. **No size labels on defects; defects are for evaluation only** (2026-11-14 ≈02:00). The user pointed out that we don't know the cross sizes and asked whether defect images are needed at all.
    - **Training uses good images only.** PatchCore and EfficientAd are one-class models: they learn "normal" and flag anything that differs. So no defect labels are needed for training, and none go into the training set.
    - **We still collect about 20–30 defect shots,** for three reasons:
      1. To **set the threshold.** With good images only, the threshold would be a guess (for example a percentile of held-out good scores). That risks either missed crosses or false alarms.
      2. To **verify detection before the demo.**
      3. To **measure the smallest cross detected**, for the report and the score.
    - **Sizes are measured afterwards from the pixels** (overhead ≈0.48 mm/px), not labelled at capture. So the tool's four size keys (10/5/3/2 mm) became a single **D = defect** key, saved to `defect/`. Draw crosses of any size and mix big and small ones.
    - The fixed 10/5/3/2 mm ladder (decision 10) becomes "the measured size of each defect brick", and the report shows detection rate against measured size.
    - Targets per camera: 150 good, 30 defect.

14. **Inspect the brick in the gripper; the robot collects its own training data** (2026-11-14 ≈02:30).
    - **Inspection flow:**
      - **Plates:** the orange plate is the input; bricks are thrown on it at random, and the cross can be on any of the 6 faces.
      - **Gripper hides 2 faces:** the gripper always covers two faces, so the flow is:
        1. The top camera checks the face pointing up. This is free, since that camera already locates the brick.
        2. Pick the brick and show it to the front camera at fixed poses: 2 free sides plus the bottom.
        3. Only if it's still clean: set it down turned 90° at a fixed spot, pick it up again, and show the remaining 2 faces.
        4. Drop it in the good or defect plate.
      - **Stop early:** a defect found at any step ends the inspection, so about 4 in 6 defective bricks skip the regrasp.
      - **Rejected: drop-and-rotate.** Where the brick lands is random, so it has to be found and picked again, and the views never repeat.
    - **Why holding the brick helps:**
      - Same pose every time, so the views are well aligned for PatchCore.
      - Close to the camera, so more pixels per mm, which helps detect small crosses (worth the most points).
      - The gripper fingers look the same in every image, so they count as normal.
      - **Risk:** people moving behind the held brick. Fix: a plain matte card behind the inspection pose, plus a tight crop on the brick's colour.
    - **Motion:**
      - **Fixed moves:** inspection poses, set-down and bins are always at the same places, so they're taught joint waypoints that are replayed, with no learning needed.
      - **Variable move:** only the pick from the random pile needs to handle variation (ACT, or the top camera locating the brick plus aligning the wrist roll).
    - **Training data rule: capture with exactly the moves used at sorting time.** Tool: `anomaly_capture/arm.py`, page `http://127.0.0.1:7870/arm`, in the same process as the capture tool so it shares the locked cameras.
      - **Teach:** the follower mirrors the leader, or goes limp to be guided by hand. Each click adds a waypoint; 📷 marks a capture pose.
      - **Run:** replays the routine N times with smooth (minimum-jerk), speed-limited motion. At every 📷 pose it waits 0.25 s, then saves a 16-frame averaged full-res frame to `training_images/<cam>/inhand/<label>/`. Each meta line gets `stage: inhand`, view, cycle, run and joint positions.
      - **Variation:** a routine that ends by putting the brick back on the pick spot loops unattended. "Wait for me between cycles" lets you nudge the brick for grip variety; optional ±N° jitter on the 📷 poses.
      - **Safety:**
        - Space/Esc/STOP freezes the arm in place, with torque kept on.
        - Every command is rate-limited.
        - A move still more than 14° off after settling is reported as BLOCKED and the routine stops.
        - The gripper uses LeRobot's 50 % torque limit, and the arm counts as arrived once the gripper stalls on the brick.
        - Mirroring creeps at about 22°/s until the follower catches up with the leader.
      - **Positions are raw servo ticks of the follower,** so no calibration file is needed. Mirroring uses LeRobot/Studio's "degrees" convention: offset from the midpoint of each joint's EEPROM range, and the gripper as a fraction of its range.
      - **Tested:**
        - With a simulated bus: moves, gripper stall, STOP, and a 2-cycle run that saves files and meta.
        - On the real follower: connect, hold with 0 drift and 0 bus errors over 2 s, then limp.
        - Read-only leader→follower mapping at rest: the elbow differs by 63°, everything else by 15° or less.
    - **Models:** one PatchCore per brick per camera (top and front); the brick's colour selects the model.
      - The top model trains on the existing top-view shots on the plate (manual tool).
      - Front and wrist shots of bricks sitting on a plate are obsolete for the in-hand design.
    - **Order:** capture all good data before drawing any cross (the pen is permanent). Then draw crosses from the smallest up and run the same routine with label = defect.

15. **One-step collection: a human teleoperates only the pick, the robot does the rest; demos are cleaned before ACT** (2026-11-14 ≈02:45).
    - **Page and code:**
      - Page: `http://127.0.0.1:7870/collect`.
      - Collector: `anomaly_capture/collect.py`. Cleaning and export: `export_act.py`. Shared preprocessing: `prep.py`.
    - **Each episode, start to finish:**
      1. Robot goes to the taught `home` pose.
      2. You move the leader to match it (gap shown per joint; it starts when every joint is within 8°, so there's no jump).
      3. You teleoperate the pick. It records at 30 fps: top, wrist and front images at 640×480, follower state, commanded action and leader position.
      4. The hand-off is automatic once the gripper is commanded at least 40 ticks more closed than it actually is, has stopped moving and isn't fully shut, for 0.3 s. G forces it; X discards the episode.
      5. The robot lifts, checks the brick is still in the gripper (otherwise the episode is marked `missed`), and runs the taught `inspect` routine, saving images at each 📷 pose.
      6. It drops the brick at a random taught `drops` pose with a random ±45° wrist turn, so the next pick starts from a new position and orientation. Then it goes home and the next episode starts.
    - **Why ACT learns only the pick:**
      - The fixed moves (lift, inspection poses, set-down, regrasp, drop) are replayed exactly. That's faster and repeatable, which matters for the anomaly views to line up.
      - The policy's capacity goes to the only variable part.
      - At run time the same stall-based hand-off switches from ACT to the scripted routine.
    - **Removing the human's slowness and mistakes (export_act.py; raw episodes are never modified):**
      - **Rejected:**
        - failed, missed, discarded or timed-out picks;
        - episodes you unticked;
        - more grasp attempts than allowed (default 1);
        - slower than 2.5× the median pick;
        - a camera stalled for more than 0.5 s.
      - **Cleaned:**
        - Idle time at the start is trimmed.
        - Commanded targets are smoothed (Gaussian σ = 1.5 frames, gripper untouched).
        - The timeline is shortened: pauses compressed 10×, free-space motion sped up 1.5×, the last 1.0 s before the gripper closes kept at full rate, and no joint moves more than 6° per step.
        - Frames are *selected* from the real recording, so images and states stay genuine.
    - **Studio format:**
      - It's written through Studio's own `RecordingMutation`, with the same features as Studio's recorder.
      - Features: `observation.state` and `action` with `<joint>.pos` names in Studio's "normalized" units (follower calibration from EEPROM, which matches Studio's robot config), plus `observation.images.top` and `.wrist` as 480×640 video; robot type `SO101_Follower`.
      - The dataset is created via `POST /api/dataset`.
      - Mirroring was switched to the same normalized convention, so the recorded actions mean exactly what Studio expects.
    - **Background movement (the cameras are fixed, people walk past):**
      - **ACT inputs:** only the top camera, cropped to a 4:3 table area you drag on `/collect` (people stand outside it), plus the wrist camera. The front camera is excluded from the policy.
      - **Anomaly images:** the front D415 now also streams depth, aligned to colour at about 10 Hz. `prep.inhand_mask` keeps the nearest depth cluster (gripper and brick, ±70 mm), fills holes so the pen cross stays inside the mask, and turns the rest flat grey. Each 📷 capture also saves `_depth.png` and `_masked.jpg`.
      - The run-time code must use the same `prep.py` functions.
    - **Waypoint fix:** a taught pose now stores the *commanded* gripper value, not the actual one. While holding a brick the actual value is the brick's width, and replaying that would hold it with zero force. Mode switches also keep the squeeze.
    - **Tested:**
      - **Simulated arm and fake teleoperator, 3 episodes:**
        - All 3 hand-offs were automatic.
        - The fumbled pick was counted as 2 attempts and rejected at export.
        - Each clean pick went from 4.07 s to 2.97 s, with 0.47 s of idle and 0.6 s of hesitation removed.
      - **Real export into Studio:** it listed 2 episodes with the right features. The test dataset was then deleted.
      - **Live front depth mask:** about 96 % of an empty scene greyed out.
      - **Not yet tested:** a real teleoperated episode on the arm.

16. **The task is split into phases; only phases 1 and 2 are built now** (2026-11-14 ≈02:55, the user's decision). This replaces the multi-view `inspect` routine and the regrasp from decisions 14–15 for now; they move to later phases.
    - **Phase 1, learned:** pick a brick lying in any position or orientation on the orange plate and bring it to **waypoint 1 (WP1), above the orange plate**. This is what ACT learns.
      - A recorded episode = human-teleoperated pick (until the gripper stall triggers the hand-off) + the robot's own **automatic rise and carry to WP1**, recorded as one episode.
      - Rise = WP1's lift, elbow and wrist-flex at the current pan and roll, so the carry never sweeps low across the plate.
      - The carry part of every demonstration is therefore clean and identical, and export keeps it as recorded (it isn't sped up).
      - At run time the policy can either do the whole phase, or the same stall hand-off can switch to the exact scripted carry.
    - **Phase 2, scripted:** WP1 → **waypoint 2 (WP2)** in front of the front camera with the **gripper command locked** at the value it had at hand-off. The clamp force doesn't change; only the other joints move. The gripper value taught in WP2 is ignored on purpose.
      - At every WP2 pose: averaged full-res front image + aligned depth + depth-masked image → `training_images/front/inhand/<label>/`.
    - **Reset, collection only:** back via WP1, drop at a random `drops` pose with a ±45° random wrist turn, then back to the start pose.
    - **Routines to teach on `/arm`:**
      - `home`: start pose, gripper open, must not block the top camera's view of the plate.
      - `wp1`.
      - `wp2`: one or more photo poses.
      - `drops`: 3–5 poses.
    - **Tested in simulation:**
      - 3 episodes, all ending exactly at WP1.
      - Gripper command constant through the carry and the WP2 capture, even though WP2 was deliberately taught with a different gripper value.
      - The fumble was rejected; clean episodes went from 5.5 s to 4.4 s.
    - **Amendment, ≈03:15: approach WP2 from above.** Swinging straight from WP1 to WP2 pushed the yellow plate.
      - Phase 2 now goes WP1 → **above WP2** at height → **straight down into WP2 as the last motion**, at half speed. Leaving WP2 is the reverse: straight up first, then over to WP1.
      - "Above WP2" is a new taught slot `wp2_above`, about 5 cm straight above WP2. Without IK, a taught pose is the only way to guarantee a vertical last motion.
      - If it isn't taught, an estimate is used: WP2's pan and roll at WP1's height. A warning is shown in that case.
      - The gripper stays locked throughout; the gripper value stored with `wp2_above` is ignored.
      - New button "test WP1 → WP2 path" (at most 30 % speed, saves nothing).
      - Checked in simulation: correct move order, gripper constant.
    - **Amendment, ≈03:20: tape-spot 90° re-grip, 2 orientations per brick** (user). Phase 2 is now:
      1. WP1 → above WP2 → straight down into WP2 → photo `<view>`.
      2. Straight up → **above the orange tape spot** (`tape_above`) → slowly straight down to **on the tape spot** (`tape_place`, brick just touching) → open slowly → up.
      3. Wrist roll +90° (or −90° if +90° would exceed the joint limit) → slowly down → close with the **same clamp command** as the original grasp → up. If the brick isn't held, the re-grip has failed: the brick stays on the tape and the episode is still counted.
      4. Above WP2 → down → photo `<view>_r90` of the rotated brick, where the faces that were under the fingers now show.
      - **Reset:** with optional reset spots on the orange plate (`drops`), a random drop as before. Without them, the brick is set back on the tape spot and the human moves it to the orange plate during MATCH.
      - **Collection needs** `home`, `wp1` and `wp2`, plus either the tape spot or reset spots.
      - **Test button "test re-grip"**: at most 30 % speed, saves nothing.
      - **Simulated one episode:** the order was as above, the roll changed exactly 90° (1024 ticks), and the clamp command was constant (1875) on every move while holding. Photos `front.png` + `front_r90.png`.

17. **One control page: everything lives on `http://127.0.0.1:7870/arm`** (2026-11-14 ≈03:05, the user's request). `ui.py` serves one long page, top to bottom in working order:
    1. Setup: connect, mirror, hold, limp; live cameras; joints.
    2. Teach the waypoints as 4 fixed slots: start pose `home`, `wp1`, `wp2` (photo poses, with a slow test run) and `drops`. Each pose has go, set and delete.
    3. What the models see: the top crop for the policy, set by dragging; the wrist view; the depth-masked front view.
    4. Phase 1 + 2 collection.
    5. Episodes, with use/don't-use ticks, and the last saved images.
    6. Clean, export and train ACT.
    7. Anomaly image counts per brick (target at least 60 good and 10–20 defect).
    - Below those, phases 3–7 are shown as dashed "later" cards with disabled controls, so the full plan is visible in one place.
    - The optional manual still photos are in a collapsed section at the bottom.
    - A sticky header holds STOP (Space/Esc), the state banner and links within the page.
    - `/` and `/collect` redirect to `/arm`. The old pages still exist at `/manual` and `/arm/routines` but aren't linked.
    - The backend takes a `routine=` parameter on the waypoint and test-run endpoints, so each slot is edited directly by name.

18. **Code lives on GitHub for collaboration** (2026-11-14 ≈03:30): private repo `https://github.com/lucabottelli/junction-2026`, branch `main`.
    - **In git:** code, notes, camera settings, taught waypoints, scripts and report images.
    - **Not in git** (`.gitignore`): `training_images/` (390 MB and growing) and `episodes/` (GBs). Copy those separately, with rsync or a Hugging Face dataset.
    - **Commits** use the GitHub no-reply email (repo-local git config).
    - **On another computer:** `gh repo clone lucabottelli/junction-2026 ~/junction26`. Running the tool there needs Physical AI Studio and the station hardware. The station-specific serials and devices are at the top of `capture.py` and `arm.py`.
    - **Teammates:** invite them under GitHub → Settings → Collaborators.

19. **Focus on the pick: the episode is now pick → WP1 → put back. Photos and re-grip are optional and off** (2026-11-14 ≈03:30, user).
    - **The tape-spot re-grip doesn't work on the real arm** (issue 12). The user wants to first teach the pick properly, with many iterations per specific block, and use ACT together with a vision model.
    - **Episode:** start pose → MATCH → recorded pick → robot rises and carries to WP1 (recorded) → **put-back**.
      - Put-back goes from WP1 to above the original grasp spot, turns by a random wrist roll (±"put-back turn"), descends slowly to the recorded grasp pose, opens and rises. The path is known to be clear and the block gets a new orientation each time.
      - With reset spots (`drops`) taught, it drops the block at a random one instead.
    - **Phase 2 photos and the tape re-grip** are switches (`phase2`, `regrasp` in `collect_settings.json`), off by default, in a collapsed "optional extras" box. `wp2` and the tape spot are only required when the switches are on.
    - **Per block:**
      - Each episode stores the task `pick up the <block> brick`, written per frame into the Studio dataset.
      - Export can select one block ("red only" → one dataset/ACT per block) or all.
      - Plan per block:
        - **Round 1:** the block alone, 30–50 episodes, every position and orientation.
        - **Round 2:** the other blocks present as distractors, 20–30 episodes.
    - **ACT + vision:**
      - ACT is itself a vision policy (top + wrist images + joints → actions). What gets added is a top-camera **block finder** (colour, or a small detector) that decides which block to pick and where.
      - **Option A:** one ACT per block, chosen by the finder.
      - **Option B:** one language-conditioned policy (SmolVLA, "pick up the red block") on the same data.
      - Shown as a "Next" card on `/arm`.
    - **Simulated:** 2 episodes; put-back went to the pick spot with a random roll, descended slowly and released; task stored per episode; `--bricks red` export filter works.
20. **Inspect the whole surface of any shape, then sort: phases 3-5 are built** (2026-11-14 ≈04:30, the user's rule). Code: `anomaly_capture/sorter.py` + `coverage.py`, page **step 8 "Inspect & sort" at `/sort`** (was section 8 on `/arm`; see Decision 22), API `/sort/*`.
    - **The rule:**
      - An object goes to the **good plate only after its whole surface was seen and judged clean**.
      - **The first photo that shows a defect sends it to the defect plate at once**, without checking the rest.
      - Anything that can't be confirmed clean never counts as good: an uncertain score after retakes, a photo without a depth mask, or a part of the surface no pose can show. These go to the defect plate.
      - A dropped object or a re-grip that fails twice stops the cycle and asks the human to put the object back.
    - **Any shape, not only cubes** (`coverage.py`):
      - The surface is tracked as **directions**, not named faces: 1500 points on a sphere fixed to the object, one per direction a surface can face.
      - A photo covers every surface that faces the camera within 65° (minus a 5° safety margin), except what lies within 35° (+5°) of the finger axis.
      - The object is "fully inspected" when every direction has been seen in a clean photo. That holds for a cube, a long box, a triangular prism or a cylinder without knowing the shape.
      - On the page each WP2 photo pose is ticked with what it shows in the gripper's frame: **front**, **back**, **down**, **up** (two ticks = a diagonal view). The page shows the percentage covered, the planned photo sequence, and refuses START if the poses can't reach 100 %.
      - **Minimum set of poses:** front, back (wrist turned 180°) and down (gripper pointing at the camera), plus the tape spot.
      - The servo direction of the 90° turn isn't assumed: both directions are tracked and a direction only counts when it's seen under both.
      - Background: this is the visibility-matrix / set-cover formulation used in inspection view planning, and the object-level "any view anomalous" rule used in multi-view anomaly detection (Real-IAD). Berkeley's Omni-Scan follows the same pattern: grasp, rotate in front of a fixed camera, then change the grasp (there by handing over to a second gripper) to see what the first grasp hid.
    - **One cycle:**
      1. **Pick:** hand-feed (put the object in the open gripper at WP1 and press G) or teleop with the leader. ACT comes in phase 6.
      2. **Grip 1:** WP1 → above WP2 → down → a photo at each WP2 pose that still shows something unseen → up.
      3. **Tape spot, only if something is still unseen:** set down → **top photo of the released object** (wrist camera, looking straight down, nothing in the way) → grip again with the wrist turned 90°.
      4. **Grip 2:** only the WP2 poses that show what's still missing. Steps 3–4 repeat up to `max re-grips` (default 2) if needed.
      5. **Sort:** `sort_<bin>_above` → `sort_<bin>` → open → up → home.
      - A clean object with the minimum poses takes 6 photos and 1 re-grip. In simulation a defective one needs 2.7 photos on average, and 28 % are caught by the first photo.
    - **Depends on the tape re-grip, which doesn't work on the real arm yet (issue 12).** Until it does, a clean object can't be confirmed good: grip 1 alone sees about 60 % of the surface. Two of issue 12's four candidate causes are addressed by the new re-grip below (opening too narrow, turned block too wide); the height of `tape_place` and centring still have to be checked with "test re-grip". MANUAL judge + hand-feed is the way to debug it step by step.
    - **Re-grip for any width** (`collect.grip_on_tape`): the gripper opens to 90 %, closes until the fingers stall on the object, then squeezes 80 ticks past that width. The old "same clamp command" only worked when both widths were equal (a 2×4 brick turned 90° is twice as wide). Collection uses the same function.
    - **Per photo:**
      - The same 16-frame average as collection, then **`prep.inspect_crop()`** (held object: depth mask + square crop) or **`prep.release_crop()`** (object on the tape spot: fixed crop per camera, set on the page, stored in `inspect_settings.json`). **Training crops come from the same two functions:** `arm.capture` saves them as `*_crop.png`.
      - Score ≥ `t_high` → defect. Score ≤ `t_low` → clean. In between → retake (up to `retries`, mean of the scores); still in between → defect plate.
    - **Judge:**
      - **MANUAL:** a human clicks C / D / U on each photo. This tests all the motions and the coverage logic before any model exists.
      - **MODEL:** Anomalib OpenVINO models, `models/front/<brick>/` for held photos and `models/wrist/<brick>/` for the top photo (fallback `<cam>/all`). They run in `score_worker.py` inside the Anomalib venv as a persistent subprocess, so the two torch installs never meet. Device CPU, GPU or NPU per run.
      - **Cascade (decision 10, optional):** `models/<cam>/<brick>_efficientad/` answers first on its own device (NPU); only photos it is unsure about go to the main model. The fast model is used only once it has calibrated thresholds.
    - **Training and thresholds:**
      - `~/anomalib/.venv/bin/python anomaly_capture/train_anomaly.py --brick <name> [--cam wrist] [--model efficientad] [--xpu]`. Every 5th good crop is held out (`holdout.json`).
      - **Calibrate** on the page scores the held-out good crops and the defect crops through the live path. For each defect brick only its highest-scoring photo counts, because the cross is on one face and the other photos of that brick look clean. Result: `thresholds.json` next to the model, with AUROC and how many defect bricks never showed their cross to that camera.
    - **Collection change:** during the tape re-grip, phase 2 now also saves the **top photo of the released object** from the wrist camera (`tape_top`, in `training_images/wrist/inhand/`). Set the crop on the page (section 8) **before** collecting, so the training crops match.
    - **Log for the report:** `runs/<session>/log.jsonl`, one line per object: verdict, reason, every photo with scores per cascade stage, inference ms, load ms and device, surface % seen, re-grips, and pick / inspect / sort / cycle times. Photos and heat-maps are saved per cycle.
    - **Tested without hardware:** `python3 anomaly_capture/sim_test.py` (fake arm, camera and Anomalib worker; the fake world knows where the defect really is):
      - 400 random defect directions, both turn directions → 400/400 to DEFECT, never GOOD;
      - cube, triangular prism, tetrahedron, cylinder and wedge in random orientations, each face defective in turn, with the width changing at the re-grip → all DEFECT; the clean ones → GOOD;
      - uncertain photo, missing depth mask, bottom never shown, no top photo → DEFECT (fail-safe);
      - object dropped, re-grip failing twice → cycle stopped;
      - model judge, cascade (clean photos settled by the fast model, a subtle defect escalated), threshold calibration, HTTP API.
    - **Not tested:** the real arm and cameras, the real Anomalib 2.7 inferencer and `train_anomaly.py`.
    - **Assumptions that the simulation can't check:**
      - The ticks on the page are right (look at the photos in MANUAL mode once).
      - The object doesn't slip in the gripper.
      - The shape is roughly convex (deep hollows can hide a cross).
      - The object can be gripped again after a 90° turn. A triangular prism gripped on two slanted faces may squeeze out; the cycle then stops with "re-grip failed".

21. **A simulated twin of the station generates pick demos for ACT; training mixes sim and real** (2026-11-14 ≈04:00, the user's decision). Code: `~/junction26/sim/`. Plan: `~/.claude/plans/dreamy-beaming-bear.md`.
    - **Why:**
      - Real demos cost a human teleoperating each pick; sim gives thousands of auto-labelled picks.
      - The user chose **sim + real mix**, because a policy trained only on rendered images usually fails on real cameras (and our top camera is noisy at max gain).
      - The sim task is **phase 1 only:** pick a brick lying anywhere on the orange plate → WP1, the same episode as the real data.
    - **Stack:** `physicalai-mujoco-plugin` 0.5.0 (added to Studio's backend with `uv add`; mujoco 3.17, nothing downgraded). It provides the SO-101 twin (same joint names) and the composer. Our own scene sits on top of it, plus the plugin's `ArmIK` (damped least-squares, fingers down, jaw yaw).
    - **Files:**
      - `station.py`: scene (table, 3 plates as dishes with rims, the real bricks (user-confirmed 04:55; **red 2×4 ridge roof swapped for a pink/salmon 2×2 on 2026-11-14 afternoon**, see decision 29): **pink 2×2**, **green 2×3**, **blue 2×2**, top camera, wrist camera) and the units mapping.
      - `calib.py` + `overlay.py`: fit to the real station; they write `calibration.json` and `calib/overlay_*.jpg`.
      - `expert.py`: scripted pick → WP1.
      - `generate.py`: randomized episodes in the **same raw format** as `episodes/`, written to `episodes_sim/`.
    - **Units:**
      - Real ticks → sim rad: `(ticks − calibrated midpoint) × 2π/4096`. This is the real driver's degree convention; zero is at the range midpoint.
      - Gripper: same servo slope, anchored so that 1979 ticks (real, holding a 31.7 mm brick) = 0.14 rad (sim, holding it).
      - Checked by FK: `wp1` lands 18 cm in front, 14 cm left, fingers 6° from vertical, above the orange plate.
    - **Top camera fit:**
      - Keypoints are the gripper at grasp/WP1 frames and the base, read off the real top frames. Height comes from the measured scale (≈0.68 m). The fit is top-down: xc 0.233, yc −0.086, yaw −11.6°.
      - The arm overlays the real one to about 10–25 px (2–5 cm); randomization covers the rest.
      - **Precise calibration still needs a session with the real arm:** keypoints spread over the whole table.
    - **Wrist camera:** placed behind the jaws (gripper frame (0.022, −0.05, −0.05), tilted 26° toward the fingers, fovy 72°), so both jaws show at the bottom, as on the real camera.
    - **Expert:**
      - Grasp across the narrowest horizontal width, for any shape and pose: the brick's corner points are projected on closing directions every 3°. Grasp height is 45% up the brick. Choose the jaw flip that is reachable and upright with the smallest wrist turn.
      - Motion: min-jerk joint move home → above the brick, correct a deliberate approach error, descend, squeeze past the brick, lift, carry to WP1.
      - Randomized speed, hover height and pauses.
      - Success without rendering: **97%** (29/30) with the first, wrong shapes. With the real shapes it is **83%** (50/60): red ridge 75%, green 93%, blue 89%. Most failures are crowding, where the target gets pushed by a neighbour or by the jaw. Failed episodes are discarded, never saved.
    - **Domain randomization:**
      - 1 brick (50%), 2 (35%) or 3 (15%); pose anywhere within 70% of the plate radius; 45% studs up, otherwise any face up.
      - Colour shades, light position and intensity.
      - Arm recoloured to the real white, jaws dark red.
      - Top camera ±7 mm / ±1°; wrist camera ±3 mm / ±3° / ±3° FOV.
      - **Real backgrounds composited in** via segmentation: sim table pixels in the top view are replaced by real top frames (sim shadows kept; the real orange plate and its bricks are painted out first), and empty-sky pixels in the wrist view by real wrist frames. Live `/snap` previews are not used, because they carry UI overlays.
      - Per-camera look: top noisy at max gain and bluish; wrist dark, warm and vignetted.
    - **Export:** `export_act.py --sim-episodes ~/junction26/episodes_sim [--sim-max N] [--real-repeat K]`. Sim episodes skip the human-demo cleaning; `--real-repeat` up-weights the few real demos.
    - **Speed:** about 20 episodes/min with 12 workers (CPU-bound), about 30 MB per episode.
    - **Runs:**
      - `gen1` (600, wrong brick shapes) was stopped at 480 and moved to `episodes_sim_old/`; don't train on it.
      - `gen2` (700 attempts, real shapes) started at 04:58.

22. **The control site was redesigned: dark, minimal, one page per step, served at `/`** (2026-11-14 ≈04:30, the user's request; this replaces Decision 17's "one long page on /arm").
    - **Look:**
      - A single near-black theme: background `#0A0A0A`, ink `#EDEBE6`, one champagne accent `#CBB68C` and a refined red for STOP/REC. There is no light theme.
      - Instrument Serif for titles and big numbers, Inter for body text. Both are self-hosted in `anomaly_capture/web/static/fonts/`, so no internet is needed.
    - **Pages:**
      - Overview: the journey at a glance plus "continue where you left off".
      - `/` `/connect` `/waypoints` `/crop` `/record` `/episodes` `/train` `/anomaly` `/sort`: steps 1–8, each with "What to do" and previous/next links. Step 8 is the teammate's inspect + sort (Decision 20, PR #1), ported into the new design.
      - `/later/target|autonomous|measure`: the remaining future phases, one page each (phases 3–5 are built now, so their "later" pages were removed).
      - `/stills`: the manual still photos.
      - `/arm` and `/collect` redirect, and old `#hash` links are mapped to the new paths. `/manual` (the old capture page) is unchanged.
    - **Code layout:**
      - `anomaly_capture/web/` holds `shell.html`, `style.css` and `js/core.js`. Core does routing, polling, the top bar, STOP and the keys.
      - Each page has its own `pages/NN-slug.html`, `js/<slug>.js` and `css/<slug>.css`. A page's JS registers hooks in `PAGE.<slug>`, and an error in one page can't break the others.
      - `ui.py` assembles everything from disk on every request, so editing a page needs only a browser reload. A server restart is needed only when a page path is added.
      - The contract is in `web/DESIGN.md`.
    - **Demo previews:** `/record?demo=<state>` and `?demo=1` on Episodes, Train and Anomaly render sample data and send nothing.
    - **No API behaviour changed.** Built by 6 parallel agents (foundation + 5 page groups), which sent only GET requests and moved nothing. The buttons were not click-tested on the robot.
    - **Noticed while building:** the live **wrist** view shows the room and people at the top of the image. The ACT policy can therefore still see background movement through the wrist camera. Consider masking or cropping the top of the wrist image (in `prep.py`, so training and runtime use the same image).


23. **Autonomous pick (phase 6) is built: pick = ACT on `/sort`** (2026-11-14 night, built overnight without hardware). Code: `anomaly_capture/policy.py`, `sorter.pick_act`, test `tests/test_policy.py`, sim check `sim/act_loop.py`.
    - **Loading:** physicalai `InferenceModel(export_dir, backend, device)`, the same call Studio's model worker makes; a direct OpenVINO / ONNX Runtime path takes over if physicalai is missing. Devices CPU / GPU / NPU / AUTO.
    - **No training/serving skew:** the observation is built by the same functions the dataset export uses (`export_act.normalize`, `prep.act_policy_image`: crop → JPEG q93 → decode → RGB). A test checks byte equality. The page warns when the current policy crop differs from the crop of the newest recorded episode.
    - **Loop:** 30 Hz; modes `chunk` (Studio's "Run model" behaviour, default), `async`, `ensemble` (ACT temporal ensembling).
    - **Safety:** targets go through the arm's own rate-limited loop; body targets stay within 25° of the real position; STOP is checked every tick and again right before each write; a blocked arm or a camera frame older than 0.5 s ends the pick.
    - **Hand-off:** the collector's stall rule (now `collect.GraspDetector`), then the scripted rise and carry to WP1. Timeout or missed grasp → let go, back to start, retry; after the last retry the cycle is logged as aborted and the page asks for help.
    - **Evidence for free:** inference p50/p95 per pick are written to the cycle log (`act_infer_ms_p50`, `act_infer_ms_p95`, `act_attempts`).
    - **Assumed, not seen:** the export layout (`<id>/exports/<backend>/manifest.json`, inputs `state` [1,6] + `images.top` / `images.wrist` [1,480,640,3] uint8 RGB, output `action` [1,n,6]) comes from reading Studio's and physicalai's source, not from a real export.
    - **Known weak spot:** in the MuJoCo check (expert wrapped as a policy, 6 tries) 4 picks were detected correctly, 1 brick was out of reach, and **1 real grasp was never detected** (timed out). If this shows on the arm, also accept "`holding()` true for 0.5 s" as the hand-off. Not built yet; G forces the hand-off meanwhile.

24. **Measurement harness (phase 7): `bench/`** (2026-11-14 night). One command, `bench/run_all.sh`, finds the anomaly models and the Studio policy exports, benchmarks each on every device (CPU, GPU, NPU, AUTO; latency and throughput hints; FP16 and INT8), samples power (RAPL) and temperature while it runs, and writes `bench/results/REPORT.md`, charts and `section9_rows.md` for §9. `bench/cycle_stats.py` adds cycle time, accuracy and cascade share from the sorter's logs. Runbook: `bench/README.md`.
    - Verified on CPU only, with tiny models and a fake sysfs tree. GPU, NPU and the real sensors are untested. The NPU and xe sysfs file names were taken from the kernel source.
    - RAPL counters are root-only: after every reboot run the `chmod` in `bench/README.md`.
    - **INT8 uses the Anomalib venv:** installing nncf into Studio's venv would downgrade numpy.

25. **Anomaly pipeline: one command per brick and camera** (2026-11-14 night). `anomaly_capture/anomaly_pipeline.sh <brick> [cam]` trains PatchCore (main) + EfficientAd (fast), exports OpenVINO FP16, evaluates through the run-time path (`scorer.py` → `score_worker.py`) on held-out **episodes**, and writes `thresholds.json` where the sorter reads it. `eval_anomaly.py --compare` is the "does this change earn its place" table (decision 9). `synth_data.py` writes a synthetic `training_images/` in the station's format for testing.
    - **Three bugs fixed that would have cost time on the station:** PatchCore exported with a fixed batch of 2 (the worker feeds 1 image); Anomalib clamps exported scores to [0,1], so hard goods and defects tied at 1.000 and thresholds could not separate them (now unclamped, same scale); EfficientAd would have crashed (it needs train batch size 1).
    - **Split:** by episode `(run, cycle)`, about 20 % of good episodes held out, defects never trained on.
    - **Verified** end to end with real Anomalib on CPU, on synthetic images with untrained backbones (the weight downloads were blocked there). Those numbers test the pipeline only and say nothing about detection quality.
    - **Check on a real image:** `prep.fill_holes` may fill a background hole enclosed by the jaws and the brick, letting people in the background into the crop. Look at one `_masked.jpg`.
    - Measured cross sizes go in `training_images/defect_sizes.jsonl` to get the detection-rate-against-size curve.

26. **Mock station, tests and CI** (2026-11-14 night). `make mock` runs the real control server with simulated arms, cameras and Anomalib (`anomaly_capture/mock_hw.py`, `JUNCTION_MOCK=1`) on port 7871. All data goes to `mock_data/`, with waypoints pre-taught, so nothing of the station's is touched. `make test` (fast, about 2 min), `make test-flows` (collect + sort end to end with STOP and recovery, about 11 min), `make test-ui` (Chromium smoke test with screenshots). GitHub Actions runs them on every push. See `docs/DEV.md`.
    - **Real-hardware path changes to review once on the station** (untestable here): `capture.py` imports `pyrealsense2` only when not in mock mode, reads the port from `JUNCTION_PORT` (default 7870) and calls `mock_hw` only in mock mode; `arm.py` imports lerobot only when not in mock mode; `collect.py`'s export passes `--episodes` explicitly (same folder as before).
    - `sim_test.py` used to overwrite `anomaly_capture/sort_settings.json` on every run; fixed.

27. **Recording runs always finish: failures pause, STOP is resumable, and a guided rest-position cycle** (2026-11-14 ≈13:00, the user's request: "it must get all of the runs done").
    - **Failures pause, they don't end the run.** A BLOCKED move, a motion error, a missed brick, a discard (X) or a timeout now all go to the same pause (`reset`, `info.why` = blocked / missed / discard / timeout / error).
      - The follower copies the leader (creeping slowly at first), the human fixes the scene, and **N** continues with the next episode.
      - Only STOP, or losing the arm or leader, ends the run.
      - Before, any exception ended the run, and "missed" went straight on without waiting.
    - **Resumable sessions.** Each session writes `episodes/<session>/session.json` (brick, label, target).
      - `/collect/status.resume` is the most recent session that ended below its target. It is worked out from disk, so it survives restarts.
      - `POST /collect/start?resume=<session>&n=` continues the same folder and count.
      - A success counts the moment it is saved (end of the carry), so a STOP or BLOCKED during the put-back can't lose it.
    - **Match guide.** `info.signed` gives the signed degrees leader − follower per joint. The record page draws one bar per joint with the ±8° zone, so you can see which way to move each leader joint.
    - **The cycle, made explicit:** robot to the start pose → leader to the same pose → pick → let go, and put the leader back in its rest position while the robot carries and puts the brick back.
      - **Tip:** teach `home` with the leader resting in its natural rest position. Each match is then just "set the leader down". This tip is on the Record and Waypoints pages.
    - **Tested** on the mock station: BLOCKED in the carry pauses → N continues; STOP → resume → 3/3 done; STOP during the put-back keeps the count. New flow test `test_blocked_carry_pauses_run_then_stop_and_resume` in `tests/test_mock_flows_collect.py`. Not yet run on the real arm.

28. **Hand-off by lifting; grasp detection fixed; the put-back varies the start pose** (2026-11-14 ≈13:30, the user's request: "as soon as I pick it up, move to waypoint 1").
    - **Measured on the real arm:**
      - In the first 3 blue episodes every hand-off needed G. Squeezing a Duplo, the follower gripper stays only **28–37 ticks** short of its command, because the leader's closed end limits it.
      - The stall rule needed `HOLD_GAP` 40, so it never fired. **`HOLD_GAP` is now 20.**
    - **New rule, `collect.HandoffDetector`:**
      - **Main trigger, lift.** The gripper must be commanded closed with the fingers not fully shut, so something is between them. The robot takes over once the arm has been raised `lift_frac` (default **0.2**, about 3 cm, setting `/collect/cfg?lift_frac=`) of the way from the pose where it closed toward WP1, held for 0.1 s.
        - This is measured on shoulder_lift / elbow_flex / wrist_flex, projected on the grasp→WP1 direction, so it needs no calibration.
      - **Fallback, stall.** The old rule, held 1.5 s (close firmly and wait). G remains the manual trigger.
      - **`policy.py` uses the same rule at run time.** The policy imitates "grasp, squeeze, lift a little", and the lift hands over to the scripted carry.
      - Unit-checked: lift fires; an empty shut gripper or an open moving gripper never fires; stall fires after about 1.7 s. tests/test_policy.py gives the same results as before (the 2 ONNX IR-version failures are environment-only). Mock collection 4/4.
    - **Put-back for pick training: every episode starts from a new pose.**
      - Turn: random, between 30% of `roll_jitter` and all of it, either direction. The default `roll_jitter` is now 90°, so the block ends up at every orientation.
      - Position: a sideways shift of up to ±5° of shoulder pan, pulled 30% toward WP1's pan (the plate centre), so it never walks off the edge.
      - Lowered slowly (0.3 × speed).
      - Each sub-step 1/4–4/4 is shown in the status line and the page.
      - The robot does not re-grip by itself: only human picks are training data, so the robot's job is to vary the start pose.
29. **The block set changes: a pink/salmon 2×2 replaces the red 2×4 ridge roof ("triangle")** (2026-11-14 afternoon, the user's decision). Set now: **pink 2×2, green 2×3, blue 2×2**.
    - **Recorded so far (verified on disk):** blue `20261114_131414_blue` 30/30 and green `20261114_142128_green` 30/30. Every episode loads, has all 3 cameras, and has one frame per camera per step at 30 Hz. No red episodes had been recorded, so nothing is lost.
      - Blue ep0001 is 71 s long (pick 66 s, the manual-G hand-off before decision 28), against a median of about 23 s. It's a candidate to leave out of training.
    - **Name it `pink`** when recording (block field on `/record`). The UI swatch is salmon (`#E8877C`; `salmon` works as an alias).
    - **Changed:** the sim brick set (`sim/station.py`: pink 2×2, 6 g; the ridge-roof shape code stays), the mock station (spawns pink, green, blue), `synth_data.py` defaults (pink, green, blue; `red` kept only for the tests), and the "e.g." hints.
    - **Red data deleted** (2026-11-14 ≈15:00, the user's request):
      - The real session `20261114_144657_red` (3 saved, 4 failed, 1.7 GB).
      - The 202 red picks in `episodes_sim/gen2`, leaving 193 blue and 209 green sim picks (12 GB).
      - Studio's training set `pick-sim604-real1x10` (`datasets/cc7a59ef…`) still contains the red sim picks. It's left as it is and gets replaced by a new export (sim blue/green plus real blue/green/pink) before the real training run.
      - `episodes_sim_old` (the older wrong-shape sim runs) is untouched.
    - **Sim regenerated for the new set** (≈15:00, sim session):
      - **Pink colour calibrated on the live top camera.** The real brick measures RGB ≈ (180, 85, 94) at maximum gain. The sim value is `(0.78, 0.32, 0.29)`, chosen so that after the camera degradation the sim brick has the camera's G/R and B/R ratios, with the same blue bias as the calibrated blue brick. The first guess, (0.91, 0.53, 0.49), rendered too light and too orange.
      - **Expert with the new set:** 85% (102/120; pink 36/43, blue 41/49, green 25/28). Unchanged, and only successes are kept.
      - **The remaining gen2 blue/green picks are moved out of the training folder** to `episodes_sim_old/gen2_bluegreen_ridge_distractors` (402 episodes, 12 GB). Their scenes still have the ridge roof as a distractor brick.
      - **`gen3` (700 attempts, seed 2000000, `-j 8` to leave CPU for `capture.py`)** rebuilds all three bricks with pink in the scene. `export_act.py --sim-episodes ~/junction26/episodes_sim` picks it up as is.
    - **Pink recorded** (2026-11-14 15:01): `20261114_150142_pink` **30/30** (15 failed attempts skipped: 10 dropped, 5 missed), 4.8 GB. Verified: every episode loads, all 3 cameras, one frame per camera per step; picks 13–25 s, median 18 s.
    - **Real pick data is complete: blue 30, green 30, pink 30** (90 episodes). Next: export them (plus the sim blue/green picks if mixing) as a new Studio dataset for the ACT run.
    - **All data is on Hugging Face** (2026-11-14 17:09): private dataset `lux13crystal/junction26-so101-duplo` (https://huggingface.co/datasets/lux13crystal/junction26-so101-duplo), 1,718 files, 71.4 GB, every file checked by size. One tar per episode: `real/` (5 sessions), `sim/gen3/`, `sim_old/`, `stills/`, `studio/` (the 2 Studio exports). `README.md` describes the format; `unpack.py` restores the station layout. Keep it private: the front and wrist cameras show people.
    - **Note for the pick policy:** pink and blue are now the same shape and size and differ only in colour. One ACT policy should pick both the same way, and choosing between them ("pick up the pink one") becomes purely a colour problem.
30. **The real picks are cleaned with Qualia before training** (2026-11-14 ≈15:40–16:00, the user's decision).
    - **Setup:** `uv tool install --prerelease allow qualia-cli`, then `qualia auth login --base-url https://junction-api.qualiastudios.dev --api-key …`. The key is stored in `~/.config/qualia/config.toml`.
      - The key is **project-restricted**. Commands that touch data need `QUALIA_WORKSPACE=af6e6844-0231-45d7-8448-dfcebcd2a517`.
      - The key **cannot** create Tasks (`task create`), publish curated versions (`dataset curate`), or read `curation-suggest`. `reject` works.
    - **Upload:** a real-only export (`export_act.py --name real76-for-qualia`, Studio `datasets/2d9b29bf…`, LeRobot v3) went up as `junction-real76-pick`: upload `f569d7f4…`, version `d6d9646c…`, 76 episodes, 29,171 frames, 695 MB. It took about 6 min.
      - Qualia episode index → raw episode: `~/junction26/qualia_real76_index.txt`.
    - **Tasks:** Qualia's automatic grouping merged the colours into 2 generic tasks (66 + 10 episodes). `confirm-task-names` set the real ones (green 28, blue 26, pink 21).
      - The blue Task could not be created with this key. The heuristic and rule checks don't need it.
    - **Checks** (free; the VLM check would cost 1,659 credits and the account has 100):
      - **heuristic:** 1 flag. Blue ep16 is 22.6 s against a fence of 22.3 s. Kept.
      - **episode-rules** (`check run -c -`): 19 flags on 15 episodes:
        - gripper flicker ×4 (Qualia-recommended);
        - "corrections near gripper events" ×11;
        - joint lag ×6 (133 ms on five episodes, 200 ms on one).
    - **Rejected (6)** and listed in `~/junction26/qualia_exclude.txt`. They were also marked rejected on the Qualia upload.
      - The four gripper-flicker episodes: blue ep22, ep44, ep45; pink ep31.
      - Pink ep42 (2 corrections) and pink ep08 (200 ms lag).
      - Kept: the 1-correction episodes and the 133 ms lags, which are normal leader→follower lag.
    - **`export_act.py`:**
      - new `--exclude FILE` (`<session>/<episode>  # reason`);
      - the generic task "pick up the brick" of the 03:25 test pick now becomes "pick up the blue brick" from its `brick`, so the dataset has exactly 3 tasks.
    - **Training set `pick-sim610-real69x2-qualia`:**
      - **69 real** picks (blue 23, green 28, pink 18), each written twice, about 31% of the frames. The slow-outlier rule also dropped blue ep27 (35.4 s against a limit of 35.3 s).
      - **610 sim** picks from gen3.
      - Next: ACT on Modal, A100-80GB, **4 epochs**, batch 32, augmentation on.
    - **Pink camera stalls:** 8 of the 30 pink successes were rejected for camera stalls of 0.5–8.8 s. The blue and green sessions had none. Most likely cause: sim generation (gen3, 8 workers) ran during that recording. Don't run the generator while recording.
31. **Anomaly inspection setup: front D415 = held-brick camera, wrist = top face, top = locating only; close-range depth mask and a taught re-grip pose** (2026-11-14 ≈17:00–19:00). Fixes issues 12 and 13 in code; **nothing below has run on the real arm or cameras yet.**
    - **Camera choice (measured on the stored crops, same blue 2×2):** front ≈ 250–300 px per brick on the table and ≈ 475 px held, best image; wrist ≈ 100–120 px, yellow cast, dark, some blur → only for the top face on the tape spot (the one view nothing covers); top ≈ 60–70 px → too few pixels for a small cross, used to find bricks and for ACT.
    - **Depth is for the mask, not for the cross:** ink has no relief, the D415 depth noise is far larger. Depth is used to grey out everything except gripper + brick.
    - **Close-range depth (`capture.py`, new `rs_depth.py`):** front depth now 424×240 (Min-Z ≈ 14 cm instead of ≈ 45 cm at 1280×720), plus an optional **disparity shift** through RealSense advanced mode (window near = fx·B/(shift+126), far = fx·B/shift, computed from the real intrinsics and baseline). Settings in `cam_settings.json` `front_depth` (`width`, `height`, `shift`, `near_mm`, `advanced`, `expect_z_mm`, `max_obj_mm`); bad values fall back to defaults with a warning; any failure → shift 0 or colour-only, as before. Endpoints `GET /depth/status`, `POST /depth/autotune` (brick held at WP2; sweeps shifts live, keeps the nearest consistent cluster, stores shift + expected distance), `POST /depth/reset`; "Auto-tune depth" on the Crop page. Auto-tune is refused while the arm, collector or sorter runs, and photos taken during it are flagged unmasked.
    - **`prep.inhand_mask` fails loudly:** min depth 50 mm (was 150); only inside the front ROI; the cluster must be near (`max_obj_mm`, default 400; **set ≈ 200–250 on the station**), object-sized (≥ 30×30 mm, ≤ 300×300 mm projected), mostly with real depth, and without a brick-sized no-depth hole; holes inside the brick (pen cross) are still filled, but background seen between the jaws is not. Otherwise `None` + reason → the photo counts as unmasked → never GOOD. The real issue-13 capture is now refused instead of masked wrongly (regression test). Capture meta records `depth_cfg`, `inhand_cfg` and `depth_mask_why`.
    - **Re-grip diagnosis (issue 12): the SO-101 has one fixed jaw.** The brick is pushed against it in the first grip, so its centre is off the wrist-roll axis; turning only `wrist_roll` by 90° puts that offset under the jaws. Fix: optional taught slots **`tape_regrip`** (+ `tape_regrip_above`) on Waypoints: the open, turned gripper straddling the released brick, fixed jaw a few mm from it. Untaught → the old computed turn.
    - **Coverage safety rules (from the review):** a taught pose is accepted only if | |Δroll| − 90° | + |Δpan| ≤ 5° (coverage's margin), because coverage credits every re-grip as exactly 90°; the four `tape_*` poses are frozen per cycle and waypoint edits are refused while a routine runs; the pose is validated **before** the brick is set down; a retry keeps the same turn direction, and **a re-grip that needed a retry can never give GOOD** (the brick may have moved): clean photos → defect plate with "orientation not verified".
    - **Safer motions:** release opens only partly at table height, slow first lift, a failed grip lets go at table height before lifting; after a slip the retry doesn't open at height.
    - **Step-by-step "Test re-grip"** on Waypoints: pauses after every sub-step (N = next, 120 s timeout = STOP), shows joint errors (height diagnosis), gripper stall vs first grip and the real holding state, and writes `runs/regrip_test_<time>/` (`steps.jsonl`, `summary.json`, wrist/top jpgs).
    - **Verified:** `make test` 185 passed / 3 skipped (was 72), `make test-flows` 15/15, `sim_test.py` passes; two independent reviews (findings above fixed, each with a regression test). **Not verified:** RealSense advanced mode on the real D415 (re-enumeration time, live depth-table changes), the real fx/baseline, the mask thresholds on real depth, and the re-grip on the real arm. The mock still has symmetric jaws, so it can't reproduce the fixed-jaw failure.
    - **Station checks, in order:** restart the tool → log shows `front depth: 424x240 …` and no warning → `/depth/status` → brick at WP2 → Auto-tune depth → `mask_ok` at every WP2 pose → one capture, look at `_masked.jpg`. Then Waypoints → Step by step re-grip → STOP after "lifted" → teach `tape_regrip` by mirroring → card shows roll ±90°, pan ≈ 0 → Step by step again → plain Test re-grip.


32. **Several bricks on the plate: always pick the brick CLOSEST TO THE ROBOT** (2026-11-14 ≈19:40, the user's decision).
    - **Why:** ACT has no language input. Sim evaluation of the first ACT model (`5d03db18`):
      - target alone on the plate: 7/13;
      - 2–3 bricks: 1/17.
      The demos picked different colours in similar-looking scenes, so the policy hovers between bricks. A fixed spatial rule makes every demo agree. The station then picks, inspects and sorts the nearest brick, and repeats until the plate is empty.
    - **Sim:**
      - `expert.Scene.reset_nearest` makes the target the brick with the smallest distance to the robot base (origin, table plane). Layouts where the two nearest are within 1.5 cm are redrawn.
      - `generate.py --rule nearest` is now the default. Plates are more crowded (1/2/3 bricks at 35/35/30 %), and the rule plus `n_bricks` go into each episode's meta.
      - `evaluate.py --rule nearest` (default) reports success by brick count.
      - Expert with the rule: 51/60 (85 %), 16/19 with 3 bricks.
    - **Real picks:** every real demo had 2–4 bricks on the plate, and the target was the session's colour.
      - `sim/real_nearest.py` checks each episode's first top frame:
        - detects the bricks by colour, only inside the detected orange plate;
        - tells pink from glare by saturation: brick S≈134, glare 76–101;
        - projects each brick through the calibrated top camera onto the table and compares distances to the base.
      - Only 13 of 86 follow the rule; 13 more couldn't be checked (target not detected).
      - The rest go into `~/junction26/nearest_rule_exclude.txt` (79 lines).
      - With the Qualia list as well, **12 real picks remain (green 8, pink 4, blue 0)**.
    - **`export_act.py --exclude` is repeatable:** `--exclude ~/junction26/qualia_exclude.txt --exclude ~/junction26/nearest_rule_exclude.txt`.
    - **New real picks must follow the rule:**
      - Put 1–3 bricks on the plate and always pick the one closest to the robot.
      - Name the session after the brick you actually pick.
      - Record them with "Front-camera photos at WP2" on, so they also give the anomaly photos.
      - Aim for about 15–20 per colour.
    - **Next ACT round:** `gen4` (`--rule nearest`, about 700 attempts), the rule-following real picks, and 6 epochs on the A100 with 16 cores. The run waits until the real-arm test of `5d03db18` is done, so EGL rendering doesn't compete with the policy on the Intel GPU.
    - **Real-arm test of `5d03db18`** (intel-2f, ≈19:35): **0/10** with one pink brick.
      - A false lift hand-off was fixed in `policy.py`.
      - After that the arm reached the plate and closed twice beside the brick, or went straight into the carry with the gripper shut.
      - Inference: 11 ms p50.
    - **Root cause found in the sim calibration** (≈19:45):
      - The first fit (`calib.py`) used the 03:25 episode, whose top `act_roi` was the FULL frame (centre 4:3 crop).
      - Every training session since 12:42 uses `act_roi` top = [388, 113, 1677, 1080].
      - So **the sim top view never matched the real training images**: different zoom and framing, and the plate about 4–5 cm off. It was also fitted on only 6 points.
    - **New hand-eye calibration from real grasps:**
      - `sim/handeye_fit.py` pairs the target brick pixel (first top frame) with the follower FK at the grasp. The grasp is the lowest TCP in the 4 s before hand-off; the hand-off fires after the lift.
      - Pixel→xy homography: 78 episodes, 52 inliers, leave-one-out **7.5 mm median / 10.7 mm p90**. Saved to `sim/calib/handeye.json`; intel-2f uses it for the scripted vision+IK pick.
      - `sim/calib_handeye.py` refits the sim top camera on the same grasps with strong top-down and height priors: rms 44 → 11 px, table error 8 mm median. Plates re-measured in the crop view; orange is now (0.184, 0.132), r 0.0975 (was 0.21, 0.117, r 0.069).
      - The old fit is kept at `sim/calib/calibration_v1_6keypoints.json`. Sim and real overlays now line up for the plates and the arm (grasp and WP1).
      - The joint map is still not fitted per joint: the old raised-arm keypoints disagree by 30–80 px. FK is trustworthy near the plate only, and the homography absorbs this.
    - **Expert with the new geometry and the nearest rule:** 55/60 (92 %).
    - **Old sim data archived** (old calibration): `episodes_sim_old/gen3_old_calibration_random_rule`, `gen4_old_calibration`.
    - **`gen5` started 19:46:** 750 attempts, seed 4000000, `--rule nearest`, nice 19, `-j 8`.

33. **Real-arm test of the ACT pick, a pick-only test tool, and a scripted vision pick started and parked** (2026-11-14 ≈19:30–20:00, intel-2f).
    - **Pick-only test:** `POST /sort/act_test?n=5[&put_back=1]` (`sorter.act_test`). The full `/sort` cycle can't start before the plates are taught, so this runs only: start pose → policy → hand-off → carry to WP1 → still held?
      - A held brick is put back with the collector's put-back (new spot, turned). A miss opens the gripper and goes back to the start pose.
      - Log: `runs/acttest_<time>/trials.jsonl` plus `trialNN_trace.npy` (t, follower position ×6, policy target ×6, in ticks).
      - Settings used: openvino on GPU, chunk mode, `act_speed` 0.5, `act_max_s` 30, `act_retries` 0.
    - **Result for `5d03db18`, one pink brick: 0/10.**
      - Run 1 (`acttest_20261114_193448`): 5/5 hand-offs fired after about 1.8 s with an empty gripper.
        - Cause: in the demos the leader's gripper rests shut at the start (commanded 3–5 %) while the follower starts 35 % open. The policy learned that and commands "closed" from step 0. Fingers closing in the air count as "closed on a brick" for the lift rule, and the move from home to the plate projects toward WP1, so it reads as a lift.
        - Fix (`policy.ActPick.attempt`): a hand-off also needs the fingers stalled on something (not moving for 0.25 s, not shut, commanded tighter than they are). Checked against the demos: true near the hand-off in 92/92 successful picks.
        - A first idea, "arm the detector only after the gripper opened above 40 %", was wrong: the humans open only 19–40 % before grasping. `tests/test_policy.py` caught it.
      - Run 2 (`acttest_20261114_193859`, with the fix): trial 1 was a real attempt (over the plate, opened, closed twice beside the brick, 16 s). Trials 2–5 went straight into the carry-to-WP1 motion with the gripper shut.
      - Inference 11–14 ms p50, 30 Hz, no late ticks: timing is not the problem.
    - **Scripted vision pick (`anomaly_capture/vpick.py`), parked by the user at ≈20:00 in favour of the next ACT round.** Not run on the arm.
      - Built because the user asked for a 90 % pick rate with one or more bricks on the plate.
      - `detect()`: bricks on the orange plate from the top camera. The plate, its shadows and its glare all keep an orange hue (H 0–23), so everything else inside the plate hull is a brick; pink needs S ≥ 108 (glare is 76–101). Gives centre, angle and size per brick.
      - `Calib`: pixel → table xy as a homography fitted in the kinematic model's own frame, from the teleop picks (brick pixel in the first top frame ↔ FK of the joints where the fingers stalled on it). 89 picks, 61 inliers, **6.7 mm median, 10.1 mm p90**. The sim session's independent fit gave 7.5 mm.
      - `plan_grasps()`: nearest brick to the robot first; jaws across the short side (both directions for a 2×2); the fixed jaw is kept 9 mm off the brick and the moving jaw pushes the brick across; the direction with the most room to the neighbours and the rim wins.
      - `VisionPick`: hover, **servo the measured joints onto the IK pose** (the servos sag up to 3.5° under gravity), step down with a contact stop, close until the fingers stall, squeeze, rise, WP1, held?
      - `place()` + `sorter.place_and_look()`: set the brick down at a random spot and look again. Each round gives one exact pixel ↔ arm pair and refits the calibration.
      - Tested only offline (detector on 35 demo frames, the whole sequence on a fake arm). `pick = vision` and `/sort/act_test?mode=vision` exist but are untested on hardware.
    - **Mistake to avoid:** I ran `git stash` to compare test results while two other sessions had uncommitted edits in the tree. It was restored within about 10 s, but never stash, checkout or reset in this shared tree.


34. **The robot records its own pick dataset; SmolVLA becomes the final pick model** (2026-11-14 ≈21:15, the user's decision; ACT is parked).
    - **Why:**
      - Real-arm ACT tests of `a87d9501` (quick run, crop-matched sim):
        - 1 brick: 0/5, then 1/5;
        - 3 bricks: trial 1 missed, trial 2 STOPPED by the user.
      - Traces: the gripper's lowest point was the same spot every trial ((186–191, 121–130) mm) while the brick was at (219, 161). It succeeds only when the brick happens to lie there.
      - Open-loop probe (`sim/probe_policy.py`): on sim frames it aims at the brick to 7–9 mm (corr 0.99, also on unseen scenes); on real frames it doesn't. **It learned sim-specific appearance**, and more sim won't fix that.
      - The real teleop data was thin after the nearest-rule filter (12).
    - **SmolVLA** (Studio policy `smolvla`, base `lerobot/smolvla_base`, SmolVLM2-500M, public weights) reads the task, so "pick up the <colour> brick" with distractors is valid. All clean real picks count again (69 after Qualia), and the nearest rule isn't needed.
    - **Self-recording loop** (intel-2f, `vpick.py` + `sorter.py`): `POST /sort/act_test?mode=vision&record=1&target=random&n=N`.
      - Scripted vision pick: detection, hand-eye fit about 7 mm, IK.
      - Each attempt is recorded as an episode in the teleop format: `episodes/<time>_vpick/`, `source: "vpick"`, task = target colour.
      - Reward = still held at WP1 (+ `verified` = seen where it was set down).
      - Random put-back and hand-eye refit from the robot's own placements.
      - **Not yet run on the real arm:** needs a 15–20 min shakedown (grasp height, jaw clearance) and the user's go.
    - **Learning layer:** `anomaly_capture/grasp_tuner.py`, a factorized Thompson-sampling bandit (the simplest RL).
      - It tunes per colour: grasp depth, jaw-axis shift, yaw offset, opening margin, descent speed.
      - On a synthetic station: 30–70% → 82–92% success within 150 picks (`tests/test_grasp_tuner.py`).
      - **Disabled** until the base pick works (`python grasp_tuner.py --enable`). Results are recorded from trial 1.
    - **Training data for SmolVLA:**
      - vpick episodes with outcome success (and `verified` != false);
      - the 69 clean real teleop picks;
      - sim 0–20 % at most.
    - **SmolVLA trial run** (pipeline check): dataset `real69-smolvla-trial` (`4f0b0852`), job `ff382962`, A100, 1500 steps.
    - **First real-arm SmolVLA tests (2026-11-14 22:24–22:28): 0/6.** Model `2ab68eb4` (≈3,900 steps, chunk 50, OpenVINO on GPU, 92–96 ms per inference, speed 0.5), 3 bricks on the orange plate, `/sort/act_test?n=3`.
      - Task "pick up the pink brick" (`runs/acttest_20261114_222421`): 0/3. Task "pick up the green brick" (`runs/acttest_20261114_222659`): 0/3. Every trial: "closed on nothing".
      - **The gripper never opens:** it starts at 2412 ticks and only closes (commanded 1890–2300 of the 1875–3395 range), so it can't get around a brick.
      - **It doesn't aim at the named brick:** green lay at (125, 88) mm; the gripper's lowest points were (163, 101), (164, 124), (189, 76), i.e. 40–65 mm away and 26–54 mm above the table. The pink run ended in the same area ((169–232, 99–105)), which is also where ACT `a87d9501` went.
      - Trials end after 3–7 s through the lift rule, not the 30 s timeout.
      - Note: `vpick_calib.json` pixels are full resolution (1920×1080); `/snap` images are 640×360 (×3).


35. **New pick dataset in `~/junction26/tDataset2`: all three blocks on the plate, one target colour per session, trained per block** (2026-11-14 ≈22:25, the user's decision).
    - **Why:** SmolVLA trial `2ab68eb4` (69 old real picks, about 3,900 steps; val/loss best at epoch 3, then overfitting) went 0/3 on the real arm.
      - Plans were jittery: the commanded position jumped > 8 mm between 33 ms ticks on 56–69 % of ticks. That's the flow-matching sampling of an undertrained model, plus a jump at every chunk switch.
      - It went down 3–5 cm away from the pink brick.
      - It commanded the gripper closed from step 0, learned from the teleop demos where the leader rests shut at the start.
      - Speed on the Intel GPU is fine: about 71 ms per 50-action chunk.
    - **Recorder:** new key `episodes_dir` in `anomaly_capture/collect_settings.json`, read once at start by `collect.Collector`. Set to `~/junction26/tDataset2`; 7870 restarted ≈22:28. Teleop sessions and the vpick self-recording (`C.EP`) both write there. The Episodes page shows only that folder now; the old `episodes/` stays as it is. `tDataset2/` is in `.gitignore`.
    - **Recording:** all 3 blocks on the plate, Block = the colour picked in that session (task "pick up the <colour> brick"), at least 30–50 per block, layout varied.
    - **Export per block:** `export_act.py --episodes ~/junction26/tDataset2 --bricks <colour>`. The export also gets a gripper-start fix: the early gripper action is kept at least as open as the follower until the human opens it.
    - **Recorded 22:33–23:11:** pink 20/30, blue 20/29, green 20/29 successes (the rest missed or discarded), 7.5 GB. The leader now starts OPEN (≈2407 ticks = follower), so the start-closed gripper quirk is gone in this data.
    - **Qualia:** `junction-tdataset2-real60` (upload `c7eccd87…`).
      - Ingest 60/60 ok. Task names confirmed.
      - episode-rules raised 7 flags; none was a "recommended" gripper flicker.
      - **5 rejected** (≥ 2 corrections near the grasp ×4, end-pose outlier ×1), listed in `tDataset2/qualia_exclude.txt` and rejected on the upload.
      - Index map: `tDataset2/qualia_index.txt`.
      - Result: **55 clean** (pink 20, blue 17, green 18).
    - **Exporter:**
      - `--episodes` is repeatable.
      - Gripper-start fix by default (`--no-gripper-fix` to skip): until the human first opens the leader, the gripper action is kept ≥ the follower's opening. This fixes the old sessions; tDataset2 is unaffected.
    - **The top view is unchanged** vs the 15:00 sessions (plates and base in the same place; the front tripod now shows at the lower left), so the old clean picks can be combined.
    - **Training (≈23:30):** one SmolVLA on **124 clean real picks** (blue 40, green 46, pink 38: old 69 + tDataset2 55), dataset `real124-smolvla`, A100, 5 epochs. One model with the colour instruction rather than three per-block models: each block has only about 40 demos, and the shared approach/grasp/lift skill benefits from all 124.
    - **Short runs and real-arm test (≈23:50–00:05):**
      - Two short SmolVLA runs trained in parallel: A = new 55 only, B = old+new 124. A100, 5 epochs each, ≈2.4–3.6k steps.
      - **A on the real arm** (OpenVINO GPU, async, speed 0.5): **pink 0/2, blue 0/3.**
        - It picks the right colour, and one try went down within ≈3 mm of the blue brick.
        - But it closes the gripper fully, on nothing, ≈5 s in, every try.
        - The plans jitter ≈10 mm/tick (human commands 0.1–0.8).
    - **Why nothing gets picked (diagnosis 00:15):**
      1. **Undertrained:** ≈2.4k steps vs the usual ≈20k for a SmolVLA fine-tune. It learned the coarse reach but not the fine grasp.
      2. **The top view wastes pixels:** in the 640×480 policy crop, the pick plate is ≈10% of the frame, in the top-right corner, with bricks ≈35 px.
         - Planned fix: crop around the pick plate, in software, the same way in training and at runtime.
      3. **The wrist camera** is dark and orange-tinted; blue reads as teal.
      4. Little data: ≈18 clean demos per colour in tDataset2.
    - **Long runs (user's decision ≈00:15), launched in parallel without waiting for B's short run:**
      - Two runs, both H100, 48 CPU, 44 workers, batch 32, no crop change (they match the current runtime):
        - `smolvla-new55-20k`: 39 epochs, job dc23272a, trainer modal-2.
        - `smolvla-real124-20k`: 16 epochs, job 47f3be5e, new trainer **modal-3** (63d17c0e).
      - Both are about 20k steps. Rate ≈0.7%/min, so ≈2.2 h each, ≈$25–30 together.
      - **Short-run comparison (00:25, `sim/compare_smolvla.py`, 12+12 episodes):**
        - **B (old+new, model 68a4bade):**
          - Aim: new 23.2 mm, old 14.6 mm. Jerk 6.0–6.9 mm/tick. Spread 4.2–4.6 mm.
          - Val loss 0.13 (but a different val set from A).
        - **A (new only, model 6b63fa17):**
          - Aim: new 15.7 mm, old 23.0 mm. Jerk 10.3–11.0 mm/tick.
        - **Verdict:** mixed. B is smoother and steadier. A aims better on the current three-block setup. Both long runs are kept.
      - **B on the real arm (01:05–01:08, short run `68a4bade`, OpenVINO GPU, async, speed 0.5, 3 bricks on the orange plate): green 0/3, blue 1/3.** First SmolVLA pick on the real arm (blue, trial 3, 6.3 s).
        - Logs: `runs/acttest_20261115_010542` (green), `runs/acttest_20261115_010714` (blue). Inference 92–94 ms p50.
        - Aim is much closer than `2ab68eb4`: it now goes down to the table (TCP 7–20 mm) near the named brick. Green: about 17 mm and 10 mm off in trials 1 and 3 (start position; misses can nudge the bricks), trial 2 went between pink and blue. Blue trial 1: about 14 mm off.
        - The gripper still never opens beyond its start (max 2401–2443 ticks, about 35 % of the range), so a 10–17 mm aim error is enough to miss.
      - **B on pink (01:11, same model and settings): 0/3**, all three "no grasp within 30 s" (`runs/acttest_20261115_011059`).
        - Each trial went down at ≈5 s to (209, 143), (190, 166), (197, 157) mm, closed the gripper fully on nothing (1882 ticks), rose to ≈95 mm and hovered until the timeout.
        - Before the run pink lay at (144, 178) and blue at (194, 148): the low points are 15–25 mm from blue and 45–65 mm from pink, so it probably went for blue. Not certain: all three bricks lay elsewhere after the run.
      - **Long-run results (finished 02:08):**
        - **`smolvla-new55-20k` (model 73282bd0):**
          - Val loss bottoms at 0.143 around epoch 16 (≈8k steps), then rises to 0.24 at epoch 39: **it overfits**.
          - Aim: new 25.2 mm, old 31.6 mm. Spread 2.5 mm. Jerk 3.3–3.6 mm/tick.
        - **`smolvla-real124-20k` (model d8a6fe45):**
          - Val loss flat at ≈0.155 across all 16 epochs.
          - Aim: new 28.7 mm, old 14.5 mm. Spread 2.2–2.6 mm. Jerk 2.6–3.4 mm/tick.
        - **Reading:** longer training made the plans about 3× smoother and about 2× more consistent; **the offline aim did not improve.** The real-arm test decides. real124-20k goes on the arm first: no overfitting, and the smoothest.
        - intel-48 was told: 40k steps on new55 is wasted.
      - **20k-step models on the real arm (02:08–02:15, OpenVINO GPU, async, speed 0.5, 3 bricks, 2 trials per colour, `/sort/act_test?n=2`):**
        | Model | Blue | Green | Pink | Total | Logs (`runs/acttest_20261115_…`) |
        |---|---|---|---|---|---|
        | `73282bd0` = `smolvla-new55-20k` (39 epochs, 20.1k steps, val loss 0.24) | 0/2 | 0/2 | 0/2 | **0/6** | `020811`, `020939`, `021027` |
        | `d8a6fe45` = `smolvla-real124-20k` (16 epochs, 20.7k steps, val loss 0.16) | 1/2 | 1/2 | 1/2 | **3/6** | `021212`, `021312`, `021345` |
        - new55: 5 of 6 trials went down at 4–7 s, shut the gripper fully on nothing (1881 ticks) and hovered to the 30 s timeout. Low points 24–42 mm from the blue and green start positions. Plans are smooth now (median 1.9 mm/tick).
        - real124: every trial ended through the lift rule; the 3 picks took 5.1–6.8 s. Inference 110–112 ms p50 (new55: 91–93 ms).
        - **Caveats:** which colour was actually picked was not checked. The pick plate was swapped during these tests (orange at 02:08, yellow with the bricks at 02:11, orange again at 02:15), so some trials may have run on the yellow plate, which the training data doesn't contain. 2 trials per colour is a small sample.
      - **`d8a6fe45` second round (02:15–02:20, same settings, 3 trials per colour, orange plate): blue 1/3, green 1/3, pink 0/3** (`runs/acttest_20261115_021510`, `021614`, `021744`). A further blue run (`021932`) was 0/2 when the user stopped it.
        - **Wrong colour confirmed once:** the wrist photo of the green "pick" shows the pink brick in the gripper, with green still on the plate. The blue pick did hold the blue brick.
        - Running total for `d8a6fe45`: 5 holds in 17 trials, at least one of them the wrong colour.
      - **`d8a6fe45` blue, 4 single trials (02:21–02:24, same settings, `put_back=0`): 0/4** (`runs/acttest_20261115_022142`, `022216`, `022254`, `022329`): 3× closed on nothing (4.0–22.4 s), 1× timeout. Running total now 5 holds in 21 trials (24%).
      - **Gotcha:** Studio ignores `max_steps` whenever `max_epochs` is set (`schemas/job.py` `resolve_training_limit`). The first launch with `max_steps=20000, max_epochs=1000` would have run 1000 epochs. It was cancelled before the GPU started, then relaunched with an epoch count derived from the frames: steps/epoch = frames × 0.95 / 32.
36. **Scripted vision pick (`vpick.py`) first run on the real arm; it becomes the pick to push to 90 %** (2026-11-15 02:27–02:42, the user's go).
    - **Why:** `d8a6fe45` (SmolVLA real124-20k) missed 4 more blue picks (5/21 in total); longer training had not improved its aim.
    - **Result so far, blue target, 3 bricks on the orange plate, speed 0.5:** `runs/acttest_20261115_023739` 3/3 really held (logged 2/3), `023946` 3/5.
    - **Fixes made on the way (all in `anomaly_capture/vpick.py` unless noted):**
      - **Calibration refit** on the 59 tDataset2 picks (45 inliers, 6.4 mm median). The 19:50 fit was 5.9 mm off in x against them. Old file: `vpick_calib.pre_refit_0230.json`.
      - **Jaw angle:** the model reads the jaw yaw 11.5° off (FK yaw minus brick axis over 99 human grasps: median −11.5°, IQR −17…−6). `YAW_BIAS_DEG = -11.5` is added to the commanded yaw. Before, the jaws stood about 20° across the brick.
      - **"Closed on nothing" was wrong:** a 2×2 held by this pick stalls at only 1908–1927 ticks (calibrated closed end 1875; empty jaws read 1881–1885). The old rule (`< 120` ticks above the end) let go of every good grasp. Now `EMPTY_TICKS = 25`. The user saw the clamp each time; the photos confirmed it.
      - **"Still held at WP1" was wrong for the same reason:** held reads 1868–1899 after the squeeze. Now held = fingers at least `HELD_TICKS = 25` short of the squeeze command, and the squeeze may go past the calibrated end (down to the servo's 1755 limit).
      - `collect.HOLD_MIN_OPEN` 50 → 15 (shared with the teleop/ACT hand-off rules; empty jaws read 6–10 above the end).
    - **Gripper servo overload (02:33):** servo 6 latched an overload error (position 1848, load 200) and the arm would not connect (`Failed to read 'Min_Position_Limit' on id_=6 … Overload error`). Cleared by torque off/on on the gripper only.
    - **Auto-refit from the robot's own placements broke the calibration (02:46) and is switched off** (`sorter.AUTO_REFIT = False`).
      - After 10 put-backs `vpick.refit` replaced the demo fit with one on 10 placement pairs: 0.1 mm/px instead of 0.45. Pink then missed 2/2 (planned width 8.8 mm for a 32 mm brick). The user noticed it as "pink needs better calibration".
      - Cause: the placement pairs scatter 5–27 mm against the demo calibration (mean −7.8 mm in x), so they are not exact pairs; the brick moves when it is let go.
      - Restored the tDataset2 fit (45 inliers, 6.4 mm median). Bad pairs kept as `vpick_pairs.bad_0250.jsonl`.
    - **Pick rate with the fixed checks and the demo calibration (3 bricks on the orange plate, speed 0.5, by the gripper rule):**
      | Target | Held | Run (`runs/acttest_20261115_…`) |
      |---|---|---|
      | blue | 9/10 | `024302` (the miss: trial 1, brick lying on its side on the rim from the run before; the refit landed mid-run) |
      | pink | 5/5 | `025038` |
      | green (2×3) | 10/10 | `025324` |
      - Together 24/25. Photos are now kept per trial (time prefix).
    - **Open:** trial 4 of `023946` was a real miss (cause unknown: the photos were overwritten by trial 5). It knocked the brick onto its side on the plate rim, and trial 5 missed that too. The put-backs landed 16–27 mm from their target.



37. **Re-clamp flow checked against the coverage logic; not yet run on the arm** (2026-11-15 ≈03:50, the user's spec, no code change).
    - **The user's flow:** photo at WP2 → gentle set-down → re-clamp turned 90° → judge at WP2 → defect plate if anomalous, else re-clamp another 90° → good plate after the third showing passes.
    - **Already built:** `sorter._inspect` does exactly this loop (early stop on a defect, `collect.place_on_tape` for the set-down, `collect.regrip` with one retry, `coverage.py` tracks seen and missing sides). A re-clamp that needed its retry never gives GOOD.
    - **Gap in the flow:** a block has 4 side faces and each grip turns one new side to the camera, so 3 showings (2 re-clamps) leave one side unseen. `coverage.plan` results:
      | WP2 poses ticked | Re-clamps | Complete | Surface seen |
      |---|---|---|---|
      | front only (today: no ticks set) | 2 | no | 47 % |
      | front + up + down | 2 | no (both finger sides uncredited) | 79 % |
      | front + up + down | 3 | yes | 100 % |
      | front + back (wrist roll 180°) + up + down | 1 | yes | 100 % |
      | front + back + down + wrist top photo | 1 | yes | 100 % |
    - **Recommendation:** add a back pose at WP2 (roll ≈ 875 + 2048 = 2923 ticks, limits 0–4090; a 180° roll at WP2 was already done on the arm at 21:14, issue 14). One re-clamp then covers everything, and the re-clamp is the least reliable step.
    - **Still open on the station:** `tape_regrip` (+ `_above`), `sort_good`, `sort_defect` are not taught; no `runs/regrip_test_*` exists; the side ticks on `/sort` are empty; all good crops so far are grip-1 views (auto-collect and hand-feed never re-clamp); no models are trained.
    - **User's decision (≈04:05): four showings = 3 re-clamps, no back pose.** Set on the live server: `max_regrasps` 3; side ticks `wp2-1` front, `wp2-3` front + down, `wp2-2` down. The ticks are an estimate from the kinematic model and the stored photos and need a check with a marked block.
    - **The palm-side face (top when picked) is not seen at WP2:** finger elevation is −19° (`wp2-1`), +11° (`wp2-3`), +40° (`wp2-2`), a 59° spread toward the fingertip side only, and every re-clamp turns about the vertical, so that face stays under the palm in all four showings. It is covered only by the wrist top photo on the tape spot (taken at the first set-down). Plan with it: complete, 3 re-clamps, 10 photos. Without it: 91 %, top missing. So a wrist model per block and `tape_top` training photos are required.
    - **Correction (≈04:25, the user was right): the first showing does reveal three faces.** Checked on the unmasked front frames of run `20261115_033435_auto` (blue cycles 4 and 15, pink 14): `wp2-1` = one free side head-on, `wp2-3` = the fingertip-side face head-on, `wp2-2` = the fingertip-side face again plus the opposite free side at a steep angle. The user calls these top, front and bottom. The earlier ticks (and the claim "front and bottom only") came from the kinematic model, whose absolute angles are unreliable. Ticks now: `wp2-1` front, `wp2-3` down, `wp2-2` back + down. Plan: showing 1 takes all three poses, 4 showings, 3 re-clamps, 8 photos including the wrist top photo, complete.
    - **Unchanged:** the face against the palm is hidden in all four showings (re-clamps turn about the vertical). Without the wrist top photo the plan stays at 91 %, top missing.
    - **Not done on purpose:** a re-clamp option for auto-collect. `autocollect.py` was being edited by another session (03:46) and the mock has no vision pick to test it on.
37. **Vision pick inside the automatic photo collection: every pick recorded to WP1; grasp depth, held check by camera, drag recovery** (2026-11-15 03:05–03:56, intel-f4 with intel-ce).
    - **Recording (the user's request):** intel-ce's `autocollect.py` records each vision pick with `sorter.VRecorder` from the start pose to WP1 (`retries=0`), then takes its WP2 photos and puts the block back. Episodes: `tDataset2/<time>_vpick/`, `source: "vpick"`. First three checked: all cameras complete, largest frame gap 34 ms, end within 53 ticks of WP1.
    - **No training run starts before the user has seen the pick counts** (the user's rule, 03:12).
    - **Grasp depth (03:18):** pink held only 5 of 14 in the first collection runs. Photos: block between the jaws, but the tips reached only halfway down and pinched it out. `Z_SEEK` 0.004 → 0.012 (descent ends ≈12 mm below the calibrated grasp height; it never touched the plate). Pink after: 12/15 (`runs/acttest_20261115_031834` 5/5, `032144` 7/10).
    - **What the remaining misses are:** a block tipped on its side, usually leaning on the plate rim; the arm then misses it again and again (pink 4× at 03:32–03:34). Distance from the plate centre alone does not predict a miss (upright blocks at the same radius are picked), so no rim rule was added.
    - **Drag recovery (03:50, `vpick.drag_in`, `LAST_MISS`):** after a failed pick, if the same-colour block is still within 45 px at the next pick (within 3 min), the arm straddles it with open jaws, drags it toward the plate's middle, looks again and picks. Before the recorded "start", so not in the episode. One case so far: `035033` trial 1 pink missed, trial 2 dragged and held.
    - **Held check by camera (03:50):** two blue "holds" at 03:40 and 03:44 had empty jaws (intel-ce: no block in the WP2 photos). Ticks cannot separate them: empty jaws jam on each other at 1860–1865, a held block reads 1867–1899. Now at WP1 the top camera looks at the plate; a block of the picked colour still there = "dropped" (`seen_after` in the log). Needs exactly one block per colour on the plate. The two episodes were relabelled `missed` (`20261115_033435_vpick/ep0007_034050`, `ep0012_034445`).
    - **Test with all of it, random colour:** `runs/acttest_20261115_035033` 7/8 held.
    - **Open:** why blocks tip over (suspect: the put-back after the photo poses); the wrist camera cannot tell a held pink block from the red jaws, so the camera check uses the plate, not the gripper.
38. **Two SmolVLA runs on 159 episodes with different validation sets; smooth descent as a toggle** (2026-11-15 04:30–04:56, intel-f4, the user's go).
    - **Collection result (intel-ce's runs, ended 04:28):** 77 recorded vision picks, **35 clean** (success and seen where set down: pink 14/37, blue 12/25, green 9/15). After the user stood the tipped blocks up: pink 6/7, blue 6/7.
    - **Dataset `real124-vpick35-smolvla` (`c6c631af`):** 159 episodes, 55,807 frames = the 124 clean teleop picks + 35 verified vision picks. `export_act.py --episodes ~/junction26/episodes --episodes ~/junction26/tDataset2 --exclude qualia_exclude.txt --exclude tDataset2/qualia_exclude.txt --exclude tDataset2/vpick_unverified_exclude.txt` (the last file lists the 8 vision "successes" that were not verified). The export's pause compression takes 6–7 s of stop-and-go out of each vision pick.
    - **Runs (second Modal account, app `physicalai-trainer-b`, H200, 64 CPU, 128 GB, batch 32, 60 workers, 12 epochs ≈ 19.9k steps, `val_split` 0.05, started 04:52):**
      | Model | Job dir on the volume | Validation episodes |
      |---|---|---|
      | `smolvla-real124-vpick35-20k-A` | `jobs/f52ed47a-46a2-41c7-b7e3-71a38be69bfc` | 44, 45, 62, 68, 91, 130, 140 |
      | `smolvla-real124-vpick35-20k-B` | `jobs/4c9a6f02-22a1-460d-a765-77d6cc176fcb` | 13, 28, 40, 56, 79, 92, 156 |
      - Same data and settings; the split is drawn at random per run (no seed in the trainer), read from each container's log.
      - 12 epochs, not 13: 13 would be 21.5k steps. Status: `scripts/modal_b_status2.sh`. Launch record: `runs/modal_b_launched_0450.json`. Not in Studio: import `artifact.zip` by hand.
      - The older runs: `smolvla-new55-30k` completed 04:44; `smolvla-real124-40k` at 96 %.
    - **Smooth descent (`vision_smooth`, off by default):** one move from the hover to the grasp depth plus one sag correction, instead of six corrected steps; no contact stop. Toggle on the Sort page ("Vision pick: smooth descent") and `POST /sort/cfg?vision_smooth=0|1`; the Sort page also got a "Vision pick · scripted" pick button. First test `runs/acttest_20261115_043452`: 5/8 held (stepped the same hour: 7/8), xy error up to 4 mm instead of ≈1 mm. Small sample.
    - **Open:** one smooth "held" (trial 3 of `043452`, pink, 1865 ticks) has no `seen_after`: the camera check could not run there, so that hold is unconfirmed.



38. **Re-clamp guided by the top camera instead of the blind tape-spot turn** (2026-11-15 ≈07:05–07:35, the user: "look at the cameras to tune it better and make it more robust"). **Written and checked offline; not yet loaded on the server or run on the arm.**
    - **What failed (run `regrip_test_20261115_070017`):** no `tape_regrip` pose was saved, so the test used the computed turn (wrist roll +90° only), and the block was lying on the tape, not held (`started_holding: false`, gripper 24 % open). Both attempts closed fully on nothing (stall 0.7 %). The wrist and top frames show the jaws about one block width to the side of the block along their free axis.
    - **Why not tune the blind pose:** it depends on where the block lies after the release, which differs per block width and per pick. The vision pick already finds and grips blocks on the start plate (24/25 measured).
    - **New flow (`sorter.regrip_vision`):** set the held block down on the START plate (`vpick.place`, contact stops the descent) → arm home → top camera finds the block → vision pick with the block and the jaw direction forced: jaw yaw = yaw at the release + 90°, always the same direction (coverage assumption) → WP1. Every attempt looks first, so a retry is not blind. If it fails, the block is simply back on the start plate.
    - **Code:** `vpick.plan_grasps(..., both=True)` also offers a non-square block across its long side (up to 50 mm); `VisionPick(force=...)` + `VisionPick.turn_yaw()` (picks a set-down yaw from which the turned grasp is inside the roll limits); `Sorter.regrip_vision`, `Sorter.regrip_test`, `POST /sort/regrip_test?colour=blue&n=3` (log: `runs/regripv_<time>/steps.jsonl`).
    - **Checked:** `make test` 194 passed / 4 skipped. Offline on a stored top frame with the real calibration: a turned grasp exists at all 20 candidate spots of the plate (roll ≈ 1000 at the set-down → ≈ 1950 for the turned grasp, limits 0–4090), also for a 2×3 in both directions.
    - **First run on the arm (07:20:43, `runs/regripv_20261115_072043`, pink, 1 re-clamp): held on the first attempt.** Pick 21.9 s; re-clamp 29.4 s (set-down → home → look → turned grasp → WP1). Jaw yaw at the release −45.1°, turned grasp +44.9°; the block lay 44 px (≈ 20 mm) from the set-down target and the camera corrected for it; chosen grasp 10.7° from the exact +90° (limit 30°); stall 1921 ticks. A 2×2 looks the same after 90°, so the turn is shown by the logged jaw directions, not by the photos.
    - **Mistake (07:20:17):** the server restart to load this code cut off a vision pick test of session intel-f4, which had started between the idle check and the restart. Restarts are now announced to the other session first.
    - **Context:** the plain vision pick got 6/10 and 7/10 on single attempts in the two tests just before (07:10, 07:15); the re-clamp allows 3 attempts, each with a fresh look.
    - **Not done:** not wired into the inspect flow (`autocollect.inspect`, edited by another session) or `sorter._inspect` (still the tape spot). The front models have only seen first-grip views; green held across its long side is a new view.

39. **Production sort flow as its own module, `anomaly_capture/flow.py`** (2026-11-15 ≈07:45–08:20, the user's spec, intel-43). **Written and tested on a fake station; not yet loaded on the server or run on the arm.**
    - **The flow, per block:**
      1. vision pick off the start plate → WP1 (3 retries);
      2. WP2, first showing: all three WP2 poses, each photo judged at once; the first anomaly → reject plate;
      3. clean → re-clamp turned 90° (`sorter.regrip_vision`), then ONE photo at the side view (first WP2 pose), judged;
      4. repeat until all 4 sides are shown; top and bottom are not inspected (the user's choice);
      5. all clean → accept plate.
    - **Same turn direction for every re-clamp of one block.** +90° then −90° shows the first side again: the −90° grasp is the +90° grasp flipped 180°, so the opposite face points at the camera. The first re-clamp chooses the sign; the later ones are asked for it (`regrip_vision(sign=…)`, requested from intel-2f; the flow passes it only once the argument exists). The flow also tracks the shown side from the reported `turn_deg`, so a wrong-way turn costs one more re-clamp instead of a false "all sides seen". Limit: `max_reclamps` 4 (3 are needed).
    - **Judging one photo** = the path the colour models were trained on (`ctx.capture`, outline mask, lifted poses), the same rules as `autocollect.inspect`: visible not-block-coloured patch → anomaly; model score ≥ `t_high` with nothing visible, or between the thresholds, or block cut by the frame → doubt → one retake. `sorter.look` is not used: its `prep.inspect_crop` path is not what the models saw.
    - **A doubt that stays goes to the reject plate** (`uncertain_to` = `reject`, logged `unconfirmed`); `POST /flow/cfg?uncertain_to=accept` flips it. Open question for the user: their rule from 04:26 says a score alone is no anomaly, their flow says reject on detection.
    - **No block in the photo** after the retake = not in the jaws: cycle aborted, nothing goes on a plate. Two aborts in a row stop the run.
    - **Failed re-clamp** (after its own 3 attempts): the block is fetched from the re-clamp spot and rejected as not verifiable; if the fetch fails too, the run stops and asks for a hand.
    - **Plates:** two taught waypoints (slots `sort_good`, `sort_defect`, optional `_above`). Which is accept and which is reject is a setting: `POST /flow/cfg?accept=sort_good|sort_defect`.
    - **API:** `POST /flow/start?n=0&colour=` (n 0 = until the start plate is empty, max 50; colour empty = nearest block), `GET /flow/status`, `POST /flow/cfg`, `POST /flow/stop`. Log: `runs/flow_<time>/log.jsonl` with per-block pick, inspect, sort and cycle times and the inference ms (feeds §9).
    - **Checked:** `tests/test_flow.py`, 11 cases on a fake station (order of photos, early stop at each stage, direction kept, mixed direction, retake, plate swap, lost block, failed re-clamp, empty plate): 11/11 under Studio's Python. The dev venv skips the file (no lerobot there).
    - **Not checked:** anything on the arm. Known risks: the models have only seen first-grip views, so re-clamped views (green across its long side above all) may score high; both plate waypoints are untaught.
    - **To run:** teach the two plates on the Sort page → restart the tool (announce it to the other sessions) → `curl -X POST '127.0.0.1:7870/flow/start?n=1&colour=pink'`.

40. **User-facing page at `/`; the engineering interface moved to `/dev`** (2026-11-15 ≈08:00, the user's request, intel-43). **Checked on the mock station; the live server needs a restart for the new routes.**
    - **`anomaly_capture/web/operator.html`:** one standalone file (own CSS and JS, same dark tokens and fonts), so work on the dev pages cannot break it. It shows the three camera feeds (refreshed snapshots), "Run the task" (blocks to sort, which colour, accept-plate assignment, fast-pick toggle, Start), a readiness line that names what is missing, the live state with the sides shown so far, totals and the list of sorted blocks with the reason for each. STOP button and Space/Esc stop everything.
    - **Look (the user's brief, ≈08:12):** extremely futuristic, monochrome with light-blue accents (`#8FDBFF`), frosted glass. Black base with a faint grid and blue glows; every panel is a translucent blurred pane; camera tiles have corner brackets and a slow scan line; thin sans headings and mono labels instead of the serif. Red is used only for STOP and faults. The dev interface keeps its champagne/serif look.
    - **Layout (the user's second brief, ≈08:16): sidebar + large camera view, no explanatory text.** Left glass sidebar: arm and camera state, Blocks / Colour / Accept plate / Fast pick, Start, only the lines that say what blocks a start, Dev menu and STOP at the bottom. Main column: one camera fills the width (front by default), the other two float on it as small views and a click enlarges one (remembered per browser); under it one strip with the state, the sides-shown bar and the three counts, then the result rows. The headline, sub-lines and camera captions are gone. The entries below describe the first version where they differ.
    - **Look, third round (≈08:20, replaces the "futuristic" look above):** clean and quiet in the Apple manner. Black and greys only, system/Inter type in normal case, frosted-glass sidebar and status strip, rounded camera views without brackets, grid or scan line. Blue is used in one place: the glow of buttons (Start always when enabled; the Dev button, the switch and the small camera views on hover or focus). STOP stays red. Counts, dots and results are white to grey.
    - **Fourth round (≈08:30, the user: "it doesn't feel like a premium UI"; replaces the layout and look entries above):** one screen, no scrolling. A floating glass control panel on the left; the live camera is the stage and fills the rest of the window; the other two cameras float top right (click = make it the stage); state, a Pick → Inspect → Re-clamp → Sort progress line with the sides shown, and the three counts sit on the stage as one glass bar; a click on the counts opens the result list. Behind everything a greyscale, far out-of-focus copy of the live view gives the glass light to blur. Cool greys only; blue only as the glow of buttons; red only on Stop.
    - **Take control (the user's request):** the Arm row shows a "Take control" button while the arm is not connected (`POST /arm/connect`), and **Start connects the arm by itself** before it starts the flow. A disconnected arm no longer blocks Start; untaught plates, a dead camera or a missing flow still do.
    - **Pause (the user's request), wired end to end:** while a run is on, the main button is Pause; it becomes Resume once pause is asked for. Backend `flow.py`: `POST /flow/pause`, `POST /flow/resume`, `pause` / `paused` in `/flow/status`. The arm stands still at the next pause point (before a pick, a photo, a re-clamp or the plate) in hold mode with the block in the jaws; the move under way finishes first, which the page shows as "Pausing". STOP works while paused and clears the pause. The flow now also puts the arm in hold mode before each block. Tests: `tests/test_flow.py` 14/14 (pause holds before the next step and resume continues; STOP while paused). Not tested on the arm.
    - **Plate A and plate B are waypoints on the dev Waypoints page (the user's request, ≈08:45).** New section "Plates" on `/waypoints` with the slots `plate_a`, `plate_b` and the optional `plate_a_above`, `plate_b_above` (same cards as the other poses: Save current pose, Go, Set). The production flow drops on these (`flow.PLATE_SLOTS`), no longer on `sort_good` / `sort_defect` (those stay with the old `/sort` flow). The user page's "Accept plate" switch (Plate A / Plate B) sets `POST /flow/cfg?accept=plate_a|plate_b`; the other plate takes the rejected blocks. An old `flow_settings.json` is migrated on load. Checked: `/flow/status` shows the new slots, the section renders on the live page, flow tests 14/14. Both plates are still untaught.
    - **Dev menu** (top right): links to `/dev` (the old overview), Connect, Waypoints, Inspect & sort, Anomaly photos. The dev sidebar has "← User view".
    - **Routing:** `ui.operator()` serves `/`; `00-overview.html` has `data-path="/dev"`; `capture.py` root calls `ui.operator()`. All other dev paths are unchanged. Until the restart `/` still shows the dev overview and `/dev` is a 404.
    - **Calls only:** `/status`, `/arm/status`, `/sort/status`, `/flow/*`, `/sort/cfg?vision_fast=`, `/snap/<cam>` and the four stop endpoints. Without `/flow/*` it says the sort task is not loaded.
    - **Checked:** mock server on a spare port: `/` 200 (user page), `/dev` and `/sort` 200, `/flow/status` 200; headless-browser screenshots of both views; `tests/test_api_sweep.py` + `test_api_mock.py` 12 passed (the sweep and `ui_smoke.mjs` now use `/dev` for the overview). Not checked: a real run driven from the page (the mock has no vision pick).

41. **Re-clamp on the tape spot works on the arm: taught waypoints for the set-down, the arm's own release pose for the turned grip** (2026-11-15 07:40–08:08, the user's rules; replaces the plate set-down of decision 38).
    - **The user's rules:** the re-clamp happens at the tape spot, not on the start plate; the grip stored in `tape_above` / `tape_place` is ignored, the clamp is kept until the block stands on the tape; the turn may be +90° or −90°, whichever is in reach.
    - **Flow (`sorter.regrip_vision`):** WP1 → `tape_above` → `tape_place` slowly with the clamp kept → read the measured pose (FK) → open, fixed jaw 6 mm off the block, straight up → `vpick.grasp_world` at that pose with the jaws turned (hover, stepwise descent with sag correction and contact stop, close until stall, lift) → WP1 → the top camera checks that the tape spot is empty.
    - **What did not work on the way:**
      | Try | Result | Cause |
      |---|---|---|
      | Camera-guided grip on the start plate (07:32) | "no reachable grasp" ×3 | the block lay 39° off the release direction |
      | Camera-guided set-down on the tape spot (07:40) | block 34° off | `vpick.place` let go 39 mm above the table (no sag correction, read as contact); the block fell |
      | Camera-guided grip at the tape spot (07:42, 07:44) | closed on nothing ×5 | the pixel calibration is fitted on the plate and is about 37 mm off at the tape spot |
      | Release pose + 90° turn, no correction (07:49) | closed on nothing ×2 | jaws 14 mm beside the block: the model's grasp point is not where the block is after a wrist turn |
    - **Offsets (mm, release gripper frame, x = fixed → moving jaw, y = 90° left):** block centre against the model's grasp point `a = (−21, 7)`, jaw closing point `b = (2, 7)`; target correction `c(turn) = a − R(turn)·b`, so +90° → (−14, 5), 180° → (−19, 14), −90° → (−28, 9). Fitted from two misses seen in the top frames; −90° is not tested. Settings `regrip_a_mm`, `regrip_b_mm`.
    - **Results with the final code (every re-clamp held on its first attempt, about 31 s each):**
      | Block | Turn | Held | Run |
      |---|---|---|---|
      | blue 2×2 | +90° | 4 of 4 | `regripv_20261115_075222`, `_075332` |
      | pink 2×2 | +90° | 3 of 3 | `_080511` |
      | green 2×3 | 180° | 2 of 2 | `_080239` |
      | green 2×3 along its long side | +90° | 1 of 2 (then the block was pushed away twice) | `_075543` |
    - **Green (the user's observation, confirmed in the frames):** gripped along its long side the block is pushed out sideways when the jaws close. A block that is not square is therefore turned 180° and gripped across its short side again (`regrip_long` setting turns the long-side grip back on). Its two long sides stay under the fingers and are not shown to the camera by any showing.
    - **Other:** `regrip_vision(..., sign=±1)` restricts the turn direction (asked for by session intel-43 for `flow.py`); a block left on the tape spot is fetched at the start of `regrip_test` (worked for green on the third attempt, failed for blue: the calibration offset there is only a guess, `regrip_off_px`); the plain vision pick needed 2 attempts for pink and missed twice on blocks at the spot.
    - **Open:** not chained into the inspection yet; no photo of the block standing free on the tape spot (the only way to see green's long sides); the front models have only seen first-grip views.

41. **Review of the production flow from start to finish; the clamp opens only at plate A or B** (2026-11-15 ≈08:45, the user's rule: "the clamp should not release until it is at plate A or B"; intel-43). **Read against the current `sorter.py` / `vpick.py` / `collect.py` / `arm.py`; fixed in `flow.py`; tested on the fake station only.**
    - **Found: three ways the clamp could open away from a plate.**
      1. `Flow.drop` opened right after `C.goto(plate)` returned. `arm.goto` returns once the joints are within 40 ticks, and after 1.5 s it returns anyway unless they are more than 160 ticks (14°) off. So a slowed or lightly blocked arm could let go up to 14° short of the plate.
      2. An aborted block ("no block seen in the jaws") went home with `gripper=open`. `arm.goto` moves the gripper together with the arm, so a block the camera had missed would have been dropped on the way.
      3. A new run after a STOP: the first move of every pick is "home, gripper open" (`flow.cycle` and `VisionPick.run`), so a block still held from the stopped run was dropped in the air.
    - **Fixes:**
      1. `Flow.at(pose)`: before opening, the MEASURED joints must be within `AT_PLATE` = 60 ticks (≈5°) of the taught plate pose (waits up to 2.5 s). If not: `RuntimeError`, the run stops, the clamp stays closed on the block.
      2. An abort with the clamp still closed carries it to the reject plate and opens there (`released_at` in the log). It still counts as aborted, not rejected.
      3. `Flow.clear_gripper` at the start of a run: a closed clamp (below 20 % open) is raised straight up, taken over waypoint 1 to the reject plate and opened there before the first pick. An arm that merely rests with closed empty jaws makes the same trip once (about 8 s).
    - **The clamp now opens in two places only:** on the tape spot inside a re-clamp (the block is gripped again there) and at plate A/B. STOP, pause and errors keep it closed (`hold_here` keeps the commanded gripper).
    - **Checked and found consistent:** the pick ends at waypoint 1 and every showing returns there, which is where `regrip_vision` and `drop` expect to start; `regrip_vision` returns `turn_deg` at the top level and takes `sign`; an empty start plate ends the run (`"brick found on"` in the pick's message, the detector only looks inside the orange plate, so blocks on plate A/B are not picked again); two aborts in a row or a block left on the tape spot stop the run; hold mode is set before the first move (the arm was in mirror mode when I looked).
    - **Known weak points, not fixable without the arm:**
      - "No block seen" uses the colour-mask size (pink needs ≥ 120,000 px because the jaws are red too). A re-clamped pink block that shows less area would be aborted and end on the reject plate.
      - A block cut by the frame edge is a doubt → retake → reject (unconfirmed). Green was cut often in earlier collections.
      - The models have only seen first-grip views; re-clamped views may score high.
      - `Flow.clear_gripper` rises straight up from wherever the arm stands; from a pose low in front of the front camera that path is untested.
    - **Tests:** `tests/test_flow.py` 18/18 under Studio's Python (new: clamp opens only at the plate in accept, reject and late-reject cycles; arm short of the plate keeps the block; aborted block opened at the reject plate; a closed clamp at the start is emptied at the reject plate before the first pick).
    - **State at ≈08:46:** both plates taught (`plate_a`, `plate_b`), tape spot taught, server restarted with this code; the restart dropped the arm connection.

42. **First real runs of the production flow; the arm no longer returns to waypoint 1 between the steps** (2026-11-15 ≈08:41–08:48, the user started the runs from the user page; route fix by intel-43, loaded 08:48:19).
    - **First runs on the arm (code of ≈08:40, before the clamp review of decision 41):**
      | Run | Block | Verdict | Photos / re-clamps | Pick | Inspect | Sort | Cycle |
      |---|---|---|---|---|---|---|---|
      | `runs/flow_20261115_084134` | pink | accept: all 4 sides clean | 6 / 3 (turns +90, +90, +90) | 13.1 s | 162.1 s | 6.5 s | 181.6 s |
      | `runs/flow_20261115_084454` | blue | reject, not confirmed clean | 5 / 1 | 17.0 s | 76.9 s | 8.6 s | 102.5 s |
      - Pink scores: −0.11 … 0.24 over the six photos (clean ≤ 0.59), visible-patch share ≤ 0.003.
      - Blue: first grip −0.35, −0.11, −0.01 (clean ≤ 0.70); after the first re-clamp 0.702 on two photos, 0.002 above `t_low`, so "doubt" → retake → reject. This is the expected weakness: the models have only seen first-grip views. Whether that blue block carried a mark is not recorded.
      - Inspection takes about 54 s per re-clamp cycle (162 s for three).
    - **The user's bug report: "it's always returning to waypoint one after each task".** Each showing ended with above WP2 → WP1, and each re-clamp ended at WP1, so the arm crossed the table to WP1 up to seven times per block.
    - **Fix:** `flow.show` ends at the pose above WP2 and stays there; `sorter.regrip_vision` got `end=None` (the pose to carry the block to after the lift instead of WP1; default unchanged, so `/sort/regrip_test` is as before) and the flow passes the pose above WP2; the plate move starts from above WP2. WP1 is now visited once, at the end of the pick. Route per block: pick → WP1 → above WP2 → photos → above WP2 → tape spot → above WP2 → photo → … → above WP2 → plate.
    - **Tests:** `tests/test_flow.py` 19/19 (new: nothing after the pick goes to WP1; every re-clamp ends above WP2). The two green runs of decision 43 (08:49, 08:51) ran with this route and the clamp review; no collision was reported.

43. **Green rejected although it was clean: the photo was cut by the frame, not the model** (2026-11-15 08:51–08:57, the user: "it was a false positive"; intel-43).
    - **Runs (code with the clamp review and the new route):**
      | Run | Block | Verdict | What happened |
      |---|---|---|---|
      | `runs/flow_20261115_084922` | green | aborted, 145.4 s | First grip clean (0.06, 0.04, 0.12; clean ≤ 0.64). Re-clamp 1 failed ("the fingers closed on nothing"); the fetch from the tape spot closed on nothing 3 times; block left on the spot, run stopped. |
      | `runs/flow_20261115_085157` | green | reject, 93.7 s | First grip clean (−0.08, −0.11, −0.09). After re-clamp 1 (turned +90°) both photos at `wp2-1` were "cut by the bottom of the frame", so no score was computed; doubt → retake (same pose, same result) → reject, logged `unconfirmed`. |
    - **Cause of the reject:** after the +90° re-clamp the 2×3 is held along its long side. It hangs lower and nearer to the front camera (mask area 515,000 px against 400,000 at the first grip) and runs out of the frame at the bottom and the right (`training_images/front/inhand/test/20261115_085320_249_green_wp2-1_masked.jpg`). The anomaly model never scored that view. The flow treated "photo unusable" like "block doubtful".
    - **Fix (`flow.look`):** a block cut by the bottom (top) edge is a bad photo. The photo pose is raised (lowered) 20 mm at a time, at most 4 times (`NUDGE_MM`, `NUDGE_MAX`), with the same pose-lifting code the collection uses (`feed.lifted`), and the photo is retaken; this costs no retake. The lift that worked is remembered per block, pose and first grip / re-clamp for the rest of the session (`lift_mm` in the log). Only if the pose can't go further and the block is still cut does it end as before (reject, unconfirmed).
    - **Not fixed here (reported to intel-2f, their code):** the failed green re-clamp and fetch of the first run; and that this green block was turned 90° although `regrip_vision` is meant to turn a non-square block 180°.
    - **Limits:** `edge_cut` only tests the top and bottom edges; in the photo the block also leaves the frame on the right, which a lift does not fix. A view of the green block held lengthwise is also new to the green model, so once it is in the frame it may still score high.
    - **Tests:** `tests/test_flow.py` 21/21 (new: a cut block is moved into the frame and accepted, the lift is reused on the later showings; a cut block that can't be moved is still rejected). Not run on the arm.

44. **Green always turns 180° on a re-clamp; blended moves through the clearance poses** (2026-11-15 ≈08:59–09:13, the user's two requests; intel-43). **Loaded on the server (start 09:13:28, commit `a158b46`); no block has run on it yet.**
    - **Green 180°:** `sorter.regrip_turns()` decides the turn; `vpick.LONG_BRICKS = {"green"}` makes green non-square whatever the top camera measured (its outline read as square in the 08:51 run and it was turned 90°). Non-square → 180°, opening = the width the pick held. Pink and blue unchanged. Loaded since the 09:10 restart.
    - **The user: "remove the pauses between waypoints except for when it is capturing the images".** Every `goto` stopped at its waypoint and then waited for the joints to settle (0.1–1.5 s), also at pure clearance poses.
    - **`arm.Arm.path(pts, speed)` + `Collector.path(poses, **override)`:** one time law over the whole way (the same `profile()` as `goto`, so "Fast moves" applies), the via points rounded over `CORNER` = 120 ticks before and after (the arm passes about 3.5° inside the corner instead of stopping on it), exact stop and the usual settle / BLOCKED test only at the last pose. For clearance poses only.
    - **The flow uses it for every transfer:** WP1 → above WP2 → photo pose; last photo pose → above WP2 → above the tape spot; last photo pose → above WP2 → above the plate → plate; plate → above it → home. `flow.show` no longer leaves WP2 itself: it records the way out (`exit_via`) and the next move flies through it. Removed: the separate stop above WP2, the extra "home" move at the start of each block, the 0.3 s wait between blocks; the move between two photo poses is 0.6 instead of 0.4 of the speed.
    - **Where the arm still stands still:** at each photo pose (settle before the photo), on the tape spot (set-down and re-grip, inside `regrip_vision`), at the plate (measured arrival, then release), and during the pick.
    - **Tests:** `tests/test_flow.py` 24/24 (new: the path lists per block; `Arm.path` on a fake clock keeps moving through the via point, ends exactly, does not overshoot and is not faster than a single move; green 180° rule). The fast suite on the mock station ran through without a reported failure; I did not capture its count.
    - **Not measured:** the time saved. Reference before the change: pink, 3 re-clamps, inspect 114.6 s, cycle 134.7 s (`runs/flow_20261115_090042`).
    - **Risk on the first block:** the rounded corners cut about 3.5° inside each clearance pose, and the descents into WP2 and onto the plate are now the end of one move instead of a separate slow move.

45. **Photos are judged in the background while the arm moves on; user page: block count 1–3, tinted plate pills, camera views without boxes** (2026-11-15 ≈09:20–09:35, the user's requests; intel-43). **Loaded on the server; no block has run on the background judge yet.**
    - **The user: "the camera should keep capturing and the movements should go on, and not wait for the computation; if the computation says reject there should be an interrupt and it moves to the reject plate".**
    - **What a photo cost before (runs of 09:00–09:21):** about 0.8 s at the pose (≈0.5 s for the 16 averaged frames, the rest mask, crop and files) plus 30–190 ms for the score.
    - **Now (`flow.show_async`, cfg `async_judge`, on by default; `POST /flow/cfg?async_judge=0` = wait as before):**
      - at the pose: settle + the frames only (`flow.grab`: averaged colour, the depth of that moment, the joints);
      - a worker thread (one, in order) does mask, crop, files and the score (`arm.capture(..., shots=, depth=, joints=, out=)`, then `flow.judge`);
      - the arm goes on at once: next pose, then out of WP2 toward the tape spot, or toward the accept plate after the last showing;
      - a result that is not clean sets `flow.alarm`; `arm.Arm.intr` makes the running `goto` / `path` raise `arm.Interrupted` and the arm holds;
      - anomaly → reject plate from where the arm stands (first back over WP2 if it had left it);
      - doubt, block cut by the frame, no block seen → back to that pose and the careful look (`flow.look`: retakes, re-posing), then on;
      - every result is collected before a re-clamp sets the block down and before a plate release.
    - **Expected gain:** about 0.3–0.5 s per photo, a few seconds per block. Not measured.
    - **Tests:** `tests/test_flow.py` 29/29 (new: a clean block is never held up; a late anomaly interrupts the move to the tape spot and ends at the reject plate; a doubt goes back, is cleared and the flow carries on; a doubt that stays is rejected; the re-clamp waits for a verdict that is still out); mock-station capture and motion tests 35 passed. **Not run on the station:** `capture` with frames grabbed earlier on the real camera. If it throws, the photo counts as a doubt and the arm goes back for the normal look.
    - **Runs on the blended moves and the green rule (before this change), `runs/flow_20261115_091718` … `092121`:**
      | Block | Verdict | Re-clamps | Inspect | Cycle |
      |---|---|---|---|---|
      | blue | accept, 4 sides clean | 3 | 94.4 s | 110.7 s |
      | blue | reject: visible patch, 8 % of the block, 4th side | 3 | 96.3 s | 112.0 s |
      | blue | reject: visible patch, 11 %, 4th side | 3 | 97.6 s | 143.6 s (pick 42.0 s) |
      | green | accept: both ends clean, long sides not inspected | 1, turned 180° | 46.9 s | 65.3 s |
      | pink | accept, 4 sides clean | 3 | 94.2 s | 112.9 s |
      - Pink with three re-clamps: inspect 114.6 s before the blended moves (09:00), 94.2 s after: 20 s less.
      - The last three were one run of three blocks ("start plate empty" ended it).
    - **User page (`web/operator.html`):**
      - Blocks: 1, 2, 3. With a colour chosen only 1 can be selected (three blocks, one per colour); with Any all three.
      - Accept-plate switch: the accept plate's pill is tinted green, the other one slightly red.
      - Camera views without the detection boxes: new `GET /view/<cam>` (`capture.py`: the frame before the boxes are drawn, 1280 px wide instead of 640). The dev pages keep `/snap/<cam>` with the boxes.

46. **Where a block's time goes; shorter photo stop, photo poses in least-travel order, "Ludicrous" speeds** (2026-11-15 ≈09:35–09:50, the user's choices from the list below; intel-43). **Loaded on the server; not yet run on the arm.**
    - **First run on the background judge (`runs/flow_20261115_092948`, 3 blocks, start plate emptied):**
      | Block | Verdict | Re-clamps | Pick | Inspect | Sort | Cycle | Energy |
      |---|---|---|---|---|---|---|---|
      | blue | reject: tape, visible patch 6.4 % (limit 5 %) on the side shown after re-clamp 1 | 1 | 12.7 s | 40.6 s | 3.6 s | 56.8 s | 1,158 J |
      | pink | accept, 4 sides clean | 3 | 14.7 s | 93.0 s | 2.7 s | 110.5 s | 2,292 J |
      | green | accept, both ends clean (180° turn) | 1 | 15.1 s | 42.8 s | 2.7 s | 60.6 s | 1,247 J |
      - All 14 photos judged in the background, none failed, none looked at again. Per photo: frames 0.78–1.13 s (arm still), mask + crop + files 0.23–0.38 s and score 28–437 ms in the background.
      - The blue reject was right: woven tape across the block (`training_images/front/inhand/test/20261115_093041_529_blue_wp2-1_masked.jpg`). **The Anomalib model scored that photo 0.596, below its clean limit of 0.70**: the tape was caught by the visible-patch check (`prep.foreign_share`), not by the model. The blue thresholds were set from clean photos only (decision of 05:40) and need defect photos.
      - Pink's highest clean score was 0.53 against its limit of 0.59 (after re-clamp 1).
    - **Timeline of the pink block (from the file times):** pick 14.7 s; WP1 → first photo 5.3 s; two more angles 8.3 s; three re-clamp cycles of 26 s photo to photo (≈10 s to the tape spot, set-down and line-up; 4 s descent; 4 s grip and lift; 8 s back to the photo); plate + home 2.7 s. Pure travel between the taught poses is about 1 s per leg (`arm.profile` on `routines.json`: tape_above → WP1 87° = 1.06 s, plate → home 112–122° = 1.3 s). About 70 % of the block is re-clamping, and most of that is stops, not travel.
    - **Options put to the user (estimates, none measured):** (1) show the opposite side by a 180° wrist roll so 4 sides need 1 re-clamp instead of 3, ≈44 s, needs photos and a retrain; (2) trim the re-clamp (set-down speed, ≈1 s of fixed waits, merged moves), 12–18 s; (3) shorter photo stop, ≈4 s; (4) photo poses at full speed in a better order, ≈2 s; (5) skip WP1 after the pick, ≈2 s; (6) faster moves, 3–4 s. Not worth it: moving the tape spot (travel is ≈1 s). **The user chose (3) and (4), then asked for the speed options.**
    - **(3) Photo stop (`flow.settle`, `flow.grab`, cfg `photo_frames` 8, `photo_still_s` 0.25, `photo_wait_s` 0.2):** 8 averaged frames instead of 16 (`State.averaged(cams, n)`), still for 0.25 s instead of 0.5 s, 0.2 s for the swing instead of 0.4 s. The careful look and the waiting flow use the same stop. `POST /flow/cfg?photo_frames=16&photo_still_s=0` restores the collection's values. **Open: the models were trained on 16-frame photos; 8 frames carry ≈1.4× the sensor noise. Check on the first runs that clean blocks keep their scores (pink was already at 0.53 of 0.59).** `frames` and `settle_ms` are in every shot of the log.
    - **(4) Photo poses (`flow.order`):** the first pose stays first, then always the nearest: wp2-1 → wp2-3 → wp2-2, 1,363 instead of 2,017 ticks of joint travel (wp2-1 → wp2-2 alone is 120°). Moves between poses at full speed instead of 0.6. The arm leaves straight up from the last pose (`above_wp2` of that pose) instead of first going back to wp2-1.
    - **"Ludicrous" (user page, row under Fast moves):** Off / 1.3× / 1.5× → `POST /sort/cfg?motion_fast=1&motion_speed=1|1.3|1.5`. Uses intel-2f's fast profile; 1.5 is its cap (`arm.FAST_VMAX`, ≈195°/s). Not tried on the arm; a held block may swing or slip at these speeds.
    - **Tests:** `tests/test_flow.py` 30/30 (new: the pose order).

---

## 5. Issues and fixes log

| # | When | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 0 | 11-14 | `follower_torque_off.py` targeted the leader | It was written while the arm roles were still reversed | Fixed to follower serial 5B3D040988 when moved to `junction26/scripts/` |
| 1 | 11-13 | Arms not accessible | User `intel` isn't in the `dialout` group | `sudo setfacl -m u:intel:rw /dev/ttyACM*` (temporary) |
| 2 | 11-13 | The "top" camera saw nothing useful | The D415 was physically blocked, and later sat at table height | Unblocked; still needs an overhead mount |
| 3 | 11-13 13:34 | "Studio not working properly" | Unclear at the time | Restarted backend and UI. In hindsight, issue 4 was probably the real cause. |
| 4 | 11-13 23:31 | No camera feed in Environments; follower rigid | Replugging the arms reset the port permissions and swapped ttyACM0/1. The robot worker couldn't open the port, so the **whole environment** failed to start, cameras included. | Re-ran `setfacl`; the environment loaded with all cameras |
| 5 | 11-13 | Leader/follower confusion | Early notes had the serials reversed, so a torque-off went to the leader by mistake | The teleop test confirmed Studio's config (leader 5B14111392, follower 5B3D040988). Notes corrected. |
| 6 | 11-14 00:08 | No Modal option under Training Targets | The installed Studio was older than the `junction2026` branch | Backed up, synced to `junction2026`, ran `uv sync --extra xpu` and `npm install`, restarted |
| 7 | 11-14 00:15 | Modal "Deploy and connect" failed with a generic error | Studio discards the deploy output. Running the deploy by hand showed **"Please add a payment method to use A10G GPU functions."** | Modal declined every card, so Modal was dropped and we train locally on the XPU |
| 8 | 11-14 | OpenVINO lists only CPU and GPU, no NPU | The kernel driver is loaded, but the **user-space NPU driver** isn't installed and isn't in Ubuntu's repositories | Open: install Intel linux-npu-driver v1.38.0 (§6) |
| 9 | 11-14 01:40 | Capture tool: no brick detected and the zoom panels blank on every camera | B (empty background) was pressed with the brick still on the spot, so the brick became part of the "table" and the difference detector couldn't see it | Deleted the bad backgrounds. B now refuses if colour detection finds anything in the area (Shift+B forces it), and `/background/clear` was added |
| 10 | 11-14 01:50 | Auto-capture took duplicate shots, then got stuck at 6/60 | (a) "Brick moved" was a raw pixel difference against the last shot, so noise, shadows and people's lighting changes triggered it (3 identical-pose shots at 01:48:48/52/55), while a real 90° turn of the square brick looked "the same". (b) The front camera found 0 bricks: a plate nudged after B left a thin rim arc that merged with the brick, and the merged blob touched the area edge and was rejected. | Auto mode now tracks the brick's **geometric pose** (centre + minAreaRect angle, respecting 90°/180° symmetry). It shoots after 0.8 s of stable pose, if it moved ≥12 px (~6 mm) or turned ≥10°. After 4 s of "not moved enough" it shows the measured move and offers **Space = shoot anyway**. The tick is wrapped in try/except so it can't die. Detection removes thin structures with a per-camera filter (top 9, front 15, wrist 13 px at ¼ scale). New **"BACKGROUND OUTDATED"** warning when >4% of the area differs from the background. Also fixed a stillness-timer bug found while testing. |
| 11 | 11-14 ≈02:10 | User frustrated with auto-capture; then the front preview looked frozen | Auto mode was too complex and unreliable. The page held 6 never-ending MJPEG streams, at the browser's ~6-connections-per-host limit, so one feed could starve. A camera read error also silently killed that camera's thread. | **Simplified to manual only:** one click = one capture from all cameras, never refused (no brick → the whole area is saved, flagged `brick_found: false` in meta), clicks are queued and never dropped, auto mode removed. Previews are now refreshed snapshots with no open connections. Camera threads catch errors and **reconnect after 3 s without frames**, and the page shows "FROZEN … auto-reconnecting" with frame age per camera. |
| 12 | 11-14 ≈03:25 | Tape-spot 90° re-grip doesn't work on the real arm | Not diagnosed yet. Candidates: `tape_place` too high or too low (the block tilts or drops when released), gripper not open wide enough for the turned block, block not centred under the fingers after the turn, or the turned block too wide for the gripper | Disabled by default (decision 19). To debug later with the "test re-grip" button, watching each step. 04:30: the re-grip now opens to 90 % and closes until the fingers stall (decision 20), which covers the two width-related candidates; not yet retried on the arm. 18:30: likely cause found (one fixed jaw, decision 31): optional taught `tape_regrip` pose, step-by-step test with logs, coverage-safe retry; not yet tried on the arm |
| 13 | 11-14 ≈16:45 | In-hand front photo: the depth mask keeps the wrong object | The only real WP2 capture (`training_images/front/inhand/good/20261114_032612_113_blue_wp2-1`) holds the brick so close to the front D415 (≈ 475 px across) that it is inside the depth camera's minimum range: only 9 % of the brick area has depth (24 % of the whole frame). `inhand_mask` therefore kept a small blob at the top of the image and greyed out the brick. Every live held photo would come back as "no depth mask" → uncertain → defect plate. Measured brick size per camera (crops of the same blue 2×2): top ≈ 60–70 px, wrist ≈ 100–120 px (yellow cast, dark, blur), front on the table ≈ 250–300 px. Top camera: 34 reconnects since start. | Fixed in code, not yet on the camera (decision 31): front depth at 424×240 + optional disparity shift with auto-tune; `inhand_mask` refuses implausible clusters instead of masking the wrong object |
| 14 | 11-14 ≈21:45 | Front camera moved higher: everything tied to its old position is stale | The user raised the front D415 so the arm can articulate freely at WP2 and the camera can see the brick's hollow bottom after a 180° wrist roll (flex alone reaches only −17° to +1° finger elevation at WP2). Measured from two 21:14 photos: the block shifts ≈ 11 mm sideways under roll 180, opposite to the sim model (the model's gripper is mirrored: TCP +15 mm on the wrong side). The last three masks also included the red jaws and a strip of orange tape (colour mask). | Old depth background cleared at 21:45 so it can't be used by mistake (backups: `anomaly_capture/front_depth_bg.before_front_move.png`, `cam_settings.before_front_move.json`). **To redo after the move, in order:** restart the capture tool (front depth dropped to colour-only on the unplug) → set the front crop (ROI) → store the depth background (arm parked away from WP2) → re-teach WP2 + `wp2_above` → brick held at WP2 → Auto-tune depth → check `mask_ok` and one `_masked.jpg` per WP2 pose → then exclude jaws and tape from the colour mask. Old front in-hand photos don't match the new view. **21:57:** camera in place, the user taught three new WP2 poses by hand (wrist flex 2194 / 889 / 2874, block ≈ 17 cm above the table); new depth background stored (84 % valid, all shifts; the arm at home is in it at ≈ 350–480 mm). Still open: front crop, auto-tune with a brick at `wp2-1`, jaws/tape in the colour mask. Until then the depth window is 75–174 mm and photos use the colour mask. **22:10, first photos at the new position (21:59, pink):** the held brick is 115–135 mm from the camera, inside the depth window, but only 15–50 % of it has depth, so the colour mask stays the one in use and auto-tune is optional. `wp2-2` shows the hollow bottom well. `wp2-3` is cut off by the bottom frame edge (only the top studs visible) and `wp2-1` touches it: re-teach both higher. Colour mask fixed in `prep.brick_mask`: a brightly lit brick face (saturation ≈ 170) was dropped at `wp2-2`, now kept by growing into connected same-hue pixels ≥ 140; pink/red hue range narrowed to 168–1, which drops the orange wrist-camera mount (hue 1–7) at `wp2-3`. The red jaws stay in the mask: same hue as the pink brick (175–179), can't be split by colour. Checked on all 9 front in-hand photos; 194 tests pass. Front crop set to [750, 280, 1700, 1080]. Loaded by the ≈22:07 restart. **22:08 handling:** all three masks correct (lit face kept, no orange mount); the computed `wp2-3` (halfway between poses 1 and 2) is fully in frame. Left: `wp2-1` still cut by the bottom edge (raise 10–15 mm); the `wp2-3` photo is ghosted, the arm was probably still settling during the 16-frame average. **22:13 handling: passes.** `wp2-1` raised 12 mm, `wp2-3` recomputed, and the feed now waits for the joints to stop before each photo: the whole brick is in frame at all three poses, no ghosting, masks hold only brick + jaws. Only the jaw tips touch the bottom edge at `wp2-1`. Front in-hand photos before 22:13 are from older poses or framing: keep them out of training. **23:17:** all front in-hand photos so far (42 poses × 4 files, 21:14–22:20, plus `meta.jsonl`) uploaded to the private HF dataset as `stills/training_images_inhand_20261114_2220.tar` (93 MB, size checked). `unpack.py` only restores `stills/training_images.tar`; untar this one by hand into `~/junction26`. |
| 15 | 11-15 00:00 | Green block: in-hand photos got no mask | The first green handling (3 poses) was refused with "no green blob in the picture", so no masked image or crop was saved. The lime-green block measures hue 23–30 and saturation 120–200; the colour range was hue 40–88 with saturation ≥ 190. Its glossy hollow underside also reflects grey, which splits the colour blob. The block is larger and was clamped by one end: it sits right of centre and its bottom is cut by the frame edge at `wp2-1`. | `prep.brick_mask`: green hue 22–88 (the yellow plate is 18–20), per-brick saturation thresholds (150 / 100), and for green the mask is the blob's convex hull. Checked on the 3 green photos; pink and blue unchanged; 70 mask tests pass. Loads at the next capture-tool restart. Open: re-take the 3 green photos; raise `wp2-1` ≈ 15 mm for green or grip it centred. **01:35, check of 24 green handlings (01:03–01:31, 72 photos, all masked):** not good enough to train on, retake. Position: the block was clamped in at least six different ways (studs up, hollow side up, on its side, by one end, deep inside the jaws); its centre moves ±83–123 px vertically and its visible area varies ±30 %; the bottom frame edge cuts it in 19 of 24 photos at `wp2-1` and 8 of 24 at `wp2-3`. Segmentation: the convex-hull mask took in the red jaw where it sits in front of the block (up to 9 % of the mask) and background in the hull's corners. Mask refined: inside the hull, jaw-red and saturated non-green pixels are removed (red in mask now ≤ 0.7 % on all 75 green photos; pink and blue unchanged; 70 mask tests pass). Needs a restart to load. For the retake: one fixed way of clamping the green block, and `wp2-1` higher for green. **01:38:** the user chose to recapture: all 75 old green photos (294 files) and their `meta.jsonl` lines deleted; pink (276 files) and blue (310) untouched. Green now has its own lift in `anomaly_capture/feed_lift.json`: `wp2-1` +28 mm, `wp2-3` +13 mm, same angles; refined mask live since the 01:36 restart. Check the first new green handling before collecting the rest. **01:40–01:50:** first green handling with the lift is whole at all three poses. A green block photographed at 01:39 with the selector still on pink (reset by the restart) gave jaw-only crops labelled pink: deleted at the user's request; the feed page will keep the selector across restarts and warn on a pink mask under 120k px. The user found the green mask not clean enough (ragged edge, bites where shadows were taken for background, a grey halo): reworked so that everything inside the hull is block except a red jaw in front of it and the bright orange plate, with a smoothed outline 3 px inside the edge. 70 mask tests pass; pink and blue unchanged; existing green masked images and crops regenerated. **Pink clean-up (11-15 ≈02:00, the user's decision):** the 14 pink captures from before 22:12 on 11-14 (old camera position or old poses, incl. the flex and roll test views; 48 files) and their meta lines are deleted. Pink now has 19 handlings × 3 views = 57 crops, all at the current poses. The HF tar `stills/training_images_inhand_20261114_2220.tar` still contains the old ones. |
| 16 | 11-15 03:00–04:30 | Anomaly photos of blocks picked by the arm | Until now every anomaly photo was hand-fed, but the sorter will see blocks the arm picked itself, which hang differently in the jaws. New `anomaly_capture/autocollect.py` (`POST /autocollect/start?per_block=12&since=…`): top-camera look → vision pick of the colour with the fewest picked handlings → WP2 photos at the three poses → put back at a random spot → repeat; each pick is also recorded as an episode up to WP1 (`sorter.VRecorder`). Found on the way: a picked block hangs lower than a hand-fed one (poses lifted: `wp2-1` +22 mm, `wp2-3` +12 mm; green `wp2-1` −6 mm on top of its feed lift); two blue "held" picks had empty jaws (now caught by intel-f4's top-camera held check); blocks tip over after some put-backs and are then missed until stood up. | Done at 04:28: 12 arm-picked handlings per block, 112 photos kept after checking each by eye (pink 47, blue 39, green 26; green has only 5 at `wp2-1`). The loop deletes photos with no block or cut by the frame edge on the spot and skips a colour after two misses in a row. Picks in these runs: pink 12 held / 11 missed, green 13 / 2, blue 14 / 9. Total good crops now: pink 104, green 119, blue 116. |
| 16 | 11-15 ≈01:35 | Green 2×3 cut off by the frame's bottom edge in the hand-feed photos | intel-ce's check of the 24 green handlings (01:03–01:31): cut in 19/24 at wp2-1 and 8/24 at wp2-3, 0/24 at wp2-2. The poses were taught with a 2×2; the longer block hangs lower. Also at least six different clamp orientations, so size and centre vary. | Per-brick lift for the hand-feed capture only: `feed_lift.json` (`green`: wp2-1 +28 mm, wp2-3 +13 mm, straight up at the same angle, by the kinematic model), `Feeder.lifted`, `POST /feed/lift?brick=&pose=&up_mm=` (no restart). Taught poses and pink/blue unchanged. Loaded 01:36 with intel-ce's refined green mask; Confirmed on the arm 01:40 (intel-ce): block whole at all three poses (mask y 171–928, 288–775, 244–971 of 1080), no jaw in the mask. The old green photos were deleted at the user's request (01:38) and are being retaken. (intel-2f) |
| 17 | 11-15 04:33–04:45 | PatchCore models trained, clean photos only | The user chose to train without defect photos. `train_anomaly.py --brick <b>` (PatchCore, wide_resnet50_2, 256 px, Intel GPU, about 40 s each, OpenVINO export) on hand-fed + arm-picked crops, 20 % of handlings held out: blue 31 train / 8 held-out handlings, pink 29 / 7, green 34 / 9. The page's calibrate button needs defect crops, so thresholds were set by script. | Models in `models/front/{blue,pink,green}`. Held-out clean scores (min / median / max): blue −0.32 / 0.16 / 0.52 (24 crops), pink −0.38 / 0.09 / 0.56 (19), green −0.14 / 0.11 / 0.53 (26); hand-fed and arm-picked score alike. Synthetic check: a black pen cross drawn on each held-out crop scored 1.32–1.89 (blue), 1.53–2.34 (pink), 1.27–2.22 (green), all above every clean score. `thresholds.json`: t_low = clean max + 0.15, t_high = clean max + 0.35 (blue 0.67 / 0.87, pink 0.71 / 0.91, green 0.68 / 0.88). **Not tested on a real defect or in the live sorter.** |
| 18 | 11-15 04:45–05:42 | A blue block with grey tape on it was judged clean | Live test (`POST /autocollect/inspect?colour=blue`: pick → 3 photos → score → put back). Three causes, found one after the other. (1) The mask was built from the block's COLOUR, so tape (not blue) was cut out of the picture and the model never saw it; with tape over the whole face no blue blob was found and the photo was thrown away. (2) With tape across the block the colour blob split and the mask kept only one half. (3) PatchCore alone does not separate tape from the normal spread between views: the taped photos scored 0.07–0.72 against clean held-out scores up to 0.65, and 384 / 512 px variants were no better (512 px also needs 3.7 s per photo). Synthetic test (`anomaly_capture/eval_defects.py`, paints tape, pen lines, crosses, dots on held-out photos): pen crosses are caught, most tape and thin lines are not. | `prep.brick_mask` now returns the block's OUTLINE (convex hull over every piece of the block's colour within ~2 cm, minus jaws and plate that reach outside it; pink: gaps up to ~15 mm closed), all crops regenerated and the three models retrained (05:38). New direct check `prep.foreign_share`: largest patch inside the outline that is not block-coloured; limit 6 % (clean photos: at most 4.2 % over 339; the real tape: 22–25 %). Verdict per photo: foreign patch ≥ 6 % or model ≥ t_high → anomalous; block held but its colour hidden at a pose → anomalous; cut by the frame or between thresholds → uncertain. Thresholds: blue 0.70 / 0.90, pink 0.59 / 0.79, green 0.64 / 0.84. **Live result 05:42: the taped block → ANOMALOUS** (22 % foreign at `wp2-1`). Still weak: black tape, thin pen lines and dots (dark marks rely on the model alone), a defect on a face that no pose shows, yellow tape on green. **Further live tests on the same taped blue block, 05:44–05:55:** (1) CLEAN, tape under the jaw (hidden); (2) CLEAN but WRONG: tape visible on the lower side face at `wp2-2`, missed because the outline ends where the blue ends and the tape took the table's colour (foreign 0.7 %); (3) CLEAN, tape on a face no pose showed (confirmed by the user: correct for the visible faces). After (2): the foreign check counts darker coloured patches (`v > 45`) and the limit is 5 % (missed photo now 5.5 %; worst of 339 clean photos 4.4 %). A depth-aided outline was tried by a subagent and rejected: depth covers only ≈ 8 % of the block and almost none of the tape, and it gave 24 false alarms on 235 clean photos (`anomaly_capture/eval_outline_depth.py`; `brick_mask(depth=…)` exists but `inspect_crop` does not pass depth). Open: hidden faces need the re-grip; tape running off a block edge is caught only barely. **Blind live tests, 05:59–06:08 (the user did not say beforehand whether tape was on):** pink without tape → CLEAN (correct); pink with tape under a jaw, a sliver showing → UNCERTAIN at `wp2-1` (0.71; correct reject); green with tape on the top face, visible at `wp2-1` → the checks MISSED it (0.50, foreign 1.4 %: the tape runs to the face's edge and falls outside the outline) and the block was rejected only because `wp2-3` was cut by the top of the frame; green without tape → CLEAN (correct; foreign 4.6 % from a reflection, close to the 5 % limit). **06:17–06:23, all three in sequence:** blue with a large tape inside the hollow → ANOMALOUS at all three poses (8 / 24 / 8 %; correct); pink without tape → ANOMALOUS from the model alone (0.89 at `wp2-1`, nothing visible): a FALSE ALARM; green without tape → picked only at the fifth attempt (it was pressed against the pink block), then clean on two poses and cut by the top of the frame at `wp2-3`. **The user's rules, now in `autocollect.inspect`:** never call a block anomalous when nothing is visible on it (a model score ≥ t_high without a foreign patch is only "uncertain"); a failed pick is tried again (3 retries); green `wp2-3` lowered by 12 mm for picked blocks. |
| 15 | 11-14 ≈22:20–23:30 | Hand-feed capture rejects a 2×2 held on edge ("the clamp closed on nothing") | The follower's jaws are flexible truss fingers: the gripper reading says little about the object's width. A 2×2 held across 32 mm reads 1919–1932, a green 2×3 1955–1979, the tips touch at ≈1882 and the empty clamp stalls at 1849. With the 2×2 on edge (across its 19 mm side) the fingers wrap around it and stall at 1853: 4 ticks from empty (one try, 22:32), inside the noise. Closing further does not help, because the tips already press on each other. | (a) 22:20: gripper servo `Min_Position_Limit` lowered 1875 → 1755 (`arm.set_grip_floor`, `gripper_floor.json`, `POST /feed/clamp?extra=120`) and the empty-clamp stop measured once (1849); `Feeder.held` = 8 ticks above that stop. Not enough for the on-edge grip. (b) 23:30: a grip inside those 8 ticks is no longer rejected and reopened. The clamp stays shut, state `thin`; a second Close clamp confirms the block is there and the photos go on; later thin grips in the same session don't ask (reset by Get ready); Open clamp cancels. The final message says "thin grip: check the block is in the photos". Tested with a fake arm (`t_thin.py`); **not yet tried with a real on-edge block**, so whether the wrapped fingers hold it through the three WP2 poses is open. The sorter's hand-feed still uses the old 50-tick rule. (intel-2f) |

---

## 6. Open blockers

### Modal: dropped, then working again (2026-11-14 ≈12:58)
- **Update ≈12:58:** the manual deploy (command below, with the saved target's app/volume names `physicalai-9449e57a…`) now **succeeds**.
  - Studio kept showing "degraded", because it holds a failed deploy in memory until the backend restarts.
  - After a backend restart (kill by PID, then the usual `serve` command) the target reports **healthy**.
  - First run: job `75217bde…`, ACT on `pick-sim604-real1x10`, **A100-80GB** (the user's choice), 15 epochs, batch 32, augmentation on, val split 5%. Submitted 12:59 via `POST /api/jobs:train` with `training_target: remote` and `modal_gpu: A100-80GB`.
  - **Confirmed: Modal runs GPU jobs now.**
    - Timeline: upload of the 4 GB dataset to the volume took about 7 minutes. The container started at 13:06, and training started at 13:07 on an NVIDIA A100-SXM4-80GB.
    - Speed: **about 3.5 steps/s at batch 32**, probably limited by video decoding, not the GPU. 15 epochs = 54,450 steps ≈ 4.3 h (≈ $11–15).
    - The user canceled at 13:10 to wait for the real picks (sim-only test run not needed).
  - **Gotcha: canceling in Studio doesn't stop the Modal container.**
    - The container kept running (and billing) after Studio said "canceled".
    - Check with `.venv/bin/python -m modal container list --env main`.
    - Stop with `.venv/bin/python -m modal container stop --yes <id>`.
  - **Sizing the real run:** about 4–6 epochs is ≈1.5–2 h for this data size. Check the steps/s in `modal app logs` after 2 minutes.
- Original note:
- The $100 credit was redeemed, but Modal requires a payment method before allowing GPU functions, and it declined every card we tried.
- **We decided to stop pursuing Modal and train locally** (decision 5).
- An unused Studio training target named "Modal" (status: degraded) is still saved. It's harmless; delete it in Settings → Training Targets if it's in the way.
- If Modal ever becomes possible, its real errors can be seen by running the deploy by hand:
  ```bash
  cd ~/physical-ai-studio/application/backend
  PHYSICALAI_MODAL_APP=physicalai-a6963d32074d4bdcb9939eddf65e7f71 \
  PHYSICALAI_MODAL_VOLUME=physicalai-training-a6963d32074d4bdcb9939eddf65e7f71 \
  PHYSICALAI_MODAL_IMAGE=ghcr.io/open-edge-platform/physicalai-trainer-cuda:main \
  .venv/bin/python -m modal deploy --env main src/trainer/modal_app.py
  ```

#### Real training run and the CPU bottleneck (2026-11-14 16:24)
- **Run:** job `f7a3207a`, model `act-pick-sim610-real69x2-qualia`.
  - Dataset `e25af90c`: 748 episodes, 172,502 frames.
  - A100-80GB, 4 epochs = 20,620 steps, batch 32, augmentation on, `val_split` 0.05.
- **Timing:** dataset upload about 5 min; training started at 16:30.
- **GPU:** PyTorch reports `NVIDIA A100 80GB PCIe`. Modal's function card still says "A10G", but that is the deployed default; Studio overrides the GPU per call.
- **Speed:** 3.3 steps/s. Expected finish about 18:15, cost about $5.50.
- **Bottleneck = CPU:**
  - The container uses 9.6 of its 8 reserved cores to decode two AV1 frames per sample plus augmentation.
  - `num_workers="auto"` is capped at `min(cpu_count, 8)` (`library/src/physicalai/data/lerobot/datamodule.py:173`).
  - So neither a second GPU nor an H100 would help, and more cores only help together with more workers.
- **Prepared for the next run** (Studio, `src/services/training_backends/modal.py`; backup in the session scratchpad):
  - Optional per-call overrides `PHYSICALAI_MODAL_CPU` / `PHYSICALAI_MODAL_MEMORY_MB`, passed with `with_options`. They merge with the GPU override, and no redeploy is needed.
  - It takes effect after a backend restart, **only once no Modal job is running**:
    `cd ~/physical-ai-studio/application/backend && PHYSICALAI_MODAL_CPU=16 uv run --no-sync physicalai-studio serve >> ~/.local/share/physicalai/logs/studio-backend.log 2>&1 &`
    (stop the old backend by its PID first).
  - In the training request, also send `"num_workers": 16`. The API accepts 0–16.
  - Expect roughly 2× faster (about 50–60 min for the same run) for about $0.40/h more. Unmeasured: compare steps/s against 3.3 in the first 2 minutes.
- **A running container cannot be resized,** and remote training cannot resume, so the current run keeps its 8 cores.
- **Preempted** (17:26):
  - The first container stopped at step 10,660/20,620 (52%) after "Modal monitoring unavailable: Deadline exceeded".
  - Modal re-ran the same call (`fc-01M7D7DG…`) on a new container at 17:27 from step 0, even with `retries=0`. That's a GPU preemption.
  - The new container ran 2.4 steps/s. Studio's progress bar kept showing the old 54%, so it hid the restart; check the step counter in the logs.
  - About $3 of compute lost.
- **Cancelled and relaunched** (≈18:00, the user's choice):
  - `POST /api/jobs/{id}:interrupt`; 0 containers left.
  - Backend restarted with `PHYSICALAI_MODAL_CPU=16` (PIDs 561526/561531).
  - New job `cd72ad9a`: same settings plus `"num_workers": 16`.
  - **Measured:** `NVIDIA A100-SXM4-80GB`, step 1 → 1,560 in 299 s = **5.2 steps/s**, against 3.3 with 8 cores and 8 workers: **1.6× faster**. This confirms the CPU/decoding bottleneck.
    - 20,488 steps take about 65 min, for about $3.50/h (A100 $2.50 + 16 cores ≈ $0.75 + memory) ≈ **$4 per run**.
    - The dataset was still on the Modal volume, so the upload was skipped and training started 2.5 min after submit.
    - It isn't 2×, so the GPU and the main process now share the limit. 32 cores would probably add little: the API caps `num_workers` at 16.
  - **Finished 19:10.** 20,488 steps, no preemption. The container scaled down by itself (0 left).
    - Model `5d03db18-3f04-41c9-88be-0ec3a5dec7b4` (1.2 GB) with exports torch, onnx, **openvino**, executorch.
    - val/loss per epoch: 0.271 → 0.233 → 0.221 → **0.217**. Still improving slightly at epoch 4, so 6 epochs might help a little.
    - Sim evaluation: `sim/evaluate.py --backend openvino --device GPU -n 30` → `sim/eval/20261114_191137/`.
      - First looks: pink alone, picked to WP1 in 5.5 s. Green with 3 bricks on the plate, hovers and times out.
      - ACT has no language input, so it can't choose a colour on a crowded plate.

### NPU: user-space driver missing
- **Install it** (needs sudo):
  ```bash
  sudo apt install -y libtbb12
  sudo dpkg -i ~/Downloads/npu-driver-v1.38.0/*.deb
  sudo modprobe -r intel_vpu && sudo modprobe intel_vpu   # or reboot
  ```
- **Verify:** `python -c "import openvino as ov; print(ov.Core().available_devices)"` should list `NPU`.
- **Why we're leaving Level Zero alone:** we keep the existing Level Zero loader (`libze1` 1.28), because the GPU stack uses it. Intel's guide suggests 1.32 from a PPA; only move to it if the NPU doesn't show up.

---

## 7. Next steps

### 7.0 Morning flow (built overnight 2026-11-14; do these in order)

Everything below was built and tested without hardware. Nothing has run on the real arm, cameras, iGPU or NPU yet.

1. **Get the code and check the install** (5 min):
   ```bash
   cd ~/junction26 && git pull          # or: git fetch && git checkout claude/stoic-fermat-r030it
   PY=~/physical-ai-studio/application/backend/.venv/bin/python
   $PY anomaly_capture/sim_test.py
   $PY -m pip install pytest httpx && $PY -m pytest tests -q -m "not slow and not ui"
   ```
2. **Start the tool once and confirm the cameras and the arm still come up** (`anomaly_capture/restart.sh`). This is the first run of the changed import lines in `capture.py` / `arm.py` (decision 26). If it fails, `git diff be90d9f -- anomaly_capture/capture.py anomaly_capture/arm.py` shows the few lines.
3. **NPU driver + sensors** (§6, then `bench/README.md`): `$PY -m bench.devices` should say "NPU: OK"; `$PY -m bench.sensors --idle 10` should show package watts after the RAPL `chmod`.
4. **Collect** as planned in step 5 below (pick demos per brick, then WP2 photos, then crosses).
5. **Train ACT** in Studio on the XPU and export to OpenVINO. Then, without the arm:
   ```bash
   $PY anomaly_capture/policy.py --list
   $PY anomaly_capture/policy.py --model ~/.local/share/physicalai/models/<id> --device GPU --bench 100   # also CPU, NPU
   ```
   The printed inputs must be `images.top`, `images.wrist` and `state`.
6. **Anomaly models:**
   ```bash
   ~/anomalib/.venv/bin/python anomaly_capture/train_anomaly.py --brick red --dry-run   # check the episode split
   anomaly_capture/anomaly_pipeline.sh red          # front camera
   anomaly_capture/anomaly_pipeline.sh red wrist    # tape-spot top photo
   ~/anomalib/.venv/bin/python anomaly_capture/eval_anomaly.py --brick red --compare --device GPU
   ```
7. **First autonomous cycle:** `/sort` → Pick = ACT policy, choose the export, device GPU, speed 0.5, 1 object, judge MANUAL. Keep a hand on Space; press it during the first pick to confirm the arm freezes. Then judge = MODEL, cascade on.
8. **Evidence:** close the camera previews, run `bench/run_all.sh` (about 15–25 min, `--quick` about 5; both estimates), run sorting sessions, then `bench/run_all.sh --report-only` and paste `bench/results/section9_rows.md` into §9.


1. [x] Mount the overhead D415 above the workspace (done; about 66 cm, top-down). The setup has been locked since 2026-11-14 ≈01:15.
2. [ ] Add a read-only Hugging Face token in Settings → General → Hugging Face.
3. [x] Install the NPU driver (§6). Done: OpenVINO lists CPU, GPU, NPU (checked 2026-11-14 18:32).
4. [ ] Get the **kickoff details**: the objects, the defects and the sorting rules.
5. [ ] **Now: phases 1 + 2 (decision 16).**
   1. [ ] On `/arm`, teach `home`, `wp1`, `wp2` and `drops`. TEST RUN each slowly.
   2. [ ] On `/collect`:
      - drag the top-camera policy crop (table only);
      - check that the masked front view shows only the gripper and brick at WP2 (brick about 35 cm or more from the camera, so depth works).
   3. [ ] Per brick (3 bricks): 30–50 good episodes. Throw the brick by hand during MATCH now and then (side, upside down, near the edges).
   4. [ ] Untick bad episodes, run the export preview, then EXPORT to Studio. Train ACT in Studio on the XPU, with cameras top + wrist.
   5. [ ] Only after all good runs: draw crosses (small first, then bigger) and run defect episodes. These add both pick data and defect images.
   6. [ ] Train PatchCore per brick on the masked WP2 good images; evaluate on the WP2 defect images; export to OpenVINO.
6. [ ] **Inspect + sort (decision 20, built, test it now):**
   1. [ ] `python3 anomaly_capture/sim_test.py` on the station (checks the install; needs no hardware).
   2. [ ] **Get the tape re-grip working (issue 12)** with "test re-grip": it now opens to 90 % and closes until it feels the block.
   3. [ ] Teach `sort_good`, `sort_defect` (+ `_above`) on `/arm` section 8.
   4. [ ] Add WP2 photo poses for **back** (wrist turned 180°) and **down** (gripper pointing at the camera), tick them, until the page says the whole surface can be checked.
   5. [ ] Set the top-photo crop (wrist camera, object lying on the tape spot) **before collecting more data**.
   6. [ ] Run with **pick = hand-feed, judge = MANUAL**: defect on a side, defect under a finger, defect on top, all clean. Try a long brick too.
   7. [ ] After the first anomaly models exist (front + wrist): judge = MODEL → calibrate → run; compare devices (CPU, GPU, NPU) and the cascade.
7. [ ] **Later phases (phases 3–7 are now built, see decisions 20 and 23–25; the list below is the original plan):**
   1. **Phase 3, decision at WP2:** anomaly score per brick (the colour selects the model), with a threshold from the defect images. Measure the smallest cross detected.
   2. **Phase 4, sort:** WP2 → good plate or defect plate (scripted waypoints with the gripper still locked, then release), then back to the start pose.
   3. **Phase 5, all faces:** the cross can be on any face. Add more WP2 views (wrist roll and tilt for 2 sides + bottom), the top-camera check of the upward face before the pick, and a set-down + 90° regrasp for the 2 gripped faces, only if still clean. Stop as soon as a cross is found (decision 14).
   4. **Phase 6, run-time loop (own runtime, not Studio's):**
      - ACT pick with **the same `prep.py` crops**;
      - the stall hand-off → scripted carry/WP2;
      - anomaly inference → sort;
      - safe stop.
      - Optional: run ACT faster than real time, using temporal ensembling or chunked execution.
   5. **Phase 7, efficiency:**
      - Assign CPU, iGPU or NPU per model (NPU driver first, §6).
      - Measure latency per step and per cycle, power and temperature; record them in §9.
      - Possibly EfficientAd on the NPU for the cascade (decision 10).
   6. Stretch: robot-generated pick demos (the robot places the brick at a known random pose, then records the pick back), SmolVLA.
8. [ ] Optional: `sudo usermod -aG dialout intel`, then log in again, so you don't have to re-run `setfacl` after every replug.

---

## 8. Watch out for

- **Every replug of the arms resets the port permissions.** Until `intel` is in `dialout`, Environments will fail and show **no camera feed**. Fix: `sudo setfacl -m u:intel:rw /dev/ttyACM*`
- **If any device in an environment fails to start, the whole environment fails, cameras included.** Check the backend log for the real error.
- **The wrist camera has no unique serial number.** If it's replugged and comes back as a different `/dev/videoN`, Studio may ask you to reselect it.
- **Each D415 shows up twice in Studio's camera list:** once as `realsense` (serials 2436…) and once as `usb_camera` (serials 3442…). Always use the `realsense` entries.
- **Don't stop Studio with `pkill -f "physicalai-studio serve"` from a shell whose own command contains that string;** it kills that shell too. Kill by PID instead.
- **Restart the capture tool with `~/junction26/anomaly_capture/restart.sh`.** A plain `kill` used to hang while a browser tab had a preview open, which left two copies fighting over the cameras (only the wrist came up). Fixed with `timeout_graceful_shutdown=1`, plus a force-kill in the script.
- **The capture tool and Studio both need exclusive access to the cameras** (`capture.py` opens all three: top, front and wrist). Close Studio's Environments preview before starting `capture.py`, and stop `capture.py` before opening the preview.
- **If the lighting changes, re-capture the empty background (B)** and ideally re-check the dataset. A changed light changes what "good" looks like.
- **Studio's Modal deploy hides the error output.** Run the deploy by hand (§6) to see the real error.

---

## 9. Measurements (fill in during Step 7)

| Model | Device | Precision | Latency per step | Cycle time | Power | Temperature | Notes |
|---|---|---|---|---|---|---|---|
| ACT `act-pick-sim610-real69x2-qualia` (export 5d03db18, OpenVINO, chunk 100, cams top+wrist) | GPU | export default | 7.8 ms p50 / 9.2 ms p95 | – | – | – | `policy.py --bench 20`, 2026-11-14 19:31, no arm. Loads via physicalai in 1.2 s; inputs `images.top`, `images.wrist` + state, as assumed in decision 23 |
| same | NPU | export default | 10.2 ms p50 / 15.1 ms p95 | – | – | – | loads in 1.7 s |
| same | CPU | export default | 81.9 ms p50 / 89.6 ms p95 | – | – | – | loads in 0.6 s |
| PatchCore `models/front/pink` (wide_resnet50_2, 256×256, OpenVINO FP16) | GPU | FP16 | 4.8 ms p50 / 5.0 ms p95 | – | – | – | 2026-11-15, raw OpenVINO `LATENCY` hint, 40 runs, random input, batch 1, no IPC. Compile 1.0 s. `capture.py` was running (≈70 % of one core) |
| same | NPU | FP16 | 83.1 ms p50 / 83.8 ms p95 | – | – | – | compile 5.9 s. Runs, but 17× slower than the GPU: PatchCore's nearest-neighbour search does not suit the NPU |
| same | CPU | FP16 | 71.2 ms p50 / 82.0 ms p95 | – | – | – | compile 0.2 s |
| SmolVLA run B (`4abb7b9d`, OpenVINO export, task "pick up the pink brick") | GPU | export default | 105 ms p50 / 158 ms p95 per action chunk | – | – | – | 2026-11-15, `Policy.chunk()` on random frames, 7 timed runs, observation build included, no arm. Loads in 13.2 s |
| same | CPU | export default | 1862 ms p50 / 1888 ms p95 per chunk | – | – | – | loads in 6.9 s. Too slow for closed-loop control |
| same | NPU | export default | not measured | – | – | – | did not finish loading within 7 min (killed by the timeout); cause not looked into |
| PatchCore `models/front/pink` through the run-time path (`score_worker.py`, real 3 crops × 8) | GPU | FP16 | 18.9 ms p50 / 27.9 ms max | – | – | – | 2026-11-15. This is the `infer_ms` the flow logs: PNG read + pre/post-processing + inference. Model load 2.5 s (first photo only) |
| same | CPU | FP16 | 86.7 ms p50 / 100.7 ms max | – | – | – | load 2.5 s |
| same | NPU | FP16 | no result | – | – | – | the worker returned no score on the NPU (error not captured; the export has a dynamic input size, which the raw test above fixed to 256×256 first) |

**Device placement (2026-11-15 ≈09:00, the user's request "run everything on the most efficient device"):** nothing had to move. Every model in the live pipeline already runs on the iGPU (`device: GPU`, `act_device: GPU`), which is the fastest device for PatchCore and SmolVLA in the rows above; the vision pick, arm control and cameras are not neural networks and stay on the CPU; the NPU has no model that suits it yet (EfficientAd is not trained).
**Latency on the user page:** the status bar of `web/operator.html` now shows, for the last sorted block, inference ms + device, pick / inspect / sort / cycle seconds; the results list has an inference column. `flow.py` puts `t_pick_s`, `t_inspect_s`, `t_sort_s`, `infer_ms_mean`, `device` into `/flow/status` → `history`. Checked against stub data in headless Firefox only; the live server (started 08:48) needs a restart to send the new fields.
**Power, temperature and device load on the user page (2026-11-15 ≈09:35):** new `anomaly_capture/telemetry.py` samples the same sysfs sources as the bench (`bench/sensors.py`) at 1 Hz: RAPL package / core / psys W, CPU package temperature and headroom to TjMax (100 °C), iGPU render-engine busy %, NPU busy %, CPU %. `GET /telemetry` (`?history=1` adds the last 2 min). The user page shows a "System" card in the sidebar (power with a 1-minute sparkline, temperature + headroom, iGPU / NPU / CPU bars); on screens under 880 px tall the label and sparkline drop out so Start and Stop stay visible. The flow now logs **`energy_j` per block** (package energy from start to end of the cycle) and the status bar shows it next to the cycle time. First live reading, idle station while EfficientAd trained on the iGPU: 21 W package, 38 W psys, 62 °C (38 °C headroom), iGPU 87 %, NPU 0 %.
- **RAPL needs `sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj` after every reboot** (or the udev rule in `bench/sensors.RAPL_FIX_PERSIST`). Without it the power fields are null and the card shows "–"; temperature and loads still work.
**EfficientAd on the NPU (2026-11-15 ≈09:50):** `models/front/pink_efficientad` (small, 256×256, 85 clean crops, 20 epochs, trained on the XPU in 474 s, exported to OpenVINO FP16). Raw OpenVINO, random input, LATENCY hint: **NPU 6.6 ms p50 / 11.0 p95** (compile 1.5 s), GPU 3.2 / 3.8 ms, CPU 83.4 / 89.0 ms. Through the run-time path (`score_worker.py`, 12 real pink crops × 2): **NPU 8.3 ms p50**, GPU 15.3 ms p50 (the GPU path feeds the full-size crop and resizes inside the graph).
- **Fix: the NPU failed in the worker** (`ZE_RESULT_ERROR_INVALID_ARGUMENT`): the export has a dynamic `[1,3,?,?]` input and the NPU compiles static shapes only. `score_worker.npu_model()` now writes a sibling `model_static256x256.xml` (input fixed at the anomaly-map size) and the worker resizes each crop to 256×256 before inference, for the NPU only; CPU and GPU still get the untouched dynamic export.
- **NPU vs GPU scores on the same 12 crops:** largest difference 0.027 (e.g. −0.309 vs −0.282); no crop changes side of 0.
- **Not wired into the decision yet:** EfficientAd has no thresholds (no defect photos; clean-only would let a missed defect skip PatchCore), so the sort flow still judges with PatchCore on the GPU. Blue and green EfficientAd training started 09:50.


---

## 10. Research: open-source libraries for moving, turning and re-gripping bricks (2026-11-14)

The question: which libraries help the follower turn a brick and grip it again so the hidden faces show (phase 5, issue 12).
**No open-source library does "re-grip to show hidden faces" on an SO-101.** Regrasp planners (WRS, ReorientBot, MoveIt Grasps) are built for 6–7-DOF arms with full 3D object models.
Libraries can supply the building blocks. The sequence itself (set the brick down, turn the wrist 90°, grip again, show the 2 faces that were under the fingers) stays our own scripted routine.

| Need | Library | Status on this PC | Verdict |
|---|---|---|---|
| Cartesian moves: straight-down set-down, exact place height, wrist roll matched to the brick angle | **LeRobot `RobotKinematics`** (`lerobot/model/kinematics.py`, wraps **placo**) + Studio's `so101_follower.urdf` (in `physicalai-so101-plugin/urdf/`) | lerobot 0.6.0 is installed. placo 0.11.0 isn't yet, but a dry run shows it installs cleanly into Studio's venv (adds 17 packages, nothing downgraded) | **Use first.** It turns "teach every pose by hand" into "go to x,y,z with roll θ". That fixes the most likely causes of issue 12 (place height, the brick not centred under the fingers) |
| Debug the re-grip without risking the real arm | **physicalai-mujoco-plugin**: SO-101 digital twin from MuJoCo Menagerie, same joint names and units as the real driver, cube scenes, object dragging | Already in Studio's plugin tree | **Use second.** Rehearse set-down → turn → re-grip in simulation, then replay on the real arm |
| Camera ↔ robot-base calibration, so top-camera pixels become arm x,y | **OpenCV** `minAreaRect` (already used), D415 depth for z, **pupil-apriltags** for a tag on the table, OpenCV `calibrateHandEye` | OpenCV is installed. pupil-apriltags installs cleanly | Needed only if the robot picks at camera-chosen spots instead of taught ones |
| Alternatives to placo for IK | **ikpy** (simple position IK), **robokin** (SO-101 wrapper: placo / PyRoki / Viser), **PyRoki** (JAX), **mink** (MuJoCo) | ikpy installs cleanly | Fallbacks only |
| Collision-aware motion planning | **MPlib** (pip, no ROS), MoveIt 2 (ROS) | MPlib would **downgrade numpy and scipy** in Studio's venv, so it would need its own venv | Not needed. The table is open, so taught via-points are enough |
| Flipping against a fixed edge (pivoting) | Research code only (Stanford TML extrinsic manipulation; Yang et al. 2023) | n/a | Not a library. The idea is useful, though: a fixed corner or L-stop on the tape spot makes the set-down repeatable |
| Learning the re-grip instead of scripting it | ACT / SmolVLA in Studio (teleoperated demos of the re-grip) | In place | Possible later. A scripted re-grip with IK is faster to get working |

**5-DOF limit:** the arm has no independent yaw at the wrist, so it can't reach every orientation. Re-gripping after a 90° turn of the wrist roll about the vertical axis is still possible. That swaps which pair of faces the fingers cover, which is exactly what phase 5 needs.

---

## 11. Second Modal account: two longer SmolVLA runs outside Studio (2026-11-15 ≈01:50–02:05, intel-48)

- **Why outside Studio:** the Studio backend uses one Modal account (the active profile in `~/.modal.toml`), and it was still monitoring jobs dc23272a / 47f3be5e. Restarting it or switching the profile could have broken them, so neither was done.
- **Second account:** profile `andreacevenini-aldini`, added with `modal token set --no-activate`. `lucabottelli` stays the active profile. Use the second one only with `MODAL_PROFILE=andreacevenini-aldini` in the command's own environment.
- **Deployed there:** app `physicalai-trainer-b`, volume `physicalai-training-b`, env `main`, same `src/trainer/modal_app.py` and image.
- **Data:** `dataset.zip` + `spec.json` copied read-only from the first account's volume (the exact files of dc23272a and 47f3be5e); only `max_epochs` and `num_workers` (60) changed.
- **Runs (both H200, 64 CPU, 128 GB, batch 32, started training ≈02:01):**
  | Model | Dataset | Epochs | Steps | Job dir on the volume | Modal call |
  |---|---|---|---|---|---|
  | `smolvla-new55-30k` | f5af7a53 (55 episodes) | 59 | ≈30k | `jobs/e11bf6a0-88ff-4218-8cf2-d7e937b4d72a` | `fc-01M7E82XVATJDKH7RAD6DTZXPY` |
  | `smolvla-real124-40k` | 15bbb56b (124 episodes) | 31 | ≈40k | `jobs/17ac5059-c5e6-408a-9b7d-0309aecae2a5` | `fc-01M7E838PSD925D8G1VH1145EB` |
- **Why 30k and not 40k for new55:** 40k would be 79 passes over 55 episodes, and its training loss was already 0.02 at 16k steps (overfit risk, flagged by the session that ran the 20k jobs).
- **B200 was tried first** (the most expensive GPU): both calls stayed queued 8 min with no container (no capacity). H200 got a container at once. `with_options(gpu=[...])` (fallback list) is not accepted by modal 1.6.3.
- **Status:** `~/junction26/scripts/modal_b_status.sh`. Launch record: `~/junction26/runs/modal_b_launched.json`.
- **Not in Studio:** these runs don't show under Jobs/Models. When done, `jobs/<id>/artifact.zip` on the volume has to be downloaded and imported by hand.
- **Stop a run:** `MODAL_PROFILE=andreacevenini-aldini .venv/bin/python -m modal container stop --yes <id>` (from the Studio backend folder).
- **Both finished (2026-11-15):** `smolvla-new55-30k` at 04:44 (30,443 steps, last val/loss 0.073, best 0.064 at epoch 11), `smolvla-real124-40k` at 05:01 (40,238 steps, last val/loss 0.182, best 0.178 at epoch 14). No preemption. val/loss was flat after ≈epoch 5 (new55) and ≈epoch 12 (real124): the extra steps did not improve it.
  - Downloaded to `~/junction26/models/modal_b/<name>/` (`artifact.zip` 3.4 GB each, `metrics.csv`, OpenVINO export unpacked in `exports/openvino/`). Not imported into Studio.
  - Two more 12-epoch SmolVLA runs (`f52ed47a`, `4c9a6f02`) were started on `physicalai-trainer-b` at 04:51–04:53 by another session; intel-48 did not launch or touch them. `jobs/35e3c72a…` on the volume is an unused upload from intel-48's first (failed) launch attempt.

- **06:40 — both 159-episode SmolVLA runs finished** (H200, 12 epochs, model uploaded ≈06:22 / 06:25). Run A `f52ed47a` (val eps 44, 45, 62, 68, 91, 130, 140): 19,991 steps, train loss 0.026, val loss 0.182 → 0.146 (flat from epoch 5). Run B `4c9a6f02` (val eps 13, 28, 40, 56, 79, 92, 156): 20,063 steps, train loss 0.029, val loss 0.187 → 0.137 (best 0.133 at epoch 7). Val losses are on different episodes (7 each), so the 0.009 gap is not a ranking; train losses are equal. Neither imported into Studio nor arm-tested yet.
  - **06:59 — both imported into Studio** with `physicalai-studio models import-dir` (project `a0877272`, dataset `c6c631af`): A = model `e5c26856-ccb8-4e22-acb7-0d644c0440d5`, B = model `4abb7b9d-f08a-4df1-b426-bf3728732b69`. Zips kept in `~/junction26/models/modal_b/smolvla-real124-vpick35-20k-{A,B}/artifact.zip`. Not arm-tested yet.
- **07:05 — learned model vs scripted pick on the pink brick (plate, same session).** SmolVLA run B (`4abb7b9d`, OpenVINO GPU, speed 0.5, task "pick up the pink brick"): **0/3**, all "closed on nothing" (`runs/acttest_20261115_070505`). Scripted vision pick, target pink, stepped descent: **3/3** held, camera-confirmed (`seen_after` = green only), 18.3 / 20.2 / 21.6 s per pick (`runs/acttest_20261115_070633`). Run A not arm-tested. User decision: the scripted vision pick is the pick from now on; Sort page pick set to `vision`.
- **07:10–07:28 — vision pick "fast" mode + false "dropped" verdicts.**
  - **Fast mode** (`VisionPick(fast=)`, cfg `vision_fast`, default on, switch "Vision pick: fast" on the Sort page): 3 descent steps instead of 6, travel to the hover ×1.8, jaw close 0.5 instead of 0.3, move to WP1 at 0.8. The lift itself stays at the old speed. Pick time ≈14 s (13.0–16.0) against 18–22 s.
  - **False "lost it on the way to WP1":** 3 picks in 20 (07:10 and 07:15 runs) were held (jaws 51–54 ticks short of the squeeze command, brick visible in the wrist photo) but the top-camera check at WP1 reported the colour "still on the plate". Cause: at WP1 the top camera sees a pink blob at ≈(1565, 255) px, the gripper with or without a pink brick in it (also there when green is held). Fix: `SURE_TICKS = 40`; with that margin the pick is held whatever the camera sees, below it the camera still decides. Recorded held picks have margins 31–56, the false ones 51–54.
  - **Result with fast + fix:** 9/10 on random colours, 1 real miss (blue, closed on nothing) (`runs/acttest_20261115_072304`). Before the fix: 7/10 and 6/10 reported, 8/10 and 8/10 real. No same-code baseline at normal speed (the control run was cut off by another session's restart at 07:20).
  - Not committed yet: `vpick.py` / `sorter.py` also hold another session's uncommitted re-clamp edits.

## Features for the user-facing interface (to build once everything works)

List of controls that exist in the engineering tool (`127.0.0.1:7870`) and are to be carried over to the final user-facing interface. Add to it as features are decided.

- **Vision pick: fast (toggle).** Decided by the user 2026-11-15 07:30. On = ≈14 s per pick (3 descent steps, quicker travel), off = 18–22 s (6 corrected steps). Backend: cfg key `vision_fast` (`POST /sort/cfg?vision_fast=0|1`, default on), `VisionPick(fast=)` in `anomaly_capture/vpick.py`. Today's switch: "Vision pick: fast" on the Sort page (`web/pages/08-sort.html`, id `so-vfast`). Measured 9/10 with it on; no same-code baseline with it off yet.
- **Accept / reject plate assignment.** Decided by the user 2026-11-15 07:45. Two plate waypoints; the interface lets the user say which one takes accepted blocks and which one rejected blocks. Backend: `POST /flow/cfg?accept=sort_good|sort_defect` (the other slot becomes reject), read back in `GET /flow/status` → `cfg.accept`, `cfg.reject`, `plates`. No control in the engineering tool yet.
- **Start / stop of the production sort flow** and its live result list: `POST /flow/start`, `POST /flow/stop`, `GET /flow/status` (`cur`, `history`, `totals`). No page yet.
- **08:09 — second speed step tried, no gain.** Fast mode with the lift at 0.55 (instead of `rise()`'s 0.3) and a 0.12 s pause in `_servo_in`: 7/10, mean held pick 15.2 s (pink 13.0, green 14.7, blue 16.0) against 14.2 s in the 07:23 run (`runs/acttest_20261115_080906`). Misses: 2 closed on nothing, 1 "lost on the way" with an ambiguous margin (29 ticks, camera decided). Pick time is dominated by where the brick lies, not by the lift. Faster lift reverted; the shorter `_servo_in` pause stays. Fast mode totals with the held-check fix: 16/20.
- **Fast moves (toggle), decided by the user 2026-11-15 ≈08:55.** Long moves ramp up smoothly to a target speed, hold it, and ramp down smoothly, instead of the minimum-jerk law that eases the whole way (mean speed only 53 % of its peak). Backend: `arm.profile()` (smoothstep speed ramps, peak acceleration 1.5 × speed / ramp), used by `Arm.goto` only when the move is long enough to reach the cruise speed and comes out shorter; short moves (pick steps, set-down) are unchanged. Settings in the sorter cfg: `POST /sort/cfg?motion_fast=0|1&motion_speed=0.3..1.5&motion_ramp=0.15..1.5` (speed as a share of `VMAX` = 1500 ticks/s for a move at the robot speed, slower moves keep their ratio; cap `FAST_VMAX` 1.5; ramp in seconds). Defaults: off, 1.0, 0.4 s. Switches: "Fast moves" on the operator page (`sw-mfast`) and on the dev Sort page (with target speed and ramp). Dry runs (`/collect/test`) ignore it.
  - Simulated arm (mock bus, robot speed 0.8, target 1.0, ramp 0.4 s): home → WP1 (1142 ticks) 1.90 s → 1.27 s; WP1 → plate A (643 ticks) 1.12 → 0.94 s; a 0.3× gentle move over 643 ticks 3.46 → 1.93 s. **Not run on the real arm yet**: pick rate and held-brick behaviour with it on are unmeasured.
  - Tests: `tests/test_profile.py` (5), cfg round trip in `tests/test_api_mock.py`. Full fast suite before the two added tests: 198 passed.
- **≈09:03 — fetch from the re-clamp spot closed on the yellow plate's rim 3× (flow 08:49, green; intel-2f).** `runs/flow_20261115_084922/085055_a1_look.jpg`: the green block had been pushed against the yellow plate; the "green" blob was 122 mm long = block + plate rim, so the grasp point lay on the rim. Cause: hue class green is 24–75 and the yellow plate measures 22–32 (inner 25–32, rim 22–30); the green block's top is 41–57. Fix: `vpick.SPOT_GREEN_MIN = 34`, the lower hue bound of green in the look at the re-clamp spot only (`detect(at=)`); the look at the orange plate is unchanged. Replayed on that frame: one green blob at (1022, 481) px, 53 × 27 mm. On an earlier frame of a green block standing on the spot (08:02) its size reads 47 × 33 mm instead of 51 × 38 (real 48 × 32), centre 3 px away. Test: `tests/test_profile.py::test_spot_look_does_not_merge_a_green_block_with_the_yellow_plate`. Not run on the arm yet.
- **≈09:09 — re-clamp shortcuts behind the "Fast moves" switch (the user: "implement 1-3 behind the fast moves switch"; intel-2f).** Why: fast moves alone barely change a re-clamp, its moves are too short to cruise; one re-clamp took 31–35 s (08:02–08:05 tests), of which the descent onto the block 6–7 s and close + lift 7 s (frame times of the 08:49 / 08:51 flows). With `motion_fast` on, `vpick.grasp_world` and `sorter.regrip_vision` now:
  1. go down in **3 corrected steps** instead of 6 (`RG_STEPS_FAST`);
  2. **close the jaws at 0.5** instead of 0.3 (`RG_CLOSE_FAST`, the fast pick's value). The jaw opening is NOT made narrower (it was part of the proposal): the 9 + 14 mm clearances are what keeps the fingers from landing on the block;
  3. **rise 38 mm** after letting go and turn the jaws there, instead of 55 mm (`RG_HOVER_FAST`; fingertips ≈18 mm over the studs). The 3 steps are then 14 mm each (were 10 mm).
  - Also: a move of the jaws alone never uses the cruise profile (`Arm.goto`), in fast mode too: the jaws meet the block at the eased speed. This changes the 08:57 fast-moves behaviour, where a long jaw move could cruise.
  - Offline replay (`grasp_world` at the taught tape spot, real kinematics, ideal arm without sag): +90°, −90° and 180° all in reach at the 38 mm hover; 17 → 11 moves per re-grip, estimated motion time 13.8 → 9.8 s (+90°). **Not run on the real arm**: hold rate with 3 steps and the 18 mm clearance while turning are unmeasured. `info` of every re-grip now logs `quick` and `hover_mm`.
  - Tests: `tests/test_profile.py` (7) + `tests/test_flow.py`, 31 passed in 10 s.
