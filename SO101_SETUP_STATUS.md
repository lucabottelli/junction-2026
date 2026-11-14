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
      - `station.py`: scene (table, 3 plates as dishes with rims, the real bricks (user-confirmed 04:55): **red 2×4 ridge roof** (tent-shaped prism, no studs), **green 2×3**, **blue 2×2**, top camera, wrist camera) and the units mapping.
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
    - **Known weak spot:** in the MuJoCo check (expert wrapped as a policy, 6 tries) 4 picks were detected correctly, 1 brick was out of reach, and **1 real grasp was never detected** (timed out).
      - **Second hand-off rule added** (2026-11-14 midday, `policy.SOFT_GAP` / `SOFT_HOLD_S`): fingers stalled on something with the target at least 20 ticks past them for 0.6 s → hand-off, logged as `grasp_soft`, and the clamp is firmed up to 80 ticks past the object before the lift. The demos' rule (40 ticks, 0.3 s) still fires first when it applies.
      - A wrong `grasp_soft` costs one retry, not a wrong sort: the "held at WP1" check still follows.
      - Tested with a simulated policy that squeezes only 30 ticks (timed out before, hands off now). **Not re-run in MuJoCo** (the plugin isn't available off the station), so whether it covers the missed case there is unconfirmed: run `sim/act_loop.py -n 6` once. G still forces the hand-off.

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
| 12 | 11-14 ≈03:25 | Tape-spot 90° re-grip doesn't work on the real arm | Not diagnosed yet. Candidates: `tape_place` too high or too low (the block tilts or drops when released), gripper not open wide enough for the turned block, block not centred under the fingers after the turn, or the turned block too wide for the gripper | Disabled by default (decision 19). To debug later with the "test re-grip" button, watching each step. 04:30: the re-grip now opens to 90 % and closes until the fingers stall (decision 20), which covers the two width-related candidates; not yet retried on the arm |

---

## 6. Open blockers

### Modal: dropped, then working again (2026-11-14 ≈12:58)
- **Update ≈12:58:** the manual deploy (command below, with the saved target's app/volume names `physicalai-9449e57a…`) now **succeeds**.
  - Studio kept showing "degraded", because it holds a failed deploy in memory until the backend restarts.
  - After a backend restart (kill by PID, then the usual `serve` command) the target reports **healthy**.
  - First run: job `75217bde…`, ACT on `pick-sim604-real1x10`, **A100-80GB** (the user's choice), 15 epochs, batch 32, augmentation on, val split 5%. Submitted 12:59 via `POST /api/jobs:train` with `training_target: remote` and `modal_gpu: A100-80GB`.
  - **Not yet confirmed: whether Modal accepts the GPU run without a card.**
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
3. [ ] Install the NPU driver (§6).
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
| | | | | | | | |

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
