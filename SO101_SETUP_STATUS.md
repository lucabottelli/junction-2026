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

Last updated: 2026-11-14 01:00 (EET)
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
| 6. Close the loop | ⏳ Not started | |
| 7. Optimize on Intel hardware | 🔴 NPU not usable yet | NPU user-space driver missing. See §6. |

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

---

## 6. Open blockers

### Modal: dropped
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
6. [ ] **Later phases (not built yet):**
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
7. [ ] Optional: `sudo usermod -aG dialout intel`, then log in again, so you don't have to re-run `setfacl` after every replug.

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
