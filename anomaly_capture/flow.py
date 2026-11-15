"""Production sort flow (the user's spec, 2026-11-15 07:45). API under /flow/*; served by capture.py.

One block:
  1. vision pick off the start plate -> WP1 (a failed pick is tried again, vpick retries). This is the only visit
     to WP1: after it the arm goes above WP2 -> photos -> above WP2 -> tape spot -> above WP2 -> ... -> plate.
  2. WP2, first showing: every WP2 pose in turn (the three angles). Each photo is judged at once;
     the first anomaly ends the inspection -> REJECT plate.
  3. All clean -> re-clamp turned 90 deg (sorter.regrip_vision: set down on the re-clamp spot, look, grip turned).
     Every re-clamp of one block turns the SAME way: the first one chooses the sign, the later ones are asked for
     it. (+90 then -90 would show the first side again.) The shown side is tracked from the reported turn, so a
     re-clamp that had to turn the other way costs one more re-clamp instead of a wrong "all sides seen".
  4. After each re-clamp: ONE photo (the side view, first WP2 pose), judged. Anomaly -> REJECT plate.
     Top and bottom are not inspected.
  5. All four sides shown and clean -> ACCEPT plate.
     A block that is not square (green 2x3) is only turned end for end (180 deg, regrip_vision's rule: gripped along
     its long side it is pushed out). Its two ends are shown; the two long sides stay under the fingers. Clean ends
     -> ACCEPT, with the uninspected sides named in the reason and in sides_hidden (the user's rule: a face no
     showing can reveal is not counted against the block).

Plates: two taught waypoints, plate A and plate B (slots plate_a / plate_b, each with an optional *_above; dev
Waypoints page, section Plates). Which one takes the accepted blocks is a setting the user page changes
(cfg accept / reject, POST /flow/cfg?accept=plate_a|plate_b); the other one takes the rejected blocks.

Judging one photo (same photo path and rules as autocollect.inspect, i.e. what the colour models were trained on):
  visible patch that is not the block's colour (prep.foreign_share >= FOREIGN_MAX)  -> anomaly
  model score >= t_high but nothing visible                                          -> doubt (the user's rule: a
                                                                                        score alone is no anomaly)
  model score between t_low and t_high, block cut by the frame, no photo             -> doubt
  model score <= t_low                                                               -> clean
A block cut by the frame's edge is a bad photo, not a doubt about the block: the photo pose is moved up (or down)
20 mm at a time, at most 4 times, until the whole block is in the frame (first real run, green after a re-clamp:
it hung lower, the bottom edge cut it, and it was rejected as "not confirmed clean").
A doubt is photographed again (cfg retakes); a doubt that stays goes where cfg uncertain_to says: "reject"
(default, fail-safe, logged as unconfirmed) or "accept".
No block in the photo (no colour mask) after the retakes = the block is not in the jaws: the cycle is aborted,
nothing is put on a plate.

Judging in the background (cfg async_judge, on by default; POST /flow/cfg?async_judge=0 goes back to waiting).
At a photo pose the arm only waits for the frames (settle + about 0.5 s). Mask, crop, files and the model score
run in a worker thread while the arm already moves to the next pose, out of WP2 and toward the tape spot (or,
after the last showing, toward the accept plate). A result that is not clean interrupts the move under way:
an anomaly -> straight to the reject plate from where the arm stands; a doubt, a block cut by the frame or no
block seen -> back to that pose for the careful look (retakes, re-posing), then on. Every result is in before
the block is set down for a re-clamp or let go at a plate.

The photo stop (cfg photo_frames 8, photo_still_s 0.25, photo_wait_s 0.2; POST /flow/cfg?photo_frames=16&
photo_still_s=0 restores the collection's values): the arm is still for 0.25 s, waits 0.2 s for the swing and
averages 8 frames instead of 16. The poses of a showing are visited in the order with the least travel, at full
speed, and the arm leaves straight up from the last one.

Motion. The arm stands still only where something happens: for a photo, on the tape spot, at the plate, and at
the pick. Everything in between is one blended move through its clearance poses (Collector.path / arm.Arm.path):
WP1 -> above WP2 -> photo pose;  photo pose -> above WP2 -> above the tape spot;  photo pose -> above WP2 ->
above the plate -> plate;  plate -> above it -> home.

The clamp. From the pick on, every move carries the clamp command of the grip. It opens in exactly two places:
on the tape spot inside a re-clamp (the block is gripped again there), and at plate A or B, after the measured
arm stands at the taught plate pose (Flow.drop). An arm that does not reach the plate keeps the block and the
run stops. An aborted block whose clamp is still closed is opened at the reject plate; a run that starts with a
closed clamp first empties it at the reject plate. STOP, pause and errors never open the clamp.

Pause (POST /flow/pause, /flow/resume): the arm stands still at the next pause point - before a pick, a photo, a
re-clamp or the plate - holding the block; the move that is under way finishes first. STOP works while paused.

Log: runs/flow_<time>/log.jsonl (one line per block: verdict, reason, every photo with score and times).
"""
import inspect as _inspect
import concurrent.futures as _cf
import json, threading, time
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import APIRouter

import autocollect as ac
import feed
import policy as pol
import prep
from arm import BODY, Interrupted, Stopped

HERE = Path(__file__).resolve().parent
PLATE_SLOTS = ("plate_a", "plate_b")     # taught on the dev Waypoints page (section Plates); optional <slot>_above
SIDES = 4                  # side faces of a block; one is shown per grip
NUDGE_MM, NUDGE_MAX = 20.0, 4    # a block cut by the frame: move the photo pose this far, at most this often
AT_PLATE = 60              # ticks (~5 deg): every arm joint this close to the taught plate pose = "at the plate"



def _energy():
    """Package energy counter (J) from telemetry.py; None when RAPL is not readable or telemetry is not loaded."""
    try:
        import telemetry
        return telemetry.energy_j()
    except Exception:
        return None

class NotHeld(Exception):
    """No block in the jaws at a photo pose."""


class Flow:
    def __init__(self, S, A, ctx, C, T, F, AC, DATA):
        self.S, self.A, self.ctx, self.C, self.T, self.F, self.AC, self.DATA = S, A, ctx, C, T, F, AC, DATA
        self.cfg_path = HERE / "flow_settings.json"
        self.cfg = {"accept": "plate_a", "reject": "plate_b", "max_reclamps": 4, "retakes": 1,
                    "uncertain_to": "reject", "side_view": "", "async_judge": True,
                    # the stop at a photo pose: frames averaged (the collection uses 16), still time, wait for the swing
                    "photo_frames": 8, "photo_still_s": 0.25, "photo_wait_s": 0.2}
        try: self.cfg.update(json.loads(self.cfg_path.read_text()))
        except Exception: pass
        if self.cfg["accept"] not in PLATE_SLOTS:                  # settings from before the plates had their own slots
            self.cfg["accept"] = PLATE_SLOTS[0]
        self.cfg["reject"] = PLATE_SLOTS[1 - PLATE_SLOTS.index(self.cfg["accept"])]
        self.running = False; self.session = None; self.cur = {}; self.history = []
        self.pause = threading.Event(); self.paused = False        # pause asked for / the arm is standing at a gate
        self.g_lock = None                                         # clamp command while a block is (thought to be) held
        self.exit_via = []                                         # poses to fly through to get out of WP2 (set by show)
        self.alarm = threading.Event()                             # a background photo was not clean (interrupts the move)
        self.pool = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge")   # one at a time, in order
        self.lift_mem = {}                                         # (block, pose, re-clamped?) -> extra mm that put it in the frame
        self.totals = {"accept": 0, "reject": 0, "aborted": 0}

    def save_cfg(self): self.cfg_path.write_text(json.dumps(self.cfg, indent=1))

    # ------------------------------------------------------------------ pause
    def gate(self):
        """Pause point between two steps (before a pick, a photo, a re-clamp, the plate): while pause is set the arm
        stands still in hold mode, the block stays in the jaws. A move that is under way finishes first. STOP still works."""
        if not self.pause.is_set(): return
        T = self.T; st, msg, ok = T.state, T.msg, T.ok
        self.paused = True; T.say("PAUSED: the arm holds its position. Resume to continue.")
        try:
            while self.pause.is_set():
                T.check(); time.sleep(0.1)
        finally:
            self.paused = False
        T.state = st; T.say(msg, ok)

    # ------------------------------------------------------------------ photo poses
    def poses(self, brick, extra=None):
        """The WP2 poses for a block picked off the plate (same lift as autocollect.photos: it hangs lower in the
        jaws than a hand-fed one). extra = {pose name: mm} raises (+) or lowers (-) single poses further; with it an
        unreachable pose raises instead of falling back."""
        C, F = self.C, self.F; extra = extra or {}
        taught = C.wps("wp2"); old = F.lift
        try:
            F.lift = {**old, brick: {w["name"]: float((old.get(brick) or {}).get(w["name"]) or 0)
                                     + ac.AUTO_LIFT_BRICK.get(brick, ac.AUTO_LIFT).get(w["name"], 0)
                                     + float(extra.get(w["name"], 0)) for w in taught}}
            if extra: return F.lifted(taught, brick)
            try: return F.lifted(taught, brick)
            except Exception:
                F.lift = old; return F.lifted(taught, brick)
        finally:
            F.lift = old

    # ------------------------------------------------------------------ one photo -> verdict
    def grab(self):
        """The still frames of one photo; the arm must stand still for this (about 0.5 s): the averaged colour frame,
        the depth of that moment and the joints. Everything after it (mask, crop, files, score) needs no arm."""
        S = self.S; n = int(self.cfg.get("photo_frames") or 0) or None; cam = S.cams["front"]
        try: shots = S.averaged(["front"], n)
        except TypeError: shots = S.averaged(["front"])
        dep = getattr(cam, "depth", None)
        if dep is not None and time.time() - getattr(cam, "depth_t", 0) >= 1.0: dep = None
        return {"shots": shots, "depth": {"front": None if dep is None else dep.copy()}, "frames": n,
                "joints": {j: int(x) for j, x in self.A.pos.items()}}

    def settle(self):
        """Stand still before a photo, as short as it can be (the user, 2026-11-15): no body joint moved more than
        2 ticks for cfg photo_still_s, then cfg photo_wait_s more for the block to stop swinging. feed.settle (the
        collection's 0.5 s + 0.4 s) is the fallback when photo_still_s is 0."""
        still, wait = float(self.cfg.get("photo_still_s") or 0), float(self.cfg.get("photo_wait_s") or 0)
        if still <= 0: return self.F.settle()
        A = self.A; t0 = time.time(); last = dict(A.pos); since = t0
        while time.time() - t0 < 3.0:
            time.sleep(0.03); self.T.check(); now = dict(A.pos)
            if max(abs(now[j] - last[j]) for j in BODY) > 2: since = time.time(); last = now
            elif time.time() - since >= still: break
        time.sleep(wait)

    def snap(self, brick, view, idx, model, t_low, t_high):
        """Photo for the background judge: settle, grab the frames -> a function that turns them into the judged
        photo record (run by the worker thread while the arm moves on)."""
        T, F = self.T, self.F
        ts = time.perf_counter(); self.settle(); T.check(); prep.configure_inhand(brick=brick)
        t0 = time.perf_counter(); g = self.grab(); g_ms = round((time.perf_counter() - t0) * 1000, 1)
        s_ms = round((t0 - ts) * 1000, 1)
        def finish():
            t1 = time.perf_counter(); recs = []
            nf = self.ctx.capture(["front"], "test", brick, view, idx, self.session, shots=g["shots"], depth=g["depth"],
                                  joints=g["joints"], out=recs) if "front" in g["shots"] else 0
            p = self._photo(brick, view, nf, recs[0] if recs else {}, t1)
            p.update({"grab_ms": g_ms, "settle_ms": s_ms, "frames": g.get("frames")})
            return self.judge(p, brick, model, t_low, t_high)
        return finish

    def shot(self, brick, view, idx):
        """One photo, waited for (the careful look and the waiting flow): same stop and same frames as snap()."""
        T = self.T
        ts = time.perf_counter(); self.settle(); T.check(); prep.configure_inhand(brick=brick)
        t0 = time.perf_counter(); g = self.grab(); recs = []
        nf = self.ctx.capture(["front"], "test", brick, view, idx, self.session, shots=g["shots"], depth=g["depth"],
                              joints=g["joints"], out=recs) if "front" in g["shots"] else 0
        p = self._photo(brick, view, nf, recs[0] if recs else {}, t0)
        p.update({"settle_ms": round((t0 - ts) * 1000, 1), "frames": g.get("frames")}); return p

    def _photo(self, brick, view, nf, rec, t0):
        """Photo record from what ctx.capture wrote (meta record `rec`): is a block there, is it cut by the frame."""
        F = self.F
        dep = rec.get("depth") if isinstance(rec.get("depth"), dict) else {}
        area = dep.get("area_px")
        ok = bool(nf) and bool(dep) and (area or 0) >= max(ac.MIN_AREA, feed.MIN_AREA.get(brick, 0))
        why = rec.get("depth_mask_why") if nf else "no fresh front frame"
        cut = self.AC.edge_cut(rec.get("file")) if ok else None
        if cut: why = f"block cut by the {cut} of the frame"
        elif not ok and why is None: why = "mask too small: no block in the jaws"
        F._grab_last(rec)
        return {"view": view, "file": rec.get("file"), "ok": ok, "cut": bool(cut), "why": why, "area_px": area,
                "photo_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def judge(self, p, brick, model, t_low, t_high):
        """Adds verdict ('clean' | 'defect' | 'uncertain' | 'nomask') and reason to the photo record."""
        if not p.get("file") or p.get("cut"):
            p["verdict"], p["reason"] = "uncertain", p.get("why") or "no photo"
        elif not p["ok"]:
            p["verdict"], p["reason"] = "nomask", p.get("why") or "no block seen"
        else:
            f = self.DATA / p["file"]; crop = f.with_name(f.stem + "_crop.png")
            fs = prep.foreign_share(cv2.imread(str(f)), brick)
            p["foreign"] = None if fs is None else round(fs[0], 3)
            r = self.T.scorer.score(model, crop, self.T.cfg["device"])
            p["score"] = round(float(r["score"]), 3); p["infer_ms"] = r.get("ms")
            if p["foreign"] is not None and p["foreign"] >= prep.FOREIGN_MAX:
                p["verdict"], p["reason"] = "defect", f"a patch that is not {brick} covers {p['foreign']:.0%} of the block"
            elif p["score"] >= t_high:
                p["verdict"], p["reason"] = "uncertain", f"model score {p['score']:.2f} >= {t_high} but no visible patch"
            elif p["score"] > t_low:
                p["verdict"], p["reason"] = "uncertain", f"model score {p['score']:.2f} between {t_low} and {t_high}"
            else:
                p["verdict"], p["reason"] = "clean", f"model score {p['score']:.2f} <= {t_low}"
        return p

    def look(self, brick, view, grip, idx, model, t_low, t_high, nudge=None):
        """Photo + verdict at the pose the arm stands in; a doubt is photographed again. -> step record.
        A block cut by the frame's bottom / top edge is not a doubt about the block but a bad photo: with
        nudge(mm) (moves this pose up / down, False if out of reach) the arm is moved NUDGE_MM at a time, up to
        NUDGE_MAX times, and the photo is taken again; that costs no retake. The lift that worked is remembered per
        block, pose and first grip / re-clamp. Raises NotHeld when no block is seen there."""
        T = self.T; shots = []; k = 0; moves = 0
        key = (brick, view, grip > 1); lift = float(self.lift_mem.get(key, 0.0))
        if lift and nudge and not nudge(lift): lift = 0.0
        while True:
            T.check(); self.gate(); T.state = "inspect"; T.say(f"block {idx}: grip {grip}, {view}, photo {len(shots) + 1}")
            p = self.judge(self.shot(brick, view, idx), brick, model, t_low, t_high); shots.append(p); p["lift_mm"] = lift
            if p.get("cut") and nudge and moves < NUDGE_MAX:
                d = -NUDGE_MM if "top" in str(p.get("why")) else NUDGE_MM
                if nudge(lift + d):
                    lift += d; moves += 1; T.say(f"block {idx}: the block is cut by the frame at {view}, moving it {'up' if d > 0 else 'down'}"); continue
            if p["verdict"] in ("clean", "defect"): break
            k += 1
            if k > max(0, int(self.cfg["retakes"])): break
        if p["verdict"] in ("clean", "defect") and lift != self.lift_mem.get(key, 0.0): self.lift_mem[key] = lift
        if p["verdict"] == "nomask": raise NotHeld(f"no {brick} block in the jaws at {view} (grip {grip}): {p['reason']}")
        return {"grip": grip, "view": view, "verdict": p["verdict"], "reason": p["reason"], "score": p.get("score"),
                "foreign": p.get("foreign"), "lift_mm": lift, "shots": shots}

    # ------------------------------------------------------------------ motions
    def show(self, wp1, wp2, views, g_lock, fn):
        """Into WP2 in ONE move (through the pose above it, no stop there), fn(pose name, nudge) at every pose in
        `views` until one is not clean. The arm STAYS at the last photo pose: the way out (self.exit_via: back to the
        first pose if it moved on, then the pose above WP2) is flown through by the next move (re-clamp or plate), so
        there is no stop above WP2 and no detour over WP1 (the user, 2026-11-15). The arm stands still only for the
        photos. -> the step that stopped it (or None = all clean)."""
        C = self.C; above = C.above_wp2(wp1, wp2[0]["pos"])[0]; at = None; bad = None
        for w in self.order([x for x in wp2 if x["name"] in views]):
            if at is None: C.path([above, w["pos"]], gripper=g_lock)
            else: C.goto(w["pos"], gripper=g_lock)
            at = w; self.exit_via = [C.above_wp2(wp1, w["pos"])[0]]            # the way out: straight up from this pose
            def nudge(mm, w=w):
                try: pose = next(x for x in self.poses(self.brick_now, {w["name"]: mm}) if x["name"] == w["name"])["pos"]
                except Exception: return False
                C.goto(pose, C.cfg["speed"] * 0.6, gripper=g_lock); return True
            st = fn(w["name"], nudge)
            if st["verdict"] != "clean": bad = st; break
        return bad

    def _judge_job(self, finish):
        """Worker thread: one photo -> judged record. Anything but 'clean' raises the alarm, which cuts the arm's
        current move short (arm.Arm.intr)."""
        try: p = finish()
        except Exception as e:
            p = {"view": None, "file": None, "ok": False, "cut": False, "why": f"photo could not be judged: {e}",
                 "verdict": "uncertain", "reason": f"photo could not be judged: {e}"}
        if p.get("verdict") != "clean": self.alarm.set()
        return p

    def show_async(self, wp1, wp2, views, g_lock, grip, idx, model, t_low, t_high, onward, steps, lenient):
        """The showing without waiting for the judge (the user, 2026-11-15): the arm stops at each photo pose only
        for the frames (settle + about 0.5 s), hands them to the worker thread and moves on at once - to the next
        pose, then out of WP2 toward `onward` (above the tape spot, or above the accept plate after the last
        showing). The worker masks, crops, saves and scores meanwhile.
        A result that is not clean sets self.alarm, which interrupts whatever move is under way; the arm holds.
          anomaly  -> returned at once: the caller takes the block to the reject plate from where it stands.
          doubt / block cut by the frame / no block seen -> back to that pose and the careful look there
            (flow.look: retakes, moving a cut block into the frame); clean after that -> the flow carries on.
        Every result is in before this returns, so nothing is set down or let go on an unjudged photo.
        -> the step that stopped it (or None = all clean)."""
        C, A = self.C, self.A; brick = self.brick_now
        above = C.above_wp2(wp1, wp2[0]["pos"])[0]; jobs = []; at = None; left = False
        def pose_of(w, lift):
            if not lift: return w["pos"]
            try: return next(x for x in self.poses(brick, {w["name"]: lift}) if x["name"] == w["name"])["pos"]
            except Exception: return w["pos"]
        self.alarm.clear(); A.intr = self.alarm
        try:
            for w in self.order([x for x in wp2 if x["name"] in views]):
                if self.alarm.is_set(): raise Interrupted()
                lift = float(self.lift_mem.get((brick, w["name"], grip > 1), 0.0)); pose = pose_of(w, lift)
                if at is None: C.path([above, pose], gripper=g_lock)
                else: C.goto(pose, gripper=g_lock)
                at = w; self.exit_via = [C.above_wp2(wp1, pose)[0]]            # the way out: straight up from this pose
                self.T.check(); self.gate(); self.T.state = "inspect"
                self.T.say(f"block {idx}: grip {grip}, {w['name']}, photo (judged while the arm moves on)")
                jobs.append((w, lift, self.pool.submit(self._judge_job, self.snap(brick, w["name"], idx, model, t_low, t_high))))
            if onward and at is not None:
                left = True; C.path(self.exit_via + list(onward), gripper=g_lock); self.exit_via = []
        except Interrupted:
            if left: self.exit_via = [above]         # somewhere between WP2 and `onward`: first back over WP2
        finally:
            A.intr = None
        for w, lift, fut in jobs:
            p = fut.result(timeout=60)
            st = {"grip": grip, "view": w["name"], "verdict": p["verdict"], "reason": p["reason"], "score": p.get("score"),
                  "foreign": p.get("foreign"), "lift_mm": lift, "shots": [p], "background": True}
            if p["verdict"] in ("clean", "defect"):
                steps.append(st)
                if p["verdict"] == "defect": return st
                continue
            # doubt: back to that pose, the careful look there
            self.alarm.clear()
            back = pose_of(w, lift)
            C.path(self.exit_via + [C.above_wp2(wp1, back)[0], back], gripper=g_lock)
            self.exit_via = [C.above_wp2(wp1, back)[0]]
            def nudge(mm, w=w):
                try: pose = next(x for x in self.poses(brick, {w["name"]: mm}) if x["name"] == w["name"])["pos"]
                except Exception: return False
                C.goto(pose, C.cfg["speed"] * 0.6, gripper=g_lock); return True
            st = self.look(brick, w["name"], grip, idx, model, t_low, t_high, nudge)
            st["shots"] = [p] + st["shots"]; st["looked_again"] = True
            if st["verdict"] == "uncertain" and lenient: st["verdict"], st["passed_doubt"] = "clean", True
            steps.append(st)
            if st["verdict"] != "clean": return st
        return None

    def plate_poses(self, which):
        """(above, release pose, letter) of the plate assigned to 'accept' | 'reject'."""
        C = self.C; slot = self.cfg[which]; tgt = C.slot(slot); L = slot[-1].upper()
        if not tgt: raise RuntimeError(f"teach plate {L} first (dev Waypoints page)")
        p = tgt[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]; ab = C.slot(slot + "_above")
        above = ab[0]["pos"] if ab else {**p, "shoulder_lift": wp1["shoulder_lift"], "elbow_flex": wp1["elbow_flex"],
                                          "wrist_flex": wp1["wrist_flex"]}
        return above, p, L

    @staticmethod
    def order(poses):
        """The photo poses in the order with the least travel: the first one stays first (the arm comes in over it),
        then always the nearest one not yet visited (largest joint difference = what sets a move's time)."""
        if len(poses) < 3: return list(poses)
        dist = lambda a, b: max(abs(float(a["pos"].get(j, 0)) - float(b["pos"].get(j, 0))) for j in BODY)
        out, rest = [poses[0]], list(poses[1:])
        while rest:
            nxt = min(rest, key=lambda w: dist(out[-1], w)); rest.remove(nxt); out.append(nxt)   # min keeps the taught order on a tie
        return out

    def reclamp(self, g_lock, k, sign):
        """Re-clamp from where the arm stands (above WP2) and end above WP2 again, ready for the next showing."""
        T, C = self.T, self.C; pr = _inspect.signature(T.regrip_vision).parameters
        kw = {"sign": sign} if sign and "sign" in pr else {}
        if "end" in pr:
            wp1 = C.wps("wp1")[0]["pos"]; kw["end"] = C.above_wp2(wp1, self.wp2_now[0]["pos"])[0]
        if self.exit_via:                            # out of WP2 and over to the tape spot in one move
            try: C.path(self.exit_via + [C.tape_poses()[0]], gripper=g_lock)
            finally: self.exit_via = []
        return T.regrip_vision(g_lock, k, **kw)

    at_wait = 2.5

    def at(self, pose, wait=None):
        """Is the arm really AT this pose (measured joints, not the command)? Waits up to `wait` s for it to settle.
        -> (yes, largest joint error in ticks)."""
        A = self.A; t0 = time.time()
        while True:
            e = max(abs(A.pos[n] - pose[n]) for n in BODY if n in pose)
            if e <= AT_PLATE: return True, e
            if time.time() - t0 > (self.at_wait if wait is None else wait): return False, e
            self.T.check(); time.sleep(0.1)

    def drop(self, which, g_lock, g_open):
        """which = 'accept' | 'reject' -> the plate assigned to it. THE CLAMP OPENS ONLY HERE, and only once the
        measured arm stands at the taught plate pose (the user's rule): every move before it carries g_lock, and an
        arm that did not get there keeps the block and stops the run instead of letting go somewhere else."""
        A, C, T = self.A, self.C, self.T
        slot = self.cfg[which]; tgt = C.slot(slot); L = slot[-1].upper()
        if not tgt: raise RuntimeError(f"teach plate {L} first (dev Waypoints page)")
        p = tgt[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]; ab = C.slot(slot + "_above")
        above = ab[0]["pos"] if ab else {**p, "shoulder_lift": wp1["shoulder_lift"], "elbow_flex": wp1["elbow_flex"],
                                          "wrist_flex": wp1["wrist_flex"]}
        T.state = "sort_good" if which == "accept" else "sort_defect"
        T.say(f"block → plate {L} ({which})")
        via, self.exit_via = self.exit_via, []
        C.path(via + [above, p], gripper=g_lock)          # out of WP2, over the plate and down in one move
        ok, e = self.at(p)
        if not ok:
            raise RuntimeError(f"the arm did not reach plate {L} ({e / AT_PLATE * 5:.0f}° off): clamp kept closed, block still held")
        C.goto(dict(A.cmd), 1.0, gripper=g_open); time.sleep(0.2)        # the only release of a sorted block
        self.g_lock = None
        C.path([above, C.wps("home")[0]["pos"]], gripper=g_open)         # up and home in one move

    def clear_gripper(self, g_open):
        """Before a run: a closed clamp may still hold a block (a run that was stopped). The first move of a pick
        opens the gripper on the way home, so whatever is in the jaws is carried to the reject plate and let go
        there first. -> True if it did that."""
        A, C, T = self.A, self.C, self.T
        gmin, gmax = A.fl["gripper"]
        if A.pos["gripper"] > gmin + 0.2 * (gmax - gmin): return False        # open: nothing can be held
        g = A.cmd["gripper"]; wp1 = C.wps("wp1")[0]["pos"]
        T.state = "sort_defect"; T.say("the clamp is closed: taking whatever it holds to the reject plate first")
        C.rise(wp1); C.goto(wp1, gripper=g)
        self.exit_via = []; self.drop("reject", g, g_open)
        return True

    # ------------------------------------------------------------------ one block
    def inspect(self, brick, g_lock, idx, model, t_low, t_high):
        """Steps 2-5 of the flow for the held block. -> result dict; verdict 'accept' | 'reject' | 'aborted'."""
        T, C = self.T, self.C
        wp1 = C.wps("wp1")[0]["pos"]; wp2 = self.poses(brick); self.wp2_now = wp2; self.brick_now = brick
        side = self.cfg["side_view"] if self.cfg["side_view"] in [w["name"] for w in wp2] else wp2[0]["name"]
        steps = []; sides = {0}; cum = 0; sign = None; k = 0; turns = []; self.g_lock = g_lock
        need = {0, 90, 180, 270}; hidden = None      # side = the block's turn in the jaws against the first grip
        lenient = self.cfg["uncertain_to"] == "accept"
        self.cur.update({"steps": steps, "grip": 1, "sides": 1})

        def out(verdict, reason, **kw):
            return {"verdict": verdict, "reason": reason, "steps": steps, "reclamps": k, "turns_deg": turns,
                    "sides_shown": len(sides), "sides_hidden": hidden, "g_lock": g_lock, **kw}

        def step(view, nudge=None):
            st = self.look(brick, view, k + 1, idx, model, t_low, t_high, nudge)
            if st["verdict"] == "uncertain" and lenient: st["verdict"], st["passed_doubt"] = "clean", True
            steps.append(st); return st

        def rejected(st):
            if st["verdict"] == "defect":
                return out("reject", f"anomaly at grip {st['grip']}, {st['view']}: {st['reason']}")
            return out("reject", f"not confirmed clean at grip {st['grip']}, {st['view']}: {st['reason']}", unconfirmed=True)

        def showing(views, onward):
            if self.cfg.get("async_judge"):
                return self.show_async(wp1, wp2, views, g_lock, k + 1, idx, model, t_low, t_high, onward, steps, lenient)
            return self.show(wp1, wp2, views, g_lock, step)
        try: to_tape = [C.tape_poses()[0]]
        except Exception: to_tape = []

        bad = showing([w["name"] for w in wp2], to_tape)                           # first showing: all angles
        if bad: return rejected(bad)
        while not need <= sides:
            if k >= int(self.cfg["max_reclamps"]):
                return out("reject", f"only {len(sides)} of {len(need)} sides shown after {k} re-clamps (fail-safe)", unconfirmed=True)
            k += 1; self.cur["grip"] = k + 1; self.gate()
            try:
                self.g_lock = None                   # the re-clamp lets go on the tape spot
                g_lock, info = self.reclamp(g_lock, k, sign); self.g_lock = g_lock
            except pol.PickFailed as e:              # the block lies on the re-clamp spot: fetch it, it can't be accepted
                why = f"re-clamp {k} failed ({e})"
                prev = T._vopt; T._vopt = {"target": brick, "retries": 2, "force": T.spot_force(None)}
                try: g_lock = T.pick_vision(tag=f"fetch{k}_"); self.g_lock = g_lock
                except pol.PickFailed as e2:
                    return out("aborted", f"{why}; the block is still on the re-clamp spot ({e2})", on_spot=True)
                finally: T._vopt = prev
                return out("reject", f"{why}: {len(sides)} of {SIDES} sides shown (fail-safe)", unconfirmed=True)
            turn = info.get("turn_deg", (info.get("forced") or {}).get("turn_deg"))
            if turn is None or round(abs(float(turn))) not in (90, 180):
                return out("reject", f"re-clamp {k}: the turn was not reported, sides not verifiable (fail-safe)", unconfirmed=True)
            turn = int(round(float(turn))); turns.append(turn)
            if abs(turn) == 180:                     # not square (green 2x3): only turned end for end, the two long
                need = {0, 180}; hidden = info.get("hidden") or "two sides stay under the fingers"   # sides are never shown
            elif sign is None: sign = 1 if turn > 0 else -1
            cum = (cum + turn) % 360
            last = need <= (sides | {cum})           # the last showing heads for the accept plate, the others for the tape spot
            bad = showing([side], [self.plate_poses("accept")[0]] if last else to_tape)   # after a re-clamp: the side only
            if bad: return rejected(bad)
            sides.add(cum); self.cur["sides"] = len(sides); self.cur["sides_need"] = len(need)
        if hidden:                                   # the user's rule: a face no showing can reveal is not counted against it
            return out("accept", f"both showable sides clean ({len(steps)} photos, {k} re-clamp{'s' * (k != 1)}); not inspected: {hidden}")
        return out("accept", f"all {SIDES} sides clean ({len(steps)} photos, {k} re-clamps)")

    def cycle(self, idx, colour):
        T, C = self.T, self.C
        home = C.wps("home")[0]["pos"]; g_open = T.g_open()
        self.cur = {"block": idx, "steps": [], "grip": 1, "sides": 0, "verdict": None}
        t0 = time.perf_counter(); brick = colour or None; self.g_lock = None
        e0 = _energy(); self.exit_via = []
        self.gate()
        if self.A.mode != "hold": self.A.set_mode("hold", force=True)
        T._vopt = {"target": colour or None, "retries": 3}                        # a failed pick is tried again
        try:
            g_lock = T.pick_vision()
        except pol.PickFailed as e:
            empty = "brick found on" in str(e)
            res = {"verdict": "empty" if empty else "aborted", "reason": str(e), "steps": [], "pick_failed": True}
            if getattr(e, "retry", True): C.goto(home, gripper=g_open)
            g_lock = None
        finally:
            T._vopt = None
        t1 = time.perf_counter(); t2 = t1
        if g_lock is not None:
            brick = ((T.act.attempts or [{}])[-1]).get("brick") or colour
            self.cur["brick"] = brick
            try:
                model, root = T.scorer.find("front", brick); th = T.scorer.thresholds(root) or {}
                if th.get("t_low") is None or th.get("t_high") is None: raise RuntimeError(f"no thresholds.json for the {brick} model")
                res = self.inspect(brick, g_lock, idx, model, th["t_low"], th["t_high"])
                t2 = time.perf_counter()
                if res["verdict"] in ("accept", "reject"):
                    self.gate(); self.drop(res["verdict"], res["g_lock"], g_open)
            except NotHeld as e:
                # No block was SEEN in the jaws. The clamp is still closed, so it is opened at the reject plate, not on
                # the way home: if the camera was wrong and a block is held, it lands on the reject plate.
                t2 = time.perf_counter()
                res = {"verdict": "aborted", "reason": str(e), "steps": self.cur.get("steps", []), "lost": True}
                if self.g_lock is not None:
                    self.drop("reject", self.g_lock, g_open); res["released_at"] = self.cfg["reject"]
            if res["verdict"] == "aborted" and not res.get("released_at"): C.goto(home, gripper=g_open)   # drop() ends at home
        t3 = time.perf_counter(); res.pop("g_lock", None)
        infer = [s["infer_ms"] for st in res["steps"] for s in st["shots"] if s.get("infer_ms") is not None]
        rec = {"block": idx, "time": datetime.now().isoformat(timespec="seconds"), "brick": brick, **res,
               "plate": self.cfg.get(res["verdict"]), "photos": sum(len(st["shots"]) for st in res["steps"]),
               "t_pick_s": round(t1 - t0, 1), "t_inspect_s": round(t2 - t1, 1), "t_sort_s": round(t3 - t2, 1),
               "t_cycle_s": round(t3 - t0, 1), "infer_ms_mean": round(sum(infer) / len(infer), 1) if infer else None,
               "device": T.cfg["device"]}
        e1 = _energy()
        rec["energy_j"] = round(e1 - e0, 1) if e0 is not None and e1 is not None else None
        self.cur["verdict"] = res["verdict"]
        if res["verdict"] != "empty":
            with open(T.RUNS / self.session / "log.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
            self.totals[res["verdict"]] = self.totals.get(res["verdict"], 0) + 1
            self.history = ([{k: rec.get(k) for k in ("block", "brick", "verdict", "reason", "plate", "photos", "reclamps",
                                                      "t_cycle_s", "t_pick_s", "t_inspect_s", "t_sort_s", "infer_ms_mean",
                                                      "device", "energy_j")}] + self.history)[:20]
            T.say(f"block {idx} ({brick}): {res['verdict'].upper()} - {res['reason']} ({rec['t_cycle_s']} s)", res["verdict"] != "aborted")
        return rec

    def run(self, n, colour):
        T, C, A = self.T, self.C, self.A
        self.running = True; T.active = True; A.running = True; C.active = True
        for e in (T.stop_flag, A.stop, C.stop_flag): e.clear()
        self.pause.clear(); self.paused = False
        self.session = T.session = datetime.now().strftime("flow_%Y%m%d_%H%M%S")
        d = T.RUNS / self.session; d.mkdir(parents=True, exist_ok=True); T._vdir = d
        (d / "settings.json").write_text(json.dumps({"flow": self.cfg, "sort": T.cfg, "n": n, "colour": colour}, indent=1))
        self.totals = {"accept": 0, "reject": 0, "aborted": 0}; self.history = []; aborted = 0; end = ""
        try:
            if A.mode != "hold": A.set_mode("hold", force=True)
            if hasattr(T, "apply_motion"): T.apply_motion()           # the "Fast moves" switch (sorter cfg motion_fast)
            self.clear_gripper(T.g_open())
            for i in range(1, n + 1):
                rec = self.cycle(i, colour)
                if rec["verdict"] == "empty": end = " · start plate empty"; break
                aborted = aborted + 1 if rec["verdict"] == "aborted" else 0
                if rec.get("on_spot") or aborted >= 2:
                    raise RuntimeError(f"needs a hand: {rec['reason']}")
            t = self.totals
            T.state = "done"; T.say(f"✓ DONE: {t['accept']} accepted, {t['reject']} rejected, {t['aborted']} aborted{end} · log: runs/{self.session}")
        except Stopped:
            T.state = "stopped"; T.say("STOPPED: arm holds its position", False)
        except Exception as e:
            T.state = "error"; T.say(f"ERROR in the sort flow: {e} (arm holds its position)", False)
        finally:
            try: C.hold_here()
            except Exception: pass
            T._vopt = None; self.pause.clear(); self.paused = False
            T.active = False; A.running = False; C.active = False; self.running = False

    def status(self):
        return {"running": self.running, "pause": self.pause.is_set(), "paused": self.paused,
                "session": self.session, "cfg": self.cfg, "cur": self.cur,
                "totals": self.totals, "history": self.history, "state": self.T.state, "msg": self.T.msg, "ok": self.T.ok,
                "plates": {s: bool(self.C.slot(s)) for s in PLATE_SLOTS}}


def make_router(S, A, ctx, C, T, F, AC, DATA):
    FL = Flow(S, A, ctx, C, T, F, AC, DATA)
    r = APIRouter()

    @r.get("/flow/status")
    def status(): return FL.status()

    @r.post("/flow/start")
    def start(n: int = 0, colour: str = ""):
        """n blocks (0 = until the start plate is empty, at most 50); colour '' = whichever block is nearest."""
        def bad(m): return {**FL.status(), "started": False, "msg": m, "ok": False}
        if colour and colour not in ac.BRICKS: return bad(f"colour must be one of {ac.BRICKS} or empty")
        if AC.busy(): return bad("busy: STOP the running job first")
        if A.f is None: return bad("connect the arm first")
        try:
            for k in ("home", "wp1", "wp2"): C.wps(k)
            T.regrip_spot()
        except Exception as e: return bad(str(e))
        if not T.tape_ok(): return bad("teach 'Above the tape spot' and 'On the tape spot' first (the re-clamp needs them)")
        miss = [c for c in ("top", "front") if c not in S.live]
        if miss: return bad(f"camera not live: {', '.join(miss)}")
        for which in ("accept", "reject"):
            if not C.slot(FL.cfg[which]): return bad(f"teach plate {FL.cfg[which][-1].upper()} first (dev Waypoints page)")
        for b in ([colour] if colour else ac.BRICKS):
            try:
                _, root = T.scorer.find("front", b)
                if not (T.scorer.thresholds(root) or {}).get("t_high"): return bad(f"no thresholds.json for the {b} model")
            except Exception as e: return bad(str(e))
        threading.Thread(target=FL.run, args=(n if 0 < n <= 50 else 50, colour), daemon=True).start()
        time.sleep(0.1); return {**FL.status(), "started": True}

    @r.post("/flow/cfg")
    def cfg(accept: str = "", uncertain_to: str = "", max_reclamps: int = -1, retakes: int = -1, side_view: str = None,
            async_judge: int = -1, photo_frames: int = -1, photo_still_s: float = -1, photo_wait_s: float = -1):
        """accept = the plate that takes the clean blocks (plate_a | plate_b); the other one takes the rejected ones."""
        if FL.running: return {**FL.status(), "msg": "not while the flow runs", "ok": False}
        if accept in PLATE_SLOTS:
            FL.cfg["accept"] = accept; FL.cfg["reject"] = PLATE_SLOTS[1 - PLATE_SLOTS.index(accept)]
        if uncertain_to in ("reject", "accept"): FL.cfg["uncertain_to"] = uncertain_to
        if 3 <= max_reclamps <= 6: FL.cfg["max_reclamps"] = max_reclamps
        if 0 <= retakes <= 3: FL.cfg["retakes"] = retakes
        if side_view is not None: FL.cfg["side_view"] = side_view.strip()
        if async_judge in (0, 1): FL.cfg["async_judge"] = bool(async_judge)      # 0 = wait for every verdict at the photo pose
        if 2 <= photo_frames <= 16: FL.cfg["photo_frames"] = photo_frames        # 16 = what the training photos used
        if 0 <= photo_still_s <= 1: FL.cfg["photo_still_s"] = photo_still_s      # 0 = feed.settle (0.5 s + 0.4 s)
        if 0 <= photo_wait_s <= 1: FL.cfg["photo_wait_s"] = photo_wait_s
        FL.save_cfg(); return FL.status()

    @r.post("/flow/pause")
    def pause():
        """Stand still at the next pause point (after the move under way). The block stays in the jaws."""
        if FL.running: FL.pause.set()
        return FL.status()

    @r.post("/flow/resume")
    def resume():
        FL.pause.clear(); return FL.status()

    @r.post("/flow/stop")
    def stop():
        T.stop_flag.set(); A.stop.set(); C.stop_flag.set(); FL.pause.clear()
        return FL.status()

    r.flow = FL
    return r
