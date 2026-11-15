"""Automatic clean-photo collection for the anomaly models (no hands): the arm feeds itself.

One cycle:  look at the plate (top camera) -> vision pick (vpick.py) of the colour that still needs the most photos
            -> WP1 -> WP2, a photo at every taught WP2 pose (same capture path, mask, per-brick lift and settle wait
            as the hand-feed page, label "good") -> back over WP1 -> set the brick down at a random free spot of the
            start plate with a random turn -> home.
Repeats until every colour on the plate has `target` good crops (training_images/front/inhand/good/*_<brick>_*_crop.png)
or `max_cycles` is reached. Stops by itself after 3 failed picks in a row, 3 handlings in a row without a usable
mask, or when a colour that still needs photos is no longer seen on the plate.

It runs as a sorter job (Sorter.active), so the top-bar STOP / Space / Esc and /sort/stop stop it like any other
routine and nothing else can start meanwhile. Log: runs/autocollect_<time>/cycles.jsonl.
record=1 (default): every pick is also saved as a training episode (sorter.VRecorder), from the start pose until the
arm is at WP1; the WP2 photos and the put-back are not in it. Episodes: episodes/<time>_vpick/.

    POST /autocollect/start?target=100&max_cycles=200&record=1      (per_block=40&since=20261115_0330: instead count
                                                 handlings PICKED BY THE ARM since that time, 40 per block)
       GET /autocollect/status     POST /autocollect/stop
"""
import json, threading, time
from datetime import datetime

from fastapi import APIRouter

import policy as pol
import prep
from arm import Stopped

BRICKS = ("pink", "green", "blue")
MIN_AREA = 40_000        # px: a colour mask smaller than this at a WP2 pose is not a held brick (pink: see feed.MIN_AREA)
AUTO_LIFT = {"wp2-1": 22.0, "wp2-3": 12.0}    # mm, see AutoCollect.photos
# green: its feed_lift.json entry (+28 mm at wp2-1) was tuned for the hand-fed end grip; picked off the plate, +50 mm
# in total pushed it out of the TOP of the frame at wp2-1 (2026-11-15, 4 of 4 handlings)
AUTO_LIFT_BRICK = {"green": {"wp2-1": -6.0, "wp2-3": 12.0}}
MAX_FAILS = 3            # failed picks in a row / handlings without a usable mask in a row -> stop and ask


class AutoCollect:
    def __init__(self, S, A, ctx, C, T, F, DATA):
        self.S, self.A, self.ctx, self.C, self.T, self.F, self.DATA = S, A, ctx, C, T, F, DATA
        self.running = False; self.target = 100; self.cycles = []; self.session = None; self.start_counts = {}; self.rec_session = None; self.per_block = 0; self.since = ""

    def counts(self):
        d = self.DATA / "front" / "inhand" / "good"
        return {b: len(list(d.glob(f"*_{b}_*_crop.png"))) for b in BRICKS}

    def picked(self, since=""):
        """Handlings PICKED BY THE ARM per brick (meta.jsonl runs ending in _auto, capture time >= since) that gave at
        least one good crop. This is what the sorter will see, unlike the hand-fed photos."""
        h = {b: set() for b in BRICKS}; d = self.DATA
        try:
            for line in open(d / "meta.jsonl"):
                j = json.loads(line)
                if (j.get("stage") == "inhand" and j.get("label") == "good" and str(j.get("run", "")).endswith("_auto")
                        and j.get("brick") in h and j.get("time", "") >= since
                        and (d / j["file"]).with_name((d / j["file"]).stem + "_crop.png").exists()):
                    h[j["brick"]].add((j["run"], j.get("cycle")))
        except Exception: pass
        return {b: len(v) for b, v in h.items()}

    def progress(self):
        """What the run counts toward its goal: picked handlings (per_block mode) or all good crops."""
        return self.picked(self.since) if self.per_block else self.counts()

    def busy(self):
        T, C, A, F = self.T, self.C, self.A, self.F
        return bool(T.active or C.active or A.running or F.active or prep.TUNING)

    def edge_cut(self, file, top=40, bottom=1076):
        """'top' / 'bottom' if the mask of this capture reaches the frame edge there (1080 rows), else None."""
        try:
            import cv2, numpy as np
            p = self.DATA / file; m = cv2.imread(str(p.with_name(p.stem + "_masked.jpg")))
            ys = np.where((np.abs(m.astype(int) - np.array(prep.GREY)).sum(2) > 30).any(1))[0]
            s = m.shape[0] / 1080.0
            return "top" if ys.min() <= top * s else "bottom" if ys.max() >= bottom * s else None
        except Exception:
            return None

    def photos(self, brick, g_lock, wp1, idx, run_id):
        """WP1 -> WP2, one masked photo per WP2 pose, back up over WP2 and to WP1. -> list of photo records."""
        T, C, F = self.T, self.C, self.F
        import feed
        # a block picked off the plate hangs lower in the jaws than a hand-fed one (2026-11-15: the frame's bottom edge
        # cut it at wp2-1 and wp2-3), so these poses go up by AUTO_LIFT on top of the brick's own feed_lift.json entry
        taught = C.wps("wp2"); old = F.lift
        try:
            F.lift = {**old, brick: {w["name"]: float((old.get(brick) or {}).get(w["name"]) or 0) + AUTO_LIFT_BRICK.get(brick, AUTO_LIFT).get(w["name"], 0)
                                     for w in taught}}
            try: wp2 = F.lifted(taught, brick)
            except Exception as e:                       # lifted pose out of reach: the hand-feed poses
                F.lift = old; wp2 = F.lifted(taught, brick); T.say(f"auto lift not reachable ({e}): using the feed poses", False)
        finally:
            F.lift = old
        T.state = "to_wp2"; T.say(f"cycle {idx}: {brick} to WP2 (clamp locked)")
        C.to_wp2(wp1, wp2, g_lock)
        out = []
        for i, w in enumerate(wp2):
            T.check()
            if i: C.goto(w["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)
            T.state = "photo"; T.say(f"cycle {idx}: {brick} photo {i + 1}/{len(wp2)} at {w['name']}")
            F.settle(); T.check()
            prep.configure_inhand(brick=brick)
            nf = self.ctx.capture(["front"], "good", brick, w["name"], idx, run_id)
            rec = next((m for m in reversed(F._last_meta(3)) if m.get("cam") == "front"), {}) if nf else {}
            dep = rec.get("depth") if isinstance(rec.get("depth"), dict) else {}
            area = dep.get("area_px")
            ok = bool(nf) and bool(dep) and (area or 0) >= max(MIN_AREA, feed.MIN_AREA.get(brick, 0))
            why = rec.get("depth_mask_why") if nf else "no fresh front frame"
            if ok:                                       # a block cut by the frame's top or bottom edge is no training photo
                cut = self.edge_cut(rec.get("file"))
                if cut: ok, why = False, f"block cut by the {cut} of the frame"
            elif why is None: why = "mask too small: no block in the jaws"
            out.append({"view": w["name"], "file": rec.get("file"), "mask_ok": bool(dep), "area_px": area, "ok": ok, "why": why})
            F._grab_last(rec)
            if not ok and rec.get("file"):               # nothing (or not this brick) in the jaws: not a training photo
                stem = (self.DATA / rec["file"]).with_suffix("")
                for f in stem.parent.glob(stem.name + "*"):
                    try: f.unlink()
                    except OSError: pass
                out[-1]["deleted"] = True
        T.state = "returning"
        if len(wp2) > 1: C.goto(wp2[0]["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)      # leave the way it came in
        C.up_wp2(wp1, wp2, g_lock)
        C.goto(wp1, gripper=g_lock)
        return out

    def run(self, target, max_cycles, record=True, per_block=0, since=""):
        T, C, A = self.T, self.C, self.A
        self.running = True; self.cycles = []; self.per_block = per_block; self.since = since
        if per_block: target = per_block
        self.target = target; unit = "handlings picked by the arm" if per_block else "good crops"
        T.active = True; A.running = True; C.active = True
        for e in (T.stop_flag, A.stop, C.stop_flag): e.clear()
        T.session = self.session = datetime.now().strftime("autocollect_%Y%m%d_%H%M%S")
        d = T.RUNS / T.session; d.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_auto"
        self.start_counts = self.counts()
        (d / "settings.json").write_text(json.dumps({"target": target, "max_cycles": max_cycles, "run": run_id,
                                                     "counts_at_start": self.start_counts}, indent=1))
        T.trials = []; T._vdir = d; pick_fails = mask_fails = 0; n = 0; vrec = None; miss = {b: 0 for b in BRICKS}
        try:
            vp, kin, cal = T.get_vision()
            if record:                                   # every pick = one training episode, start pose -> WP1
                from sorter import VRecorder
                vrec = VRecorder(T, max_cycles, d); self.rec_session = vrec.session
            home = C.wps("home")[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]; g_open = T.g_open()
            while n < max_cycles:
                T.check()
                cnt = self.progress(); need = [b for b in BRICKS if cnt[b] < target]
                if not need:
                    T.state = "done"; T.say(f"✓ AUTO-COLLECT DONE: every block has {target}+ {unit} ({cnt})"); return
                T.state = "look"; C.goto(home, gripper=g_open)
                _, bricks, _ = vp.VisionPick(A, C, self.S, kin, cal, check=T.check, say=T.say).look()
                seen = {b["colour"] for b in bricks}
                cand = [b for b in need if b in seen]
                if not cand:
                    T.state = "help"
                    T.say(f"auto-collect stopped: {', '.join(need)} still need photos but the top camera doesn't see "
                          f"{'them' if len(need) > 1 else 'it'} on the plate (sees: {', '.join(sorted(seen)) or 'nothing'}). "
                          f"Put the block(s) on the plate and start again. Counts: {cnt}", False); return
                # a block that was missed twice in a row is probably tipped over or on the plate rim: leave it for a
                # person and go on with the other colours instead of hammering it
                skip = [b for b in cand if miss[b] >= 2]; cand = [b for b in cand if miss[b] < 2]
                if not cand:
                    T.state = "help"
                    T.say(f"auto-collect stopped: {', '.join(skip)} missed twice in a row each (tipped over or on the plate "
                          f"rim?). Stand the block(s) upright near the middle of the plate and start again. {unit}: {cnt}", False); return
                colour = min(cand, key=lambda b: cnt[b])
                n += 1; t0 = time.time(); rec = {"cycle": n, "colour": colour, "counts": cnt}
                T._vopt = {"target": colour, "retries": 1}
                if vrec:                                     # one attempt = one episode (a retry would restart the buffer)
                    vrec.begin(n); T._vopt = {"target": colour, "retries": 0, "on_event": vrec.on_event, "tuner": vrec.tuner}
                try:
                    g_lock = T.pick_vision()
                    if vrec: vrec.end(True, g_lock)          # the cut at WP1: photos and put-back are not in the episode
                except pol.PickFailed as e:
                    if vrec: vrec.end(False)
                    pick_fails += 1; miss[colour] += 1; rec.update(ok=False, error=str(e), t_s=round(time.time() - t0, 1)); self._log(d, rec)
                    if not e.retry or pick_fails >= 2 * MAX_FAILS:
                        T.state = "help"
                        T.say(f"auto-collect stopped after {pick_fails} failed pick(s) in a row: {e}", False); return
                    C.goto(dict(A.cmd), 0.4, gripper=g_open); C.rise(wp1); C.goto(home, gripper=g_open); continue
                except BaseException:
                    if vrec: vrec.end(False, discard=True)
                    raise
                pick_fails = 0; miss[colour] = 0
                brick = ((T.act.attempts or [{}])[-1]).get("brick") or colour
                ph = self.photos(brick, g_lock, wp1, n, run_id)
                good = sum(p["ok"] for p in ph)
                mask_fails = 0 if good else mask_fails + 1
                placed = None
                for k in range(3):                                   # a random spot can be out of reach: draw another
                    try: placed = T.place_and_look(n, g_lock, home, g_open, d); break
                    except pol.PickFailed as e:
                        if k == 2: raise RuntimeError(f"no reachable spot to set the {brick} block down: {e}")
                        C.goto(wp1, gripper=g_lock)
                if vrec: vrec.verify(bool(placed))
                rec.update(ok=True, brick=brick, photos=ph, good_photos=good, placed_seen=placed,
                           t_s=round(time.time() - t0, 1)); self._log(d, rec)
                T.say(f"cycle {n}: {brick} · {good}/{len(ph)} photos masked · {unit} now {self.progress()}", good == len(ph))
                if mask_fails >= MAX_FAILS:
                    T.state = "help"
                    T.say(f"auto-collect stopped: {mask_fails} handlings in a row without a usable mask (last: "
                          f"{ph[-1].get('why') or 'mask too small'}). Check the front camera view and the block.", False); return
                time.sleep(0.3)
            T.state = "done"; T.say(f"auto-collect: {max_cycles} cycles done · {unit} {self.progress()}")
        except Stopped:
            T.state = "stopped"; T.say("STOPPED: arm holds its position", False)
        except Exception as e:
            T.state = "error"; T.say(f"ERROR in auto-collect: {e} (arm holds its position)", False)
        finally:
            try: C.hold_here()
            except Exception: pass
            T.active = False; A.running = False; C.active = False; self.running = False

    def _log(self, d, rec):
        self.cycles.append({k: v for k, v in rec.items() if k != "photos"})
        with open(d / "cycles.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")

    def status(self):
        T = self.T; c = self.counts()
        return {"running": self.running, "state": T.state if self.running else "idle", "msg": T.msg, "ok": T.ok,
                "target": self.target, "per_block": self.per_block, "since": self.since, "picked": self.picked(self.since), "counts": c, "added": {b: c[b] - self.start_counts.get(b, c[b]) for b in BRICKS},
                "session": self.session, "episodes": self.rec_session, "cycles": self.cycles[-10:], "n_cycles": len(self.cycles)}


def make_router(S, A, ctx, C, T, F, DATA):
    AC = AutoCollect(S, A, ctx, C, T, F, DATA)
    r = APIRouter()

    @r.get("/autocollect/status")
    def status(): return AC.status()

    @r.post("/autocollect/start")
    def start(target: int = 100, max_cycles: int = 200, record: bool = True, per_block: int = 0, since: str = ""):
        def bad(m): return {**AC.status(), "started": False, "msg": m, "ok": False}
        if AC.busy(): return bad("busy: STOP the running job first")
        if A.f is None: return bad("connect the arm first")
        try:
            for k in ("home", "wp1", "wp2"): C.wps(k)
        except Exception as e: return bad(str(e))
        miss = [c for c in ("top", "front") if c not in S.live]
        if miss: return bad(f"camera not live: {', '.join(miss)}")
        threading.Thread(target=AC.run, args=(max(1, min(1000, target)), max(1, min(1000, max_cycles)), bool(record), max(0, min(500, per_block)), since), daemon=True).start()
        time.sleep(0.1); return {**AC.status(), "started": True}

    @r.post("/autocollect/stop")
    def stop():
        T.stop_flag.set(); A.stop.set(); C.stop_flag.set()
        return AC.status()

    r.autocollect = AC
    return r
