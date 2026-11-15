"""Production sort flow (the user's spec, 2026-11-15 07:45). API under /flow/*; served by capture.py.

One block:
  1. vision pick off the start plate -> WP1 (a failed pick is tried again, vpick retries).
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
A doubt is photographed again (cfg retakes); a doubt that stays goes where cfg uncertain_to says: "reject"
(default, fail-safe, logged as unconfirmed) or "accept".
No block in the photo (no colour mask) after the retakes = the block is not in the jaws: the cycle is aborted,
nothing is put on a plate.

Pause (POST /flow/pause, /flow/resume): the arm stands still at the next pause point - before a pick, a photo, a
re-clamp or the plate - holding the block; the move that is under way finishes first. STOP works while paused.

Log: runs/flow_<time>/log.jsonl (one line per block: verdict, reason, every photo with score and times).
"""
import inspect as _inspect
import json, threading, time
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import APIRouter

import autocollect as ac
import feed
import policy as pol
import prep
from arm import Stopped

HERE = Path(__file__).resolve().parent
PLATE_SLOTS = ("plate_a", "plate_b")     # taught on the dev Waypoints page (section Plates); optional <slot>_above
SIDES = 4                  # side faces of a block; one is shown per grip


class NotHeld(Exception):
    """No block in the jaws at a photo pose."""


class Flow:
    def __init__(self, S, A, ctx, C, T, F, AC, DATA):
        self.S, self.A, self.ctx, self.C, self.T, self.F, self.AC, self.DATA = S, A, ctx, C, T, F, AC, DATA
        self.cfg_path = HERE / "flow_settings.json"
        self.cfg = {"accept": "plate_a", "reject": "plate_b", "max_reclamps": 4, "retakes": 1,
                    "uncertain_to": "reject", "side_view": ""}
        try: self.cfg.update(json.loads(self.cfg_path.read_text()))
        except Exception: pass
        if self.cfg["accept"] not in PLATE_SLOTS:                  # settings from before the plates had their own slots
            self.cfg["accept"] = PLATE_SLOTS[0]
        self.cfg["reject"] = PLATE_SLOTS[1 - PLATE_SLOTS.index(self.cfg["accept"])]
        self.running = False; self.session = None; self.cur = {}; self.history = []
        self.pause = threading.Event(); self.paused = False        # pause asked for / the arm is standing at a gate
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
    def poses(self, brick):
        """The WP2 poses for a block picked off the plate (same lift as autocollect.photos: it hangs lower in the
        jaws than a hand-fed one)."""
        C, F = self.C, self.F
        taught = C.wps("wp2"); old = F.lift
        try:
            F.lift = {**old, brick: {w["name"]: float((old.get(brick) or {}).get(w["name"]) or 0)
                                     + ac.AUTO_LIFT_BRICK.get(brick, ac.AUTO_LIFT).get(w["name"], 0) for w in taught}}
            try: return F.lifted(taught, brick)
            except Exception:
                F.lift = old; return F.lifted(taught, brick)
        finally:
            F.lift = old

    # ------------------------------------------------------------------ one photo -> verdict
    def shot(self, brick, view, idx):
        T, F = self.T, self.F
        F.settle(); T.check(); prep.configure_inhand(brick=brick)
        t0 = time.perf_counter()
        nf = self.ctx.capture(["front"], "test", brick, view, idx, self.session)
        rec = next((m for m in reversed(F._last_meta(3)) if m.get("cam") == "front"), {}) if nf else {}
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

    def look(self, brick, view, grip, idx, model, t_low, t_high):
        """Photo + verdict at the pose the arm stands in; a doubt is photographed again. -> step record.
        Raises NotHeld when no block is seen there."""
        T = self.T; shots = []
        for k in range(1 + max(0, int(self.cfg["retakes"]))):
            T.check(); self.gate(); T.state = "inspect"; T.say(f"block {idx}: grip {grip}, {view}, photo {k + 1}")
            p = self.judge(self.shot(brick, view, idx), brick, model, t_low, t_high); shots.append(p)
            if p["verdict"] in ("clean", "defect"): break
        if p["verdict"] == "nomask": raise NotHeld(f"no {brick} block in the jaws at {view} (grip {grip}): {p['reason']}")
        return {"grip": grip, "view": view, "verdict": p["verdict"], "reason": p["reason"], "score": p.get("score"),
                "foreign": p.get("foreign"), "shots": shots}

    # ------------------------------------------------------------------ motions
    def show(self, wp1, wp2, views, g_lock, fn):
        """WP1 -> WP2, fn(pose name) at every pose in `views` until one is not clean, back up to WP1.
        -> the step that stopped it (or None = all clean)."""
        C = self.C; at = wp2[0]; bad = None
        C.to_wp2(wp1, wp2, g_lock)
        try:
            for w in wp2:
                if w["name"] not in views: continue
                if w is not at: C.goto(w["pos"], C.cfg["speed"] * 0.4, gripper=g_lock); at = w
                st = fn(w["name"])
                if st["verdict"] != "clean": bad = st; break
        finally:
            if not (self.T.stop_flag.is_set() or self.A.stop.is_set()):       # leave WP2 the way it came in
                if at is not wp2[0]: C.goto(wp2[0]["pos"], C.cfg["speed"] * 0.4, gripper=g_lock)
                C.up_wp2(wp1, wp2, g_lock); C.goto(wp1, gripper=g_lock)
        return bad

    def reclamp(self, g_lock, k, sign):
        T = self.T
        kw = {"sign": sign} if sign and "sign" in _inspect.signature(T.regrip_vision).parameters else {}
        return T.regrip_vision(g_lock, k, **kw)

    def drop(self, which, g_lock, g_open):
        """which = 'accept' | 'reject' -> the plate waypoint assigned to it."""
        A, C, T = self.A, self.C, self.T
        slot = self.cfg[which]; tgt = C.slot(slot)
        if not tgt: raise RuntimeError(f"teach plate {slot[-1].upper()} first (dev Waypoints page)")
        p = tgt[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]; ab = C.slot(slot + "_above")
        above = ab[0]["pos"] if ab else {**p, "shoulder_lift": wp1["shoulder_lift"], "elbow_flex": wp1["elbow_flex"],
                                          "wrist_flex": wp1["wrist_flex"]}
        T.state = "sort_good" if which == "accept" else "sort_defect"
        T.say(f"block → {which.upper()} plate")
        C.goto(above, gripper=g_lock)
        C.goto(p, C.cfg["speed"] * 0.5, gripper=g_lock)
        C.goto(dict(A.cmd), 1.0, gripper=g_open); time.sleep(0.25)
        C.goto(above, C.cfg["speed"] * 0.6, gripper=g_open)

    # ------------------------------------------------------------------ one block
    def inspect(self, brick, g_lock, idx, model, t_low, t_high):
        """Steps 2-5 of the flow for the held block. -> result dict; verdict 'accept' | 'reject' | 'aborted'."""
        T, C = self.T, self.C
        wp1 = C.wps("wp1")[0]["pos"]; wp2 = self.poses(brick)
        side = self.cfg["side_view"] if self.cfg["side_view"] in [w["name"] for w in wp2] else wp2[0]["name"]
        steps = []; sides = {0}; cum = 0; sign = None; k = 0; turns = []
        need = {0, 90, 180, 270}; hidden = None      # side = the block's turn in the jaws against the first grip
        lenient = self.cfg["uncertain_to"] == "accept"
        self.cur.update({"steps": steps, "grip": 1, "sides": 1})

        def out(verdict, reason, **kw):
            return {"verdict": verdict, "reason": reason, "steps": steps, "reclamps": k, "turns_deg": turns,
                    "sides_shown": len(sides), "sides_hidden": hidden, "g_lock": g_lock, **kw}

        def step(view):
            st = self.look(brick, view, k + 1, idx, model, t_low, t_high)
            if st["verdict"] == "uncertain" and lenient: st["verdict"], st["passed_doubt"] = "clean", True
            steps.append(st); return st

        def rejected(st):
            if st["verdict"] == "defect":
                return out("reject", f"anomaly at grip {st['grip']}, {st['view']}: {st['reason']}")
            return out("reject", f"not confirmed clean at grip {st['grip']}, {st['view']}: {st['reason']}", unconfirmed=True)

        bad = self.show(wp1, wp2, [w["name"] for w in wp2], g_lock, step)          # first showing: all angles
        if bad: return rejected(bad)
        while not need <= sides:
            if k >= int(self.cfg["max_reclamps"]):
                return out("reject", f"only {len(sides)} of {len(need)} sides shown after {k} re-clamps (fail-safe)", unconfirmed=True)
            k += 1; self.cur["grip"] = k + 1; self.gate()
            try:
                g_lock, info = self.reclamp(g_lock, k, sign)
            except pol.PickFailed as e:              # the block lies on the re-clamp spot: fetch it, it can't be accepted
                why = f"re-clamp {k} failed ({e})"
                prev = T._vopt; T._vopt = {"target": brick, "retries": 2, "force": T.spot_force(None)}
                try: g_lock = T.pick_vision(tag=f"fetch{k}_")
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
            bad = self.show(wp1, wp2, [side], g_lock, step)                       # after a re-clamp: the side only
            if bad: return rejected(bad)
            sides.add(cum); self.cur["sides"] = len(sides); self.cur["sides_need"] = len(need)
        if hidden:                                   # the user's rule: a face no showing can reveal is not counted against it
            return out("accept", f"both showable sides clean ({len(steps)} photos, {k} re-clamp{'s' * (k != 1)}); not inspected: {hidden}")
        return out("accept", f"all {SIDES} sides clean ({len(steps)} photos, {k} re-clamps)")

    def cycle(self, idx, colour):
        T, C = self.T, self.C
        home = C.wps("home")[0]["pos"]; g_open = T.g_open()
        self.cur = {"block": idx, "steps": [], "grip": 1, "sides": 0, "verdict": None}
        t0 = time.perf_counter(); brick = colour or None
        self.gate()
        if self.A.mode != "hold": self.A.set_mode("hold", force=True)
        C.goto(home, gripper=g_open)
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
                t2 = time.perf_counter()
                res = {"verdict": "aborted", "reason": str(e), "steps": self.cur.get("steps", []), "lost": True}
            C.goto(home, gripper=g_open)
        t3 = time.perf_counter(); res.pop("g_lock", None)
        infer = [s["infer_ms"] for st in res["steps"] for s in st["shots"] if s.get("infer_ms") is not None]
        rec = {"block": idx, "time": datetime.now().isoformat(timespec="seconds"), "brick": brick, **res,
               "plate": self.cfg.get(res["verdict"]), "photos": sum(len(st["shots"]) for st in res["steps"]),
               "t_pick_s": round(t1 - t0, 1), "t_inspect_s": round(t2 - t1, 1), "t_sort_s": round(t3 - t2, 1),
               "t_cycle_s": round(t3 - t0, 1), "infer_ms_mean": round(sum(infer) / len(infer), 1) if infer else None,
               "device": T.cfg["device"]}
        self.cur["verdict"] = res["verdict"]
        if res["verdict"] != "empty":
            with open(T.RUNS / self.session / "log.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
            self.totals[res["verdict"]] = self.totals.get(res["verdict"], 0) + 1
            self.history = ([{k: rec.get(k) for k in ("block", "brick", "verdict", "reason", "plate", "photos", "reclamps",
                                                      "t_cycle_s")}] + self.history)[:20]
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
            for i in range(1, n + 1):
                rec = self.cycle(i, colour)
                if rec["verdict"] == "empty": end = " · start plate empty"; break
                aborted = aborted + 1 if rec["verdict"] == "aborted" else 0
                if rec.get("on_spot") or aborted >= 2:
                    raise RuntimeError(f"needs a hand: {rec['reason']}")
                time.sleep(0.3)
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
    def cfg(accept: str = "", uncertain_to: str = "", max_reclamps: int = -1, retakes: int = -1, side_view: str = None):
        """accept = the plate that takes the clean blocks (plate_a | plate_b); the other one takes the rejected ones."""
        if FL.running: return {**FL.status(), "msg": "not while the flow runs", "ok": False}
        if accept in PLATE_SLOTS:
            FL.cfg["accept"] = accept; FL.cfg["reject"] = PLATE_SLOTS[1 - PLATE_SLOTS.index(accept)]
        if uncertain_to in ("reject", "accept"): FL.cfg["uncertain_to"] = uncertain_to
        if 3 <= max_reclamps <= 6: FL.cfg["max_reclamps"] = max_reclamps
        if 0 <= retakes <= 3: FL.cfg["retakes"] = retakes
        if side_view is not None: FL.cfg["side_view"] = side_view.strip()
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
