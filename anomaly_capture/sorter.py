"""Phases 3-5: inspect the WHOLE surface of the held object (any shape), then sort it (good plate / defect plate).

Served by capture.py (same process: shares the locked cameras, the arm and the collector's motion helpers).
UI: section "Inspect & sort" on http://127.0.0.1:7870/arm ; API under /sort/*.

RULE (the user's): an object goes to the GOOD plate only after its whole surface has been seen and judged clean.
As soon as one photo shows a defect it goes to the DEFECT plate, without looking at the rest (object-level decision
= "any view anomalous", the max-over-views rule of multi-view anomaly detection). Anything that cannot be confirmed
clean - an uncertain score after retakes, a part of the surface no pose can show, a photo without a usable depth
mask - is never called good. A lost brick or a failed re-grip stops the cycle and asks for help.

Any shape. Nothing here knows about cubes: coverage.py tracks which surface DIRECTIONS have been seen (points on a
sphere fixed to the object). Each WP2 photo pose is annotated on the page with the sides it shows (front / back /
up / down, several ticks = diagonal view). What the fingers hide in one grip is seen after the tape-spot re-grip
(wrist turned 90 deg); the top of the object is photographed from above while it lies released on the tape spot
(wrist camera). The same poses therefore cover a cube, a long box, a triangular prism or a cylinder.

One cycle:  PICK (hand-feed / teleop / ACT policy = phase 6, policy.py) -> WP1
            repeat:  WP2 poses that still show something unseen (the rest are skipped)   -> defect? stop
                     all seen? -> GOOD.   nothing more can be seen? -> DEFECT plate (fail-safe)
                     tape spot: set down -> top photo of the released brick -> grip again turned 90 deg
            SORT:    above the plate -> down -> open -> up -> home
Per photo:  score >= t_high -> DEFECT (stop) ; score <= t_low -> CLEAN ; in between -> retake (mean of the scores)
            up to `retries` times, still in between -> UNCERTAIN -> defect plate.
Cascade (optional): a fast model (e.g. EfficientAd on the NPU) answers first; only photos it is unsure about go
to the accurate model (PatchCore on the GPU).

Every cycle is logged to ~/junction26/runs/<session>/log.jsonl (photos, scores, latencies, times) for the report.
"""
import json, threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response

import coverage as cov
import policy as pol
import prep
from arm import BODY, TICK_PER_DEG, Stopped
from collect import (HOLD_GAP, HOLD_MIN_OPEN, HOLD_STILL, HOLD_TIME, MATCH_DEG, MAX_PICK_S, REGRIP_SQUEEZE, TAIL_S)
from scorer import Scorer

HERE = Path(__file__).resolve().parent
BINS = ("good", "defect")


class Lost(Exception):
    """The brick is no longer in the gripper."""


def decide(scores, t_low, t_high):
    """Mean of the scores so far -> ('defect' | 'clean' | 'uncertain', mean)."""
    m = float(np.mean(scores))
    return ("defect" if m >= t_high else "clean" if m <= t_low else "uncertain"), m


def auroc(neg, pos):
    if not len(neg) or not len(pos): return None
    n, p = np.asarray(neg), np.asarray(pos)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def thresholds_from(good, defect_episode_max):
    """Two thresholds from scores. good: scores of clean photos (ideally held out from training).
    defect_episode_max: per defect brick, the highest score among its photos (the cross is only on one face, so
    most photos of a defect brick look clean; the highest one is the photo that shows it).
    Returns dict with t_low, t_high and diagnostics; raises if the model does not separate at all."""
    g = np.asarray(good, float); d = np.asarray(defect_episode_max, float)
    g_max, g_ref = float(g.max()), float(np.percentile(g, 95))
    vis = d[d > g_ref]
    if not len(vis):
        raise RuntimeError("no defect brick scores above the clean range: the model does not see the crosses "
                           "(or they were never facing this camera)")
    d_min = float(vis.min())
    if d_min > g_max:                       # separated: uncertain band = middle third of the gap
        gap = d_min - g_max; t_low, t_high = g_max + gap / 3, d_min - gap / 3
    else:                                   # overlap: everything in the overlap is uncertain -> retake -> fail-safe
        t_low, t_high = max(0.0, d_min - 0.01), g_max + 0.01
    return {"t_low": round(t_low, 4), "t_high": round(t_high, 4), "good_max": round(g_max, 4), "good_p95": round(g_ref, 4),
            "defect_min": round(d_min, 4), "overlap": bool(d_min <= g_max), "defect_bricks": int(len(d)),
            "defect_bricks_not_visible": int(len(d) - len(vis)),
            "good_uncertain_share": round(float((g > t_low).mean()), 3), "auroc": auroc(g, vis)}


class Sorter:
    def __init__(self, S, A, ctx, C, DATA, ROOT):
        self.S, self.A, self.ctx, self.C, self.DATA = S, A, ctx, C, DATA
        self.RUNS = ROOT / "runs"; self.RUNS.mkdir(exist_ok=True)
        self.scorer = Scorer(ROOT / "models", ROOT / "runs" / "score_worker.log")
        self.cfg_path = HERE / "sort_settings.json"
        self.cfg = {"pick": "handfeed", "judge": "manual", "brick": "", "cam": "front", "device": "GPU",
                    "t_low": 0.4, "t_high": 0.6, "retries": 2, "faces": {}, "allow_incomplete": False,
                    "release": True, "release_cam": "wrist", "cascade": False, "fast_variant": "_efficientad",
                    "fast_device": "NPU", "view_deg": cov.DEFAULTS["view_deg"], "finger_deg": cov.DEFAULTS["finger_deg"],
                    "max_regrasps": cov.DEFAULTS["max_regrasps"],
                    # pick = "act" (policy.py): Studio export dir, OpenVINO device, export backend, limits per pick
                    "act_model": "", "act_device": "GPU", "act_backend": "auto", "act_mode": "chunk",
                    "act_max_s": 20.0, "act_retries": 2, "act_speed": 1.0}
        try: self.cfg.update(json.loads(self.cfg_path.read_text()))
        except Exception: pass
        self.state, self.msg, self.ok = "idle", "teach the two plates, tick what each WP2 pose shows, then START", True
        self.active = False; self.stop_flag = threading.Event()
        self.flags = {"grasped": False, "next": threading.Event()}
        self.judge = {"img": None, "caption": "", "answer": None, "evt": threading.Event(), "n": 0}
        self.cur = {}; self.history = []; self.totals = {"good": 0, "defect": 0, "aborted": 0}
        self.photos = []; self.session = None; self.calib = {"running": False, "result": None}
        self._pol = None; self.clock = pol.RealClock(); self.act = None; self.act_info = None; self._roiw = (0, None)

    def save_cfg(self): self.cfg_path.write_text(json.dumps(self.cfg, indent=1))
    def say(self, m, ok=True): self.msg, self.ok = m, ok

    def check(self):
        if self.stop_flag.is_set() or self.A.stop.is_set(): raise Stopped()

    # ------------------------------------------------------------------ coverage model
    def params(self):
        return {"view_deg": float(self.cfg["view_deg"]), "finger_deg": float(self.cfg["finger_deg"]),
                "margin_deg": cov.DEFAULTS["margin_deg"], "max_regrasps": int(self.cfg["max_regrasps"])}

    def wp2(self): return self.C.wps("wp2")
    def sides(self, view): return self.cfg["faces"].get(view) or ["front"]
    def tape_ok(self): return bool(self.C.slot("tape_above") and self.C.slot("tape_place"))
    def release_on(self): return bool(self.cfg["release"] and self.tape_ok() and self.cfg["release_cam"] in self.S.live)

    def poses(self):
        try: return [(w["name"], self.sides(w["name"])) for w in self.wp2()]
        except Exception: return []

    def plan(self):
        return cov.plan(self.poses(), self.tape_ok(), self.release_on(), self.params())

    def g_open(self):
        gmin, gmax = self.A.fl["gripper"]
        return max(self.C.wps("home")[0]["pos"]["gripper"], gmin + 0.35 * (gmax - gmin))

    def held(self):
        if not self.C.holding(): raise Lost()

    # ------------------------------------------------------------------ picking
    def pick_handfeed(self):
        """Test mode: the robot waits at WP1 with the gripper open, you put the brick between the fingers, G closes."""
        A, C = self.A, self.C
        wp1 = C.wps("wp1")[0]["pos"]; gmin, gmax = A.fl["gripper"]
        self.state = "handfeed"; self.say("robot to waypoint 1, gripper open")
        C.goto(wp1, gripper=max(self.g_open(), gmin + 0.8 * (gmax - gmin)))
        self.flags["grasped"] = False
        self.say("HAND-FEED: hold the object between the open fingers (any orientation), then press G")
        while not self.flags["grasped"]:
            self.check(); time.sleep(0.05)
        self.say("closing on the object: keep holding it until the gripper has it")
        C.goto(dict(A.cmd), 0.25, gripper=gmin)                  # stops when it stalls on the object
        g_lock = max(gmin, A.pos["gripper"] - REGRIP_SQUEEZE)    # squeeze a bit past the object's width
        C.goto(dict(A.cmd), 0.5, gripper=g_lock); time.sleep(0.3)
        if not C.holding(): raise RuntimeError("hand-feed: the gripper did not close on an object")
        return g_lock

    def pick_teleop(self):
        """Human pick with the leader (same grasp detection as collection, not recorded), then rise + WP1."""
        A, C = self.A, self.C
        home = C.wps("home")[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]; g_open = self.g_open()
        gmin = A.fl["gripper"][0]
        self.state = "home"; self.say("robot to the start pose")
        C.goto(home, gripper=g_open)
        A.ensure_leader(); self.state = "match"; ok_since = None
        while True:
            self.check()
            lt = A.leader_target()
            gaps = {n: abs(lt[n] - A.cmd[n]) / TICK_PER_DEG for n in BODY}
            worst = max(gaps, key=gaps.get)
            ok_since = (ok_since or time.time()) if gaps[worst] <= MATCH_DEG else None
            self.say(f"move the LEADER to the robot's pose: biggest gap {worst} {gaps[worst]:.0f}° (needs ≤{MATCH_DEG:.0f}°)")
            if ok_since and time.time() - ok_since > 0.4: break
            time.sleep(0.05)
        A.set_mode("mirror", force=True)
        self.state = "pick"; self.flags["grasped"] = False; t0 = time.time(); gq = []; hold_since = None
        while True:
            self.check(); now = time.time()
            g, c = A.pos["gripper"], A.goal["gripper"]
            gq = [(t, v) for t, v in gq if now - t < 0.25] + [(now, g)]
            still = max(v for _, v in gq) - min(v for _, v in gq) <= HOLD_STILL and now - gq[0][0] > 0.2
            hold_since = (hold_since or now) if (c < g - HOLD_GAP and still and g - gmin > HOLD_MIN_OPEN) else None
            self.say(f"PICK the object with the leader · {now - t0:.0f} s (G if the robot doesn't notice the grasp)")
            if (hold_since and now - hold_since >= HOLD_TIME) or self.flags["grasped"]: break
            if now - t0 > MAX_PICK_S: raise RuntimeError("pick timed out")
            time.sleep(0.02)
        time.sleep(TAIL_S)
        C.hold_here(); g_lock = A.cmd["gripper"]
        self.state = "carry"; self.say("hands off: robot lifts the object to waypoint 1")
        C.rise(wp1); C.goto(wp1, gripper=g_lock); time.sleep(0.3)
        if not C.holding(): raise RuntimeError("object not in the gripper at waypoint 1")
        return g_lock

    def get_policy(self):
        """The ACT export chosen on the page, loaded once (reloaded when model / device / backend change)."""
        c = self.cfg; key = (c["act_model"], c["act_device"], c["act_backend"])
        if self._pol and self._pol[0] == key: return self._pol[1]
        self.drop_policy()
        if not c["act_model"]: raise RuntimeError("pick = ACT: choose a policy export first")
        self.state = "act_load"; self.say(f"loading the ACT policy on {c['act_device']} (the first load compiles, can take a minute)")
        self._pol = (key, pol.load(c["act_model"], c["act_device"], c["act_backend"]))
        return self._pol[1]

    def drop_policy(self):
        if self._pol: self._pol[1].close()
        self._pol = None

    def pick_act(self):
        """Autonomous pick (phase 6): start pose -> the trained ACT policy until the fingers stall on the object
        (the collector's hand-off rule) -> the scripted rise + carry to WP1 -> still held? Retries; after the last
        one policy.PickFailed (the cycle asks for help). Latencies go into the cycle log (act_info)."""
        P = self.get_policy(); c = self.cfg; b = c["brick"]
        self.state = "act"; self.flags["grasped"] = False; self.act_info = None
        self.act = pol.ActPick(self.A, self.C, self.S, P, max_s=c["act_max_s"], retries=c["act_retries"], mode=c["act_mode"],
                               speed=c["act_speed"], task=f"pick up the {b} brick" if b else "pick up the brick",
                               check=self.check, grasped=lambda: bool(self.flags["grasped"]), say=self.say, clock=self.clock)
        try:
            g_lock, self.act_info = self.act.run()
        except pol.PickFailed as e:
            self.act_info = e.info; raise
        return g_lock

    def pick(self):
        p = self.cfg["pick"]
        return self.pick_act() if p == "act" else self.pick_teleop() if p == "teleop" else self.pick_handfeed()

    def roi_warning(self):
        if time.time() - self._roiw[0] > 10:
            self._roiw = (time.time(), pol.roi_warning(self.C, getattr(self.C, "EP", self.RUNS.parent / "episodes")))
        return self._roiw[1]

    # ------------------------------------------------------------------ one photo -> decision
    def _photo(self, kind, cam):
        shots = self.S.averaged([cam])
        if cam not in shots: raise RuntimeError(f"camera {cam} gave no frames")
        f, first, last = shots[cam]
        if kind == "release":
            crop, info = prep.release_crop(f, cam)
        else:
            cobj = self.S.cams[cam]
            dep = getattr(cobj, "depth", None)
            if dep is not None and time.time() - getattr(cobj, "depth_t", 0) > 1.0: dep = None
            crop, info = prep.inspect_crop(f, dep)
        info["motion"] = round(float(np.abs(cv2.blur(first, (5, 5)) - cv2.blur(last, (5, 5))).mean()), 2)
        return crop, info

    def _remember(self, caption, img):
        sc = 220 / img.shape[0]
        jb = cv2.imencode(".jpg", cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))[1].tobytes()
        self.photos = ([(caption, jb)] + self.photos)[:8]

    def _manual(self, crop, caption):
        j = self.judge
        j["img"] = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        j["caption"], j["answer"] = caption, None; j["n"] += 1; j["evt"].clear()
        self.state = "judge"; self.say(f"MANUAL JUDGE: look at the photo ({caption}): CLEAN, DEFECT or UNSURE?")
        while not j["evt"].wait(0.1): self.check()
        j["img"] = None
        return j["answer"]

    def stages(self, cam):
        """Models to ask, in order: [optional fast model (only if it has calibrated thresholds), main model]."""
        brick = self.cfg["brick"]; out = []
        main, root = self.scorer.find(cam, brick)
        if self.cfg["cascade"]:
            try:
                fm, froot = self.scorer.find(cam, brick, self.cfg["fast_variant"]); ft = self.scorer.thresholds(froot)
                if ft: out.append({"name": "fast", "model": fm, "device": self.cfg["fast_device"],
                                   "t_low": ft["t_low"], "t_high": ft["t_high"]})
            except RuntimeError: pass
        t = self.scorer.thresholds(root) or {}
        out.append({"name": "main", "model": main, "device": self.cfg["device"],
                    "t_low": t.get("t_low", self.cfg["t_low"]), "t_high": t.get("t_high", self.cfg["t_high"])})
        return out

    def look(self, kind, grip, view, cam, shows, cycle_dir, step):
        """Photograph what the camera sees now (retakes while uncertain) -> step record with the decision."""
        manual = self.cfg["judge"] != "model"
        stages = [] if manual else self.stages(cam)
        rec = {"kind": kind, "grip": grip, "view": view, "cam": cam, "shows": shows, "shots": [], "decision": "uncertain"}
        cap = f"grip {grip} · {view} · shows {', '.join(shows) or '-'}"
        final = []; dec = "uncertain"; mean = None
        for k in range(1 + max(0, int(self.cfg["retries"]))):
            self.check(); time.sleep(0.25)                       # let the brick stop swinging
            self.state = "inspect"; self.say(f"{cap} · photo {k + 1}")
            t0 = time.perf_counter()
            crop, info = self._photo(kind, cam)
            p = cycle_dir / f"{step:02d}_g{grip}_{view}_{k}.png"; cv2.imwrite(str(p), crop)
            shot = {"file": p.name, "photo_ms": round((time.perf_counter() - t0) * 1000, 1), "motion": info.get("motion")}
            shown = crop
            if manual:
                ans = self._manual(crop, cap)
                dec = {"clean": "clean", "defect": "defect"}.get(ans, "uncertain")
                mean = {"clean": 0.0, "defect": 1.0}.get(ans, 0.5); shot.update({"judge": "manual", "score": mean})
            elif not info.get("masked"):                         # no depth mask: the background is in the picture
                dec = "uncertain"; shot["warning"] = f"no depth mask ({info.get('why')}): photo not trusted"
            else:
                shot["stages"] = []
                for si, st in enumerate(stages):
                    heat = p.with_name(p.stem + "_heat.jpg")
                    r = self.scorer.score(st["model"], p, st["device"], heatmap=heat)
                    if si == len(stages) - 1:
                        final.append(r["score"]); dec, mean = decide(final, st["t_low"], st["t_high"])
                    else:
                        dec, mean = decide([r["score"]], st["t_low"], st["t_high"])
                    shot["stages"].append({"stage": st["name"], "device": st["device"], "score": round(r["score"], 4),
                                           "infer_ms": r["ms"], "load_ms": r.get("load_ms"), "decision": dec})
                    if heat.exists(): shown = cv2.imread(str(heat))
                    if dec != "uncertain": break
                shot["score"] = round(mean, 4); shot["decided_by"] = shot["stages"][-1]["stage"]
            rec["shots"].append(shot); rec.update({"decision": dec, "score": None if mean is None else round(mean, 4)})
            self._remember(f"g{grip} {view}: {'' if mean is None else f'{mean:.2f} '}{dec}", shown)
            if dec != "uncertain": break
        return rec

    # ------------------------------------------------------------------ inspection of the whole surface
    def inspect(self, g_lock, g_open, cycle_dir):
        """The tape poses are frozen (and validated) once per cycle, before anything is set down: every re-grip of the
        cycle turns the same way, and an unusable / missing pose means "no re-grip" (fail-safe), not a stopped run."""
        C = self.C; tape_err = None
        if self.tape_ok():
            try: C.freeze_tape()
            except Exception as e: tape_err = str(e) or type(e).__name__
        try: return self._inspect(g_lock, g_open, cycle_dir, tape_err)
        finally: C.thaw_tape()

    def _inspect(self, g_lock, g_open, cycle_dir, tape_err=None):
        C = self.C; p = self.params()
        wp1 = C.wps("wp1")[0]["pos"]; wp2 = self.wp2(); poses = self.poses()
        can_tape = self.tape_ok() and not tape_err
        release = self.release_on() and can_tape; maxk = p["max_regrasps"] if can_tape else 0
        seen = cov.Seen(); steps = []; k = 0; regrips = []; retried = []
        self.cur.update({"steps": steps, "regions": seen.regions(), "grip": 1, "seen_pct": 0})

        def result(verdict, reason, **kw):
            return {"verdict": verdict, "reason": reason, "steps": steps, "regions": seen.regions(),
                    "seen_pct": round(100 * seen.fraction(), 1), "regrasps": k, "g_lock": g_lock,
                    "regrip_log": regrips, **kw}

        def regrip(turned):
            """collect.regrip (one retry that first lets go and lifts) -> (held, clamp, why-text of the failed attempts).
            Any other motion error after the release (the object lies on the tape) -> not held, with the error."""
            try:
                ok, g, whys = C.regrip(g_open, turned=turned)
                fail = C.why_text(whys)
            except RuntimeError as e:
                ok, g, fail = False, None, f"{C.sub or 'start'}: {e}"
            regrips.append(dict(C.regrip_info or {}))
            return ok, g, fail

        def why(rec):
            w = "defect seen" if rec["decision"] == "defect" else "uncertain after retakes (fail-safe)"
            return f"{w}: grip {rec['grip']}, {rec['view']} ({', '.join(rec['shows']) or '-'})"

        def clean(sides, held):
            seen.add(sides, k, held, p)
            self.cur.update({"regions": seen.regions(), "seen_pct": round(100 * seen.fraction(), 1)})

        while True:
            # ---- this grip: only the poses that still show something unseen
            at_wp2 = False
            for w in wp2:
                sides = self.sides(w["name"])
                if not seen.gain(sides, k, True, p): continue
                if not at_wp2:
                    C.to_wp2(wp1, wp2, g_lock); at_wp2 = True        # above WP2 -> straight down to the first pose
                    if w is not wp2[0]: C.goto(w["pos"], gripper=g_lock)
                else:
                    C.goto(w["pos"], gripper=g_lock)
                self.held()
                rec = self.look("inhand", k + 1, w["name"], self.cfg["cam"], seen.new_regions(sides, k, True, p),
                                cycle_dir, len(steps) + 1)
                steps.append(rec)
                if rec["decision"] != "clean":
                    C.up_wp2(wp1, wp2, g_lock)
                    return result("defect", why(rec))
                clean(sides, True)
            if at_wp2: C.up_wp2(wp1, wp2, g_lock)
            if seen.complete():
                done = f"{len(steps)} photos, {k} re-grip{'s' * (k != 1)}"
                if retried:                  # a failed attempt may have spun / tipped it: the coverage can't be trusted
                    return result("defect", f"defect plate: re-grip {', '.join(map(str, retried))} needed a retry, "
                                            f"orientation not verified (every photo was clean: {done})", orientation_unverified=True)
                return result("good", f"whole surface seen clean ({done})")
            if k >= maxk or not cov.potential(seen, poses, k + 1, maxk, release, p):
                miss = seen.missing()
                return result("defect", f"{', '.join(miss)} can't be seen by any pose: not verifiable (fail-safe)"
                                        + (f" - re-grip not possible: {tape_err}" if tape_err else ""), unverified=miss,
                              **({"orientation_unverified": True} if retried else {}))
            # ---- tape spot: set down, top photo of the released object, grip again turned 90 deg
            self.state = "regrasp"
            self.say(f"{', '.join(seen.missing())} not fully seen yet: re-grip on the tape spot, turned 90°")
            self.held()
            C.place_on_tape(g_lock, g_open)
            if release and seen.gain(["up"], 0, False, p):
                rec = self.look("release", k + 1, "tape_top", self.cfg["release_cam"],
                                seen.new_regions(["up"], 0, False, p), cycle_dir, len(steps) + 1)
                steps.append(rec)
                if rec["decision"] != "clean":                    # pick it up again as it lies, to carry it to the plate
                    self.state = "regrasp"
                    ok, g, fail = regrip(False)
                    if not ok: return result("aborted", why(rec) + f" - but it could not be picked up again ({fail})", on_tape=True)
                    g_lock = g
                    return result("defect", why(rec))
                clean(["up"], False)
            self.state = "regrasp"
            ok, g, fail = regrip(True)
            if not ok:
                twice = len((regrips[-1] if regrips else {}).get("failed") or []) >= 2
                return result("aborted", f"re-grip failed{' twice' if twice else ''} ({fail}): object left on the tape spot", on_tape=True)
            if (C.regrip_info or {}).get("attempts", 1) > 1: retried.append(k + 1)
            g_lock = g; k += 1; self.cur["grip"] = k + 1

    # ------------------------------------------------------------------ sorting
    def drop_in(self, bin_, g_lock, g_open):
        A, C = self.A, self.C
        tgt = C.slot(f"sort_{bin_}")
        if not tgt: raise RuntimeError(f"teach the {bin_}-plate pose (sort_{bin_}) first")
        p = tgt[0]["pos"]; wp1 = C.wps("wp1")[0]["pos"]
        ab = C.slot(f"sort_{bin_}_above")
        above = ab[0]["pos"] if ab else {**p, "shoulder_lift": wp1["shoulder_lift"], "elbow_flex": wp1["elbow_flex"],
                                          "wrist_flex": wp1["wrist_flex"]}
        self.state = f"sort_{bin_}"; self.say(f"sorting: object → {bin_.upper()} plate")
        C.goto(above, gripper=g_lock)
        self.held()                                              # never "sort" an empty gripper
        C.goto(p, C.cfg["speed"] * 0.5, gripper=g_lock)
        C.goto(dict(A.cmd), 1.0, gripper=g_open); time.sleep(0.25)       # release
        C.goto(above, C.cfg["speed"] * 0.6, gripper=g_open)

    # ------------------------------------------------------------------ one cycle / session
    def cycle(self, idx):
        A, C = self.A, self.C
        home = C.wps("home")[0]["pos"]; g_open = self.g_open()
        cycle_dir = self.RUNS / self.session / f"cycle{idx:03d}"; cycle_dir.mkdir(parents=True, exist_ok=True)
        self.cur = {"cycle": idx, "steps": [], "regions": {}, "grip": 1, "verdict": None, "seen_pct": 0}
        if A.mode != "hold": A.set_mode("hold", force=True)
        t0 = time.perf_counter(); failed = None
        try: g_lock = self.pick()
        except pol.PickFailed as e: failed = e                  # the autonomous pick gave up: ask for help below
        t1 = time.perf_counter()
        try:
            if failed: raise Lost()
            res = self.inspect(g_lock, g_open, cycle_dir)
            t2 = time.perf_counter()
            self.cur["verdict"] = res["verdict"]
            if res["verdict"] in BINS: self.drop_in(res["verdict"], res["g_lock"], g_open)
        except Lost:
            t2 = time.perf_counter()
            res = {"verdict": "aborted", "reason": "the object is no longer in the gripper (dropped)", "lost": True,
                   "steps": self.cur.get("steps", []), "regions": self.cur.get("regions", {}),
                   "seen_pct": self.cur.get("seen_pct", 0), "regrasps": self.cur.get("grip", 1) - 1}
            if failed: res.update({"reason": str(failed), "lost": False, "pick_failed": True})
            self.cur["verdict"] = "aborted"
        res.pop("g_lock", None)
        if not failed or failed.retry: C.goto(home, gripper=g_open)      # BLOCKED / camera gone: don't move
        t3 = time.perf_counter()
        infer = [s["infer_ms"] for st in res["steps"] for sh in st["shots"] for s in sh.get("stages", [])]
        rec = {"cycle": idx, "time": datetime.now().isoformat(timespec="seconds"), "brick": self.cfg["brick"],
               "judge": self.cfg["judge"], "device": self.cfg["device"], "cascade": self.cfg["cascade"],
               "pick": self.cfg["pick"], **res,
               "photos": sum(len(s["shots"]) for s in res["steps"]),
               "t_pick_s": round(t1 - t0, 2), "t_inspect_s": round(t2 - t1, 2), "t_sort_s": round(t3 - t2, 2),
               "t_cycle_s": round(t3 - t0, 2), "t_auto_s": round(t3 - t1, 2),
               "infer_ms_mean": round(float(np.mean(infer)), 2) if infer else None,
               "infer_ms_max": round(float(np.max(infer)), 2) if infer else None,
               "infer_ms_total": round(float(np.sum(infer)), 2) if infer else None}
        if self.cfg["pick"] == "act" and self.act_info:              # policy latency per inference step (efficiency)
            rec.update({"act": self.act_info, "act_infer_ms_p50": self.act_info.get("infer_ms_p50"),
                        "act_infer_ms_p95": self.act_info.get("infer_ms_p95"), "act_attempts": self.act_info.get("n_attempts")})
        with open(self.RUNS / self.session / "log.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
        self.totals[res["verdict"]] = self.totals.get(res["verdict"], 0) + 1
        self.history = ([{k: rec.get(k) for k in ("cycle", "verdict", "reason", "photos", "t_auto_s", "infer_ms_mean", "t_pick_s",
                                                  "act_infer_ms_p50")}]
                        + self.history)[:15]
        self.say(f"object {idx}: {res['verdict'].upper()} - {res['reason']} ({rec['t_auto_s']} s after the pick)",
                 res["verdict"] != "aborted")
        if res["verdict"] == "aborted":
            self.state = "help"; self.flags["next"].clear()
            where = "is on the tape spot" if res.get("on_tape") else "was not picked" if res.get("pick_failed") else "was dropped"
            self.say(f"the object {where} ({res['reason']}): put it back on the input plate, then NEXT", False)
            while not self.flags["next"].wait(0.1): self.check()
        return rec

    def run(self, n):
        self.active = True; self.A.running = True; self.C.active = True
        for e in (self.stop_flag, self.A.stop, self.C.stop_flag): e.clear()
        self.session = datetime.now().strftime("sort_%Y%m%d_%H%M%S")
        (self.RUNS / self.session).mkdir(parents=True, exist_ok=True)
        (self.RUNS / self.session / "settings.json").write_text(json.dumps({**self.cfg, "plan": self.plan()}, indent=1))
        self.totals = {"good": 0, "defect": 0, "aborted": 0}
        try:
            if self.cfg["pick"] == "act": self.get_policy()           # load / compile before the first cycle
            for i in range(1, n + 1):
                self.cycle(i)
                time.sleep(0.5)
            self.state = "done"
            self.say(f"✓ DONE: {self.totals['good']} good, {self.totals['defect']} defect, {self.totals['aborted']} aborted · log: runs/{self.session}")
        except Stopped:
            self.state = "stopped"; self.say("STOPPED: arm holds its position", False)
        except Exception as e:
            self.state = "error"; self.say(f"ERROR: {e} (arm holds its position)", False)
        finally:
            try: self.C.hold_here()
            except Exception: pass
            self.judge["img"] = None
            self.active = False; self.A.running = False; self.C.active = False

    # ------------------------------------------------------------------ threshold calibration (same path as runtime)
    def _episode_of(self):
        """crop stem -> (run, cycle): which brick-handling a photo belongs to (from training_images/meta.jsonl)."""
        m = {}
        try:
            for line in open(self.DATA / "meta.jsonl"):
                j = json.loads(line); m[Path(j["file"]).stem] = (j.get("run"), j.get("cycle"))
        except Exception: pass
        return m

    def calibrate(self, brick, cam, variant=""):
        self.calib = {"running": True, "result": None, "progress": "starting"}
        try:
            model, root = self.scorer.find(cam, brick, variant)
            device = self.cfg["fast_device"] if variant else self.cfg["device"]
            files = {}
            for lab in ("good", "defect"):
                fs = sorted((self.DATA / cam / "inhand" / lab).glob("*_crop.png"))
                files[lab] = [f for f in fs if not brick or (len(f.stem.split("_")) > 4 and f.stem.split("_")[3] == brick)]
            held_out = None
            try: held_out = set(json.loads((root / "holdout.json").read_text()))
            except Exception: pass
            if held_out: files["good"] = [f for f in files["good"] if f.name in held_out]
            if not files["good"]: raise RuntimeError(f"no good crops for '{brick or 'all'}' on {cam}")
            if not files["defect"]: raise RuntimeError(f"no defect crops for '{brick or 'all'}' on {cam}: collect defect episodes first")
            sc = {"good": [], "defect": []}
            for lab in ("good", "defect"):
                for i, f in enumerate(files[lab]):
                    self.calib["progress"] = f"{lab} {i + 1}/{len(files[lab])}"
                    sc[lab].append(self.scorer.score(model, f, device)["score"])
            ep = self._episode_of(); groups = {}
            for f, s in zip(files["defect"], sc["defect"]):
                groups.setdefault(ep.get(f.stem[:-len("_crop")], f.stem), []).append(s)
            res = thresholds_from(sc["good"], [max(v) for v in groups.values()])
            res.update({"brick": brick or "all", "cam": cam, "model": model, "device": device, "n_good": len(sc["good"]),
                        "good_held_out": bool(held_out),
                        "note": None if held_out else "good crops were also used for training: their scores are optimistically "
                                                      "low (train with train_anomaly.py to get a held-out set)"})
            self.scorer.save_thresholds(root, {"t_low": res["t_low"], "t_high": res["t_high"],
                                               "calibrated": datetime.now().isoformat(timespec="seconds"),
                                               **{k: res[k] for k in ("n_good", "defect_bricks", "auroc", "overlap")}})
            self.calib = {"running": False, "result": res}
        except Exception as e:
            self.calib = {"running": False, "result": {"error": str(e)}}

    def status(self):
        try: models = self.scorer.models()
        except Exception: models = []
        try: wp2 = [w["name"] for w in self.wp2()]
        except Exception: wp2 = []
        return {"state": self.state, "msg": self.msg, "ok": self.ok, "active": self.active, "cfg": self.cfg,
                "plan": self.plan(), "wp2": wp2, "tape": self.tape_ok(), "release_on": self.release_on(),
                "release_roi": {c: prep.release_roi(c) for c in ("wrist", "top", "front")}, "models": models,
                "bins": {b: bool(self.C.slot(f"sort_{b}")) for b in BINS},
                "cur": self.cur, "totals": self.totals, "history": self.history, "session": self.session,
                "judge": self.judge["img"] is not None, "judge_caption": self.judge["caption"], "judge_n": self.judge["n"],
                "photos": [c for c, _ in self.photos], "calib": self.calib, "region_names": cov.REGION_NAMES,
                "live": self.S.live,
                "act": {"loaded": self._pol[1].label if self._pol else None, "live": self.act.live if self.act else {},
                        "last": {k: v for k, v in (self.act_info or {}).items() if k != "attempts"},
                        "roi_warning": self.roi_warning() if self.cfg["pick"] == "act" else None}}


def make_router(S, A, ctx, C, DATA, ROOT):
    T = Sorter(S, A, ctx, C, DATA, ROOT)
    r = APIRouter()

    @r.get("/sort/status")
    def st(): return T.status()

    @r.post("/sort/start")
    def start(n: int = 1):
        def bad(m): T.say(m, False); return T.status()
        if T.active or C.active or A.running: return bad("busy: STOP the running job first")
        if A.f is None: return bad("connect the arm first (section 1)")
        try:
            for k in ("home", "wp1", "wp2"): C.wps(k)
        except Exception as e: return bad(str(e))
        for b in BINS:
            if not C.slot(f"sort_{b}"): return bad(f"teach the {b}-plate pose first")
        if T.tape_ok() and C.regrip_check(): return bad(C.regrip_check())
        pl = T.plan()
        if not pl["complete"] and not T.cfg["allow_incomplete"]:
            return bad(f"with these poses {', '.join(pl['missing'])} can never be fully seen ({pl['seen_pct']} % of the surface "
                       "is covered): add / tick WP2 poses, teach the tape spot, enable the top photo "
                       "(or tick 'allow incomplete' to test - such objects always go to DEFECT)")
        if T.cfg["pick"] == "act":
            try:
                d = pol.describe(pol.resolve(T.cfg["act_model"] or "-", T.cfg["act_backend"])[0]) or {}
                miss = [c for c in d.get("cams", []) if c not in S.live]
                if miss: return bad(f"the ACT policy needs camera {', '.join(miss)}, which is not live")
            except Exception as e: return bad(f"pick = ACT: {e}" if T.cfg["act_model"] else "pick = ACT: choose a policy export first")
        if T.cfg["judge"] == "model":
            try:
                T.scorer.find(T.cfg["cam"], T.cfg["brick"])
                if T.release_on() and any(s.get("release_photo") for s in pl["steps"]):
                    T.scorer.find(T.cfg["release_cam"], T.cfg["brick"])
            except Exception as e: return bad(str(e))
        threading.Thread(target=T.run, args=(max(1, n),), daemon=True).start()
        time.sleep(0.1); return T.status()

    @r.get("/sort/policies")
    def policies():
        """Policy exports for pick = ACT: Studio's models dir (~/.local/share/physicalai/models/<id>/exports/<backend>)
        and ROOT/models, plus the OpenVINO devices this PC has."""
        try:
            import openvino as ov
            devs = ov.Core().available_devices
        except Exception: devs = []
        try: ex = pol.find_exports([ROOT / "models"])
        except Exception as e: ex = []; T.say(f"listing policy exports failed: {e}", False)
        return {"exports": ex, "devices": devs, "studio_dir": str(pol.STUDIO_MODELS), "selected": T.cfg["act_model"]}

    @r.post("/sort/stop")
    def stop():
        for e in (T.stop_flag, A.stop, C.stop_flag): e.set()
        T.flags["next"].set(); T.judge["evt"].set()
        time.sleep(0.06)
        if A.f:
            with A.io: A.mode = "hold"; A.goal = dict(A.cmd)
        T.say("STOPPED: arm holds its position", False); return T.status()

    @r.post("/sort/flag/{flag}")
    def flag(flag: str):
        if flag == "grasped": T.flags["grasped"] = True
        elif flag == "next": T.flags["next"].set()
        return T.status()

    @r.post("/sort/judge/{answer}")
    def judge(answer: str):
        if answer in ("clean", "defect", "unsure") and T.judge["img"] is not None:
            T.judge["answer"] = answer; T.judge["evt"].set()
        return T.status()

    @r.get("/sort/judge.jpg")
    def judge_img():
        if T.judge["img"] is None: return Response(status_code=204)
        return Response(T.judge["img"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @r.get("/sort/photo/{i}")
    def photo(i: int):
        if i >= len(T.photos): return Response(status_code=204)
        return Response(T.photos[i][1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @r.get("/sort/release_view.jpg")
    def release_view(cam: str = ""):
        """Live: exactly the crop the top-photo model gets (set the crop so the released brick fills it)."""
        cam = cam or T.cfg["release_cam"]
        f = S.frames.get(cam)
        if f is None: return Response(status_code=204)
        crop, _ = prep.release_crop(f, cam)
        sc = 300 / crop.shape[0]
        img = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        return Response(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @r.post("/sort/release_roi")
    def release_roi(x0: float, y0: float, x1: float, y1: float, cam: str = ""):
        """Crop for the top photo, in percent of the frame (0-100)."""
        try: prep.set_release_roi(cam or T.cfg["release_cam"], [x0 / 100, y0 / 100, x1 / 100, y1 / 100])
        except Exception as e: T.say(f"crop not changed: {e}", False)
        return T.status()

    @r.post("/sort/cfg")
    def cfg(pick: str = "", judge: str = "", brick: str = None, device: str = "", t_low: float = -1, t_high: float = -1,
            retries: int = -1, allow_incomplete: int = -1, release: int = -1, release_cam: str = "", cascade: int = -1,
            fast_device: str = "", view_deg: float = -1, finger_deg: float = -1, max_regrasps: int = -1,
            act_model: str = None, act_device: str = "", act_backend: str = "", act_mode: str = "", act_max_s: float = -1,
            act_retries: int = -1, act_speed: float = -1):
        if pick in ("handfeed", "teleop", "act"): T.cfg["pick"] = pick
        if act_model is not None: T.cfg["act_model"] = act_model.strip()
        if act_device in pol.DEVICES: T.cfg["act_device"] = act_device
        if act_backend in ("auto",) + pol.BACKENDS: T.cfg["act_backend"] = act_backend
        if act_mode in pol.MODES: T.cfg["act_mode"] = act_mode
        if 2 <= act_max_s <= 120: T.cfg["act_max_s"] = act_max_s
        if 0 <= act_retries <= 5: T.cfg["act_retries"] = act_retries
        if 0.1 <= act_speed <= 1: T.cfg["act_speed"] = act_speed
        if T._pol and T._pol[0] != (T.cfg["act_model"], T.cfg["act_device"], T.cfg["act_backend"]) and not T.active:
            T.drop_policy()
        if judge in ("manual", "model"): T.cfg["judge"] = judge
        if brick is not None: T.cfg["brick"] = brick.strip()
        if device in ("CPU", "GPU", "NPU", "AUTO"): T.cfg["device"] = device
        if fast_device in ("CPU", "GPU", "NPU", "AUTO"): T.cfg["fast_device"] = fast_device
        if 0 <= t_low <= 1: T.cfg["t_low"] = t_low
        if 0 <= t_high <= 1: T.cfg["t_high"] = t_high
        if T.cfg["t_high"] < T.cfg["t_low"]: T.cfg["t_high"] = T.cfg["t_low"]
        if 0 <= retries <= 5: T.cfg["retries"] = retries
        if allow_incomplete in (0, 1): T.cfg["allow_incomplete"] = bool(allow_incomplete)
        if release in (0, 1): T.cfg["release"] = bool(release)
        if release_cam in ("wrist", "top", "front"): T.cfg["release_cam"] = release_cam
        if cascade in (0, 1): T.cfg["cascade"] = bool(cascade)
        if 30 <= view_deg <= 80: T.cfg["view_deg"] = view_deg
        if 10 <= finger_deg <= 60: T.cfg["finger_deg"] = finger_deg
        if 0 <= max_regrasps <= 4: T.cfg["max_regrasps"] = max_regrasps
        T.save_cfg(); return T.status()

    @r.post("/sort/faces")
    def faces(view: str, sides: str = ""):
        """Sides (gripper frame) the front camera sees at WP2 pose `view`: comma list of front,back,up,down."""
        T.cfg["faces"][view] = [s for s in sides.split(",") if s in cov.SIDES]
        T.save_cfg(); return T.status()

    @r.post("/sort/calibrate")
    def calibrate(brick: str = "", cam: str = "", fast: bool = False):
        if T.calib.get("running"): return T.status()
        threading.Thread(target=T.calibrate, args=(brick.strip(), cam or T.cfg["cam"], T.cfg["fast_variant"] if fast else ""),
                         daemon=True).start()
        time.sleep(0.1); return T.status()

    r.sorter = T
    return r
