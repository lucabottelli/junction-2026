"""Grasp tuner: learns HOW to grasp from the robot's own success / failure (a Thompson-sampling bandit, the simplest
form of reinforcement learning). Used by the self-recording pick loop (vision pick -> carry -> "still held?" check):

    from grasp_tuner import Tuner
    t = Tuner.load()                       # state in anomaly_capture/grasp_tuner.json (survives restarts)
    p = t.suggest("pink")                  # {'dz_mm': 0, 'dxy_mm': 0, 'yaw_off_deg': 0, 'open_frac': 1.0,
                                           #  'approach_speed': 1.0, '_cells': {...}}  (neutral until --enable)
    ... grasp with p ...
    t.update(p, success=True)              # reward 1 = brick still held at WP1, 0 = missed / dropped

Each grasp setting is learned separately (a factorized bandit: one Beta(successes, failures) per option and setting,
per brick colour, with a shared all-colours prior). That learns from a few dozen picks instead of the hundreds a joint
grid would need. Thompson sampling: draw a success rate for every option from its Beta, take the best draw. Options
that keep failing are tried less and less but never ruled out completely, so the tuner keeps adapting if the
station changes (a camera bumped, lighting).

The neutral option (no change to the scripted grasp) starts with a small head start, so a fresh tuner behaves like
the plain scripted pick and only moves away from it once the data says so.

    python grasp_tuner.py --enable         start learning, once the plain scripted pick works on the real arm
    python grasp_tuner.py --stats          success rate per setting and option (what it has learned)
    python grasp_tuner.py --reset          forget everything
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "grasp_tuner.json"

# setting -> options. The first option of each list is the neutral one (= the plain scripted grasp).
# Meanings as vpick.py applies them (intel-2f, 2026-11-14 21:20):
SPACE: dict[str, list] = {
    "dz_mm": [0, -3, -6, 3],                     # grasp height offset (- = deeper)
    "dxy_mm": [0, 3, -3, 6, -6],                 # shift along the jaw axis (+ = toward the moving jaw)
    "yaw_off_deg": [0, -8, 8],                   # jaw angle offset
    "open_frac": [1.0, 1.5, 0.7],                # x the opening margin around the brick (1.0 = scripted)
    "approach_speed": [1.0, 0.7],                # x the descent speed (1.0 = scripted)
}
NEUTRAL_PRIOR = (2.0, 1.0)      # Beta(a, b) of the neutral option: a small head start
OTHER_PRIOR = (1.0, 1.0)        # uniform for the others
SHARED_WEIGHT = 0.5             # how much the all-colours counts add to a colour's own counts


class Tuner:
    def __init__(self, path: Path = STATE, space: dict | None = None, seed: int | None = None):
        self.path, self.space = Path(path), space or SPACE
        self.rng = random.Random(seed)
        self.counts: dict = {}      # counts[context][setting][option index] = [successes, failures]
        self.history: list = []     # one record per update
        self.enabled = False        # False: suggest() always returns the neutral grasp (results are still recorded)
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path = STATE, **kw) -> "Tuner":
        t = cls(path, **kw)
        if t.path.exists():
            d = json.loads(t.path.read_text())
            if d.get("space") == {k: v for k, v in t.space.items()}:
                t.counts, t.history = d.get("counts", {}), d.get("history", [])
            t.enabled = bool(d.get("enabled", False))
        return t

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"space": self.space, "enabled": self.enabled, "counts": self.counts, "history": self.history[-5000:]}, indent=1))
        tmp.replace(self.path)

    # ------------------------------------------------------------------ bandit
    def _cnt(self, ctx: str, setting: str) -> list:
        c = self.counts.setdefault(ctx, {}).setdefault(setting, [[0, 0] for _ in self.space[setting]])
        if len(c) != len(self.space[setting]): c[:] = [[0, 0] for _ in self.space[setting]]
        return c

    def _beta(self, ctx: str, setting: str, i: int) -> tuple[float, float]:
        a0, b0 = NEUTRAL_PRIOR if i == 0 else OTHER_PRIOR
        s, f = self._cnt(ctx, setting)[i]
        if ctx != "*":
            ss, ff = self._cnt("*", setting)[i]; s, f = s + SHARED_WEIGHT * ss, f + SHARED_WEIGHT * ff
        return a0 + s, b0 + f

    def suggest(self, colour: str = "", explore: bool = True) -> dict:
        """One grasp setting per parameter. explore=False: the best mean (no sampling), for a final run."""
        ctx = colour or "*"; out, cells = {}, {}
        if not self.enabled:   # learning switched off: the plain scripted grasp, tagged so update() still counts it
            out = {k: v[0] for k, v in self.space.items()}
            out["_cells"], out["_colour"] = {k: 0 for k in self.space}, colour
            return out
        with self.lock:
            for setting, opts in self.space.items():
                scores = []
                for i in range(len(opts)):
                    a, b = self._beta(ctx, setting, i)
                    scores.append(self.rng.betavariate(a, b) if explore else a / (a + b))
                i = max(range(len(opts)), key=scores.__getitem__)
                out[setting], cells[setting] = opts[i], i
        out["_cells"], out["_colour"] = cells, colour
        return out

    def update(self, params: dict, success: bool, info: dict | None = None) -> None:
        cells, colour = params.get("_cells") or {}, params.get("_colour", "")
        with self.lock:
            for ctx in {colour or "*", "*"}:
                for setting, i in cells.items():
                    if setting in self.space: self._cnt(ctx, setting)[i][0 if success else 1] += 1
            self.history.append({"t": round(time.time(), 1), "colour": colour, "success": bool(success),
                                 "params": {k: v for k, v in params.items() if not k.startswith("_")}, **(info or {})})
            self.save()

    # ------------------------------------------------------------------ reporting
    def stats(self, colour: str = "*") -> dict:
        """{setting: [(option, successes, tries, posterior mean)]} for a colour ("*" = all colours)."""
        out = {}
        for setting, opts in self.space.items():
            rows = []
            for i, o in enumerate(opts):
                s, f = self._cnt(colour, setting)[i]; a, b = self._beta(colour, setting, i)
                rows.append((o, s, s + f, round(a / (a + b), 3)))
            out[setting] = rows
        return out

    def summary(self, last: int = 50) -> dict:
        h = self.history[-last:]
        return {"enabled": self.enabled, "tries": len(self.history), "recent_tries": len(h),
                "recent_success_rate": round(sum(r["success"] for r in h) / len(h), 3) if h else None,
                "best": {k: v for k, v in self.suggest(explore=False).items() if not k.startswith("_")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true"); ap.add_argument("--reset", action="store_true")
    ap.add_argument("--colour", default="*")
    ap.add_argument("--enable", action="store_true", help="start learning (suggest() explores)")
    ap.add_argument("--disable", action="store_true", help="back to the neutral scripted grasp")
    a = ap.parse_args()
    if a.reset:
        STATE.unlink(missing_ok=True); print("tuner reset"); return
    t = Tuner.load()
    if a.enable or a.disable: t.enabled = a.enable; t.save(); print("learning", "ON" if t.enabled else "OFF")
    print(json.dumps({"enabled": t.enabled, **t.summary()}, indent=1))
    if a.stats:
        for setting, rows in t.stats(a.colour).items():
            print(f"{setting}:")
            for o, s, n, m in rows: print(f"   {str(o):10s} {s:3d}/{n:<3d} success   posterior mean {m:.2f}")


if __name__ == "__main__":
    main()
