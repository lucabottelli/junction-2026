"""Shape-agnostic "have we seen the whole surface?" bookkeeping for the held object (pure numpy, no hardware).

Instead of naming the faces of one particular shape (cube: 6 faces), the surface is represented by the DIRECTIONS
its outward normals can point in: ~1500 points spread evenly over a sphere fixed to the object (Gaussian sphere).
A photo taken from direction v sees every surface patch whose normal is within `view_deg` of v, except what the
fingers cover. The object counts as fully inspected when every direction has been seen in a clean photo.
This is the classic visibility-matrix / set-cover formulation of inspection view planning, and it holds for a
cube, a long box, a triangular prism, a cylinder or any other convex shape without knowing the shape.

Frames (gripper frame, object picked from above):
    +z  "up"    toward the gripper's palm (the face that pointed up on the table)
    -z  "down"  fingertip side (the face that lay on the table)
    +x  "front" free side facing the camera in the first WP2 pose      -x "back" the opposite free side
    +-y         the finger axis: what is pressed between the fingers is hidden while held
Each photo pose is annotated with the sides it shows (front/back/up/down); several ticks = a diagonal view
(direction = normalised sum). A re-grip on the tape spot turns the gripper 90 deg about z relative to the object,
so the old finger sides become front/back. The released brick on the tape spot can be photographed from above
(wrist camera) = view +z with no finger in the way.

Chirality: whether "wrist roll +90" turns the object's +y or -y to the front depends on servo direction. Both
hypotheses are tracked and a direction only counts as seen when it is seen under BOTH (fail-safe).
"""
import numpy as np

N_POINTS = 1500
AXES = {"front": (1, 0, 0), "back": (-1, 0, 0), "up": (0, 0, 1), "down": (0, 0, -1)}
SIDES = list(AXES)
# display regions of the sphere, named in the object's frame as it was first picked
REGIONS = {"U": (0, 0, 1), "D": (0, 0, -1), "S1": (1, 0, 0), "S2": (-1, 0, 0), "F1": (0, 1, 0), "F2": (0, -1, 0)}
REGION_NAMES = {"U": "top (up when picked)", "D": "bottom (down when picked)", "S1": "free side 1", "S2": "free side 2",
                "F1": "finger side 1", "F2": "finger side 2"}
DEFAULTS = {"view_deg": 65.0, "finger_deg": 35.0, "margin_deg": 5.0, "max_regrasps": 2}
SIGNS = (1, -1)


def sphere(n=N_POINTS):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = i * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


PTS = sphere()
_RK = list(REGIONS)
REGION_OF = np.argmax(PTS @ np.array([REGIONS[k] for k in _RK], float).T, axis=1)


def view_dir(sides):
    """Ticked sides -> unit view direction in the gripper frame (None if nothing / opposite sides ticked)."""
    v = np.zeros(3)
    for s in sides or []:
        if s in AXES: v += AXES[s]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def rotz(k, sign):
    a = sign * k * np.pi / 2
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def visible(sides, k=0, sign=1, held=True, p=DEFAULTS):
    """Directions (bool mask over PTS, object frame) a photo shows. k = number of 90 deg re-grips so far."""
    v = view_dir(sides)
    if v is None: return np.zeros(len(PTS), bool)
    R = rotz(k, sign)
    m = PTS @ (R @ v) >= np.cos(np.radians(p["view_deg"] - p["margin_deg"]))
    if held:
        m &= np.abs(PTS @ (R @ np.array([0.0, 1.0, 0.0]))) < np.cos(np.radians(p["finger_deg"] + p["margin_deg"]))
    return m


class Seen:
    """What has been confirmed clean so far (one mask per chirality hypothesis)."""
    def __init__(self):
        self.m = {s: np.zeros(len(PTS), bool) for s in SIGNS}

    def copy(self):
        c = Seen(); c.m = {s: v.copy() for s, v in self.m.items()}; return c

    def gain(self, sides, k, held=True, p=DEFAULTS):
        """Number of not-yet-seen directions this photo would add (summed over both hypotheses)."""
        return int(sum((visible(sides, k, s, held, p) & ~self.m[s]).sum() for s in SIGNS))

    def add(self, sides, k, held=True, p=DEFAULTS):
        for s in SIGNS: self.m[s] |= visible(sides, k, s, held, p)

    def both(self): return self.m[1] & self.m[-1]
    def complete(self): return bool(self.both().all())
    def fraction(self): return float(self.both().mean())

    def regions(self):
        """Percent of each display region confirmed clean."""
        b = self.both()
        return {r: int(round(100 * b[REGION_OF == i].mean())) for i, r in enumerate(_RK)}

    def new_regions(self, sides, k, held=True, p=DEFAULTS, min_share=0.15):
        """Display regions this photo contributes to noticeably (for captions/logs)."""
        g = np.zeros(len(PTS), bool)
        for s in SIGNS: g |= visible(sides, k, s, held, p) & ~self.m[s]
        return [r for i, r in enumerate(_RK) if g[REGION_OF == i].mean() >= min_share]

    def missing(self):
        b = self.both()
        return [r for i, r in enumerate(_RK) if not b[REGION_OF == i].all()]


def potential(seen, poses, k_from, k_to, release, p):
    """Could grips k_from..k_to (and the release photo) still add anything?"""
    if release and seen.gain(["up"], 0, held=False, p=p): return True
    return any(seen.gain(sides, k, True, p) for k in range(k_from, k_to + 1) for _, sides in poses)


def plan(poses, tape=True, release=False, p=DEFAULTS):
    """Dry run assuming every photo is clean -> the worst-case (all-good brick) sequence.

    poses: [(name, sides)] in taught order. Returns {"steps", "complete", "regions", "missing", "regrasps", "photos"}.
    """
    seen = Seen(); steps = []; k = 0
    maxk = int(p.get("max_regrasps", 2)) if tape else 0
    while True:
        for name, sides in poses:
            if seen.gain(sides, k, True, p):
                steps.append({"grip": k + 1, "view": name, "shows": seen.new_regions(sides, k, True, p)})
                seen.add(sides, k, True, p)
        if seen.complete() or k >= maxk or not potential(seen, poses, k + 1, maxk, release, p): break
        st = {"regrasp": k + 1}
        if release and seen.gain(["up"], 0, held=False, p=p):
            st["release_photo"] = True; seen.add(["up"], 0, held=False, p=p)
        steps.append(st); k += 1
    return {"steps": steps, "complete": seen.complete(), "regions": seen.regions(), "missing": seen.missing(),
            "regrasps": k, "photos": sum(1 for s in steps if "view" in s or s.get("release_photo")),
            "seen_pct": round(100 * seen.fraction(), 1)}
