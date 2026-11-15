"""Offline check of the depth-aided brick outline (prep.brick_mask(..., depth=...)) and foreign_share.
Reads training_images/front/inhand/{test,good} (read only); writes only a new folder under runs/.
Usage: python eval_outline_depth.py [--orig path/to/old_prep.py]  (--orig: also check no-depth masks are identical)"""
import sys, time, glob, importlib.util
from pathlib import Path
import cv2, numpy as np
import prep

ROOT = Path(__file__).resolve().parent.parent
TI = ROOT / "training_images/front/inhand"
FAIL = TI / "test/20261115_054706_551_blue_wp2-2.png"


def load(p):
    img = cv2.imread(str(p)); dp = Path(str(p)[:-4] + "_depth.png")
    d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) if dp.exists() else None
    return img, d


def both(img, d, brick):
    m0, i0 = prep.brick_mask(img, brick)
    m1, i1 = prep.brick_mask(img, brick, depth=d)
    f0 = prep.foreign_share(img, brick, m0) if m0 is not None else None
    f1 = prep.foreign_share(img, brick, m1) if m1 is not None else None
    return m0, i0, f0, m1, i1, f1


def draw(img, m, title):
    v = img.copy(); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(v, cs, -1, (0, 255, 0), 4)
    cv2.putText(v, title, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
    return v


def main():
    orig = None
    if "--orig" in sys.argv:
        sp = importlib.util.spec_from_file_location("prep_orig", sys.argv[sys.argv.index("--orig") + 1])
        orig = importlib.util.module_from_spec(sp); sp.loader.exec_module(orig)
    print("== test: blue wp2 (area px, foreign_share[0]) colour-only -> depth-aided")
    for p in sorted(glob.glob(str(TI / "test/*_blue_wp2-?.png"))):
        img, d = load(p)
        m0, i0, f0, m1, i1, f1 = both(img, d, "blue")
        if m0 is None: print(Path(p).name, "no mask:", i0); continue
        print(f"{Path(p).name}  area {i0['area_px']:7d} -> {i1['area_px']:7d}  fs {f0[0]:.3f} -> {f1[0]:.3f}"
              f"  depth_used={i1['depth_used']} zb={i1.get('block_z_mm')}")
        if Path(p) == FAIL:
            out = ROOT / "runs" / time.strftime("outline_depth_%Y%m%d_%H%M%S"); out.mkdir(parents=True)
            sbs = np.hstack([draw(img, m0, f"colour-only fs={f0[0]:.3f}"), draw(img, m1, f"depth-aided fs={f1[0]:.3f}")])
            cv2.imwrite(str(out / (FAIL.stem + "_sbs.jpg")), sbs, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print("  side-by-side:", out / (FAIL.stem + "_sbs.jpg"))
    print("\n== good (clean) photos, depth-aided masks")
    for brick in ("blue", "green"):
        rows = []; same = 0; ndiff = 0
        for p in sorted(glob.glob(str(TI / f"good/*_{brick}_wp2-?.png"))):
            img, d = load(p)
            if d is None: continue
            m0, i0, f0, m1, i1, f1 = both(img, d, brick)
            if m0 is None or m1 is None: print("  no mask", Path(p).name, i0 if m0 is None else i1); continue
            if orig is not None:
                mo, _ = orig.brick_mask(img, brick)
                if np.array_equal(mo, m0): same += 1
                else: ndiff += 1
            rows.append((Path(p).name, f1[0], i1["area_px"] / i0["area_px"], i1["depth_used"], f0[0]))
        fs = np.array([r[1] for r in rows]); ra = np.array([r[2] for r in rows]); fs0 = np.array([r[4] for r in rows])
        print(f"{brick}: n={len(rows)} depth_used={sum(r[3] for r in rows)}  fs max={fs.max():.3f} p99={np.percentile(fs, 99):.3f}"
              f" >=0.06: {(fs >= prep.FOREIGN_MAX).sum()} >=0.05: {(fs >= 0.05).sum()}  (colour-only mask: max={fs0.max():.3f})")
        for r in sorted(rows, key=lambda r: -r[1])[:3]: print(f"   worst fs {r[1]:.3f} (colour-only {r[4]:.3f})  {r[0]}")
        print(f"   area ratio depth/colour: median={np.median(ra):.3f} max={ra.max():.3f}  >1.05: {(ra > 1.05).sum()}  <0.95: {(ra < 0.95).sum()}")
        for r in sorted(rows, key=lambda r: -r[2])[:5]: print(f"   ratio {r[2]:.3f}  {r[0]}")
        if orig is not None: print(f"   no-depth mask identical to old prep.py: {same}/{same + ndiff}")


if __name__ == "__main__":
    main()
