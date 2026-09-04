#!/usr/bin/env python3
"""Does the sweep recover a real surface? Checked against the ground, not against opinion.

The road under a car-mounted camera is the one thing whose geometry is known without any other
sensor: it is flat, and it is directly below the camera. If the swept depths unproject to a plane
of small residual at a plausible height below the camera, the sweep is working. If they do not,
nothing downstream is worth running.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curbmeasure.geometry import Camera, camera_from_image  # noqa: E402
from curbmeasure.mapillary import fetch_pixels, images_in_box  # noqa: E402
from curbmeasure.stereo import SweepConfig, pick_device, sweep, unproject  # noqa: E402


def undistort(gray, camera: Camera):
    K = np.array([[camera.focal_px, 0, camera.principal[0]],
                  [0, camera.focal_px, camera.principal[1]], [0, 0, 1]])
    return cv2.undistort(gray, K, np.array([camera.k1, camera.k2, 0.0, 0.0, 0.0]))


def rescale(c: Camera, gray, width: int) -> Camera:
    r = gray.shape[1] / width
    return Camera(c.image_id, c.centre, c.rotation, c.focal_px * r,
                  (c.principal[0] * r, c.principal[1] * r), c.k1, c.k2, gray.shape[1], gray.shape[0])


def main() -> int:
    imgs = images_in_box(-122.4475, 37.7860, -122.4405, 37.7889)
    seqs: dict[str, list] = {}
    for i in imgs:
        if i.camera_type == "perspective" and i.sequence:
            seqs.setdefault(i.sequence, []).append(i)
    grp = max(seqs.values(), key=len)
    grp.sort(key=lambda i: i.captured_at or 0)
    picks = grp[6:11]
    if len({p.merge_cc for p in picks}) != 1:
        picks = grp[0:5]
    work = ROOT / "work"; work.mkdir(exist_ok=True)
    paths = [work / f"{p.id}.jpg" for p in picks]
    if not all(fetch_pixels(p, q) for p, q in zip(picks, paths)):
        print("fetch failed"); return 1

    lat0, lon0 = picks[0].lat, picks[0].lon
    grays, cams = [], []
    for pick, path in zip(picks, paths):
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        scale = 1024 / g.shape[1]
        g = cv2.resize(g, (1024, int(g.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        c = rescale(camera_from_image(pick, lat0, lon0), g, pick.width)
        grays.append(undistort(g, c)); cams.append(c)

    ref, ref_img = cams[2], grays[2]
    others = [(cams[i], grays[i]) for i in (0, 1, 3, 4)]
    print(f"reference {ref.image_id}  {ref_img.shape[1]}x{ref_img.shape[0]}  device {pick_device()}")
    print(f"baselines: {[round(float(np.linalg.norm(c.centre-ref.centre)),2) for c,_ in others]} m")

    t0 = time.time()
    depth, score, top = sweep(ref, ref_img, others, SweepConfig())
    took = time.time() - t0
    got = int((depth > 0).sum())
    print(f"swept {depth.size:,} pixels in {took:.1f}s  ({100*got/depth.size:.0f}% accepted)")

    # Where in the frame is depth actually being accepted, and at what range? For a forward
    # camera the bottom rows are the road a few metres ahead; if those rows are empty or far,
    # the sweep is matching structure rather than ground.
    print("\n  row band (top->bottom of swept region)   accepted   median depth")
    bands = 6
    for k in range(bands):
        lo = k * depth.shape[0] // bands; hi = (k + 1) * depth.shape[0] // bands
        chunk = depth[lo:hi]
        ok = chunk > 0
        med = float(np.median(chunk[ok])) if ok.any() else float("nan")
        print(f"    rows {top+lo:4d}-{top+hi:4d}   {100*ok.mean():5.1f}%      {med:6.2f} m")
    pts = unproject(ref, depth, top)
    print(f"{len(pts):,} world points")
    if len(pts) < 500:
        print("too few to judge"); return 1
    rel = pts - ref.centre
    down = -rel[:, 2]
    print(f"  height below camera: median {np.median(down):.2f} m  p10 {np.percentile(down,10):.2f}  p90 {np.percentile(down,90):.2f}")
    ground = pts[(down > 0.5) & (down < 4.0)]
    print(f"  plausible-ground points: {len(ground):,}")
    if len(ground) > 300:
        z = ground[:, 2]; A = np.column_stack((ground[:, 0], ground[:, 1], np.ones(len(ground))))
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = z - A @ coef
        print(f"  plane fit residual: rms {resid.std()*1000:.0f} mm   slope {np.hypot(*coef[:2])*100:.1f}%")
        print(f"  camera height above that plane: {ref.centre[2] - (coef[0]*ref.centre[0]+coef[1]*ref.centre[1]+coef[2]):.2f} m")
    for q in paths:
        q.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
