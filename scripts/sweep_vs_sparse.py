#!/usr/bin/env python3
"""Sweep depth against triangulated depth, on the same pixels of the same image.

Both use the same cameras, so this isolates the sweep from the poses. Sparse triangulation was
already checked by reprojection into a held-out third view and agrees to a few pixels at 10-40 m.
If the two disagree here, the fault is in the sweep.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curbmeasure.geometry import Camera, camera_from_image, triangulate  # noqa: E402
from curbmeasure.mapillary import fetch_pixels, images_in_box  # noqa: E402
from curbmeasure.stereo import SweepConfig, sweep  # noqa: E402


def rescale(c, g, w):
    r = g.shape[1] / w
    return Camera(c.image_id, c.centre, c.rotation, c.focal_px * r,
                  (c.principal[0] * r, c.principal[1] * r), c.k1, c.k2, g.shape[1], g.shape[0])


def main() -> int:
    imgs = images_in_box(-122.4475, 37.7860, -122.4405, 37.7889)
    seqs: dict[str, list] = {}
    for i in imgs:
        if i.camera_type == "perspective" and i.sequence:
            seqs.setdefault(i.sequence, []).append(i)
    grp = max(seqs.values(), key=len); grp.sort(key=lambda i: i.captured_at or 0)
    picks = grp[4:9]
    work = ROOT / "work"; work.mkdir(exist_ok=True)
    paths = [work / f"{p.id}.jpg" for p in picks]
    if not all(fetch_pixels(p, q) for p, q in zip(picks, paths)):
        print("fetch failed"); return 1
    lat0, lon0 = picks[0].lat, picks[0].lon
    grays, cams = [], []
    for pick, path in zip(picks, paths):
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        s = 1024 / g.shape[1]
        g = cv2.resize(g, (1024, int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
        grays.append(g); cams.append(rescale(camera_from_image(pick, lat0, lon0), g, pick.width))

    ref, ref_img = cams[2], grays[2]
    others = [(cams[i], grays[i]) for i in (0, 1, 3, 4)]
    # The whole frame, not just the ground band: the sparse features this is being compared
    # against are almost all above the band, which is the very reason the band needs a dense
    # method in the first place.
    depth, _, top = sweep(ref, ref_img, others, SweepConfig(ground_fraction=1.0, far_m=80.0, planes=128))

    sift = cv2.SIFT_create(nfeatures=8000)
    k0, d0 = sift.detectAndCompute(ref_img, None)
    k1, d1 = sift.detectAndCompute(grays[3], None)
    bf = cv2.BFMatcher()
    ms = [m for m, n in bf.knnMatch(d0, d1, k=2) if m.distance < 0.75 * n.distance]
    K = np.array([[ref.focal_px, 0, ref.principal[0]], [0, ref.focal_px, ref.principal[1]], [0, 0, 1]])
    q0 = np.float64([k0[m.queryIdx].pt for m in ms]); q1 = np.float64([k1[m.trainIdx].pt for m in ms])
    E, inl = cv2.findEssentialMat(q0, q1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    ms = [m for m, keep in zip(ms, inl.ravel()) if keep]
    print(f"{len(ms)} geometrically verified matches")

    axis = ref.look(); ratios = []
    for m in ms:
        u, v = k0[m.queryIdx].pt
        row = int(round(v)) - top; col = int(round(u))
        if row < 0 or row >= depth.shape[0] or col < 0 or col >= depth.shape[1]:
            continue
        d_sweep = float(depth[row, col])
        if d_sweep <= 0:
            continue
        X = triangulate([ref, cams[3]], [np.array([u, v]), np.array(k1[m.trainIdx].pt)])
        if not np.all(np.isfinite(X)):
            continue
        d_tri = float((X - ref.centre) @ axis)          # along the optical axis, same as the sweep
        if d_tri <= 0.5 or d_tri > 80:
            continue
        ratios.append((d_sweep, d_tri))
    if len(ratios) < 20:
        print(f"only {len(ratios)} comparable pixels"); return 1
    a = np.array(ratios); r = a[:, 0] / a[:, 1]
    print(f"{len(a)} pixels with both a swept and a triangulated depth")
    print(f"  triangulated depth: median {np.median(a[:,1]):6.2f} m")
    print(f"  swept depth:        median {np.median(a[:,0]):6.2f} m")
    print(f"  ratio sweep/triangulated: median {np.median(r):.3f}  p25 {np.percentile(r,25):.3f}  p75 {np.percentile(r,75):.3f}")
    for q in paths:
        q.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
