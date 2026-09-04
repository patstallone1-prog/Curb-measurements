#!/usr/bin/env python3
"""Reprojection accuracy against depth, within a single sequence.

Grouping matters more than it looks. A ``merge_cc`` is a joint reconstruction that can span
several separate drives, so sorting its images by capture time interleaves passes that went in
different directions -- and "consecutive" pairs drawn from it are wide-baseline pairs of
different streets. Grouping by ``sequence`` first is what makes consecutive mean consecutive.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curbmeasure.geometry import Camera, camera_from_image, triangulate  # noqa: E402
from curbmeasure.mapillary import fetch_pixels, images_in_box  # noqa: E402


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
    work = ROOT / "work"; work.mkdir(exist_ok=True)
    err: list[float] = []; dep: list[float] = []
    for grp in sorted(seqs.values(), key=len, reverse=True)[:3]:
        grp.sort(key=lambda i: i.captured_at or 0)
        lat0, lon0 = grp[0].lat, grp[0].lon
        for s in range(0, min(len(grp) - 2, 9), 3):
            picks = grp[s:s + 3]
            if len({p.merge_cc for p in picks}) != 1:
                continue
            paths = [work / f"{p.id}.jpg" for p in picks]
            if not all(fetch_pixels(p, q) for p, q in zip(picks, paths)):
                continue
            gs = [cv2.imread(str(q), 0) for q in paths]
            if any(x is None for x in gs):
                continue
            cams = [rescale(camera_from_image(p, lat0, lon0), g, p.width) for p, g in zip(picks, gs)]
            sift = cv2.SIFT_create(nfeatures=6000)
            kd = [sift.detectAndCompute(g, None) for g in gs]
            bf = cv2.BFMatcher()
            def gd(a, b):
                return [m for m, n in bf.knnMatch(kd[a][1], kd[b][1], k=2) if m.distance < 0.8 * n.distance]
            ab = gd(0, 1)
            # Geometric verification before anything is triangulated. A ratio test alone leaves
            # plenty of confident mismatches, and a mismatched pair does not fail loudly -- it
            # triangulates to a plausible-looking point somewhere and then reprojects hundreds of
            # pixels away, which reads as a pose problem rather than as the matching problem it is.
            if len(ab) >= 30:
                q0 = np.float64([kd[0][0][m.queryIdx].pt for m in ab])
                q1 = np.float64([kd[1][0][m.trainIdx].pt for m in ab])
                K = np.array([[cams[0].focal_px, 0, cams[0].principal[0]],
                              [0, cams[0].focal_px, cams[0].principal[1]], [0, 0, 1]])
                E, inl = cv2.findEssentialMat(q0, q1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None and inl is not None:
                    ab = [m for m, keep in zip(ab, inl.ravel()) if keep]
            ac = {m.queryIdx: m.trainIdx for m in gd(0, 2)}
            for m in ab:
                if m.queryIdx not in ac:
                    continue
                p0 = np.array(kd[0][0][m.queryIdx].pt); p1 = np.array(kd[1][0][m.trainIdx].pt)
                p2 = np.array(kd[2][0][ac[m.queryIdx]].pt)
                X = triangulate([cams[0], cams[1]], [p0, p1])
                if not np.all(np.isfinite(X)):
                    continue
                z = float(np.linalg.norm(X - cams[0].centre))
                bk = cams[2].project(X[None, :])[0]
                if not np.all(np.isfinite(bk)) or z <= 0.5 or z > 120:
                    continue
                err.append(float(np.linalg.norm(bk - p2))); dep.append(z)
            for q in paths:
                q.unlink(missing_ok=True)
    e, d = np.array(err), np.array(dep)
    print(f"{len(e)} triangulated points, within-sequence triples")
    print("  depth band      n    median err (px)   within 3 px")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 120)):
        s = (d >= lo) & (d < hi)
        if s.sum() < 3:
            print(f"  {lo:3d}-{hi:3d} m  {int(s.sum()):5d}          -")
            continue
        print(f"  {lo:3d}-{hi:3d} m  {int(s.sum()):5d}     {np.median(e[s]):8.2f}        {100*(e[s]<3).mean():3.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
