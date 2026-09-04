#!/usr/bin/env python3
"""Reprojection error as a function of depth, which is what decides whether this is viable.

A kerb beside the camera is five to ten metres away. Distant facades triangulate badly at any
baseline and their error says nothing about whether the near ground can be measured, so the
question is not "what is the error" but "what is the error at the range we care about".
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


def rescale(camera: Camera, gray, width: int) -> Camera:
    ratio = gray.shape[1] / width
    return Camera(camera.image_id, camera.centre, camera.rotation, camera.focal_px * ratio,
                  (camera.principal[0] * ratio, camera.principal[1] * ratio),
                  camera.k1, camera.k2, gray.shape[1], gray.shape[0])


def main() -> int:
    images = images_in_box(-122.4475, 37.7860, -122.4405, 37.7889)
    groups: dict[str, list] = {}
    for i in images:
        if i.camera_type == "perspective" and i.merge_cc:
            groups.setdefault(i.merge_cc, []).append(i)

    work = ROOT / "work"; work.mkdir(exist_ok=True)
    errors, depths, alts = [], [], []
    used = 0
    for group in sorted(groups.values(), key=len, reverse=True)[:3]:
        group.sort(key=lambda i: i.captured_at or 0)
        lat0, lon0 = group[0].lat, group[0].lon
        cams = [camera_from_image(i, lat0, lon0) for i in group]
        alts += [float(i.altitude) for i in group if i.altitude is not None]
        for start in range(0, min(len(group) - 2, 9), 3):
            picks = group[start:start + 3]
            paths = [work / f"{p.id}.jpg" for p in picks]
            if not all(fetch_pixels(p, q) for p, q in zip(picks, paths)):
                continue
            grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
            if any(g is None for g in grays):
                continue
            three = [rescale(cams[start + k], grays[k], picks[k].width) for k in range(3)]
            used += 1
            sift = cv2.SIFT_create(nfeatures=8000)
            kd = [sift.detectAndCompute(g, None) for g in grays]
            bf = cv2.BFMatcher()
            def good(a, b):
                return [m for m, n in bf.knnMatch(kd[a][1], kd[b][1], k=2) if m.distance < 0.8 * n.distance]
            ab = good(0, 1)
            ac = {m.queryIdx: m.trainIdx for m in good(0, 2)}
            for m in ab:
                if m.queryIdx not in ac:
                    continue
                p0 = np.array(kd[0][0][m.queryIdx].pt); p1 = np.array(kd[1][0][m.trainIdx].pt)
                p2 = np.array(kd[2][0][ac[m.queryIdx]].pt)
                X = triangulate([three[0], three[1]], [p0, p1])
                if not np.all(np.isfinite(X)):
                    continue
                z = float(np.linalg.norm(X - three[0].centre))
                back = three[2].project(X[None, :])[0]
                if not np.all(np.isfinite(back)) or z <= 0.5 or z > 120:
                    continue
                errors.append(float(np.linalg.norm(back - p2))); depths.append(z)
            for q in paths:
                q.unlink(missing_ok=True)

    e, d = np.array(errors), np.array(depths)
    print(f"{used} image triples, {len(e)} triangulated points")
    if len(alts) > 1:
        a = np.array(alts)
        print(f"altitudes: median {np.median(a):.1f} m, spread p10-p90 {np.percentile(a,10):.1f}..{np.percentile(a,90):.1f}")
    print("\n  depth band     n    median err (px)   within 3 px")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 120)):
        s = (d >= lo) & (d < hi)
        if s.sum() < 3:
            print(f"  {lo:3d}-{hi:3d} m  {int(s.sum()):5d}          -"); continue
        print(f"  {lo:3d}-{hi:3d} m  {int(s.sum()):5d}     {np.median(e[s]):8.2f}        {100*(e[s]<3).mean():3.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
