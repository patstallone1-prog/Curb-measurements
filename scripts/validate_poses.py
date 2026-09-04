#!/usr/bin/env python3
"""Check that Mapillary's poses are usable before building anything on them.

Features are matched between two images and triangulated; the resulting 3D points are then
reprojected into a *third* image that took no part in the triangulation. If the pose convention
is right the points land where the feature actually is, within a pixel or two. If any of the
conventions is wrong -- rotation sense, frame handedness, normalisation by width instead of the
long side -- the error is enormous and unmistakable, which is the point of testing this way
rather than by inspecting a point cloud for plausibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curbmeasure.geometry import camera_from_image, triangulate  # noqa: E402
from curbmeasure.mapillary import fetch_pixels, images_in_box  # noqa: E402


def main() -> int:
    west, south, east, north = -122.4475, 37.7860, -122.4405, 37.7889
    images = images_in_box(west, south, east, north)
    print(f"{len(images)} images with a solved pose")

    groups: dict[str, list] = {}
    for image in images:
        if image.camera_type == "perspective" and image.merge_cc:
            groups.setdefault(image.merge_cc, []).append(image)
    best = max(groups.values(), key=len)
    best.sort(key=lambda i: i.captured_at or 0)
    print(f"largest perspective reconstruction: {len(best)} images (merge_cc {best[0].merge_cc})")

    lat0, lon0 = best[0].lat, best[0].lon
    cams = [camera_from_image(i, lat0, lon0) for i in best]
    spacing = [float(np.linalg.norm(cams[i + 1].centre - cams[i].centre)) for i in range(len(cams) - 1)]
    print(f"consecutive baselines: median {np.median(spacing):.2f} m")

    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    trio = None
    for start in range(len(best) - 2):
        picks = best[start : start + 3]
        paths = [work / f"{p.id}.jpg" for p in picks]
        if all(fetch_pixels(p, q) for p, q in zip(picks, paths)):
            trio = (picks, paths, [cams[start], cams[start + 1], cams[start + 2]])
            break
    if trio is None:
        print("could not fetch three consecutive images")
        return 1
    picks, paths, three = trio
    print(f"using {', '.join(p.id for p in picks)}")

    grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    # Poses describe the full-resolution frame; the thumbnails are smaller, so the intrinsics
    # have to be scaled to the pixels actually being matched or every reprojection is off by the
    # ratio between them.
    scaled = []
    for camera, gray, pick in zip(three, grays, picks):
        ratio = gray.shape[1] / pick.width
        scaled.append(
            type(camera)(
                camera.image_id, camera.centre, camera.rotation,
                camera.focal_px * ratio,
                (camera.principal[0] * ratio, camera.principal[1] * ratio),
                camera.k1, camera.k2, gray.shape[1], gray.shape[0],
            )
        )

    sift = cv2.SIFT_create(nfeatures=4000)
    kp = []
    des = []
    for gray in grays:
        k, d = sift.detectAndCompute(gray, None)
        kp.append(k)
        des.append(d)
    matcher = cv2.BFMatcher()

    def matched(a: int, b: int):
        pairs = matcher.knnMatch(des[a], des[b], k=2)
        return [m for m, n in pairs if m.distance < 0.75 * n.distance]

    ab = matched(0, 1)
    ac = {m.queryIdx: m.trainIdx for m in matched(0, 2)}
    print(f"matches 0-1: {len(ab)}   also seen in 2: {sum(1 for m in ab if m.queryIdx in ac)}")

    errors = []
    depths = []
    for m in ab:
        if m.queryIdx not in ac:
            continue
        p0 = np.array(kp[0][m.queryIdx].pt)
        p1 = np.array(kp[1][m.trainIdx].pt)
        p2 = np.array(kp[2][ac[m.queryIdx]].pt)
        X = triangulate([scaled[0], scaled[1]], [p0, p1])
        if not np.all(np.isfinite(X)):
            continue
        depth = float(np.linalg.norm(X - scaled[0].centre))
        if not (1.0 < depth < 60.0):
            continue
        back = scaled[2].project(X[None, :])[0]
        if not np.all(np.isfinite(back)):
            continue
        errors.append(float(np.linalg.norm(back - p2)))
        depths.append(depth)

    if not errors:
        print("no points survived triangulation")
        return 1
    e = np.array(errors)
    d = np.array(depths)
    print(f"\nreprojection into the held-out third view: n={len(e)}")
    print(f"  median {np.median(e):7.2f} px    p25 {np.percentile(e,25):.2f}   p75 {np.percentile(e,75):.2f}")
    print(f"  within 2 px: {100*(e<2).mean():.0f}%    within 5 px: {100*(e<5).mean():.0f}%")
    print(f"  triangulated depth: median {np.median(d):.1f} m  p10 {np.percentile(d,10):.1f}")
    for path in paths:
        path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
