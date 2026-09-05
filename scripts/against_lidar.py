#!/usr/bin/env python3
"""Measure kerbs from photographs where lidar already measured them.

This is the only test that means anything at kerb scale. Agreement with sparse triangulation was
established at 37 m, because that is the only range where sparse features exist; the five to ten
metres where a kerb sits has none to compare against, so the check has to come from outside the
photographs entirely.

The measurement code is imported from the sibling project rather than reimplemented. Running a
different estimator here would confound whether the sensors agree with whether the estimators do
-- the mistake that produced a spurious 50 mm bias between the aerial and Waymo lidar earlier.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SIBLING = Path(os.environ.get(
    "KERBSIDE_SRC", "/Users/elialbukerk/Projects/Applications/spatial-mapping-crowdsource/src"))
sys.path.insert(0, str(SIBLING))

from curbmeasure.geometry import Camera, camera_from_image  # noqa: E402
from curbmeasure.mapillary import fetch_pixels, images_along  # noqa: E402
from curbmeasure.stereo import SweepConfig, sweep, unproject  # noqa: E402
from smc.lidar.curb import LIDAR_CONFIG, MAX_STEP_M, MIN_STEP_M, find_kerb_line  # noqa: E402
from smc.measure.extract import measure_cross_section  # noqa: E402

EARTH_RADIUS_M = 6_378_137.0
NEAR_M = 18.0
#: A camera further than this from the footway line is looking at a different piece of ground.
CAMERA_LATERAL_M = 12.0
ROAD_REACH_M, WALK_REACH_M = 6.0, 4.0


def enu(lat, lon, lat0, lon0):
    return (math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * EARTH_RADIUS_M)


def rescale(c: Camera, gray, width: int) -> Camera:
    r = gray.shape[1] / width
    return Camera(c.image_id, c.centre, c.rotation, c.focal_px * r,
                  (c.principal[0] * r, c.principal[1] * r), c.k1, c.k2, gray.shape[1], gray.shape[0])


def undistort(gray, c: Camera):
    K = np.array([[c.focal_px, 0, c.principal[0]], [0, c.focal_px, c.principal[1]], [0, 0, 1]])
    return cv2.undistort(gray, K, np.array([c.k1, c.k2, 0.0, 0.0, 0.0]))


def measure_footway(way_id: str, slices: list[dict], work: Path, width: int) -> list[dict]:
    a, b = way_id.split(":")
    lon1, lat1 = (float(x) for x in a.split(","))
    lon2, lat2 = (float(x) for x in b.split(","))
    lat0, lon0 = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    origin = np.array(enu(lat1, lon1, lat0, lon0))
    span = np.array(enu(lat2, lon2, lat0, lon0)) - origin
    length = float(np.hypot(*span))
    if length < 10:
        return []
    along = span / length
    across = np.array([-along[1], along[0]])

    found = images_along(lat1, lon1, lat2, lon2, half_width_m=NEAR_M * 2)

    # Selecting by sequence length picks the longest drive crossing the area, which is routinely
    # a different street: on one footway it chose seven cameras clustered at one end and 15-23 m
    # off to the side, so the sweep reconstructed ground twenty metres from the window that was
    # supposed to measure it. What matters is that a camera is beside this footway and looking
    # along it, so that is what is tested.
    beside: dict[str, list] = {}
    for i in found:
        if i.camera_type != "perspective" or not i.sequence:
            continue
        offset = np.array(enu(i.lat, i.lon, lat0, lon0)) - origin
        station = float(offset @ along)
        lateral = abs(float(offset @ across))
        if lateral > CAMERA_LATERAL_M or not (-5.0 <= station <= length + 5.0):
            continue
        beside.setdefault(i.sequence, []).append((station, i))
    if not beside:
        return []

    # Spread along the footway is what makes a sequence useful -- ten photographs of one spot are
    # one viewpoint -- but spread alone picks pathological candidates: the widest here was two
    # cameras 98 m apart, which is no use for stereo and, being under the minimum, caused the
    # whole footway to be abandoned while a 13-camera sequence spanning 92 m sat unused beside
    # it. So the minimum is applied first and the ranking second, and more than one candidate is
    # tried before giving up.
    def spread(entries: list) -> float:
        stations = [s for s, _ in entries]
        return max(stations) - min(stations)

    candidates = [e for e in beside.values() if len(e) >= 4 and spread(e) >= 10.0]
    if not candidates:
        return []
    candidates.sort(key=spread, reverse=True)
    chosen = candidates[0]
    chosen.sort(key=lambda pair: pair[0])
    # Evenly spaced along the run rather than the first seven, which would cluster at one end.
    step = max(1, len(chosen) // 7)
    picks = [image for _, image in chosen[::step]][:7]
    paths = [work / f"{p.id}.jpg" for p in picks]
    grays: list[np.ndarray] = []
    cams: list[Camera] = []
    try:
        for pick, path in zip(picks, paths):
            if not fetch_pixels(pick, path):
                continue
            g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            s = width / g.shape[1]
            g = cv2.resize(g, (width, int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
            c = rescale(camera_from_image(pick, lat0, lon0), g, pick.width)
            grays.append(undistort(g, c))
            cams.append(c)
        if len(cams) < 4:
            return []

        cloud = []
        for k in range(1, len(cams) - 1):
            others = [(cams[j], grays[j]) for j in range(len(cams)) if j != k][:4]
            if len(others) < 3:
                continue
            depth, _, top = sweep(cams[k], grays[k], others, SweepConfig())
            pts = unproject(cams[k], depth, top)
            if len(pts):
                cloud.append(pts)
    finally:
        for q in paths:
            q.unlink(missing_ok=True)
    if not cloud:
        return []
    pts = np.concatenate(cloud)

    rel = pts[:, :2] - origin
    u = rel @ along
    w = pts[:, 2]
    lidar_by_station = {round(s["station_m"], 1): s["curb_height_m"] for s in slices}

    # The footway's winding order is arbitrary, so which side holds the road is unknown. Both are
    # tried and whichever yields kerbs is kept.
    for sign in (1.0, -1.0):
        v = (rel @ across) * sign
        band = (v > -ROAD_REACH_M) & (v < WALK_REACH_M)
        out: list[dict] = []
        for station in np.arange(2.5, length, 5.0):
            win = band & (np.abs(u - station) < 2.5)
            if int(win.sum()) < LIDAR_CONFIG.min_surface_points:
                continue
            offset = find_kerb_line(v[win], w[win])
            if offset is None:
                continue
            section = measure_cross_section(
                np.column_stack((u[win] - station, v[win], w[win])),
                float(station), config=LIDAR_CONFIG, kerb_offset_hint=offset)
            if section.kerb is None or not (MIN_STEP_M <= section.kerb.height_m <= MAX_STEP_M):
                continue
            nearest = min(lidar_by_station, key=lambda k: abs(k - station))
            if abs(nearest - station) > 6.0:
                continue
            out.append({
                "footway_id": way_id, "station_m": float(station),
                "photo_curb_m": float(section.kerb.height_m),
                "photo_sigma_m": float(section.kerb.sigma_m),
                "lidar_curb_m": lidar_by_station[nearest],
                "points": int(len(pts)), "views": len(cams),
            })
        if out:
            return out
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", type=Path,
                    default=SIBLING.parent / "data/sf_corridor/depth/lidar/curb_sections.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "photo_vs_lidar.jsonl")
    ap.add_argument("--tiles", type=int, default=20)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.sections.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("curb_height_m") is not None]
    byway: dict[str, list] = {}
    for r in rows:
        byway.setdefault(r["footway_id"], []).append(r)
    ways = sorted(byway.items(), key=lambda kv: -len(kv[1]))[: args.tiles]
    print(f"{len(rows)} lidar slices; trying the {len(ways)} best-covered footways", flush=True)

    work = ROOT / "work"; work.mkdir(exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["footway_id"])

    written = 0
    with args.out.open("a") as sink:
        for index, (way_id, slices) in enumerate(ways, start=1):
            if way_id in done:
                continue
            try:
                results = measure_footway(way_id, slices, work, args.width)
            except Exception as exc:  # noqa: BLE001 - one bad footway must not end the run
                print(f"  {index}/{len(ways)}: skipped ({type(exc).__name__}: {str(exc)[:70]})", flush=True)
                continue
            for row in results:
                sink.write(json.dumps(row) + "\n")
            sink.flush()
            written += len(results)
            print(f"  {index}/{len(ways)}: {len(results)} paired slices (total {written})", flush=True)
    print(f"{written} paired measurements -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
