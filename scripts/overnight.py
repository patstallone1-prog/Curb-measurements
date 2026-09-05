#!/usr/bin/env python3
"""Measure kerbs from photographs, unattended, against lidar ground truth.

Built to be left running. Three properties matter more than speed:

*It cannot lose work.* Poses are cached to disk as they arrive and results are appended and
flushed per footway, so an interruption at hour seven costs the footway in progress and nothing
else.

*It cannot fill the disk.* Pixels are fetched into a working directory, measured, and deleted
immediately. The run stops on its own if free space falls below a floor rather than taking the
machine down with it.

*It does not lean on the search endpoint.* That is the one that gets throttled. Every image id in
the corridor was already harvested, so this looks poses up by id, which is not throttled -- five
a second measured, run here at three out of politeness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SIBLING = Path(os.environ.get(
    "KERBSIDE_SRC", "/Users/elialbukerk/Projects/Applications/spatial-mapping-crowdsource/src"))
sys.path.insert(0, str(SIBLING))

from curbmeasure.geometry import Camera, camera_from_image  # noqa: E402
from curbmeasure.lookup import PoseCache  # noqa: E402
from curbmeasure.mapillary import fetch_pixels  # noqa: E402
from curbmeasure.stereo import SweepConfig, sweep, unproject  # noqa: E402
from smc.lidar.curb import LIDAR_CONFIG, MAX_STEP_M, MIN_STEP_M, find_kerb_line  # noqa: E402
from smc.measure.extract import measure_cross_section  # noqa: E402

EARTH_RADIUS_M = 6_378_137.0
#: A camera further than this from the footway line is looking at different ground.
CAMERA_LATERAL_M = 12.0
#: Candidate images are drawn from this far around the footway before being filtered.
SEARCH_PAD_M = 40.0
ROAD_REACH_M, WALK_REACH_M = 6.0, 4.0
#: Stop rather than fill the disk. The working set is small, but a run left overnight should not
#: be the reason the machine has no room in the morning.
MIN_FREE_GB = 3.0


def enu(lat, lon, lat0, lon0):
    return (math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * EARTH_RADIUS_M)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def rescale(c: Camera, gray, width: int) -> Camera:
    r = gray.shape[1] / width
    return Camera(c.image_id, c.centre, c.rotation, c.focal_px * r,
                  (c.principal[0] * r, c.principal[1] * r), c.k1, c.k2, gray.shape[1], gray.shape[0])


def undistort(gray, c: Camera):
    K = np.array([[c.focal_px, 0, c.principal[0]], [0, c.focal_px, c.principal[1]], [0, 0, 1]])
    return cv2.undistort(gray, K, np.array([c.k1, c.k2, 0.0, 0.0, 0.0]))


def load_catalogue(path: Path) -> list[dict]:
    """Image ids and rough positions, from the harvest the sibling project already did."""
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("latitude") is None or row.get("provider_image_id") is None:
                continue
            rows.append({"id": str(row["provider_image_id"]),
                         "lat": float(row["latitude"]), "lon": float(row["longitude"])})
    return rows


def measure_footway(way_id, slices, catalogue_index, poses, work, width, log) -> list[dict]:
    a, b = way_id.split(":")
    lon1, lat1 = (float(x) for x in a.split(","))
    lon2, lat2 = (float(x) for x in b.split(","))
    lat0, lon0 = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    origin = np.array(enu(lat1, lon1, lat0, lon0))
    span = np.array(enu(lat2, lon2, lat0, lon0)) - origin
    length = float(np.hypot(*span))
    if length < 12:
        return []
    along = span / length
    across = np.array([-along[1], along[0]])

    # Candidates come from the local catalogue, so no search request is issued at all.
    degree = SEARCH_PAD_M / 111_320.0
    nearby = [
        row for row in catalogue_index.query(lat0, lon0, degree + length / 2 / 111_320.0)
        if abs(row["lat"] - lat0) < degree + length / 2 / 111_320.0
    ]
    beside = []
    for row in nearby:
        offset = np.array(enu(row["lat"], row["lon"], lat0, lon0)) - origin
        station = float(offset @ along)
        if abs(float(offset @ across)) > CAMERA_LATERAL_M or not (-5.0 <= station <= length + 5.0):
            continue
        beside.append((station, row["id"]))
    if len(beside) < 4:
        return []

    by_sequence: dict[str, list] = defaultdict(list)
    resolved = []
    for station, image_id in sorted(beside)[:60]:
        image = poses.fetch(image_id)
        if image is None or image.camera_type != "perspective" or not image.sequence:
            continue
        resolved.append((station, image))
        by_sequence[image.sequence].append((station, image))
    if not by_sequence:
        return []

    def spread(entries):
        stations = [s for s, _ in entries]
        return max(stations) - min(stations)

    candidates = [e for e in by_sequence.values() if len(e) >= 4 and spread(e) >= 10.0]
    if not candidates:
        return []
    chosen = sorted(candidates, key=spread, reverse=True)[0]
    chosen.sort(key=lambda pair: pair[0])
    step = max(1, len(chosen) // 7)
    picks = [image for _, image in chosen[::step]][:7]

    paths = [work / f"{p.id}.jpg" for p in picks]
    grays, cams = [], []
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
    lidar = {round(s["station_m"], 1): s["curb_height_m"] for s in slices}
    log(f"      {len(cams)} views, {len(pts):,} points, cameras spread {spread(chosen):.0f} m")

    for sign in (1.0, -1.0):
        v = (rel @ across) * sign
        band = (v > -ROAD_REACH_M) & (v < WALK_REACH_M)
        out = []
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
            nearest = min(lidar, key=lambda k: abs(k - station))
            if abs(nearest - station) > 6.0:
                continue
            out.append({"footway_id": way_id, "station_m": float(station),
                        "photo_curb_m": float(section.kerb.height_m),
                        "photo_sigma_m": float(section.kerb.sigma_m),
                        "lidar_curb_m": lidar[nearest],
                        "views": len(cams), "points": int(len(pts))})
        if out:
            return out
    return []


class Grid:
    """A coarse spatial index over the catalogue, so no search request is ever needed."""

    def __init__(self, rows: list[dict], cell_deg: float = 0.002) -> None:
        self.cell = cell_deg
        self.buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in rows:
            self.buckets[(int(row["lat"] / cell_deg), int(row["lon"] / cell_deg))].append(row)

    def query(self, lat: float, lon: float, radius_deg: float) -> list[dict]:
        reach = int(radius_deg / self.cell) + 1
        base_lat, base_lon = int(lat / self.cell), int(lon / self.cell)
        found = []
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                found.extend(self.buckets.get((base_lat + dy, base_lon + dx), ()))
        return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", type=Path,
                    default=SIBLING.parent / "data/sf_corridor_mapillary_dense/observations/journal.jsonl")
    ap.add_argument("--sections", type=Path,
                    default=SIBLING.parent / "data/sf_corridor/depth/lidar/curb_sections.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "photo_vs_lidar.jsonl")
    ap.add_argument("--poses", type=Path, default=ROOT / "work" / "poses.jsonl")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--hours", type=float, default=8.0)
    args = ap.parse_args()

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    catalogue = load_catalogue(args.catalogue)
    grid = Grid(catalogue)
    log(f"{len(catalogue):,} catalogued images indexed")

    rows = [json.loads(l) for l in args.sections.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("curb_height_m") is not None]
    byway: dict[str, list] = defaultdict(list)
    for r in rows:
        byway[r["footway_id"]].append(r)
    # Ordering by lidar coverage puts the best-measured footways first, and those turn out to be
    # downtown -- outside the longitude range the interrupted photo harvest ever reached. Only 4%
    # of lidar slices have a catalogued photograph within 25 m, so the ordering that matters is
    # where both exist, not where either is richest.
    def photo_support(way_id: str) -> int:
        a, b = way_id.split(":")
        lon1, lat1 = (float(x) for x in a.split(","))
        lon2, lat2 = (float(x) for x in b.split(","))
        lat0, lon0 = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        reach = (SEARCH_PAD_M + math.hypot((lat2 - lat1) * 111_320, (lon2 - lon1) * 88_000) / 2) / 111_320.0
        found = grid.query(lat0, lon0, reach)
        return sum(
            1 for r in found
            if abs(r["lat"] - lat0) * 111_320 < reach * 111_320
            and abs(r["lon"] - lon0) * 88_000 < reach * 111_320
        )

    ranked = [(photo_support(way_id), way_id, slices) for way_id, slices in byway.items()]
    ranked = [r for r in ranked if r[0] >= 4]
    ranked.sort(key=lambda r: -r[0])
    ways = [(way_id, slices) for _, way_id, slices in ranked]
    log(f"{len(byway):,} lidar-measured footways; {len(ways):,} have photographs beside them")

    work = ROOT / "work"; work.mkdir(exist_ok=True)
    poses = PoseCache(args.poses)
    log(f"{len(poses):,} poses already cached")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["footway_id"])
    log(f"{len(done):,} footways already measured")

    deadline = time.time() + args.hours * 3600
    written = attempted = 0
    with args.out.open("a") as sink:
        for index, (way_id, slices) in enumerate(ways, start=1):
            if way_id in done:
                continue
            if time.time() > deadline:
                log("time budget reached; stopping cleanly")
                break
            if free_gb(work) < MIN_FREE_GB:
                log(f"only {free_gb(work):.1f} GB free; stopping rather than filling the disk")
                break
            attempted += 1
            try:
                results = measure_footway(way_id, slices, grid, poses, work, args.width, log)
            except Exception as exc:  # noqa: BLE001 - one footway must never end the night
                log(f"  {index}/{len(ways)} skipped: {type(exc).__name__}: {str(exc)[:70]}")
                continue
            for row in results:
                sink.write(json.dumps(row) + "\n")
            sink.flush()
            written += len(results)
            if results:
                photo = np.median([r["photo_curb_m"] for r in results]) * 1000
                truth = np.median([r["lidar_curb_m"] for r in results]) * 1000
                log(f"  {index}/{len(ways)}: {len(results)} slices  photo {photo:.0f} mm vs lidar {truth:.0f} mm"
                    f"   [total {written}]")
            elif attempted % 10 == 0:
                log(f"  {index}/{len(ways)}: nothing  [total {written}, poses cached {len(poses):,}]")
    poses.close()
    log(f"done: {written} paired measurements from {attempted} footways attempted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
