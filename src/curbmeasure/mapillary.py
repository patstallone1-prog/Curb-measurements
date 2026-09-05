"""Fetch Mapillary image metadata and pixels.

Only the fields that carry geometry are requested. Three of them are the whole reason this
project is possible and are worth naming:

``computed_rotation``   the camera's solved orientation, as an angle-axis vector
``computed_geometry``   its SfM-refined position, which is better than the raw GPS
``atomic_scale``        the metric scale of the reconstruction it belongs to

and one that decides what may be compared with what:

``merge_cc``            the connected component it was reconstructed in. Poses are mutually
                        consistent only inside one of these. Triangulating across two is
                        combining two coordinate systems that were never registered to each
                        other, and it produces a confident wrong answer rather than an error.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

API = "https://graph.mapillary.com"
TOKEN_ENV = "MAPILLARY_TOKEN"

FIELDS = (
    "id,sequence,merge_cc,captured_at,camera_type,camera_parameters,width,height,"
    "geometry,computed_geometry,computed_rotation,computed_altitude,altitude,"
    "atomic_scale,compass_angle,computed_compass_angle,thumb_2048_url,thumb_1024_url"
)

#: Kept high deliberately. The service returns *fewer* rows for a smaller limit while reporting
#: no continuation cursor -- a 20 m box gives 130 images at 1000 and 15 at 100, with nothing to
#: say the other 115 exist. Lowering this to be polite would silently discard most of the data.
PAGE_LIMIT = 1000


class MapillaryError(RuntimeError):
    pass


class QuotaExceeded(MapillaryError):
    """The box asks for more than the service will return. The remedy is a smaller box."""


@dataclass(frozen=True, slots=True)
class Image:
    id: str
    lat: float
    lon: float
    altitude: float | None
    rotation: tuple[float, float, float]
    scale: float
    focal: float
    k1: float
    k2: float
    width: int
    height: int
    camera_type: str
    merge_cc: str | None
    sequence: str | None
    captured_at: int | None
    thumb_url: str | None

    @property
    def focal_px(self) -> float:
        """Mapillary states focal length as a fraction of the larger image dimension."""
        return self.focal * max(self.width, self.height)


def token() -> str:
    value = os.environ.get(TOKEN_ENV, "")
    if not value:
        raise MapillaryError(f"set ${TOKEN_ENV} to a Mapillary application token")
    return value


#: Transient failures -- a read timeout, a reset connection, a throttle -- are ordinary over a
#: long crawl and must not end it. Only a refusal that survives every attempt means anything.
ATTEMPTS = 5


def _get(url: str, timeout: float = 60.0) -> dict:
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(1.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # Mapillary answers several unrelated conditions with HTTP 500, so the code alone
            # cannot be acted on. The size quota needs a smaller box; a throttle or a transient
            # server fault needs a wait. Telling them apart requires reading the body -- and
            # getting it wrong turns a throttle into a subdivision storm, where every refusal
            # spawns four more requests and drives the throttling harder.
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            if "reduce the amount of data" in body:
                # Measured, not assumed: the identical request for a 20 m box downtown is
                # refused with this message one minute and returns 130 rows the next. It is a
                # throttle wearing a quota's wording, so it is retried like any other transient
                # failure and only escalated to a smaller box once retries are exhausted.
                if attempt == ATTEMPTS - 1:
                    raise QuotaExceeded(body[:160]) from exc
                last = exc
                continue
            if exc.code < 500:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
    raise MapillaryError(f"{url.split('?')[0]}: {last} after {ATTEMPTS} attempts")


#: How many times a refused box may be quartered before giving up on it.
MAX_SPLITS = 6


def images_in_box(
    west: float, south: float, east: float, north: float, *, limit: int = PAGE_LIMIT, _depth: int = 0
) -> list[Image]:
    """Every image in a bounding box, following the cursor to the end.

    A box the service considers too large comes back as an HTTP 500 carrying "Please reduce the
    amount of data you're asking for". That is a quota on the response, not a fault, and the
    remedy is a smaller box rather than a retry -- so a refusal quarters the box and recurses.
    Retrying the same request, or treating the refusal as an empty result, reports a densely
    covered street as having no photographs at all.
    """
    query = urllib.parse.urlencode(
        {
            "access_token": token(),
            "fields": FIELDS,
            "bbox": f"{west},{south},{east},{north}",
            "limit": limit,
        }
    )
    url = f"{API}/images?{query}"
    out: list[Image] = []
    while url:
        try:
            payload = _get(url)
        except QuotaExceeded as exc:
            if _depth >= MAX_SPLITS:
                raise MapillaryError(f"still over quota at split depth {_depth}: {exc}") from exc
            mid_lat, mid_lon = (south + north) / 2.0, (west + east) / 2.0
            for box in (
                (west, south, mid_lon, mid_lat), (mid_lon, south, east, mid_lat),
                (west, mid_lat, mid_lon, north), (mid_lon, mid_lat, east, north),
            ):
                out.extend(images_in_box(*box, limit=limit, _depth=_depth + 1))
            # A frame on a shared edge appears in two quadrants; the caller gets one of each.
            seen: set[str] = set()
            unique = []
            for image in out:
                if image.id in seen:
                    continue
                seen.add(image.id)
                unique.append(image)
            return unique
        for row in payload.get("data") or []:
            image = _to_image(row)
            if image is not None:
                out.append(image)
        following = (payload.get("paging") or {}).get("next")
        url = following or ""
    return out


def _to_image(row: dict) -> Image | None:
    rotation = row.get("computed_rotation")
    scale = row.get("atomic_scale")
    geometry = row.get("computed_geometry") or row.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []
    # Without a solved rotation and a scale there is no metric camera, and a photograph that
    # cannot be placed in the world contributes nothing here. Dropped rather than defaulted.
    if not rotation or scale is None or len(coordinates) < 2:
        return None
    parameters = row.get("camera_parameters") or []
    focal = float(parameters[0]) if len(parameters) > 0 else 0.85
    k1 = float(parameters[1]) if len(parameters) > 1 else 0.0
    k2 = float(parameters[2]) if len(parameters) > 2 else 0.0
    return Image(
        id=str(row["id"]),
        lat=float(coordinates[1]),
        lon=float(coordinates[0]),
        altitude=row.get("computed_altitude") or row.get("altitude"),
        rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2])),
        scale=float(scale),
        focal=focal,
        k1=k1,
        k2=k2,
        width=int(row.get("width") or 0),
        height=int(row.get("height") or 0),
        camera_type=str(row.get("camera_type") or "perspective"),
        merge_cc=str(row["merge_cc"]) if row.get("merge_cc") is not None else None,
        sequence=row.get("sequence"),
        captured_at=row.get("captured_at"),
        thumb_url=row.get("thumb_2048_url") or row.get("thumb_1024_url"),
    )


def fetch_pixels(image: Image, path) -> bool:
    """Download one image's pixels. Returns False rather than raising on a dead link."""
    if not image.thumb_url:
        return False
    for attempt in range(3):
        if attempt:
            time.sleep(1.0 * (2 ** (attempt - 1)))
        try:
            with urllib.request.urlopen(image.thumb_url, timeout=90) as response:
                path.write_bytes(response.read())
            return True
        except Exception:  # noqa: BLE001 - a missing thumbnail is ordinary, not exceptional
            continue
    return False


#: Query boxes are kept at or below this. The service's limit is not a response size but a
#: footprint, and it is not enforced consistently: downtown, a 90 m box is refused outright, a
#: 45 m box returns *zero rows* while a 22 m box inside it returns 131, and only below about
#: 25 m is the answer reliable. The silent-empty case is the dangerous one -- subdividing only on
#: an error would take that 45 m box at its word and record a dense street as having no
#: photographs at all.
SAFE_BOX_M = 20.0


def images_along(
    lat1: float, lon1: float, lat2: float, lon2: float, *, half_width_m: float = 25.0
) -> list[Image]:
    """Every image near the segment from one point to the other.

    Tiles small boxes along the line rather than asking for one box around it, because a box
    large enough to hold a city block is refused or silently emptied. Overlapping tiles are
    deduplicated by image id.
    """
    import math  # noqa: PLC0415

    span_lat = lat2 - lat1
    span_lon = lon2 - lon1
    metres = math.hypot(span_lat * 111_320.0, span_lon * 88_000.0)
    steps = max(1, int(math.ceil(metres / SAFE_BOX_M)))
    half_lat = min(SAFE_BOX_M, half_width_m) / 2 / 111_320.0
    half_lon = half_lat / max(math.cos(math.radians(lat1)), 1e-6)

    seen: set[str] = set()
    out: list[Image] = []
    for step in range(steps + 1):
        t = step / steps
        lat = lat1 + span_lat * t
        lon = lon1 + span_lon * t
        for image in images_in_box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat):
            if image.id in seen:
                continue
            seen.add(image.id)
            out.append(image)
    return out
