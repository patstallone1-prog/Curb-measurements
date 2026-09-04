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

#: Mapillary refuses a box it considers too large with an HTTP 500 that says
#: "Please reduce the amount of data you're asking for". It is a response-size quota rather than
#: a fault, and the remedy is a smaller box, not a retry.
PAGE_LIMIT = 1000


class MapillaryError(RuntimeError):
    pass


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


def _get(url: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def images_in_box(
    west: float, south: float, east: float, north: float, *, limit: int = PAGE_LIMIT
) -> list[Image]:
    """Every image in a bounding box, following the cursor to the end."""
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
        payload = _get(url)
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
    try:
        with urllib.request.urlopen(image.thumb_url, timeout=90) as response:
            path.write_bytes(response.read())
    except Exception:  # noqa: BLE001 - a missing thumbnail is ordinary, not exceptional
        return False
    return True
