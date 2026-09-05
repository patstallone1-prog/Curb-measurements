"""Fetch image poses one id at a time, and keep them.

The bounding-box search is what gets throttled. Measured against the same token in the same
minute, a box query downtown is refused with "Please reduce the amount of data you're asking
for" while twenty-five per-id lookups return cleanly at five a second. The metadata for every
image in the corridor was already harvested into the sibling project, so the search can be
skipped entirely: the ids are known, and the ids are all the per-id endpoint needs.

Poses are cached to disk on arrival. A run that is interrupted after two hours should resume
where it stopped rather than start again, and nothing here is expensive enough to be worth
fetching twice.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from curbmeasure.mapillary import API, FIELDS, Image, MapillaryError, _to_image, token

#: Requests a second. The endpoint sustained five without complaint; this leaves room and keeps
#: an eight-hour unattended run from being the reason a free service starts refusing.
RATE_PER_S = 3.0
ATTEMPTS = 5


class PoseCache:
    """Image poses on disk, appended as they arrive."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line torn by an interrupted write; the id will be refetched
                self._rows[str(row.get("id"))] = row
        self._handle = self.path.open("a")
        self._last = 0.0

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, image_id: str) -> bool:
        return str(image_id) in self._rows

    def get(self, image_id: str) -> Image | None:
        row = self._rows.get(str(image_id))
        return _to_image(row) if row else None

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        minimum = 1.0 / RATE_PER_S
        if gap < minimum:
            time.sleep(minimum - gap)
        self._last = time.monotonic()

    def fetch(self, image_id: str) -> Image | None:
        """The pose for one image, from cache or from the service."""
        image_id = str(image_id)
        if image_id in self._rows:
            return self.get(image_id)
        query = urllib.parse.urlencode({"access_token": token(), "fields": FIELDS})
        url = f"{API}/{image_id}?{query}"
        last: Exception | None = None
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(2.0 * (2 ** (attempt - 1)) + random.uniform(0, 1.0))
            self._wait()
            try:
                with urllib.request.urlopen(url, timeout=45) as response:
                    row = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403, 404):
                    # A withdrawn or private image. Recorded as absent so a resume does not
                    # spend the rest of the night asking for it again.
                    self._remember({"id": image_id, "absent": True})
                    return None
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        else:
            raise MapillaryError(f"image {image_id}: {last} after {ATTEMPTS} attempts")
        row.setdefault("id", image_id)
        self._remember(row)
        return _to_image(row)

    def _remember(self, row: dict) -> None:
        self._rows[str(row["id"])] = row
        self._handle.write(json.dumps(row) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
