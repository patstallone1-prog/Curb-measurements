"""Mapillary poses as metric cameras in a local frame.

The conventions here are OpenSfM's, because that is what produced the poses, and getting any of
them wrong yields a reconstruction that looks reasonable and measures nothing:

* The world frame is topocentric ENU — x east, y north, z up — about a reference position.
* ``computed_rotation`` is an angle-axis vector for the **world to camera** rotation, not its
  inverse.
* The camera frame is x right, y down, z forward.
* Pixel coordinates are normalised by the *larger* image dimension, not the width, and the
  origin is the image centre. ``focal`` is a fraction of that same larger dimension.

``atomic_scale`` is deliberately not applied to positions. Positions come from
``computed_geometry``, which is latitude and longitude, so they are already metric by
construction — multiplying by the scale a second time would shrink or stretch the whole scene.
It is kept as a quality signal: a reconstruction whose scale sits far from 1.0 was stretched hard
to fit its GPS, and its relative geometry deserves less trust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


def rodrigues(rotation: tuple[float, float, float]) -> np.ndarray:
    """Angle-axis vector to a 3x3 rotation matrix."""
    r = np.asarray(rotation, dtype=np.float64)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def enu(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    east = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * EARTH_RADIUS_M
    return np.array([east, north, alt - alt0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Camera:
    """One posed, calibrated camera in a local ENU frame."""

    image_id: str
    centre: np.ndarray          # camera position, ENU metres
    rotation: np.ndarray        # 3x3, world -> camera
    focal_px: float
    principal: tuple[float, float]
    k1: float
    k2: float
    width: int
    height: int

    @property
    def projection(self) -> np.ndarray:
        """3x4 world-to-camera transform."""
        P = np.eye(4)[:3]
        P[:, :3] = self.rotation
        P[:, 3] = -self.rotation @ self.centre
        return P

    def look(self) -> np.ndarray:
        """Unit vector the camera points along, in world coordinates."""
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points to pixels. Points behind the camera come back as NaN."""
        pts = np.atleast_2d(points)
        cam = (self.rotation @ (pts - self.centre).T).T
        z = cam[:, 2]
        out = np.full((len(pts), 2), np.nan)
        front = z > 1e-6
        if not front.any():
            return out
        xn = cam[front, 0] / z[front]
        yn = cam[front, 1] / z[front]
        r2 = xn * xn + yn * yn
        distort = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        out[front, 0] = self.focal_px * distort * xn + self.principal[0]
        out[front, 1] = self.focal_px * distort * yn + self.principal[1]
        return out

    def ray(self, pixels: np.ndarray) -> np.ndarray:
        """Unit rays in world coordinates for pixel coordinates.

        The radial distortion is undone by fixed-point iteration. Inverting the polynomial in
        closed form is possible for k1 alone but not once k2 is present, and Mapillary supplies
        both; five iterations converge well inside a tenth of a pixel over the image.
        """
        px = np.atleast_2d(pixels).astype(np.float64)
        xn = (px[:, 0] - self.principal[0]) / self.focal_px
        yn = (px[:, 1] - self.principal[1]) / self.focal_px
        x, y = xn.copy(), yn.copy()
        for _ in range(5):
            r2 = x * x + y * y
            distort = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            x = xn / distort
            y = yn / distort
        directions = np.column_stack((x, y, np.ones_like(x)))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return directions @ self.rotation  # R^T applied on the right


def camera_from_image(image, lat0: float, lon0: float, alt0: float = 0.0) -> Camera:
    altitude = image.altitude if image.altitude is not None else alt0
    return Camera(
        image_id=image.id,
        centre=enu(image.lat, image.lon, float(altitude), lat0, lon0, alt0),
        rotation=rodrigues(image.rotation),
        focal_px=image.focal_px,
        principal=(image.width / 2.0, image.height / 2.0),
        k1=image.k1,
        k2=image.k2,
        width=image.width,
        height=image.height,
    )


def triangulate(cameras: list[Camera], pixels: list[np.ndarray]) -> np.ndarray:
    """Least-squares intersection of rays from several cameras.

    Solved as the point minimising squared distance to every ray, which unlike the linear DLT
    stays well conditioned when the rays are nearly parallel -- and at ten metres with a metre of
    baseline, they are nearly parallel.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for camera, pixel in zip(cameras, pixels):
        d = camera.ray(np.atleast_2d(pixel))[0]
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ camera.centre
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)
