"""Plane-sweep stereo over the ground in front of a camera.

Sparse feature matching cannot see a kerb. Measured on this corpus it returns nothing at all
inside five metres, because SIFT keys on facades and signage while the road and footway beside
the camera are flat grey concrete. Dense matching does not need distinctive points: it scores
every pixel against a window of its neighbours and lets a smoothness assumption carry the
texture-poor stretches.

The sweep is the classical one. For each hypothesised depth, the plane at that depth induces a
homography from each neighbouring view into the reference view; warping by it and comparing to
the reference gives a cost per pixel, and the depth whose warp agrees best wins. Two choices are
worth stating because they are what make it affordable here:

*Only the lower part of the frame is swept.* A kerb is on the ground, and the sky and upper
facades are most of the pixels and none of the answer.

*Depths are spaced evenly in inverse depth, not in depth.* Disparity is linear in inverse depth,
so uniform depth steps oversample the far field and undersample the near -- and the near field
is the whole point.

Cost is zero-mean normalised cross-correlation over a small window, which is invariant to the
exposure differences between frames taken seconds apart with automatic metering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from curbmeasure.geometry import Camera


def pick_device() -> torch.device:
    """Metal where it exists. There is no CUDA on this machine and none is assumed."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True, slots=True)
class SweepConfig:
    #: Nearest and furthest hypothesised depth, metres. A kerb beside a car-mounted camera sits
    #: within about ten metres; beyond twenty the triangulation is too weak to measure a step.
    near_m: float = 2.0
    far_m: float = 25.0
    planes: int = 96
    #: Correlation window. Small enough not to smear the kerb edge, large enough to be stable on
    #: plain concrete.
    window: int = 7
    #: Fraction of the frame height, measured from the bottom, that is swept.
    ground_fraction: float = 0.55
    #: Below this correlation the winning depth is not believed and the pixel is dropped.
    min_score: float = 0.5
    #: How much better the winning depth must be than the best *different* surface. Measured
    #: against the runner-up outside a neighbourhood of the winner, not against the literal
    #: second-best plane: after aggregation the cost curve is smooth, so the adjacent plane is
    #: always a close second and testing against it rejects almost everything -- it drops
    #: acceptance to 8% while telling you nothing about ambiguity.
    min_margin: float = 0.02
    #: Planes within this distance of the winner are the same surface, not a rival for it.
    margin_exclusion: int = 4
    #: Semi-global aggregation penalties, on a cost scale of 0..2. P1 is charged for moving one
    #: depth plane between neighbouring pixels -- cheap, because real surfaces slope. P2 is
    #: charged for jumping further, and is what stops a textureless patch choosing at random.
    p1: float = 0.06
    p2: float = 1.2
    #: P2 is divided down where the image has a strong edge, so a genuine depth discontinuity is
    #: not smoothed away. A kerb is precisely such a discontinuity, so this matters here more
    #: than it does in a typical stereo scene.
    edge_relief: float = 12.0


def intrinsics(camera: Camera) -> torch.Tensor:
    return torch.tensor(
        [[camera.focal_px, 0.0, camera.principal[0]],
         [0.0, camera.focal_px, camera.principal[1]],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def relative(reference: Camera, other: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation and translation taking the reference camera frame to ``other``."""
    R = torch.tensor(other.rotation @ reference.rotation.T, dtype=torch.float32)
    t = torch.tensor(other.rotation @ (reference.centre - other.centre), dtype=torch.float32)
    return R, t


def sweep(
    reference: Camera,
    reference_image: np.ndarray,
    neighbours: list[tuple[Camera, np.ndarray]],
    config: SweepConfig | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Depth for the ground region of ``reference_image``.

    Returns the depth map, a per-pixel confidence, and the first row of the swept band so the
    caller can place the result back in the full frame.
    """
    config = config or SweepConfig()
    device = device or pick_device()

    height, width = reference_image.shape
    top = int(height * (1.0 - config.ground_fraction))
    band = torch.tensor(reference_image[top:, :], dtype=torch.float32, device=device) / 255.0
    band_h, band_w = band.shape

    ys, xs = torch.meshgrid(
        torch.arange(top, height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pixels = torch.stack((xs, ys, ones), dim=-1).reshape(-1, 3).T  # 3 x N

    K = intrinsics(reference).to(device)
    Kinv = torch.linalg.inv(K)
    rays = Kinv @ pixels  # normalised reference rays

    inverse = torch.linspace(1.0 / config.far_m, 1.0 / config.near_m, config.planes, device=device)
    depths = 1.0 / inverse

    # The whole cost volume is kept, because semi-global aggregation needs to look along entire
    # scanlines and cannot be done as the planes stream past. At 96 planes over a 1024-wide band
    # it is about 125 MB, which this machine has.
    volume = torch.empty((config.planes, band_h, band_w), device=device)
    reference_patch = _local_stats(band, config.window)

    for plane, depth in enumerate(depths):
        accumulated = torch.zeros((band_h * band_w,), device=device)
        counted = torch.zeros((band_h * band_w,), device=device)
        for camera, image in neighbours:
            R, t = relative(reference, camera)
            R, t = R.to(device), t.to(device)
            Kj = intrinsics(camera).to(device)
            # Points on the plane at this depth, carried into the neighbour and projected.
            world = rays * depth
            moved = R @ world + t.unsqueeze(1)
            z = moved[2]
            valid = z > 1e-3
            projected = Kj @ moved
            u = torch.where(valid, projected[0] / z.clamp(min=1e-3), torch.full_like(z, -1e4))
            v = torch.where(valid, projected[1] / z.clamp(min=1e-3), torch.full_like(z, -1e4))

            gu = (u / (camera.width - 1)) * 2.0 - 1.0
            gv = (v / (camera.height - 1)) * 2.0 - 1.0
            grid = torch.stack((gu, gv), dim=-1).reshape(1, band_h, band_w, 2)
            source = torch.tensor(image, dtype=torch.float32, device=device).reshape(
                1, 1, image.shape[0], image.shape[1]
            ) / 255.0
            warped = torch.nn.functional.grid_sample(
                source, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )[0, 0]
            inside = (gu.abs() <= 1.0) & (gv.abs() <= 1.0) & valid
            score = _zncc(band, warped, reference_patch, config.window)
            accumulated += torch.where(inside.reshape(band_h, band_w), score, torch.zeros_like(score)).reshape(-1)
            counted += inside.float()

        score = torch.where(counted > 0, accumulated / counted.clamp(min=1.0), torch.full_like(accumulated, -1.0))
        # Correlation is a similarity; aggregation works on a cost, so it is inverted here once
        # rather than sprinkling sign flips through the recursion.
        volume[plane] = (1.0 - score).reshape(band_h, band_w)

    aggregated = _semi_global(volume, band, config)

    best, index = aggregated.min(dim=0)
    planes_axis = torch.arange(config.planes, device=aggregated.device).view(-1, 1, 1)
    far_enough = (planes_axis - index.unsqueeze(0)).abs() > config.margin_exclusion
    rival = torch.where(far_enough, aggregated, torch.full_like(aggregated, 1e6))
    second = rival.min(dim=0).values

    # Sub-pixel refinement by fitting a parabola to the winning cost and its two neighbours. The
    # planes are uniform in inverse depth, so the correction is applied there and inverted back.
    offset = _parabola(aggregated, index)
    refined = index.to(torch.float32) + offset
    step = (inverse[0] - inverse[-1]) / (config.planes - 1)
    refined_inverse = inverse[0] - refined * step
    depth_map = 1.0 / refined_inverse.clamp(min=1.0 / (config.far_m * 2))

    similarity = 1.0 - best
    keep = (similarity > config.min_score) & ((second - best) > config.min_margin)
    depth_map = torch.where(keep, depth_map, torch.zeros_like(depth_map))
    return depth_map.cpu().numpy(), similarity.cpu().numpy(), top


def _parabola(volume: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Sub-plane offset of the cost minimum, in [-0.5, 0.5]."""
    planes = volume.shape[0]
    lo = (index - 1).clamp(0, planes - 1)
    hi = (index + 1).clamp(0, planes - 1)
    c0 = volume.gather(0, lo.unsqueeze(0))[0]
    c1 = volume.gather(0, index.unsqueeze(0))[0]
    c2 = volume.gather(0, hi.unsqueeze(0))[0]
    denominator = (c0 - 2.0 * c1 + c2)
    offset = torch.where(denominator.abs() > 1e-6, 0.5 * (c0 - c2) / denominator, torch.zeros_like(c1))
    # At the ends of the range there is no bracket, and an unclamped fit there runs away.
    interior = (index > 0) & (index < planes - 1)
    return torch.where(interior, offset.clamp(-0.5, 0.5), torch.zeros_like(offset))


def _semi_global(volume: torch.Tensor, band: torch.Tensor, config: SweepConfig) -> torch.Tensor:
    """Accumulate the cost along four directions with a smoothness penalty.

    Winner-take-all gives noise one chance per hypothesis to beat the signal, and on plain
    asphalt there is no signal to beat: ninety-six planes is ninety-six lottery tickets. The
    recursion here makes a pixel's choice depend on its neighbours', so a textureless patch
    inherits the depth of the textured ground around it instead of choosing at random.
    """
    gradient_y = torch.zeros_like(band)
    gradient_x = torch.zeros_like(band)
    gradient_y[1:, :] = (band[1:, :] - band[:-1, :]).abs()
    gradient_x[:, 1:] = (band[:, 1:] - band[:, :-1]).abs()

    total = torch.zeros_like(volume)
    for axis, reverse, gradient in (
        (2, False, gradient_x), (2, True, gradient_x),
        (1, False, gradient_y), (1, True, gradient_y),
    ):
        total += _scan(volume, gradient, axis, reverse, config)
    return total / 4.0


def _scan(
    volume: torch.Tensor, gradient: torch.Tensor, axis: int, reverse: bool, config: SweepConfig
) -> torch.Tensor:
    out = torch.empty_like(volume)
    length = volume.shape[axis]
    order = range(length - 1, -1, -1) if reverse else range(length)
    previous = None
    for position in order:
        if axis == 1:
            cost = volume[:, position, :]
            edge = gradient[position, :]
        else:
            cost = volume[:, :, position]
            edge = gradient[:, position]
        if previous is None:
            current = cost
        else:
            floor = previous.min(dim=0).values
            up = torch.roll(previous, 1, dims=0)
            up[0] = 1e6
            down = torch.roll(previous, -1, dims=0)
            down[-1] = 1e6
            p2 = (config.p2 / (1.0 + config.edge_relief * edge)).clamp(min=config.p1 * 1.5)
            best = torch.minimum(
                torch.minimum(previous, up + config.p1),
                torch.minimum(down + config.p1, floor + p2),
            )
            current = cost + best - floor
        if axis == 1:
            out[:, position, :] = current
        else:
            out[:, :, position] = current
        previous = current
    return out


def _local_stats(image: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = _box(image, window)
    sq = _box(image * image, window)
    var = (sq - mean * mean).clamp(min=1e-6)
    return mean, var.sqrt()


def _box(image: torch.Tensor, window: int) -> torch.Tensor:
    pad = window // 2
    kernel = torch.ones((1, 1, window, window), device=image.device) / (window * window)
    padded = torch.nn.functional.pad(image.reshape(1, 1, *image.shape), (pad, pad, pad, pad), mode="replicate")
    return torch.nn.functional.conv2d(padded, kernel)[0, 0]


def _zncc(a: torch.Tensor, b: torch.Tensor, a_stats, window: int) -> torch.Tensor:
    """Zero-mean normalised cross-correlation, which ignores exposure differences."""
    a_mean, a_std = a_stats
    b_mean = _box(b, window)
    b_std = (_box(b * b, window) - b_mean * b_mean).clamp(min=1e-6).sqrt()
    cross = _box(a * b, window) - a_mean * b_mean
    return cross / (a_std * b_std)


def unproject(camera: Camera, depth_map: np.ndarray, top: int) -> np.ndarray:
    """Depth map to world points, dropping pixels with no accepted depth."""
    height, width = depth_map.shape
    ys, xs = np.mgrid[top : top + height, 0:width]
    keep = depth_map > 0
    if not keep.any():
        return np.empty((0, 3))
    pixels = np.column_stack((xs[keep].ravel(), ys[keep].ravel())).astype(np.float64)
    directions = camera.ray(pixels)
    # ray() is a unit vector; depth here is along the optical axis, so it is scaled by the
    # cosine between the ray and that axis rather than used as a radius.
    axis = camera.look()
    cos = directions @ axis
    scale = depth_map[keep].ravel() / np.clip(cos, 1e-6, None)
    return camera.centre + directions * scale[:, None]
