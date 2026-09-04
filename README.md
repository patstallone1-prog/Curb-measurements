# Curb measurements from street photography

Measures kerb height and footway width from ordinary street-level photographs, by multi-view
stereo on camera poses that somebody else already solved.

## Why this is tractable

The hard part of measuring anything from a photograph is scale. A picture of a kerb is a picture
of a kerb at any size; recovering the metre requires either a ranging sensor or a reconstruction
anchored to something of known length, and errors in that anchor are systematic — they do not
average away no matter how many photographs you add.

Mapillary has already solved it. Every image in their catalogue carries `computed_rotation`,
`computed_geometry` and `atomic_scale`: a solved camera orientation, an SfM-refined position, and
the metric scale of the reconstruction it belongs to. Measured over a San Francisco test box, 100%
of 789 images had all three, with `atomic_scale` clustering at 0.99 (p10 0.89, p90 1.08) and a
median nearest-neighbour camera spacing of **1.09 m**.

That spacing is what makes this work. With a 1.09 m baseline and a ~3,400 px focal length,
triangulation precision is:

```
σ_depth ≈ Z² / (f · B) · σ_disparity        at σ_disparity = 0.3 px

  Z =  5 m  →   ~2 mm
  Z = 10 m  →   ~8 mm
  Z = 20 m  →  ~32 mm
```

A kerb is about 130 mm. Inside ten metres — which is where a kerb beside a camera actually is —
this resolves it.

## What was ruled out first, and why

**Monocular metric depth.** The obvious idea, and it does not work at this scale of feature.
Benchmarked on street imagery, the best models (UniDepth-Base, Metric3D-ViT) reach 5.6–5.7 m mean
absolute error, and 1.2–2.4 m even under five metres. A kerb is 0.15 m. The error is an order of
magnitude larger than the thing being measured.

**Mapillary's published `mesh`.** Per-image and free, but it is the sparse SfM cloud triangulated:
452 vertices and 868 faces for a whole street scene, roughly one vertex per 4,000 pixels. Fine for
scene structure, far too coarse for a kerb face.

## Ground truth

Measurements are checked against 9,376 kerb heights measured from USGS 3DEP aerial lidar in the
sibling project, which were themselves cross-validated against Waymo ground-level lidar and agreed
to within 1 mm of median. Nothing from this pipeline should be believed until it has been put
against those.

## What the validation found

Poses were checked before anything was built on them, by triangulating SIFT matches between two
images and reprojecting into a **third** that took no part in the triangulation.

Mapillary's poses are good, *within a sequence*:

```
rotation disagreement vs the pixels' own essential matrix   median 0.86°   p90 3.11°
baseline direction disagreement                             median 4.79°   p90 41.0°
consecutive baseline                                        median 2.05 m
```

Reprojection accuracy after geometric verification:

```
depth band      n    median err (px)   within 3 px
  0-  5 m       0          -
  5- 10 m       9       9444             0%
 10- 20 m     117       7.95            15%
 20- 40 m     249       4.58            25%
 40-120 m     930      29.00            14%
```

**Sparse feature matching cannot measure a kerb, and the reason is not the poses.** There are
essentially no matchable features in the near field: SIFT keys on facades, signage and distant
structure, while the five metres of road and footway beside the camera are low-texture asphalt
and concrete. The band where a kerb actually lives returns nothing to triangulate.

That is what makes the dense step mandatory rather than an optimisation. Dense methods —
plane-sweep, SGBM, the patch-match in `opensfm/dense.py` — do not need distinctive points; they
match every pixel with a local window and a smoothness prior, which is exactly what a texture-poor
road surface requires.

## Three mistakes worth recording

Each produced a confident, wrong conclusion, and each was a fault in the test rather than in the
data:

1. **Comparing `t` from `recoverPose` against a baseline direction.** OpenCV returns the
   translation of the transform, not the camera centre; the centre is `-Rᵀt`. This reported a 66°
   disagreement where there was almost none.
2. **Grouping by `merge_cc` and sorting by capture time.** A merge component can span several
   separate drives, so "consecutive" pairs were wide-baseline pairs of different streets pointing
   different ways. This reported 110° of rotation error. Grouping by `sequence` first fixed it.
3. **Triangulating unverified matches.** A ratio test leaves confident mismatches, and a
   mismatched pair does not fail loudly — it triangulates somewhere plausible and reprojects
   hundreds of pixels away, which reads as a pose problem. Essential-matrix RANSAC before
   triangulation cut the 10–40 m error from 32 px to 4.6 px.

## Hardware this has to run on

Apple M4, 10 CPU cores (4P/6E), 10 GPU cores, 16 GB unified memory, Metal 3. **No CUDA**, so
COLMAP's dense stereo is unavailable and anything GPU-bound has to go through MPS or Metal.

Disk is the harder limit: **13 GB free**. At 1024 px the full 92,834-image catalogue is 32 GB and
at 2048 px it is 109 GB, so images can never all be resident. The pipeline streams — fetch a
working set, measure, delete the pixels, keep only the measurements.

## Plane sweep: built, fast, not yet correct

`src/curbmeasure/stereo.py` implements the sweep — inverse-depth plane hypotheses, homography
warps of each neighbour into the reference, ZNCC cost, winner-take-all with a best-vs-second
margin test. It runs on Metal: **325,000 pixels over 96 depth planes against 4 neighbours in
1.4–2.3 s**, which extrapolates to the whole corpus comfortably inside the machine's limits.

The output is not yet usable, and the diagnostic says why:

```
row band (top -> bottom of swept region)   accepted   median depth
  rows  259- 311    27.2%        4.89 m
  rows  311- 364    20.9%        4.56 m
  rows  364- 417    17.0%        3.43 m
  rows  417- 470    15.7%        3.43 m
  rows  470- 523    14.6%        3.43 m
  rows  523- 576    18.8%        3.07 m
```

The *sign* is right — depth falls towards the bottom of the frame, as road should. The *range* is
wrong: it should run from about 3 m at the bottom to twenty or more near the horizon, and instead
it is compressed into 3–5 m throughout. Depths are systematically too near, which then places the
reconstructed ground 0.5 m below the camera instead of the ~2.5 m a vehicle camera actually sits
at, and leaves a 570–630 mm residual against a plane that should be flat to a few centimetres.

The leading explanation is the cost, not the geometry. Winner-take-all ZNCC over 96 hypotheses is
96 chances for noise to win on a surface with no texture, and plain asphalt is exactly that.
Semi-global aggregation — a smoothness penalty accumulated along several directions, as in SGM —
exists for this failure and is the next thing to try.
