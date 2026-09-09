# Curb measurements

This repository serves one thing: the San Francisco corridor map, at
<https://patstallone1-prog.github.io/Curb-measurements/>.

Everything in `docs/` is built output — the page and the four data files it fetches — and is
overwritten wholesale by each deploy. Nothing here is edited by hand.

The map is built in [Spatial-data-mapping](https://github.com/patstallone1-prog/Spatial-data-mapping),
by:

    GITHUB_REPO=Curb-measurements python tools/build_pages.py --map-only --out <this>/docs

The plane-sweep stereo research this repository started as — measuring kerb height from
Mapillary's pre-solved camera poses — moved to that repository too, so that the two halves of
the same project stop being two checkouts. It is in `src/smc/curbmeasure/`,
`scripts/curbmeasure/` and `docs/21-curb-measurement-from-photographs.md` there.
