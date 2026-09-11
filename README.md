# malkus

`malkus` is a Python library for turning tropical-cyclone track catalogues into
wind-hazard footprints and return-period maps. It provides tools for validating
track data, interpolating storm motion, evaluating gradient winds, applying
surface-roughness downscaling, and exporting hazard maps.

## Installation

For development or reproducible use, install [pixi](https://pixi.prefix.dev/)
and run:

```bash
pixi install
```

## Example

Tracks are read from Parquet or GeoParquet into a validated `TrackSet`. The
following illustrates the core workflow:

Track inputs should contain the canonical fields required by `TrackSet`,
including track ID, UTC timestamp, latitude, longitude, maximum wind speed,
radius to maximum winds, and minimum pressure.

```python
from malkus import (
    RegularGrid,
    TrackSet,
    TrackSource,
    compute_gradient_winds,
)

bounds = (56.2, -21.8, 59.1, -18.9)  # west, south, east, north
grid = RegularGrid.from_bbox(bounds, resolution=0.1)
tracks = TrackSet.read_parquet(
    "tracks.geoparquet",
    source=TrackSource.EMANUEL,
)

footprints = compute_gradient_winds(
    tracks,
    grid,
    interpolation_frequency="30min",
)
print(footprints)
```

Return-period maps are calculated from surface-level footprints. Use
`downscale_winds` first when surface roughness is required, then pass the
result to `return_period_maps`.

For a complete example, including surface-roughness downscaling and Zarr
output, see [`scripts/trackset_to_rp_maps.py`](scripts/trackset_to_rp_maps.py).
That workflow expects tropical-cyclone tracks, a land-cover raster, and a
land-cover-to-roughness mapping table under `data/in/`.

## Development

Run the test suite with:

```bash
pixi run python -m pytest -q
```

The project supports Python 3.12 and newer.
