# tc-lib design plan

Working name: `tc-lib`.

The library should provide a clean Python interface for working with tropical
cyclone event sets, producing wind hazard fields, and deriving return-period
products for impact modelling. Preprocessors for source-specific raw model
outputs can remain separate projects for now; this library starts from a strict
canonical track table format that can be stored as GeoParquet, Parquet, or other
tabular formats.

## Core concepts

### Track

A single tropical cyclone trajectory: one storm/event represented by a sequence
of eye observations.

`Track` should be lightweight and mostly provide convenient access to one
grouped `track_id` from a `TrackSet`.

### TrackSet

A validated table of many tracks in a common schema.

`TrackSet` is the main catalogue object:

- one row per eye observation
- strict required columns and units
- optional profile-specific or source-specific columns
- validation, filtering, interpolation, and gap filling
- lightweight catalogue metadata such as source TC model, GCM, scenario, epoch,
  sample/member, and preprocessing provenance
- the assumption that `year` values are plausible and already encode the correct
  annual event frequency

In short: `TrackSet` describes both what tracks exist and the catalogue they
belong to. A separate `EventSet` class can be added later if there is a need to
combine multiple track files under a richer statistical interpretation, but it
should not be part of the v1 design.

### WindFieldSet

Gridded or point-based wind outputs derived from a `TrackSet`.

It should support:

- time-evolving gradient-level winds
- time-evolving surface winds
- max-over-time event footprints
- point/asset wind intensities
- regular-grid products for GeoTIFF, NetCDF, and Zarr export

## Canonical track schema

Version 1 should use a strict track table schema. Parsers for raw model data can
be introduced later, but the first stable contract should be the cleaned track
format. `lat` and `lon` columns should be authoritative; geometry is optional
and useful for GeoParquet/GIS compatibility.

Required or strongly preferred columns:

```text
track_id
timestep
time_utc
year
month
day
hour
lat
lon
geometry
basin_id
max_wind_speed_ms
radius_to_max_winds_km
min_pressure_hpa
```

Optional columns:

```text
event_id
sample
source_year
calibration_year
tc_number
ss_category
environmental_pressure_hpa
holland_b
outer_radius_km
translation_speed_ms
translation_heading_deg
storm_motion_u_ms
storm_motion_v_ms
```

`source_year` and `calibration_year` can be kept separate where possible. This
preserves original temporal structure from a TC model while allowing calibrated
or sampled years to drive return-period calculations. For v1, the library assumes
the main `year` column is already suitable for annual maxima.

Each wind profile should declare its required and optional columns so missing
meteorological inputs fail clearly and can be routed through gap-filling
functions.

## Core evaluation primitive

The most important implementation choice is to make wind evaluation independent
of output shape. A wind model should evaluate at arbitrary points:

```python
def evaluate_at_points(
    points,
    eye,
    observation,
    profile,
    *,
    taper=None,
    return_components=False,
):
    ...
```

The same primitive should support:

- asset locations as a 1D point cloud
- flattened cells from a regular grid
- masked sub-grids near the eye
- time-evolving wind fields
- max-over-time footprints

This avoids treating 2D rasters as the fundamental object. A regular grid is
just an ordered point cloud with shape metadata.

## Point clouds and grids

Use a small abstraction for evaluation sites:

```python
PointCloud
  lats
  lons
  indices_within_radius(lat, lon, radius_m)

RegularGrid(PointCloud)
  shape
  transform / coordinates
  to_xarray()
  to_geotiff()
```

For regular-grid footprints, each timestep can:

1. find nearby flat grid indices within a practical search radius
2. evaluate winds only at those points
3. apply roughness or other pointwise downscaling
4. scatter-max into the event footprint

This supports country, bbox, basin, and global outputs without evaluating huge
empty bounding boxes around every track.

## Wind modelling

Wind profiles should be simple callable objects or protocols, not a heavy plugin
system.

Examples:

```python
Holland1980()
Holland2010()
Willoughby2006()
RankineVortex()
```

The current sigmoid outer taper should be represented explicitly, probably as a
separate taper/decay component applied during evaluation. It is part of the wind
model behaviour, not just a computational mask. The search radius is then a
practical cutoff for evaluation.

The current quadratic interpolation to hourly timesteps should be available as
the default. The interpolation interval should be configurable because it
controls the tradeoff between computational cost and the "coffee ring" artefact
from discrete eye positions.

Potential future option:

```text
adaptive timestep based on translation speed and radius to maximum winds
```

## Gradient and surface winds

Gradient-level wind vectors should be first-class outputs because a future FNO
or Kepert-Wang emulator may consume time-evolving `u_g(t)` and `v_g(t)`.

Suggested time-resolved variables:

```text
u_gradient_ms
v_gradient_ms
wind_speed_gradient_ms
u_surface_ms
v_surface_ms
wind_speed_surface_ms
```

For many impact workflows, the derived max-over-time field remains the primary
product:

```text
max_wind_speed_ms(event, y, x)
```

## Downscaling

Downscalers should be composable callables operating on evaluated winds and
local point metadata.

Initial methods:

- surface roughness scaling from land use
- external Kepert-Wang PBL adapter when the separate library exists
- future FNO emulator consuming gradient-level wind vectors

Roughness can be assumed pre-averaged or resampled to the evaluation grid for
version 1.

## Return periods and pooling

For version 1, return-period calculations should assume that tracks already have
plausible years and the correct annual frequency. The basic calculation is:

1. compute one annual maximum field per year
2. sort annual maxima at each grid cell
3. assign plotting positions such as `(n_years + 1) / rank`
4. interpolate/extract requested return periods

This should be the primary method because it matches the intended input data
contract. Event exceedance-rate methods can be added later for catalogues that
do not have meaningful annual grouping.

Primary pooling design should be RP-map-first:

1. compute RP maps separately for each source combination
2. retain metadata for TC model, GCM, scenario, profile, epoch, sample, and
   simulated years
3. stack RP maps along an ensemble/source dimension
4. compute means, quantiles, spreads, and grouped summaries

This preserves uncertainty attribution across TC model, wind profile, GCM, and
scenario. Merging event sets before RP calculation can be supported as a
secondary shortcut, but should not be the main design target.

## I/O

Primary inputs:

- strict track tables, with GeoParquet supported but not required
- raster/vector region definitions
- optional point asset tables later

Primary storage/output:

- Zarr for large gridded event sets and time-evolving winds
- GeoTIFF for max footprints and return-period maps
- NetCDF for compatibility
- Parquet/GeoParquet for tracks and point outputs

## Public API sketch

```python
tracks = tc.TrackSet.read("tracks.gpq").validate()
tracks = tracks.with_metadata(
    source="CHAZ",
    scenario="SSP585",
    gcm="UKESM1-0-LL",
    epoch=2050,
)

grid = tc.RegularGrid.from_bbox(
    bounds=(-20, 30, 60, 5),
    resolution=0.05,
)

gradient = tc.compute_gradient_winds(
    tracks,
    grid,
    profile=tc.holland_1980,
    interpolation_frequency="1h",
)

surface = tc.downscale_winds(
    gradient,
    method=tc.SurfaceRoughness("land_cover.tif", "roughness.csv"),
)

rp = tc.return_period_maps(
    surface,
    return_periods=[10, 25, 50, 100, 250],
)
```

## Open questions

1. Should geometry be written when exporting to GeoParquet, or should exports
   remain plain Parquet by default? A: Plain parquet
2. Should global processing parallelism be event-first, tile-first, or support
   both for SLURM workflows? A: event-first
3. What metadata fields should be standardized on `TrackSet`? A: source (str), scenario (str, nullable), gcm (str, nullable), epoch (int), is_synthetic (bool)
4. What plotting position should be the default for return periods?
