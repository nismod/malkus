"""
Example script to exercise library.

Run with:
$ pixi run python trackset_to_rp_maps.py

This script is a workflow that:
- Creates an wind speed evaluation grid
- Reads parquet formatted tropical cyclone tracks
- Filters them to an area of interest
- Interpolates the track observations
- Evaluates the wind fields for each event
- Downscales these wind fields using a surface roughness technique
- Calculates the wind speeds corresponding to a set of provided return periods
- Saves the interpolated tracks, downscaled per event wind fields and return period maps 
"""


from datetime import datetime
import logging
from pathlib import Path

from malkus import (
    RegularGrid,
    ReturnPeriodMapSet,
    SurfaceRoughness,
    TrackSet,
    TrackSource,
    WindFootprintSet,
    compute_gradient_winds,
    downscale_winds,
    initialize_wind_footprints,
    return_period_maps,
)


if __name__ == "__main__":

    logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)

    logging.info("Processing tracks to return period hazard maps")
    t0 = datetime.now()

    bbox = (56.2, -21.8, 59.1, -18.9)  # Mauritius
    source = TrackSource.EMANUEL
    scenario = "585"
    gcm = "cesm2"
    epoch = 2005
    n_years = 200
    interpolation_frequency = "30min"
    grid_resolution = 0.1
    return_periods = [1, 2, 5, 10, 20]

    input_dir = Path("data/in/")
    tracks_path = input_dir / f"tracks/emanuel_ssp-{scenario}_gcm-{gcm}_epoch-{epoch}/tracks.geoparquet"
    land_cover_path = input_dir / "land_cover/glob_cover_2009/GLOBCOVER_L4_200901_200912_V2.3.tif"
    mapping_path = input_dir / "land_cover/land_cover_to_surface_roughness.csv"

    out_dir = Path("data/out/")
    interpolated_tracks_path = out_dir / f"tracks/{source}_{scenario}_{gcm}_{epoch}.pq"
    storm_qc_path = out_dir / f"tracks/{source}_{scenario}_{gcm}_{epoch}_qc.pq"
    surface_footprints_path = out_dir / f"wind_fields/{source}_{scenario}_{gcm}_{epoch}.zarr"
    rp_maps_zarr_path = out_dir / f"hazard_maps/{source}_{scenario}_{gcm}_{epoch}.zarr"
    rp_maps_tiff_path = out_dir / f"hazard_maps/{source}_{scenario}_{gcm}_{epoch}"

    grid = RegularGrid.from_bbox(bbox, grid_resolution)
    logging.info(grid)

    trackset = TrackSet.read_parquet(
        tracks_path,
        metadata={
            "scenario": scenario,
            "gcm": gcm,
            "epoch": epoch,
        },
        source=source,
    )
    logging.info(trackset)
    logging.info("Filtering trackset")
    trackset = trackset.filter_by_minimum_max_wind_speed(15.0)
    trackset = trackset.filter_by_bbox(bbox, search_radius_deg=3)
    trackset = trackset.filter_first_years(n_years)
    logging.info(trackset)

    logging.info("Compute gradient winds")
    gradient_footprints: WindFootprintSet = compute_gradient_winds(
        trackset,
        grid,
        interpolation_frequency=interpolation_frequency,
        interpolated_tracks_path=interpolated_tracks_path,
        storm_qc_path=storm_qc_path,
    )

    logging.info("Initialize footprint store")
    footprints_store: WindFootprintSet = initialize_wind_footprints(
        surface_footprints_path,
        trackset,
        grid,
        level="surface",
    )

    logging.info("Downscale winds")
    surface_roughness = SurfaceRoughness(
        land_cover_path=land_cover_path,
        mapping_path=mapping_path,
    )
    surface_footprints: WindFootprintSet = downscale_winds(
        gradient_footprints,
        method=surface_roughness,
        output=footprints_store
    )
    logging.info(surface_footprints)

    logging.info("Calculate return-period maps")
    rp_maps: ReturnPeriodMapSet = return_period_maps(
        surface_footprints,
        return_periods=return_periods,
        output=rp_maps_zarr_path,
    )
    rp_maps.write_geotiffs(rp_maps_tiff_path)
    logging.info(rp_maps)

    logging.info(f"Done in {datetime.now() - t0}")
