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
    compute_winds,
    downscale_winds,
    initialize_wind_footprints,
    return_period_maps,
)


if __name__ == "__main__":

    logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)

    logging.info("Processing tracks to return period hazard maps")
    t0 = datetime.now()

    # bbox = (56.2, -21.8, 59.1, -18.9)  # Mauritius
    # name = "mur"
    # bbox = (-61.96, 13.17, -59.90, 14.63)  # St. Lucia
    # name = "lca"
    bbox = (-66.12, 9.69, -58.44, 19.46)  # Lesser Antilles
    name = "lesser-antilles"

    n_cpu = 48
    interpolation_frequency = "30min"
    grid_resolution = 0.05

    input_dir = Path("data/in/")

#   source = TrackSource.EMANUEL
#   tracks_paths = input_dir / "tracks/emanuel_ssp-585_gcm-cesm2_epoch-2005/tracks.geoparquet"
#   scenario = "585"
#   gcm = "cesm2"
#   epoch = 2005
#   n_years = 200
#   return_periods = [1, 2, 5, 10, 20]

    source = TrackSource.CHAZ
    # tracks_path = input_dir / "tracks/CHAZ_SSP-585_GCM-CESM2_epoch-2010/tracks.geoparquet"
    tracks_paths = [input_dir / f"tracks/CHAZ_SSP-585_GCM-UKESM1-0-LL_epoch-2010/{i}/tracks.geoparquet" for i in range(5)]
    gcm = "UKESM1-0-LL"
    scenario = "SSP585"
    epoch = 2010
    return_periods = [5, 10, 20, 50, 100, 200, 500]

    land_cover_path = input_dir / "land_cover/glob_cover_2009/GLOBCOVER_L4_200901_200912_V2.3.tif"
    mapping_path = input_dir / "land_cover/land_cover_to_surface_roughness.csv"

    out_dir = Path("data/out/")
    interpolated_tracks_path = out_dir / f"tracks/{name}_{source}_{scenario}_{gcm}_{epoch}.pq"
    storm_qc_path = out_dir / f"tracks/{name}_{source}_{scenario}_{gcm}_{epoch}_qc.pq"
    wind_footprints_path = out_dir / f"wind_fields/{name}_{source}_{scenario}_{gcm}_{epoch}.zarr"
    rp_maps_zarr_path = out_dir / f"hazard_maps/{name}_{source}_{scenario}_{gcm}_{epoch}.zarr"
    rp_maps_tiff_path = out_dir / f"hazard_maps/{name}_{source}_{scenario}_{gcm}_{epoch}"

    grid = RegularGrid.from_bbox(bbox, grid_resolution)
    logging.info(grid)

    trackset = TrackSet.read_parquet(
        tracks_paths,
        metadata={
            "scenario": scenario,
            "gcm": gcm,
            "epoch": epoch,
        },
        source=source,
        bbox=bbox,
        search_radius_deg=3,
    )
    logging.info(trackset)

    logging.info("Compute winds")
    wind_footprints: WindFootprintSet = compute_winds(
        trackset,
        grid,
        interpolation_frequency=interpolation_frequency,
        interpolated_tracks_path=interpolated_tracks_path,
        storm_qc_path=storm_qc_path,
        n_workers=n_cpu,
    )

    logging.info("Initialize footprint store")
    footprints_store: WindFootprintSet = initialize_wind_footprints(
        wind_footprints_path,
        trackset,
        grid,
    )

    logging.info("Downscale winds")
    surface_roughness = SurfaceRoughness(
        land_cover_path=land_cover_path,
        mapping_path=mapping_path,
    )
    downscaled_footprints: WindFootprintSet = downscale_winds(
        wind_footprints,
        method=surface_roughness,
        output=footprints_store
    )
    logging.info(downscaled_footprints)

    logging.info("Calculate return-period maps")
    rp_maps: ReturnPeriodMapSet = return_period_maps(
        downscaled_footprints,
        return_periods=return_periods,
        output=rp_maps_zarr_path,
    )
    rp_maps.write_geotiffs(rp_maps_tiff_path)
    logging.info(rp_maps)

    logging.info(f"Done in {datetime.now() - t0}")
