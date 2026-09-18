"""Create wind fields and return-period maps from tropical-cyclone tracks."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path

from malkus import (
    RegularGrid,
    SurfaceRoughness,
    TrackSet,
    TrackSource,
    compute_winds,
    downscale_winds,
    initialize_wind_footprints,
    return_period_maps,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="+", type=Path, help="Input track Parquet/GeoParquet files")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--search-radius-deg", type=float, default=3.0, help="Track search radius in degrees (default: 3)")
    parser.add_argument("--grid-resolution-deg", type=float, required=True, help="Wind-field grid resolution in degrees")
    parser.add_argument("--source", choices=[source.value for source in TrackSource], required=True)
    parser.add_argument("--scenario", required=True, help="Scenario metadata")
    parser.add_argument("--gcm", required=True, help="GCM metadata")
    parser.add_argument("--epoch", type=int, required=True, help="Simulation epoch")
    parser.add_argument("--return-periods", nargs="+", type=float, required=True, help="Positive unique return periods in years")
    parser.add_argument("--wind-fields", type=Path, required=True, help="Output wind-footprint Zarr store")
    parser.add_argument("--rp-maps", type=Path, required=True, help="Output return-period-map Zarr store")
    parser.add_argument("--rp-geotiffs", type=Path, required=True, help="Output GeoTIFF directory")
    parser.add_argument("--interpolated-tracks", type=Path, help="Optional interpolated-track Parquet output")
    parser.add_argument("--storm-qc", type=Path, help="Optional storm-QC Parquet output")
    parser.add_argument("--max-cpus", type=int, default=1, help="Maximum worker processes (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Events per worker batch (default: 32)")
    parser.add_argument("--interp-temp-freq", help="Fixed interpolation frequency, e.g. 1h or 30min (default: 1h)")
    parser.add_argument("--interp-dist-factor", type=float, help="Adaptive spacing as a multiple of local RMW")
    parser.add_argument("--land-cover", type=Path, help="Optional land-cover raster for downscaling")
    parser.add_argument("--roughness-mapping", type=Path, help="Optional land-cover-to-roughness CSV")
    return parser


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args()
    if args.search_radius_deg <= 0 or args.grid_resolution_deg <= 0:
        parser.error("search-radius-deg and grid-resolution-deg must be positive")
    if args.max_cpus < 1 or args.batch_size < 1:
        parser.error("max-cpus and batch-size must be positive")
    if args.interp_dist_factor is not None and args.interp_dist_factor <= 0:
        parser.error("interp-dist-factor must be positive")
    if args.interp_temp_freq is not None and args.interp_dist_factor is not None:
        parser.error("interp-temp-freq and interp-dist-factor are mutually exclusive")
    if any(period <= 0 for period in args.return_periods):
        parser.error("return-periods must be positive")
    if len(set(args.return_periods)) != len(args.return_periods):
        parser.error("return-periods must be unique")
    if (args.land_cover is None) != (args.roughness_mapping is None):
        parser.error("land-cover and roughness-mapping must be supplied together")
    if args.interpolated_tracks is None and args.storm_qc is not None:
        parser.error("storm-qc requires interpolated-tracks")
    return args


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)
    started = datetime.now()
    logging.info("Processing tracks to return-period hazard maps")

    bbox = tuple(args.bbox)
    grid = RegularGrid.from_bbox(bbox, args.grid_resolution_deg)
    logging.info(grid)
    trackset = TrackSet.read_parquet(
        args.tracks,
        metadata={"scenario": args.scenario, "gcm": args.gcm, "epoch": args.epoch},
        source=TrackSource(args.source),
        bbox=bbox,
        search_radius_deg=args.search_radius_deg,
    )
    logging.info(trackset)

    for path in (args.wind_fields, args.rp_maps, args.interpolated_tracks, args.storm_qc):
        if path is not None:
            _ensure_parent(path)
    args.rp_geotiffs.mkdir(parents=True, exist_ok=True)
    frequency = args.interp_temp_freq or "1h"

    logging.info("Computing wind fields")
    compute_kwargs = dict(
        interpolation_frequency=frequency,
        interpolation_spacing_factor=args.interp_dist_factor,
        interpolated_tracks_path=args.interpolated_tracks,
        storm_qc_path=args.storm_qc,
        n_workers=args.max_cpus,
        batch_size=args.batch_size,
    )
    if args.land_cover is None:
        wind_fields = compute_winds(
            trackset,
            grid,
            output=initialize_wind_footprints(args.wind_fields, trackset, grid),
            **compute_kwargs,
        )
    else:
        computed = compute_winds(trackset, grid, **compute_kwargs)
        logging.info("Downscaling wind fields")
        wind_fields = downscale_winds(
            computed,
            method=SurfaceRoughness(
                land_cover_path=args.land_cover,
                mapping_path=args.roughness_mapping,
            ),
            output=initialize_wind_footprints(args.wind_fields, trackset, grid),
        )

    logging.info("Computing return-period maps")
    maps = return_period_maps(wind_fields, args.return_periods, output=args.rp_maps)
    maps.write_geotiffs(args.rp_geotiffs)
    logging.info("Completed in %s", datetime.now() - started)


if __name__ == "__main__":
    main()
