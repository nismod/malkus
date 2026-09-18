"""
Render event wind footprints and raw-track attribute distributions.

Example usage:

$ pixi run python scripts/plot_track_footprints.py \
    data/in/tracks/CHAZ_SSP-585_GCM-UKESM1-0-LL_epoch-2010/tracks.geoparquet \
    data/out/wind_fields/lesser-antilles_chaz_SSP585_UKESM1-0-LL_2010.zarr/ \
    data/out/wind_fields/lesser-antilles_chaz_SSP585_UKESM1-0-LL_2010_per-track \
    --most-intense \
    --max-tracks 100 \
    --max-cpus 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import multiprocessing as mp
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoLocator, MultipleLocator
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from malkus import WindFootprintSet
from malkus.hazard.footprint import WIND_VARIABLE


ATTRIBUTES = (
    ("min_pressure_hpa", "Minimum pressure [hPa]"),
    ("max_wind_speed_ms", r"Maximum wind speed [m s$^{-1}$]"),
    ("radius_to_max_winds_km", "Radius to maximum winds [km]"),
)
TRACK_SEARCH_RADIUS_DEG = 2.0
WIND_VMIN = 18.0
WIND_VMAX = 72.0
WIND_INTERVAL = 3.0

_WORKER_DATA = None
_WORKER_EVENT_INDICES = None
_WORKER_LATS = None
_WORKER_LONS = None
_WORKER_BINS = None
_WORKER_CATALOGUE_COUNTS = None
_WORKER_VMIN = None
_WORKER_VMAX = None


def _initialise_worker(
    footprint_path: str,
    bins: dict[str, np.ndarray],
    catalogue_counts: dict[str, np.ndarray],
    vmin: float,
    vmax: float,
) -> None:
    global _WORKER_DATA, _WORKER_EVENT_INDICES, _WORKER_LATS, _WORKER_LONS
    global _WORKER_BINS, _WORKER_CATALOGUE_COUNTS, _WORKER_VMIN, _WORKER_VMAX
    data = WindFootprintSet.open(footprint_path).data
    _WORKER_DATA = data[WIND_VARIABLE]
    event_ids = data["event"].values.astype(str)
    _WORKER_EVENT_INDICES = {event_id: index for index, event_id in enumerate(event_ids)}
    _WORKER_LATS = np.asarray(data["lat"].values, dtype=float)
    _WORKER_LONS = np.asarray(data["lon"].values, dtype=float)
    _WORKER_BINS = bins
    _WORKER_CATALOGUE_COUNTS = catalogue_counts
    _WORKER_VMIN = vmin
    _WORKER_VMAX = vmax


def _safe_filename(track_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(track_id)).strip("._")
    return name or "track"


def _set_equal_map_tick_resolution(axis, extent: list[float]) -> None:
    """Use the coarsest automatic tick spacing for both map axes."""

    x_ticks = AutoLocator().tick_values(extent[0], extent[1])
    y_ticks = AutoLocator().tick_values(extent[2], extent[3])
    x_step = np.median(np.diff(x_ticks))
    y_step = np.median(np.diff(y_ticks))
    axis.xaxis.set_major_locator(MultipleLocator(max(x_step, y_step)))
    axis.yaxis.set_major_locator(MultipleLocator(max(x_step, y_step)))


def _render_track(
    track_id: str,
    track_records: list[dict],
    output_directory: str,
) -> tuple[str, str]:
    if _WORKER_DATA is None or _WORKER_EVENT_INDICES is None:
        raise RuntimeError("Track-footprint worker was not initialized")
    event_index = _WORKER_EVENT_INDICES[track_id]
    field = np.asarray(_WORKER_DATA.isel(event=event_index).values, dtype=np.float32)
    track = pd.DataFrame(track_records)

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    grid_spec = fig.add_gridspec(3, 2, width_ratios=(2.0, 1.0), wspace=0.07)
    map_axis = fig.add_subplot(grid_spec[:, 0])
    attribute_axes = [fig.add_subplot(grid_spec[index, 1]) for index in range(3)]

    extent = [
        float(_WORKER_LONS.min()),
        float(_WORKER_LONS.max()),
        float(_WORKER_LATS.min()),
        float(_WORKER_LATS.max()),
    ]
    levels = np.arange(
        _WORKER_VMIN,
        _WORKER_VMAX + WIND_INTERVAL,
        WIND_INTERVAL,
    )
    cmap = plt.get_cmap("magma_r", len(levels) - 1).copy()
    cmap.set_under("white")
    norm = BoundaryNorm(levels, cmap.N, clip=False)
    image = map_axis.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
    )
    map_axis.plot(track["lon"], track["lat"], color="cyan", linewidth=1.5)
    map_axis.scatter(
        track["lon"],
        track["lat"],
        color="white",
        edgecolor="black",
        linewidth=0.3,
        s=12,
        zorder=3,
    )
    map_axis.grid(which="both", alpha=0.2)
    map_axis.set_xlim(extent[0], extent[1])
    map_axis.set_ylim(extent[2], extent[3])
    _set_equal_map_tick_resolution(map_axis, extent)
    map_axis.set_aspect("equal")
    map_axis.set_xlabel("Longitude [deg]")
    map_axis.set_ylabel("Latitude [deg]")
    fig.colorbar(image, ax=map_axis, label=r"Maximum wind speed [m s$^{-1}$]")

    for panel_index, (axis, (column, label)) in enumerate(zip(attribute_axes, ATTRIBUTES)):
        catalogue_counts = _WORKER_CATALOGUE_COUNTS[column]
        bin_widths = np.diff(_WORKER_BINS[column])
        catalogue_density = catalogue_counts / (catalogue_counts.sum() * bin_widths)
        axis.stairs(
            catalogue_density,
            _WORKER_BINS[column],
            fill=True,
            color="0.65",
            alpha=0.55,
            label="Catalogue observations" if panel_index == 0 else "_nolegend_",
        )
        current = track[column].to_numpy(dtype=float)
        current = current[np.isfinite(current)]
        axis.hist(
            current,
            bins=_WORKER_BINS[column],
            density=True,
            histtype="step",
            color="tab:red",
            linewidth=1.8,
            label=(
                f"Track observations"
                if panel_index == 0
                else "_nolegend_"
            ),
        )
        axis.grid(alpha=0.2)
        axis.set_ylabel("Density")
        axis.set_ylim(0, 0.3)
        axis.set_xlabel(label)
        if panel_index == 0:
            axis.legend(loc="best", fontsize="small")

    fig.suptitle(str(track_id))
    output_path = Path(output_directory) / f"{_safe_filename(track_id)}.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return track_id, str(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_tracks", type=Path, help="Raw track Parquet or GeoParquet file")
    parser.add_argument("footprints", type=Path, help="Complete wind-footprint Zarr store")
    parser.add_argument("output_directory", type=Path, help="Directory for per-track PNG files")
    parser.add_argument("--max-tracks", type=int, default=None, help="Maximum tracks to render")
    parser.add_argument(
        "--most-intense",
        action="store_true",
        help="Rank tracks by their highest raw wind speed within the buffered wind field",
    )
    parser.add_argument(
        "--max-cpus", type=int, default=1, help="Maximum worker processes (default: 1)"
    )
    args = parser.parse_args()
    if args.max_cpus < 1 or (args.max_tracks is not None and args.max_tracks < 1):
        parser.error("max-cpus and max-tracks must be positive")
    return args


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)
    started = datetime.now()
    logging.info("Creating TC track/field and attribute distribution plots")

    args = _parse_args()
    attribute_columns = [column for column, _ in ATTRIBUTES]
    try:
        raw = pd.read_parquet(
            args.raw_tracks,
            columns=["track_id", "lat", "lon", *attribute_columns],
        )
    except Exception:
        raw = gpd.read_parquet(
            args.raw_tracks,
            columns=["track_id", "geometry", *attribute_columns],
        )
        raw["lon"] = raw.geometry.x
        raw["lat"] = raw.geometry.y
        raw = pd.DataFrame(raw.drop(columns="geometry"))
    required = {"track_id", *attribute_columns}
    for name, frame in (("raw", raw),):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} tracks are missing columns: {sorted(missing)}")
    if not {"lat", "lon"}.issubset(raw.columns):
        raise ValueError("raw tracks must contain lat/lon or point geometry") 

    footprints = WindFootprintSet.open(args.footprints)
    footprints.require_complete()
    data = footprints.data
    event_ids = data["event"].values.astype(str)
    event_indices = {event_id: index for index, event_id in enumerate(event_ids)}
    field_min_lon = float(data["lon"].values.min())
    field_max_lon = float(data["lon"].values.max())
    field_min_lat = float(data["lat"].values.min())
    field_max_lat = float(data["lat"].values.max())
    raw_track_ids = raw["track_id"].astype(str).drop_duplicates().tolist()
    candidate_ids = [track_id for track_id in raw_track_ids if track_id in event_indices]
    candidate_raw = raw[raw["track_id"].astype(str).isin(candidate_ids)].copy()
    candidate_within_field = (
        candidate_raw["lon"].between(
            field_min_lon - TRACK_SEARCH_RADIUS_DEG,
            field_max_lon + TRACK_SEARCH_RADIUS_DEG,
        )
        & candidate_raw["lat"].between(
            field_min_lat - TRACK_SEARCH_RADIUS_DEG,
            field_max_lat + TRACK_SEARCH_RADIUS_DEG,
        )
    )
    if args.most_intense:
        peak_winds = (
            candidate_raw.loc[candidate_within_field]
            .assign(_track_id=candidate_raw.loc[candidate_within_field, "track_id"].astype(str))
            .groupby("_track_id")["max_wind_speed_ms"]
            .max()
        )
        track_ids = sorted(candidate_ids, key=lambda track_id: -peak_winds.get(track_id, -np.inf))
    else:
        track_ids = candidate_ids
    if not track_ids:
        raise ValueError("No raw track IDs match events in the footprint store")
    if args.max_tracks is not None:
        track_ids = track_ids[: args.max_tracks]

    output_names = [_safe_filename(track_id) for track_id in track_ids]
    if len(set(output_names)) != len(output_names):
        raise ValueError("Track IDs collide after filename sanitization")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    bins: dict[str, np.ndarray] = {}
    catalogue_counts: dict[str, np.ndarray] = {}
    for column, _ in ATTRIBUTES:
        values = raw[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"Raw tracks contain no finite values for {column}")
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
        bins[column] = np.linspace(low, high, 31)
        catalogue_counts[column], _ = np.histogram(values, bins=bins[column])

    vmin = WIND_VMIN
    vmax = WIND_VMAX
    selected_ids = set(track_ids)
    raw_selected = raw[raw["track_id"].astype(str).isin(selected_ids)].copy()
    within_field = (
        raw_selected["lon"].between(
            field_min_lon - TRACK_SEARCH_RADIUS_DEG,
            field_max_lon + TRACK_SEARCH_RADIUS_DEG,
        )
        & raw_selected["lat"].between(
            field_min_lat - TRACK_SEARCH_RADIUS_DEG,
            field_max_lat + TRACK_SEARCH_RADIUS_DEG,
        )
    )
    raw_selected = raw_selected.loc[within_field]
    track_groups = {
        str(track_id): group.to_dict("records")
        for track_id, group in raw_selected.groupby("track_id", sort=False)
    }
    missing_raw = [track_id for track_id in track_ids if track_id not in track_groups]
    if missing_raw:
        raise ValueError(f"Tracks have no raw observations: {missing_raw[:5]}")

    worker_count = min(args.max_cpus, len(track_ids))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialise_worker,
        initargs=(str(args.footprints), bins, catalogue_counts, vmin, vmax),
    ) as executor:
        futures = [
            executor.submit(
                _render_track,
                track_id,
                track_groups[track_id],
                str(args.output_directory),
            )
            for track_id in track_ids
        ]
        with tqdm(total=len(futures), desc="Rendering images") as progress:
            for future in as_completed(futures):
                future.result()
                progress.update(1)

    logging.info("Completed in %s", datetime.now() - started)


if __name__ == "__main__":
    main()
