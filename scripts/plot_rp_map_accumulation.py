"""
Animate cumulative return-period maps from a wind-footprint Zarr store.

Example usage:
$ pixi run python scripts/plot_rp_map_accumulation.py \
    data/out/wind_fields/lesser-antilles_chaz_SSP585_UKESM1-0-LL_2010.zarr/ \
    lesser-antilles.gif \
    --return-periods 5 10 20 50 100 200 \
    --max-cpus 56 \
    --max-years 1000 \
    --title "Lesser Antilles: CHAZ UKESM1-0-LL 2010"
"""


from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import multiprocessing as mp
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from PIL import Image
from tqdm import tqdm

from malkus import WindFootprintSet
from malkus.hazard.footprint import WIND_VARIABLE


MAX_FRAMES = 200
CMAP = "magma_r"
CMAP_UNDER = "white"
CMAP_INTERVAL = 3

_WORKER_DATA = None
_WORKER_YEARS = None
_WORKER_LATS = None
_WORKER_LONS = None


def _initialise_worker(input_path: str) -> None:
    global _WORKER_DATA, _WORKER_YEARS, _WORKER_LATS, _WORKER_LONS
    data = WindFootprintSet.open(input_path).data
    _WORKER_DATA = data[WIND_VARIABLE]
    _WORKER_YEARS = np.asarray(data["year"].values, dtype=int)
    _WORKER_LATS = np.asarray(data["lat"].values)
    _WORKER_LONS = np.asarray(data["lon"].values)


def _cumulative_maps_lazy(frame_year: int, periods: np.ndarray):
    if _WORKER_DATA is None or _WORKER_YEARS is None:
        raise RuntimeError("Frame worker was not initialized")
    shape = (_WORKER_DATA.sizes["lat"], _WORKER_DATA.sizes["lon"])
    first_year = int(_WORKER_YEARS.min())
    calendar_years = np.arange(first_year, frame_year + 1)
    annual = np.zeros((len(calendar_years), *shape), dtype=np.float32)
    current = np.zeros(shape, dtype=np.float32)
    for index, year in enumerate(calendar_years):
        event_indices = np.flatnonzero(_WORKER_YEARS == year)
        if len(event_indices):
            values = np.asarray(_WORKER_DATA.isel(event=event_indices).values, dtype=np.float32)
            annual[index] = np.max(values, axis=0)
            if year == frame_year:
                current = annual[index].copy()
    result = np.zeros((len(periods), *shape), dtype=np.float32)
    duration = len(calendar_years)
    ranked = np.sort(annual, axis=0)[::-1]
    empirical = (duration + 1) / np.arange(1, duration + 1)
    values = ranked.reshape(duration, -1)[::-1]
    axis = empirical[::-1]
    for index, period in enumerate(periods):
        if duration >= period:
            result[index] = np.array([
                np.interp(period, axis, values[:, cell])
                for cell in range(values.shape[1])
            ]).reshape(shape)
    return current, result


def _gif_palette() -> Image.Image:
    """Return one fixed palette shared by every GIF frame."""
    cmap = plt.get_cmap(CMAP, 240)
    map_colours = (cmap(np.linspace(0, 1, 240))[:, :3] * 255).round().astype(np.uint8)
    ui_colours = np.array([
        (255, 255, 255), (0, 0, 0), (32, 32, 32), (64, 64, 64),
        (96, 96, 96), (128, 128, 128), (160, 160, 160), (192, 192, 192),
        (224, 224, 224), (240, 240, 240), (255, 0, 0), (0, 0, 255),
        (0, 128, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ], dtype=np.uint8)
    palette = Image.new("P", (256, 1))
    palette.putpalette(np.concatenate((map_colours, ui_colours)).ravel().tolist())
    return palette


def _render_frame(index, year, year_count, periods, output_dir, vmin, vmax, title):
    if _WORKER_LATS is None or _WORKER_LONS is None:
        raise RuntimeError("Frame worker was not initialized")

    current, maps = _cumulative_maps_lazy(year, periods)
    levels = np.arange(vmin, vmax + CMAP_INTERVAL, CMAP_INTERVAL)
    cmap = plt.get_cmap(CMAP, len(levels) - 1).copy()
    cmap.set_under(CMAP_UNDER)
    norm = BoundaryNorm(levels, cmap.N, clip=False)

    n_panels = len(periods) + 1
    max_cols = 4
    rows = n_panels // max_cols + 1
    cols = n_panels % max_cols + 1
    fig, axes = plt.subplots(
        rows,
        cols,
        squeeze=False,
        figsize=(3.5 * cols, 3.5 * rows),
        layout="constrained",
    )
    axes_flat = axes.ravel()
    extent = [float(_WORKER_LONS.min()), float(_WORKER_LONS.max()),
              float(_WORKER_LATS.min()), float(_WORKER_LATS.max())]
    values = [current, *maps]
    images = [
        ax.imshow(value, origin="lower", extent=extent, cmap=cmap, norm=norm)
        for ax, value in zip(axes_flat, values)
    ]
    axes_flat[0].set_title("Annual maximum")
    axes_flat[0].text(0.05, 0.92, f"Year {year + 1} of {year_count}", transform=axes_flat[0].transAxes)

    for ax, period in zip(axes_flat[1:], periods):
        ax.set_title(f"{period:g}-year RP")
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    fig.colorbar(images[0], ax=axes_flat[:n_panels].tolist(), label=r"Wind speed [ms$^{-1}$]")
    fig.text(0.5, 1.05, title, ha="center", va="bottom", size=14)
    fig.text(0.5, -0.05, "Longitude [deg]", ha="center", va="bottom", size=12)
    fig.text(0, 0.5, "Latitude [deg]", ha="left", va="center", rotation="vertical", size=12)
    fig.set_constrained_layout_pads(h_pad=0.1)
    path = Path(output_dir) / f"frame_{index:06d}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    return index, str(path)


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)
    started = datetime.now()
    logging.info("Creating cumulative return-period map animation")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Input wind-footprint Zarr store (.zarr)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output animation file (.gif)",
    )
    parser.add_argument(
        "--return-periods",
        nargs="+",
        type=float,
        required=True,
        help="Return periods in years to plot (for example: 5 10 20 50)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=18,
        help="Lower wind-speed colour limit in m/s (default: 18)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=72,
        help="Upper wind-speed colour limit in m/s (default: 72)",
    )
    parser.add_argument(
        "--max-years",
        type=int,
        default=None,
        help="Maximum number of calendar years to animate from the first year",
    )
    parser.add_argument(
        "--max-cpus",
        type=int,
        default=1,
        help="Maximum number of worker processes for frame rendering (default: 1)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot title text (default: "")"
    )
    args = parser.parse_args()

    if (
        args.vmax <= args.vmin
        or args.max_cpus < 1
        or args.max_years == 0
        or (args.max_years is not None and args.max_years < 0)
    ):
        parser.error(
            "max-cpus must be positive, vmax must exceed vmin, and max-years must be positive"
        )

    footprints = WindFootprintSet.open(args.input)
    footprints.require_complete()
    data = footprints.data
    years = np.asarray(data["year"].values, dtype=int)
    periods = np.asarray(args.return_periods, dtype=float)
    if np.any(periods <= 0) or len(np.unique(periods)) != len(periods):
        parser.error("return periods must be unique and positive")
    first_year, last_year = int(years.min()), int(years.max())
    if args.max_years is not None:
        last_year = min(last_year, first_year + args.max_years - 1)

    # Which years to plot, given MAX_FRAMES?
    # Build a set of integers that slowly accelerate. Hit all the low, positive integers.
    duration = last_year - first_year + 1
    n_frames = min(MAX_FRAMES, duration)
    exponent = 3
    ideal_offsets = np.linspace(0, 1, n_frames) ** exponent * (duration - 1)
    frame_years = np.empty(n_frames, dtype=int)
    frame_years[0] = first_year
    for i in range(1, n_frames):
        frame_years[i] = min(
            first_year + round(ideal_offsets[i]),
            last_year,
        )
        frame_years[i] = max(frame_years[i], frame_years[i - 1] + 1)

    worker_count = min(args.max_cpus, n_frames)
    with tempfile.TemporaryDirectory(prefix="malkus-rp-frames-") as temp_dir:
        paths: list[str | None] = [None] * n_frames
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_initialise_worker,
            initargs=(str(args.input),)
        ) as executor:

            futures = [
                executor.submit(
                    _render_frame,
                    i,
                    int(year),
                    duration,
                    periods,
                    temp_dir,
                    args.vmin,
                    args.vmax,
                    args.title,
                ) for i, year in enumerate(frame_years)
            ]
            with tqdm(total=n_frames, desc="Rendering frames") as progress:
                for future in as_completed(futures):
                    index, path = future.result()
                    paths[index] = path
                    progress.update(1)
        ordered_paths = [path for path in paths if path is not None]
        if len(ordered_paths) != n_frames:
            raise RuntimeError("Not all animation frames were rendered")
        palette = _gif_palette()
        images = [
            Image.open(path).convert("RGB").quantize(
                palette=palette, dither=Image.Dither.FLOYDSTEINBERG
            )
            for path in ordered_paths
        ]

        try:
            logging.info("Assembling animation at %s", args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            elapsed_years = np.arange(1, n_frames + 1)
            frame_fps = np.clip(elapsed_years ** 0.25, 1, 50)
            frame_durations_ms = 1000.0 / frame_fps
            images[0].save(
                args.output,
                save_all=True,
                append_images=images[1:],
                duration=frame_durations_ms.astype(int).tolist(),
                loop=0
            )
        finally:
            for image in images:
                image.close()

    logging.info("Completed in %s", datetime.now() - started)


if __name__ == "__main__":
    main()
