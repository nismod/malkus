"""
Compare fixed-time and RMW-adaptive wind-field interpolation.

The script creates slow- and fast-moving synthetic storms, evaluates each
with two fixed temporal resolutions and several adaptive RMW spacing factors,
and writes a side-by-side comparison of the resulting maximum wind fields.

Run, for example:
$ pixi run python scripts/plot_adaptive_interpolation_demo.py demo.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malkus import RegularGrid
from malkus.hazard.footprint import _compute_event_footprint
from malkus.wind.interpolate import interpolate_track


ADAPTIVE_FACTORS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
TEMPORAL_CASES = ((f"Temporal {freq}", freq) for freq in ("15min", "30min", "1h", "6h"))


def _make_track(name: str, longitude_step: float) -> pd.DataFrame:
    n_observations = 7
    times = pd.date_range("2000-09-01", periods=n_observations, freq="6h", tz="UTC")
    return pd.DataFrame(
        {
            "track_id": name,
            "time_utc": times,
            "year": 2000,
            "lat": np.full(n_observations, 10.0),
            "lon": 120.0 + longitude_step * np.arange(n_observations),
            "basin_id": "WP",
            "max_wind_speed_ms": np.array([35, 42, 50, 55, 50, 42, 35], dtype=float),
            "radius_to_max_winds_km": np.array([45, 40, 30, 25, 30, 40, 45], dtype=float),
            "min_pressure_hpa": np.array([970, 950, 930, 920, 930, 950, 970], dtype=float),
        }
    )


def _plot_case(ax, field, grid, title, vmin, vmax):
    image = ax.imshow(
        field,
        origin="lower",
        extent=(*grid.bbox[:3:2], grid.bbox[1], grid.bbox[3]),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("adaptive_interpolation_demo.png"),
        help="Output PNG path (default: adaptive_interpolation_demo.png)",
    )
    args = parser.parse_args()

    grid = RegularGrid.from_bbox((118.0, 7.0, 130.0, 13.0), resolution=0.1)
    cases = [
        ("Slow-moving storm", _make_track("slow", 0.5)),
        ("Fast-moving storm", _make_track("fast", 1.5)),
    ]
    methods = [*TEMPORAL_CASES, *[(f"Adaptive {factor:g} RMW", factor) for factor in ADAPTIVE_FACTORS]]

    fields: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
    for storm_name, track in cases:
        for method_name, method in methods:
            if isinstance(method, str):
                field = _compute_event_footprint(
                    track, grid, interpolation_frequency=method
                )
                n_samples = len(pd.date_range(track.time_utc.iloc[0], track.time_utc.iloc[-1], freq=method))
            else:
                field = _compute_event_footprint(
                    track, grid, interpolation_spacing_factor=method
                )
                n_samples = len(
                    interpolate_track(track, spacing_factor=method)
                )
            fields[(storm_name, method_name)] = (field, n_samples)

    vmax = max(float(field.max()) for field, _ in fields.values())
    n_plots = len(cases) * len(methods)
    n_columns = min(5, n_plots)
    n_rows = math.ceil(n_plots / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.2 * n_columns, 2.8 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    image = None
    flat_axes = axes.ravel()
    for plot_index, (storm_name, _) in enumerate(cases):
        for method_index, (method_name, _) in enumerate(methods):
            axis = flat_axes[plot_index * len(methods) + method_index]
            field, n_samples = fields[(storm_name, method_name)]
            image = _plot_case(
                axis,
                field,
                grid,
                f"{storm_name}: {method_name}\n{n_samples} evaluations",
                0,
                vmax,
            )
    for axis in flat_axes[n_plots:]:
        axis.remove()

    assert image is not None
    # Use the populated axes' union so the colourbar spans exactly the plot
    # rows, including when the final row is only partially populated.
    fig.canvas.draw()
    positions = [axis.get_position() for axis in flat_axes[:n_plots]]
    top = max(position.y1 for position in positions)
    bottom = min(position.y0 for position in positions)
    right = max(position.x1 for position in positions)
    colourbar_axis = fig.add_axes([right + 0.015, bottom, 0.018, top - bottom])
    fig.colorbar(image, cax=colourbar_axis, label="Maximum wind speed (m/s)")
    fig.suptitle("Temporal versus RMW-adaptive interpolation", fontsize=16)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.25)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
