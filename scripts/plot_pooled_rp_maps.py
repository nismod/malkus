#!/usr/bin/env python3
"""Plot pooled mean and IQR return-period maps side by side."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import xarray as xr


WIND_VMIN = 18.0
WIND_VMAX = 72.0
WIND_INTERVAL = 3.0
WIND_CMAP = "magma_r"
IQR_CMAP = "viridis"
IQR_VMIN = 0.0
IQR_VMAX = 15.0
IQR_INTERVAL = 1.0


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--title", default="Pooled return-period maps")
    args = parser.parse_args()
    data = xr.open_zarr(args.input, consolidated=False)
    for variable in ("mean_wind_speed_ms", "iqr_wind_speed_ms"):
        if variable not in data:
            raise ValueError(f"Pooled store is missing {variable}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extent = [float(data.lon.min()), float(data.lon.max()), float(data.lat.min()), float(data.lat.max())]
    wind_levels = np.arange(WIND_VMIN, WIND_VMAX + WIND_INTERVAL, WIND_INTERVAL)
    wind_cmap = plt.get_cmap(WIND_CMAP, len(wind_levels) - 1).copy()
    wind_cmap.set_under("white")
    wind_norm = BoundaryNorm(wind_levels, wind_cmap.N, clip=False)

    iqr_values = data.iqr_wind_speed_ms.values
    if not np.isfinite(iqr_values).any():
        raise ValueError("IQR maps contain no finite values")
    iqr_levels = np.arange(IQR_VMIN, IQR_VMAX + IQR_INTERVAL, IQR_INTERVAL)
    iqr_cmap = plt.get_cmap(IQR_CMAP, len(iqr_levels) - 1).copy()
    iqr_cmap.set_over("white")
    iqr_norm = BoundaryNorm(iqr_levels, iqr_cmap.N, clip=False)

    for period in data.return_period.values:
        mean = data.mean_wind_speed_ms.sel(return_period=period)
        iqr = data.iqr_wind_speed_ms.sel(return_period=period)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        wind_image = axes[0].imshow(
            mean,
            origin="lower",
            extent=extent,
            cmap=wind_cmap,
            norm=wind_norm,
        )
        iqr_image = axes[1].imshow(
            iqr,
            origin="lower",
            extent=extent,
            cmap=iqr_cmap,
            norm=iqr_norm,
        )
        axes[0].set_title("Mean wind speed")
        axes[1].set_title("IQR wind speed")
        for axis in axes:
            axis.set_xlabel("Longitude [deg]")
            axis.set_ylabel("Latitude [deg]")
        fig.colorbar(
            wind_image,
            ax=axes[0],
            ticks=wind_levels,
            label=r"Wind speed [m s$^{-1}$]",
        )
        fig.colorbar(
            iqr_image,
            ax=axes[1],
            ticks=iqr_levels,
            extend="max",
            label=r"IQR [m s$^{-1}$]",
        )
        fig.suptitle(f"{args.title}: {float(period):g}-year return period")
        fig.savefig(args.output_dir / f"return_period_{float(period):g}_year.png", dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    main()
