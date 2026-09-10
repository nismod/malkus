"""Return-period maps derived from complete wind-footprint catalogues."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from tc_wind_lib.hazard.footprint import WIND_VARIABLE, WindFootprintSet


def return_period_maps(
    footprints: WindFootprintSet,
    return_periods: list[float] | np.ndarray,
    *,
    output: str | Path | None = None,
) -> xr.DataArray:
    """Calculate empirical annual-maximum wind maps at return periods.

    Rank one has return period ``n_years + 1`` and rank ``r`` has return
    period ``(n_years + 1) / r``. Requested periods between empirical ranks
    are linearly interpolated; values outside the range use the nearest rank.
    """

    if footprints.level != "surface":
        raise ValueError("return_period_maps requires surface wind footprints")
    footprints.require_complete()
    periods = np.asarray(return_periods, dtype=float)
    if periods.ndim != 1 or not len(periods):
        raise ValueError("return_periods must be a non-empty one-dimensional sequence")
    if not np.isfinite(periods).all() or (periods <= 0).any():
        raise ValueError("return_periods must be finite and positive")

    data = footprints.data
    annual = data[WIND_VARIABLE].groupby("year").max("event")
    ranked = np.sort(np.asarray(annual.values), axis=0)[::-1]
    n_years = ranked.shape[0]
    empirical_periods = (n_years + 1) / np.arange(1, n_years + 1)

    # np.interp works along one axis, so make each grid cell a column.
    values = ranked.reshape(n_years, -1)[::-1]
    period_axis = empirical_periods[::-1]
    maps = np.stack(
        [np.interp(periods, period_axis, values[:, cell]) for cell in range(values.shape[1])],
        axis=1,
    ).reshape(len(periods), *footprints.grid.shape)

    result = xr.DataArray(
        maps.astype(np.float32),
        dims=("return_period", "lat", "lon"),
        coords={
            "return_period": periods,
            "lat": data["lat"].values,
            "lon": data["lon"].values,
        },
        name=WIND_VARIABLE,
        attrs={
            **data.attrs,
            "annual_maxima_years": n_years,
            "plotting_position": "(n_years + 1) / rank",
        },
    )
    if output is not None:
        result.to_dataset().to_zarr(Path(output), mode="w", consolidated=False)
    return result
