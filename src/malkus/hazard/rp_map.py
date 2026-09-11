"""Return-period maps derived from complete wind-footprint catalogues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import indent

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from malkus.hazard.footprint import WIND_VARIABLE, WindFootprintSet


@dataclass(frozen=True, init=False)
class ReturnPeriodMapSet:
    """Return-period wind maps held in memory or persisted to Zarr."""

    _data: xr.DataArray | None
    path: Path | None

    def __init__(
        self,
        *,
        data: xr.DataArray | None = None,
        path: str | Path | None = None,
    ) -> None:
        resolved_path = None if path is None else Path(path)
        if (data is None) == (resolved_path is None):
            raise ValueError("Provide exactly one of data or path")
        if data is not None:
            _validate_return_period_maps(data)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "path", resolved_path)

    def __repr__(self) -> str:
        """Return the backing store location and xarray map summary."""

        path = None if self.path is None else str(self.path)
        return (
            "ReturnPeriodMapSet(\n"
            f"  path={path!r},\n"
            "  data=\n"
            f"{indent(repr(self.data), '    ')}\n"
            ")"
        )

    @classmethod
    def open(cls, path: str | Path) -> "ReturnPeriodMapSet":
        """Open an existing return-period Zarr store without loading maps."""

        instance = cls(path=path)
        _validate_return_period_maps(instance.data)
        return instance

    @property
    def data(self) -> xr.DataArray:
        """Return the underlying return-period map array."""

        if self._data is not None:
            return self._data
        assert self.path is not None
        return xr.open_zarr(self.path, consolidated=False, mask_and_scale=False)[
            WIND_VARIABLE
        ]

    @property
    def is_persisted(self) -> bool:
        """Whether the maps are backed by a Zarr store."""

        return self.path is not None

    def write_geotiffs(self, directory: str | Path) -> list[Path]:
        """Write one north-up EPSG:4326 GeoTIFF for each return period."""

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        data = self.data
        resolution = float(data.attrs["grid_resolution_deg"])
        lats = data["lat"].values
        lons = data["lon"].values
        transform = from_origin(
            float(lons[0] - resolution / 2),
            float(lats[-1] + resolution / 2),
            resolution,
            resolution,
        )

        paths: list[Path] = []
        for return_period in data["return_period"].values:
            period = float(return_period)
            path = output_directory / f"return_period_{period:g}_year.tif"
            north_up_values = data.sel(return_period=return_period).values[::-1, :]
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=len(lats),
                width=len(lons),
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as destination:
                destination.write(north_up_values.astype(np.float32), 1)
                destination.update_tags(
                    **{
                        **{key: str(value) for key, value in data.attrs.items()},
                        "return_period_years": f"{period:g}",
                        "units": "m s-1",
                    }
                )
            paths.append(path)
        return paths


def return_period_maps(
    footprints: WindFootprintSet,
    return_periods: list[float] | np.ndarray,
    *,
    output: str | Path | None = None,
) -> ReturnPeriodMapSet:
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
    if len(np.unique(periods)) != len(periods):
        raise ValueError("return_periods must be unique")

    data = footprints.data
    winds = data[WIND_VARIABLE].assign_coords(year=data["year"])
    annual = winds.groupby("year").max("event")
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
    if output is None:
        return ReturnPeriodMapSet(data=result)
    result.to_dataset().to_zarr(Path(output), mode="w", consolidated=False)
    return ReturnPeriodMapSet.open(output)


def _validate_return_period_maps(data: xr.DataArray) -> None:
    if data.name != WIND_VARIABLE:
        raise ValueError(f"Return-period maps must be named {WIND_VARIABLE!r}")
    if data.dims != ("return_period", "lat", "lon"):
        raise ValueError("Return-period maps must have dimensions (return_period, lat, lon)")
    for coordinate in ("return_period", "lat", "lon"):
        values = data[coordinate].values
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{coordinate} must be a one-dimensional numeric coordinate")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{coordinate} must be non-empty and finite")
    periods = data["return_period"].values
    if (periods <= 0).any() or len(np.unique(periods)) != len(periods):
        raise ValueError("return_period must contain unique positive values")
    resolution = data.attrs.get("grid_resolution_deg")
    if not isinstance(resolution, (int, float)) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Return-period maps require a positive grid_resolution_deg attribute")
    for coordinate in ("lat", "lon"):
        values = data[coordinate].values
        if len(values) > 1 and (
            not np.all(np.diff(values) > 0)
            or not np.allclose(np.diff(values), resolution)
        ):
            raise ValueError(
                f"{coordinate} must be ascending and regularly spaced at grid_resolution_deg"
            )
