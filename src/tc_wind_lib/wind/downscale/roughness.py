"""Land-cover roughness preparation and power-law wind downscaling."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds as window_from_bounds

from tc_wind_lib.hazard.grid.grid import RegularGrid


ROUGHNESS_MAPPING_COLUMNS = frozenset({"glob_cover_2009_id", "roughness_length_m"})


def power_law_scale_factors(z0: np.ndarray, z1: float = 10.0, z2: float = 18.0) -> np.ndarray:
    """Scale wind at height ``z2`` to height ``z1`` using Wieringa's power law."""

    z0 = np.asarray(z0, dtype=float)
    if z1 <= 0 or z2 <= 0:
        raise ValueError("Wind heights must be positive")
    if not np.isfinite(z0).all() or (z0 <= 0).any():
        raise ValueError("Surface roughness lengths must be finite and positive")
    exponent = 1 / np.log(np.sqrt(z1 * z2) / z0)
    return np.power(z1 / z2, exponent)


def roughness_from_land_cover(
    land_cover_path: str | Path,
    mapping_path: str | Path,
    grid: RegularGrid,
) -> np.ndarray:
    """Map categorical GlobCover data to roughness and average it onto ``grid``.

    Only the raster window overlapping the target grid is read. Returned values
    follow ``grid.shape`` and its south-to-north latitude coordinate order.
    """

    mapping = pd.read_csv(mapping_path, comment="#")
    missing_columns = ROUGHNESS_MAPPING_COLUMNS.difference(mapping.columns)
    if missing_columns:
        raise ValueError(f"Roughness mapping is missing columns: {sorted(missing_columns)}")
    if mapping["glob_cover_2009_id"].duplicated().any():
        raise ValueError("Roughness mapping has duplicate land-cover class IDs")
    class_to_roughness = dict(
        zip(mapping["glob_cover_2009_id"].astype(int), mapping["roughness_length_m"])
    )
    if not np.isfinite(np.asarray(list(class_to_roughness.values()), dtype=float)).all() or any(
        value <= 0 for value in class_to_roughness.values()
    ):
        raise ValueError("Roughness mapping values must be finite and positive")

    west = grid.lons[0] - grid.resolution / 2
    east = grid.lons[-1] + grid.resolution / 2
    south = grid.lats[0] - grid.resolution / 2
    north = grid.lats[-1] + grid.resolution / 2
    destination_transform = from_bounds(west, south, east, north, grid.nlon, grid.nlat)

    with rasterio.open(land_cover_path) as source:
        if source.crs is None:
            raise ValueError("Land-cover raster must define a CRS")
        if source.crs.to_epsg() != 4326:
            raise ValueError("v1 land-cover preparation requires an EPSG:4326 raster")
        requested = window_from_bounds(west, south, east, north, source.transform)
        full_window = Window(0, 0, source.width, source.height)
        try:
            window = requested.round_offsets().round_lengths().intersection(full_window)
        except rasterio.errors.WindowError as error:
            raise ValueError("Target grid does not overlap the land-cover raster") from error
        if window.width <= 0 or window.height <= 0:
            raise ValueError("Target grid does not overlap the land-cover raster")
        classes = source.read(1, window=window)
        source_transform = source.window_transform(window)
        valid_classes = np.unique(classes)
        if source.nodata is not None:
            valid_classes = valid_classes[valid_classes != source.nodata]
        unknown = set(valid_classes.astype(int)).difference(class_to_roughness)
        if unknown:
            raise ValueError(f"Land-cover classes lack roughness values: {sorted(unknown)}")
        source_roughness = np.full(classes.shape, np.nan, dtype=np.float32)
        for class_id, roughness_length in class_to_roughness.items():
            source_roughness[classes == class_id] = roughness_length
        destination = np.full((grid.nlat, grid.nlon), np.nan, dtype=np.float32)
        reproject(
            source=source_roughness,
            destination=destination,
            src_transform=source_transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=destination_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    # Raster row order is north-to-south; RegularGrid coordinates are ascending.
    result = np.flipud(destination)
    if not np.isfinite(result).all():
        raise ValueError("Land-cover raster leaves uncovered cells in the target grid")
    return result


def surface_roughness_factors(
    land_cover_path: str | Path,
    mapping_path: str | Path,
    grid: RegularGrid,
    *,
    surface_height_m: float = 10.0,
    gradient_height_m: float = 18.0,
) -> np.ndarray:
    """Return grid-aligned factors for downscaling gradient winds to the surface."""

    roughness = roughness_from_land_cover(land_cover_path, mapping_path, grid)
    return power_law_scale_factors(roughness, surface_height_m, gradient_height_m)


class SurfaceRoughness:
    """Callable adapter that converts land-cover data into a surface-downscaling field."""

    def __init__(
        self,
        land_cover_path: str | Path,
        mapping_path: str | Path,
        *,
        surface_height_m: float = 10.0,
        gradient_height_m: float = 18.0,
    ) -> None:
        self.land_cover_path = str(land_cover_path)
        self.mapping_path = str(mapping_path)
        self.surface_height_m = surface_height_m
        self.gradient_height_m = gradient_height_m

    def __call__(self, grid: RegularGrid) -> np.ndarray:
        return surface_roughness_factors(
            self.land_cover_path,
            self.mapping_path,
            grid,
            surface_height_m=self.surface_height_m,
            gradient_height_m=self.gradient_height_m,
        )


DownscalingMethod = Callable[[RegularGrid], np.ndarray]
