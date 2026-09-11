"""Canonical schema and validation for processed tropical-cyclone tracks."""

from collections.abc import Iterable
import logging

import numpy as np
import pandas as pd
import geopandas as gpd


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = frozenset(
    {
        "track_id",
        "time_utc",
        "year",
        "basin_id",
        "max_wind_speed_ms",
        "radius_to_max_winds_km",
        "min_pressure_hpa",
    }
)

OPTIONAL_COLUMNS = frozenset(
    {
        "timestep",
        "sample",
        "source_year",
        "calibration_year",
        "tc_number",
        "ss_category",
        "environmental_pressure_hpa",
        "holland_b",
        "outer_radius_km",
        "translation_speed_ms",
        "translation_heading_deg",
        "storm_motion_u_ms",
        "storm_motion_v_ms",
        "geometry",
        "lon",
        "lat",
    }
)

_NUMERIC_COLUMNS = frozenset(
    {
        "lat",
        "lon",
        "max_wind_speed_ms",
        "radius_to_max_winds_km",
        "min_pressure_hpa",
    }
)


def normalise_track_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical UTC timestamps and latitude/longitude coordinates.

    ``lat`` and ``lon`` are preferred when supplied. Missing coordinate columns
    are derived from point geometry, including GeoParquet inputs that only
    contain a geometry column.
    """

    result = frame.copy()
    if "time_utc" not in result.columns and isinstance(result.index, pd.DatetimeIndex):
        result["time_utc"] = result.index
    if "time_utc" in result.columns:
        result["time_utc"] = pd.to_datetime(result["time_utc"], utc=True)
    result = result.reset_index(drop=True)

    if "lat" not in result.columns or "lon" not in result.columns:
        if "geometry" not in result.columns:
            raise RuntimeError("No geometry information!")
        geometry = gpd.GeoSeries(
            result["geometry"], crs=getattr(frame, "crs", None)
        )
        if geometry.isna().any() or not geometry.geom_type.eq("Point").all():
            raise ValueError("geometry must contain Point values only")
        if geometry.crs is not None and geometry.crs.to_epsg() != 4326:
            geometry = geometry.to_crs("EPSG:4326")
        if "lat" not in result.columns:
            result["lat"] = geometry.y
        if "lon" not in result.columns:
            result["lon"] = geometry.x

    return result.sort_values(["track_id", "time_utc"], kind="stable").reset_index(drop=True)


def validate_track_frame(frame: pd.DataFrame) -> None:
    """Validate the v1 processed-track contract, raising ``ValueError`` on failure."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Track table is missing required columns: {sorted(missing)}")
    if frame.empty:
        return
    if frame["track_id"].isna().any():
        raise ValueError("track_id contains missing values")
    if frame["time_utc"].isna().any():
        raise ValueError("time_utc contains missing or invalid timestamps")
    if not isinstance(frame["time_utc"].dtype, pd.DatetimeTZDtype):
        raise ValueError("time_utc must be timezone-aware")
    for column in _NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{column} must be numeric")
    for column in frame.select_dtypes(include="number"):
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        non_finite_fraction = float((~np.isfinite(values)).mean())
        logger.info(
            "Track column %s non-finite fraction: %.6f",
            column,
            non_finite_fraction,
        )
    if not frame["lat"].between(-90, 90).all():
        raise ValueError("lat must lie between -90 and 90 degrees")
    if not frame["lon"].between(-180, 180).all():
        raise ValueError("lon must lie between -180 and 180 degrees")
    if (frame["max_wind_speed_ms"] <= 0).any():
        raise ValueError("max_wind_speed_ms must be positive")
    if (frame["radius_to_max_winds_km"] <= 0).any():
        raise ValueError("radius_to_max_winds_km must be positive")
    if (frame["min_pressure_hpa"] <= 0).any():
        raise ValueError("min_pressure_hpa must be positive")
    if frame.duplicated(["track_id", "time_utc"]).any():
        raise ValueError("A track cannot contain duplicate time_utc observations")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    """Raise a focused error when an operation needs unavailable track fields."""

    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{context} requires columns: {sorted(missing)}")
