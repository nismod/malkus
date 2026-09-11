"""Time interpolation and eye-motion derivation for canonical tracks."""

import numpy as np
import pandas as pd
import pyproj

from tc_wind_lib.tracks.schema import require_columns


INTERPOLATED_COLUMNS = (
    "lat",
    "lon",
    "min_pressure_hpa",
    "max_wind_speed_ms",
    "radius_to_max_winds_km",
)


def interpolate_track(track: pd.DataFrame, frequency: str = "1h") -> pd.DataFrame:
    """Interpolate a canonical single track to a regular UTC time interval."""

    require_columns(track, ("track_id", "time_utc", *INTERPOLATED_COLUMNS), "interpolate_track")
    if track.empty:
        raise ValueError("No track data")
    track = track.sort_values("time_utc", kind="stable").copy()
    if track["track_id"].nunique() != 1:
        raise ValueError("interpolate_track accepts exactly one track_id")
    if track["time_utc"].duplicated().any():
        raise ValueError("Track contains duplicate time_utc observations")
    if len(track) == 1:
        return track.reset_index(drop=True)

    method = "linear" if len(track) == 2 else "quadratic"
    source = track.set_index("time_utc")
    target_index = pd.date_range(source.index[0], source.index[-1], freq=frequency, tz="UTC")
    combined = source.reindex(source.index.union(target_index)).sort_index()
    combined.loc[:, INTERPOLATED_COLUMNS] = combined.loc[:, INTERPOLATED_COLUMNS].interpolate(
        method=method, limit_area="inside"
    )
    result = combined.reindex(target_index)
    metadata_columns = [column for column in result.columns if column not in INTERPOLATED_COLUMNS]
    result.loc[:, metadata_columns] = result.loc[:, metadata_columns].ffill().bfill()
    if result.loc[:, INTERPOLATED_COLUMNS].isna().any().any():
        raise ValueError("Track interpolation left missing wind or position values")
    result = result.reset_index(names="time_utc")
    if "timestep" in result.columns:
        result["timestep"] = np.arange(len(result), dtype=int)
    return result


def derive_track_motion(track: pd.DataFrame) -> pd.DataFrame:
    """Add heading, speed, and signed speed acceleration to a track."""

    require_columns(track, ("time_utc", "lat", "lon"), "derive_track_motion")
    if len(track) < 2:
        raise ValueError("At least two observations are required to derive track motion")
    result = track.copy()
    geod = pyproj.Geod(ellps="WGS84")
    heading, _, distance_m = geod.inv(
        result["lon"].iloc[:-1].to_numpy(),
        result["lat"].iloc[:-1].to_numpy(),
        result["lon"].iloc[1:].to_numpy(),
        result["lat"].iloc[1:].to_numpy(),
    )
    periods = result["time_utc"].iloc[1:].reset_index(drop=True) - result["time_utc"].iloc[:-1].reset_index(drop=True)
    seconds = periods.dt.total_seconds().to_numpy()
    if (seconds <= 0).any():
        raise ValueError("Track timestamps must be strictly increasing")
    result["translation_heading_deg"] = np.append(heading, heading[-1])
    translation_speed_ms = np.append(distance_m / seconds, distance_m[-1] / seconds[-1])
    result["translation_speed_ms"] = translation_speed_ms
    acceleration = np.zeros(len(result), dtype=float)
    acceleration[:-1] = np.diff(translation_speed_ms) / seconds
    result["translation_acceleration_ms2"] = acceleration
    return result
