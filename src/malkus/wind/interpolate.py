"""Time interpolation and eye-motion derivation for canonical tracks."""

import numpy as np
import pandas as pd
import pyproj

from malkus.tracks.schema import require_columns


INTERPOLATED_COLUMNS = (
    "lat",
    "lon",
    "min_pressure_hpa",
    "max_wind_speed_ms",
    "radius_to_max_winds_km",
)


def interpolate_track(
    track: pd.DataFrame,
    frequency: str = "1h",
    *,
    spacing_factor: float | None = None,
) -> pd.DataFrame:
    """Interpolate a track on a regular time or adaptive spatial interval.

    When ``spacing_factor`` is supplied, successive eye positions are spaced
    by approximately ``spacing_factor * radius_to_max_winds``.  The regular
    frequency path is retained when it is ``None``.
    """

    require_columns(track, ("track_id", "time_utc", *INTERPOLATED_COLUMNS), "interpolate_track")
    if track.empty:
        raise ValueError("No track data")
    track = track.sort_values("time_utc", kind="stable").copy()
    if track["track_id"].nunique() != 1:
        raise ValueError("interpolate_track accepts exactly one track_id")
    if track["time_utc"].duplicated().any():
        raise ValueError("Track contains duplicate time_utc observations")
    if spacing_factor is not None:
        if not np.isfinite(spacing_factor) or spacing_factor <= 0:
            raise ValueError("spacing_factor must be finite and positive")
        return _interpolate_track_by_distance(track, spacing_factor)
    if len(track) == 1:
        return track.reset_index(drop=True)

    source = track.set_index("time_utc")
    target_index = pd.date_range(source.index[0], source.index[-1], freq=frequency, tz="UTC")
    combined = source.reindex(source.index.union(target_index)).sort_index()

    # TODO: Consider Spherical linear interpolation (SLERP) for position and
    # Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) for other vars
    combined.loc[:, INTERPOLATED_COLUMNS] = combined.loc[:, INTERPOLATED_COLUMNS].interpolate(
        method="linear", limit_area="inside"
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


def _interpolate_track_by_distance(
    track: pd.DataFrame, spacing_factor: float
) -> pd.DataFrame:
    """
    Interpolate a track using an RMW-scaled eye-travel distance. With
    evaluation positions RMW * spacing_factor, piecewise linearly interpolate
    meteorological variables.
    """

    if len(track) == 1:
        return track.reset_index(drop=True)

    lons = track["lon"].to_numpy(dtype=float)
    lats = track["lat"].to_numpy(dtype=float)
    rmw_m = track["radius_to_max_winds_km"].to_numpy(dtype=float) * 1_000.0
    geod = pyproj.Geod(ellps="WGS84")
    azimuth, _, distances = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    if not np.isfinite(rmw_m).all() or (rmw_m <= 0).any():
        raise ValueError("radius_to_max_winds_km must be finite and positive")

    points: list[tuple[int, float]] = [(0, 0.0)]
    accumulated = 0.0
    segment = 0
    fraction = 0.0
    while segment < len(distances):
        segment_distance = float(distances[segment])
        if segment_distance <= 0:
            segment += 1
            fraction = 0.0
            continue

        local_rmw = rmw_m[segment] + fraction * (rmw_m[segment + 1] - rmw_m[segment])
        target = spacing_factor * local_rmw
        remaining = segment_distance * (1.0 - fraction)
        if accumulated >= target:
            point = (segment, fraction)
            if points[-1] != point:
                points.append(point)
            accumulated = 0.0
            continue
        if accumulated + remaining >= target:
            travel = target - accumulated
            fraction += travel / segment_distance
            fraction = min(fraction, 1.0)
            points.append((segment, fraction))
            accumulated = 0.0
            if fraction >= 1.0 - 1e-12:
                segment += 1
                fraction = 0.0
        else:
            accumulated += remaining
            segment += 1
            fraction = 0.0

    if points[-1] != (len(track) - 2, 1.0):
        points.append((len(track) - 2, 1.0))

    rows = []
    for segment, fraction in points:
        left = track.iloc[segment]
        right = track.iloc[min(segment + 1, len(track) - 1)]
        row = left.copy()
        for column in INTERPOLATED_COLUMNS:
            if column in ("lat", "lon"):
                continue
            row[column] = left[column] + fraction * (right[column] - left[column])
        lon, lat, _ = geod.fwd(
            lons[segment], lats[segment], azimuth[segment], distances[segment] * fraction
        )
        row["lon"] = lon
        row["lat"] = lat
        timestamp = left["time_utc"] + fraction * (
            right["time_utc"] - left["time_utc"]
        )
        row["time_utc"] = timestamp
        rows.append(row)

    result = pd.DataFrame(rows).reset_index(drop=True)
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
