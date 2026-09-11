"""Evaluate parametric tropical-cyclone winds at arbitrary geographic points."""

import numpy as np

from tc_wind_lib.hazard.grid.geodesic import bearing_and_great_circle_distance
from tc_wind_lib.tracks.source import WindSpeedReference

from .advection import lin_chavas_2012
from .decay import sigmoid_decay
from .profiles import WindProfile, holland_1980


def evaluate_at_points(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    eye_lon: float,
    eye_lat: float,
    max_wind_speed_ms: float,
    radius_to_max_winds_m: float,
    min_pressure_pa: float,
    env_pressure_pa: float,
    track_heading_deg: float,
    translation_speed_ms: float,
    wind_speed_reference: WindSpeedReference = WindSpeedReference.EARTH_RELATIVE,
    profile: WindProfile = holland_1980,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate gradient winds at point coordinates.

    The returned components are ``(u_east_ms, v_north_ms)``. The Lin--Chavas
    advective component is smoothly tapered with the 1,000 km sigmoid;
    the rotational profile is not tapered.
    """

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.ndim != 1 or lats.ndim != 1 or lons.shape != lats.shape:
        raise ValueError("lons and lats must be equal-length one-dimensional arrays")
    if not (0 < max_wind_speed_ms < 130):
        raise ValueError("max_wind_speed_ms must lie between 0 and 130")
    if not (0 < radius_to_max_winds_m < 1_500_000):
        raise ValueError("radius_to_max_winds_m must lie between 0 and 1,500,000")
    if not (75_000 < min_pressure_pa < 102_000):
        raise ValueError("min_pressure_pa must lie between 75,000 and 102,000")
    if env_pressure_pa <= min_pressure_pa:
        raise ValueError("env_pressure_pa must exceed min_pressure_pa")
    if not np.isfinite(translation_speed_ms) or translation_speed_ms < 0:
        raise ValueError("translation_speed_ms must be finite and non-negative")

    azimuth_deg, radius_m = bearing_and_great_circle_distance(lons, lats, eye_lon, eye_lat)
    hemisphere = 1 if eye_lat >= 0 else -1
    advective = lin_chavas_2012(track_heading_deg, translation_speed_ms, hemisphere)
    if wind_speed_reference == WindSpeedReference.EARTH_RELATIVE:
        rotational_max = max_wind_speed_ms - abs(advective)
    elif wind_speed_reference == WindSpeedReference.EYE_RELATIVE:
        rotational_max = max_wind_speed_ms
    else:
        raise ValueError(f"Unknown wind_speed_reference: {wind_speed_reference!r}")
    advective_field = advective * sigmoid_decay(radius_m / 1_000, 500, 0.004)
    rotational_speed = (
        profile(
            radius_m,
            v_max_ms=rotational_max,
            r_max_m=radius_to_max_winds_m,
            min_pressure_pa=min_pressure_pa,
            env_pressure_pa=env_pressure_pa,
            lat_deg=eye_lat,
        )
        if rotational_max > 0
        else np.zeros_like(radius_m)
    )
    angle = np.radians(azimuth_deg + hemisphere * 90)
    u_east = advective_field.real + rotational_speed * np.sin(angle)
    v_north = advective_field.imag + rotational_speed * np.cos(angle)
    if return_components:
        return u_east, v_north
    return np.hypot(u_east, v_north)


def max_over_time(field: np.ndarray) -> np.ndarray:
    """Collapse the time axis by taking the NaN-safe maximum at each grid cell.

    Two-dimensional arrays are treated as already-aggregated spatial fields and are
    returned unchanged. Higher-dimensional arrays are reduced over axis 0.
    """

    field = np.asarray(field)
    if field.ndim <= 2:
        return field.astype(np.float32, copy=False)
    return np.nanmax(field, axis=0).astype(np.float32, copy=False)
