import numpy as np
import pytest

from malkus import WindSpeedReference, evaluate_at_points


def test_point_evaluation_exposes_speed_and_components():
    kwargs = dict(
        eye_lon=120.0,
        eye_lat=10.0,
        max_wind_speed_ms=50.0,
        radius_to_max_winds_m=30_000.0,
        min_pressure_pa=94_000.0,
        env_pressure_pa=100_830.0,
        track_heading_deg=45.0,
        translation_speed_ms=5.0,
    )
    speed = evaluate_at_points(np.array([120.2]), np.array([10.0]), **kwargs)
    u_east, v_north = evaluate_at_points(
        np.array([120.2]), np.array([10.0]), return_components=True, **kwargs
    )
    assert speed[0] > 0
    assert speed[0] == pytest.approx(np.hypot(u_east[0], v_north[0]))


def test_point_evaluation_uses_input_wind_reference_for_rotational_profile():
    rotational_maxima: list[float] = []

    def profile(radius_m, *, v_max_ms, **_):
        rotational_maxima.append(v_max_ms)
        return np.full_like(radius_m, v_max_ms)

    kwargs = dict(
        eye_lon=120.0,
        eye_lat=10.0,
        max_wind_speed_ms=50.0,
        radius_to_max_winds_m=30_000.0,
        min_pressure_pa=94_000.0,
        env_pressure_pa=100_830.0,
        track_heading_deg=45.0,
        translation_speed_ms=5.0,
        profile=profile,
    )
    evaluate_at_points(
        np.array([120.2]),
        np.array([10.0]),
        wind_speed_reference=WindSpeedReference.EARTH_RELATIVE,
        **kwargs,
    )
    evaluate_at_points(
        np.array([120.2]),
        np.array([10.0]),
        wind_speed_reference=WindSpeedReference.EYE_RELATIVE,
        **kwargs,
    )
    assert rotational_maxima == pytest.approx([47.2, 50.0])


def test_point_evaluation_uses_advection_only_for_nonpositive_earth_rotation():
    def profile(*_, **__):
        raise AssertionError("The rotational profile should not be evaluated")

    speed = evaluate_at_points(
        np.array([120.2]),
        np.array([10.0]),
        eye_lon=120.0,
        eye_lat=10.0,
        max_wind_speed_ms=10.0,
        radius_to_max_winds_m=30_000.0,
        min_pressure_pa=94_000.0,
        env_pressure_pa=100_830.0,
        track_heading_deg=45.0,
        translation_speed_ms=20.0,
        wind_speed_reference=WindSpeedReference.EARTH_RELATIVE,
        profile=profile,
    )
    assert speed[0] > 0
