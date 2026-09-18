from collections.abc import Callable

import pandas as pd
import pytest

from malkus.wind.interpolate import derive_track_motion, interpolate_track


def test_hourly_interpolation_and_motion(track_frame: Callable[[], pd.DataFrame]):
    interpolated = interpolate_track(track_frame())
    assert len(interpolated) == 7
    motion = derive_track_motion(interpolated)
    assert motion.translation_speed_ms.gt(0).all()
    assert motion.translation_heading_deg.notna().all()


def test_adaptive_interpolation_retains_endpoints_and_adapts_to_rmw(
    track_frame: Callable[[], pd.DataFrame],
):
    track = track_frame().assign(radius_to_max_winds_km=[15.0, 15.0, 15.0])
    adaptive = interpolate_track(track, spacing_factor=1.5)
    assert adaptive.time_utc.iloc[0] == track.time_utc.min()
    assert adaptive.time_utc.iloc[-1] == track.time_utc.max()
    assert adaptive.time_utc.is_monotonic_increasing
    if "timestep" in track:
        assert adaptive.timestep.tolist() == list(range(len(adaptive)))

    smaller_rmw = interpolate_track(
        track.assign(radius_to_max_winds_km=7.5), spacing_factor=1.5
    )
    assert len(smaller_rmw) > len(adaptive)


def test_adaptive_interpolation_rejects_invalid_spacing(
    track_frame: Callable[[], pd.DataFrame],
):
    with pytest.raises(ValueError, match="spacing_factor"):
        interpolate_track(track_frame(), spacing_factor=0)
    with pytest.raises(ValueError, match="spacing_factor"):
        interpolate_track(track_frame(), spacing_factor=float("nan"))
