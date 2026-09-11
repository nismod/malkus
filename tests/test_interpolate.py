from collections.abc import Callable

import pandas as pd

from malkus.wind.interpolate import derive_track_motion, interpolate_track


def test_hourly_interpolation_and_motion(track_frame: Callable[[], pd.DataFrame]):
    interpolated = interpolate_track(track_frame())
    assert len(interpolated) == 7
    motion = derive_track_motion(interpolated)
    assert motion.translation_speed_ms.gt(0).all()
    assert motion.translation_heading_deg.notna().all()
