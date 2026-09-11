from collections.abc import Callable

import pandas as pd
import pytest

from tc_wind_lib import TrackSet, WindSpeedReference


@pytest.fixture
def track_frame() -> Callable[[], pd.DataFrame]:
    """Return a fresh canonical one-track table."""

    def make() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "track_id": ["storm-1", "storm-1", "storm-1"],
                "time_utc": pd.to_datetime(
                    [
                        "2000-01-01T00:00:00Z",
                        "2000-01-01T03:00:00Z",
                        "2000-01-01T06:00:00Z",
                    ]
                ),
                "year": [2000, 2000, 2000],
                "lat": [10.0, 10.1, 10.2],
                "lon": [120.0, 120.2, 120.4],
                "basin_id": ["WP", "WP", "WP"],
                "max_wind_speed_ms": [45.0, 50.0, 45.0],
                "radius_to_max_winds_km": [30.0, 30.0, 30.0],
                "min_pressure_hpa": [950.0, 940.0, 950.0],
            }
        )

    return make


@pytest.fixture
def multi_trackset(track_frame: Callable[[], pd.DataFrame]) -> Callable[[], TrackSet]:
    """Return a fresh two-event catalogue spanning two years."""

    def make() -> TrackSet:
        second = track_frame()
        second["track_id"] = "storm-2"
        second["time_utc"] = second["time_utc"] + pd.DateOffset(years=1)
        second["year"] = 2001
        second["lat"] += 0.2
        return TrackSet(
            pd.concat([track_frame(), second], ignore_index=True),
            {"source": "test"},
            source="test",
            wind_speed_reference=WindSpeedReference.EARTH_RELATIVE,
            is_synthetic=True,
        )

    return make
