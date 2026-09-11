import pytest

from malkus.hazard.grid.geodesic import bearing_and_great_circle_distance


def test_geodesic_returns_expected_equatorial_distance():
    _, distance_m = bearing_and_great_circle_distance(0.0, 0.0, 1.0, 0.0)
    assert distance_m == pytest.approx(111_195, rel=0.002)
