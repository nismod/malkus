from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from malkus import TrackSet, TrackSource, WindSpeedReference


def test_trackset_normalises_and_selects_tracks(track_frame: Callable[[], pd.DataFrame]):
    trackset = TrackSet(track_frame(), is_synthetic=False)
    assert isinstance(trackset.tracks, gpd.GeoDataFrame)
    assert trackset.tracks.crs.to_epsg() == 4326
    assert trackset.track_ids.tolist() == ["storm-1"]
    assert str(trackset.tracks.time_utc.dtype) == "datetime64[us, UTC]"
    assert len(trackset.get_track("storm-1")) == 3
    np.testing.assert_allclose(trackset.tracks.geometry.x, trackset.tracks["lon"])
    np.testing.assert_allclose(trackset.tracks.geometry.y, trackset.tracks["lat"])


def test_trackset_rejects_duplicate_times(track_frame: Callable[[], pd.DataFrame]):
    frame = track_frame()
    frame.loc[1, "time_utc"] = frame.loc[0, "time_utc"]
    with pytest.raises(ValueError, match="duplicate"):
        TrackSet(frame, is_synthetic=False)


def test_trackset_rebuilds_geometry_from_authoritative_coordinates(
    track_frame: Callable[[], pd.DataFrame],
):
    frame = gpd.GeoDataFrame(
        track_frame(),
        geometry=gpd.points_from_xy([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        crs="EPSG:4326",
    )
    trackset = TrackSet(frame, is_synthetic=False)
    np.testing.assert_allclose(trackset.tracks.geometry.x, frame["lon"])
    np.testing.assert_allclose(trackset.tracks.geometry.y, frame["lat"])


def test_trackset_derives_missing_coordinates_from_geometry(
    track_frame: Callable[[], pd.DataFrame],
):
    frame = track_frame()
    geometry = gpd.points_from_xy(frame["lon"], frame["lat"], crs="EPSG:4326")
    geometry_only = gpd.GeoDataFrame(
        frame.drop(columns=["lat", "lon"]), geometry=geometry, crs="EPSG:4326"
    )
    trackset = TrackSet(geometry_only, is_synthetic=False)
    np.testing.assert_allclose(trackset.tracks["lon"], frame["lon"])
    np.testing.assert_allclose(trackset.tracks["lat"], frame["lat"])


def test_trackset_filters_first_complete_tracks(track_frame: Callable[[], pd.DataFrame]):
    frame = pd.concat(
        [
            track_frame().assign(track_id="storm-2"),
            track_frame().assign(track_id="storm-1"),
        ],
        ignore_index=True,
    )
    trackset = TrackSet(frame, metadata={"source": "test"}, is_synthetic=True)
    selected = trackset.filter_first_tracks(1)
    assert selected.track_ids.tolist() == ["storm-1"]
    assert len(selected.tracks) == 3
    assert selected.metadata == {"source": "test"}
    assert selected.is_synthetic
    assert selected.with_metadata(scenario="ssp585").is_synthetic
    assert len(trackset.filter_first_tracks(10).track_ids) == 2
    with pytest.raises(ValueError, match="positive"):
        trackset.filter_first_tracks(0)
    with pytest.raises(ValueError, match="positive"):
        trackset.filter_first_tracks(-1)
    with pytest.raises(TypeError, match="integer"):
        trackset.filter_first_tracks(1.5)


def test_trackset_filters_observations_from_first_calendar_years(
    track_frame: Callable[[], pd.DataFrame],
):
    frame = pd.concat(
        [
            track_frame().assign(track_id="storm-2002", year=2002),
            track_frame().assign(track_id="storm-2000", year=2000),
            track_frame().assign(track_id="storm-2001", year=2001),
        ],
        ignore_index=True,
    )
    trackset = TrackSet(frame, metadata={"source": "test"}, is_synthetic=True)
    selected = trackset.filter_first_years(2002)
    assert selected.tracks["year"].unique().tolist() == [2000, 2001]
    assert selected.track_ids.tolist() == ["storm-2000", "storm-2001"]
    assert selected.metadata == {"source": "test"}
    assert selected.is_synthetic
    assert len(trackset.filter_first_years(2010).track_ids) == 3
    with pytest.raises(ValueError, match="positive"):
        trackset.filter_first_years(0)
    with pytest.raises(TypeError, match="integer"):
        trackset.filter_first_years(1.5)


def test_trackset_resolves_known_source_properties_and_rejects_conflicts(
    track_frame: Callable[[], pd.DataFrame],
):
    expected_properties = {
        TrackSource.IBTRACS: (WindSpeedReference.EARTH_RELATIVE, False),
        TrackSource.IRIS: (WindSpeedReference.EARTH_RELATIVE, True),
        TrackSource.STORM: (WindSpeedReference.EARTH_RELATIVE, True),
        TrackSource.CHAZ: (WindSpeedReference.EYE_RELATIVE, True),
        TrackSource.EMANUEL: (WindSpeedReference.EYE_RELATIVE, True),
    }
    for source, (reference, is_synthetic) in expected_properties.items():
        trackset = TrackSet(track_frame(), source=source)
        assert trackset.wind_speed_reference == reference
        assert trackset.is_synthetic is is_synthetic
        assert trackset.model_family == source.value
    with pytest.raises(ValueError, match="require earth_relative"):
        TrackSet(
            track_frame(),
            source=TrackSource.IRIS,
            wind_speed_reference=WindSpeedReference.EYE_RELATIVE,
        )
    with pytest.raises(ValueError, match="require is_synthetic=False"):
        TrackSet(track_frame(), source=TrackSource.IBTRACS, is_synthetic=True)


def test_trackset_requires_explicit_synthetic_status_for_custom_or_missing_sources(
    track_frame: Callable[[], pd.DataFrame],
):
    with pytest.raises(ValueError, match="explicit boolean is_synthetic"):
        TrackSet(track_frame())
    with pytest.raises(ValueError, match="explicit boolean is_synthetic"):
        TrackSet(track_frame(), source="custom-model")
    with pytest.raises(ValueError, match="explicit boolean is_synthetic"):
        TrackSet(track_frame(), source="custom-model", is_synthetic=1)
    with pytest.raises(ValueError, match="is_synthetic TrackSet field"):
        TrackSet(track_frame(), metadata={"is_synthetic": True}, is_synthetic=True)


def test_trackset_filters_complete_tracks_by_peak_wind_speed(
    track_frame: Callable[[], pd.DataFrame],
):
    weak = track_frame().assign(track_id="weak", max_wind_speed_ms=[8.0, 14.0, 12.0])
    boundary = track_frame().assign(
        track_id="boundary", max_wind_speed_ms=[8.0, 15.0, 12.0]
    )
    trackset = TrackSet(
        pd.concat([track_frame(), weak, boundary], ignore_index=True),
        metadata={"source": "test"},
        is_synthetic=True,
    )
    selected = trackset.filter_by_minimum_max_wind_speed(15.0)
    assert set(selected.track_ids) == {"storm-1", "boundary"}
    assert selected.tracks.groupby("track_id").size().to_dict() == {
        "boundary": 3,
        "storm-1": 3,
    }
    assert selected.metadata == {"source": "test"}


def test_trackset_peak_wind_filter_handles_empty_results_and_invalid_thresholds(
    track_frame: Callable[[], pd.DataFrame],
):
    trackset = TrackSet(track_frame(), metadata={"source": "test"}, is_synthetic=True)
    empty = trackset.filter_by_minimum_max_wind_speed(100.0)
    assert empty.tracks.empty
    assert empty.metadata == {"source": "test"}
    for threshold in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and non-negative"):
            trackset.filter_by_minimum_max_wind_speed(threshold)


def test_trackset_filter_by_bbox_rejects_antimeridian_box(
    track_frame: Callable[[], pd.DataFrame],
):
    with pytest.raises(ValueError, match="Longitude bounds"):
        TrackSet(track_frame(), is_synthetic=False).filter_by_bbox(
            (170.0, -10.0, -170.0, 10.0), search_radius_deg=1.0
        )


def test_trackset_filter_by_bbox_keeps_intermediate_departure_and_return(
    track_frame: Callable[[], pd.DataFrame],
):
    trackset = TrackSet(
        track_frame().assign(lon=[119.8, 118.0, 120.2]), is_synthetic=False
    )
    selected = trackset.filter_by_bbox(
        (120.0, 9.9, 120.3, 10.3), search_radius_deg=0.3
    )
    assert selected.track_ids.tolist() == ["storm-1"]
    assert selected.tracks["lon"].tolist() == [119.8, 118.0, 120.2]


def test_trackset_filter_by_bbox_returns_empty_when_no_track_can_affect_area(
    track_frame: Callable[[], pd.DataFrame],
):
    selected = TrackSet(track_frame(), is_synthetic=False).filter_by_bbox(
        (0.0, 0.0, 1.0, 1.0), search_radius_deg=0.1
    )
    assert selected.tracks.empty


def test_trackset_read_parquet_reads_and_validates(
    tmp_path, track_frame: Callable[[], pd.DataFrame]
):
    path = tmp_path / "tracks.parquet"
    track_frame().to_parquet(path)
    trackset = TrackSet.read_parquet(path, is_synthetic=False)
    assert isinstance(trackset, TrackSet)
    assert isinstance(trackset.tracks, gpd.GeoDataFrame)
    assert trackset.track_ids.tolist() == ["storm-1"]


def test_trackset_read_parquet_decodes_geoparquet_geometry(
    tmp_path, track_frame: Callable[[], pd.DataFrame]
):
    path = tmp_path / "tracks.geoparquet"
    frame = track_frame()
    gpd.GeoDataFrame(
        frame.drop(columns=["lat", "lon"]),
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    ).to_parquet(path)
    trackset = TrackSet.read_parquet(path, is_synthetic=False)
    np.testing.assert_allclose(trackset.tracks["lon"], frame["lon"])
    np.testing.assert_allclose(trackset.tracks["lat"], frame["lat"])
