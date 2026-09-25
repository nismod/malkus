from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from malkus import (
    RegularGrid,
    SurfaceRoughness,
    TrackSet,
    WindFootprintSet,
    WindSpeedReference,
    compute_winds,
    downscale_winds,
    initialize_wind_footprints,
)
from malkus.hazard.footprint import (
    _compute_event_footprint,
    _prepare_track_for_wind_evaluation,
)


def test_prepared_track_records_fast_motion_qc_and_terminal_acceleration(
    track_frame: Callable[[], pd.DataFrame],
):
    frame = track_frame().assign(
        lon=[120.0, 126.0, 132.0], max_wind_speed_ms=[10.0, 10.0, 10.0]
    )
    prepared = _prepare_track_for_wind_evaluation(
        frame,
        interpolation_frequency="3h",
        wind_speed_reference=WindSpeedReference.EARTH_RELATIVE,
    )
    assert prepared["rotational_wind_nonpositive"].all()
    assert np.isinf(prepared["chi"]).all()
    assert prepared["translation_acceleration_ms2"].iloc[-1] == 0.0


def test_internal_event_footprint_is_grid_shaped_and_nonzero(
    track_frame: Callable[[], pd.DataFrame],
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.1)
    footprint = _compute_event_footprint(track_frame(), grid)
    assert footprint.shape == grid.shape
    assert footprint.dtype == np.float32
    assert footprint.max() > 0
    assert np.isfinite(footprint).all()


def test_wind_catalogue_has_event_and_year_coordinates(
    multi_trackset: Callable[[], TrackSet],
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    footprints = compute_winds(multi_trackset(), grid)
    assert isinstance(footprints, WindFootprintSet)
    assert "wind_level" not in footprints.data.attrs
    assert footprints.complete
    assert footprints.data.max_wind_speed_ms.shape == (2, *grid.shape)
    assert footprints.event_ids.tolist() == ["storm-1", "storm-2"]
    assert footprints.data.year.values.tolist() == [2000, 2001]
    assert float(footprints.data.max_wind_speed_ms.max()) > 0
    assert footprints.data.attrs["model_family"] == "test"
    assert footprints.data.attrs["is_synthetic"] is True


def test_parallel_winds_match_serial_winds(multi_trackset: Callable[[], TrackSet]):
    trackset = multi_trackset()
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)

    serial = compute_winds(trackset, grid, n_workers=1, batch_size=1)
    parallel = compute_winds(trackset, grid, n_workers=2, batch_size=1)

    assert parallel.event_ids.tolist() == serial.event_ids.tolist()
    np.testing.assert_allclose(
        parallel.data.max_wind_speed_ms, serial.data.max_wind_speed_ms
    )


def test_winds_require_a_resolved_wind_reference(
    track_frame: Callable[[], pd.DataFrame],
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    trackset = TrackSet(track_frame(), source="custom-model", is_synthetic=True)
    with pytest.raises(ValueError, match="known source or an explicit"):
        compute_winds(trackset, grid)


def test_winds_write_optional_qc_parquet_outputs(
    tmp_path, multi_trackset: Callable[[], TrackSet]
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    interpolated_path = tmp_path / "interpolated_tracks.pq"
    summary_path = tmp_path / "storm_qc.pq"
    compute_winds(
        multi_trackset(),
        grid,
        interpolated_tracks_path=interpolated_path,
        storm_qc_path=summary_path,
        n_workers=2,
        batch_size=1,
    )
    interpolated = pd.read_parquet(interpolated_path)
    summary = pd.read_parquet(summary_path)
    assert {"chi", "translation_acceleration_ms2", "model_family"}.issubset(
        interpolated.columns
    )
    assert "geometry" not in interpolated.columns
    assert {
        "track_id",
        "year",
        "model_family",
        "max_advective_wind_speed_ms",
        "max_rotational_max_wind_speed_ms",
        "max_chi",
        "max_abs_translation_acceleration_ms2",
        "n_nonpositive_rotational_wind_observations",
    }.issubset(summary.columns)
    assert set(summary["model_family"]) == {"test"}


def test_partitioned_wind_fields_match_in_memory_and_reject_rewrites(
    tmp_path, multi_trackset: Callable[[], TrackSet]
):
    trackset = multi_trackset()
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    expected = compute_winds(trackset, grid)
    store = initialize_wind_footprints(tmp_path / "winds.zarr", trackset, grid)
    first = trackset._with_tracks(
        trackset.tracks.loc[trackset.tracks.track_id == "storm-1"]
    )
    second = trackset._with_tracks(
        trackset.tracks.loc[trackset.tracks.track_id == "storm-2"]
    )
    compute_winds(first, grid, output=store)
    assert not store.complete
    compute_winds(second, grid, output=store)
    assert store.complete
    np.testing.assert_allclose(
        store.data.max_wind_speed_ms.values, expected.data.max_wind_speed_ms.values
    )
    with pytest.raises(ValueError, match="already exist"):
        compute_winds(first, grid, output=store)
    unknown = trackset._with_tracks(trackset.tracks.assign(track_id="unknown"))
    with pytest.raises(ValueError, match="no events"):
        compute_winds(unknown, grid, output=store)


def test_downscale_catalogue_to_separate_store(
    tmp_path, multi_trackset: Callable[[], TrackSet]
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.1)
    land_cover_path = tmp_path / "land-cover.tif"
    mapping_path = tmp_path / "roughness.csv"
    data = np.ones((grid.nlat, grid.nlon), dtype=np.uint8)
    with rasterio.open(
        land_cover_path,
        "w",
        driver="GTiff",
        height=grid.nlat,
        width=grid.nlon,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_bounds(
            grid.lons[0] - grid.resolution / 2,
            grid.lats[0] - grid.resolution / 2,
            grid.lons[-1] + grid.resolution / 2,
            grid.lats[-1] + grid.resolution / 2,
            grid.nlon,
            grid.nlat,
        ),
    ) as dataset:
        dataset.write(data, 1)
    pd.DataFrame(
        {"glob_cover_2009_id": [1], "roughness_length_m": [0.05]}
    ).to_csv(mapping_path, index=False)
    wind_footprints = compute_winds(multi_trackset(), grid)
    downscaled = downscale_winds(
        wind_footprints,
        method=SurfaceRoughness(
            land_cover_path=land_cover_path,
            mapping_path=mapping_path,
        ),
    )
    assert "wind_level" not in downscaled.data.attrs
    assert downscaled.event_ids.tolist() == wind_footprints.event_ids.tolist()
    np.testing.assert_allclose(downscaled.data.year, wind_footprints.data.year)
    assert np.isfinite(downscaled.data.max_wind_speed_ms.values).all()
    assert float(downscaled.data.max_wind_speed_ms.max()) <= float(
        wind_footprints.data.max_wind_speed_ms.max()
    )
    downscaled_store = initialize_wind_footprints(
        tmp_path / "downscaled.zarr", multi_trackset(), grid
    )
    written = downscale_winds(
        wind_footprints, method=lambda _: np.ones(grid.shape), output=downscaled_store
    )
    assert written.complete
    np.testing.assert_allclose(
        written.data.max_wind_speed_ms, wind_footprints.data.max_wind_speed_ms
    )
