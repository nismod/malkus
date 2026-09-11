from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from tc_wind_lib import (
    RegularGrid,
    SurfaceRoughness,
    TrackSet,
    WindFootprintSet,
    WindSpeedReference,
    compute_gradient_winds,
    downscale_winds,
    initialize_wind_footprints,
)
from tc_wind_lib.hazard.footprint import (
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


def test_gradient_catalogue_has_event_and_year_coordinates(
    multi_trackset: Callable[[], TrackSet],
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    footprints = compute_gradient_winds(multi_trackset(), grid)
    assert isinstance(footprints, WindFootprintSet)
    assert footprints.level == "gradient"
    assert footprints.complete
    assert footprints.data.max_wind_speed_ms.shape == (2, *grid.shape)
    assert footprints.event_ids.tolist() == ["storm-1", "storm-2"]
    assert footprints.data.year.values.tolist() == [2000, 2001]
    assert float(footprints.data.max_wind_speed_ms.max()) > 0
    assert footprints.data.attrs["model_family"] == "test"
    assert footprints.data.attrs["is_synthetic"] is True


def test_gradient_winds_require_a_resolved_wind_reference(
    track_frame: Callable[[], pd.DataFrame],
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    trackset = TrackSet(track_frame(), source="custom-model", is_synthetic=True)
    with pytest.raises(ValueError, match="known source or an explicit"):
        compute_gradient_winds(trackset, grid)


def test_gradient_winds_write_optional_qc_parquet_outputs(
    tmp_path, multi_trackset: Callable[[], TrackSet]
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    interpolated_path = tmp_path / "interpolated_tracks.pq"
    summary_path = tmp_path / "storm_qc.pq"
    compute_gradient_winds(
        multi_trackset(),
        grid,
        interpolated_tracks_path=interpolated_path,
        storm_qc_path=summary_path,
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


def test_partitioned_zarr_gradient_matches_in_memory_and_rejects_rewrites(
    tmp_path, multi_trackset: Callable[[], TrackSet]
):
    trackset = multi_trackset()
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    expected = compute_gradient_winds(trackset, grid)
    store = initialize_wind_footprints(
        tmp_path / "gradient.zarr", trackset, grid, level="gradient"
    )
    first = trackset._with_tracks(
        trackset.tracks.loc[trackset.tracks.track_id == "storm-1"]
    )
    second = trackset._with_tracks(
        trackset.tracks.loc[trackset.tracks.track_id == "storm-2"]
    )
    compute_gradient_winds(first, grid, output=store)
    assert not store.complete
    compute_gradient_winds(second, grid, output=store)
    assert store.complete
    np.testing.assert_allclose(
        store.data.max_wind_speed_ms.values, expected.data.max_wind_speed_ms.values
    )
    with pytest.raises(ValueError, match="already exist"):
        compute_gradient_winds(first, grid, output=store)
    unknown = trackset._with_tracks(trackset.tracks.assign(track_id="unknown"))
    with pytest.raises(ValueError, match="no events"):
        compute_gradient_winds(unknown, grid, output=store)


def test_downscale_catalogue_to_separate_surface_store(
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
    gradient = compute_gradient_winds(multi_trackset(), grid)
    downscaled = downscale_winds(
        gradient,
        method=SurfaceRoughness(
            land_cover_path=land_cover_path,
            mapping_path=mapping_path,
        ),
    )
    assert downscaled.level == "surface"
    assert downscaled.event_ids.tolist() == gradient.event_ids.tolist()
    np.testing.assert_allclose(downscaled.data.year, gradient.data.year)
    assert np.isfinite(downscaled.data.max_wind_speed_ms.values).all()
    assert float(downscaled.data.max_wind_speed_ms.max()) <= float(
        gradient.data.max_wind_speed_ms.max()
    )
    surface_store = initialize_wind_footprints(
        tmp_path / "surface.zarr", multi_trackset(), grid, level="surface"
    )
    written = downscale_winds(
        gradient, method=lambda _: np.ones(grid.shape), output=surface_store
    )
    assert written.complete
    np.testing.assert_allclose(
        written.data.max_wind_speed_ms, gradient.data.max_wind_speed_ms
    )
