import numpy as np
import pandas as pd
import pytest
import rasterio
import geopandas as gpd
import xarray as xr
from rasterio.transform import from_bounds

from tc_wind_lib import (
    RegularGrid,
    TrackSet,
    SurfaceRoughness,
    WindFootprintSet,
    compute_gradient_winds,
    downscale_winds,
    evaluate_at_points,
    initialize_wind_footprints,
    return_period_maps,
)
from tc_wind_lib.hazard.footprint import _compute_event_footprint
from tc_wind_lib.hazard.grid.geodesic import bearing_and_great_circle_distance
from tc_wind_lib.wind.interpolate import derive_track_motion, interpolate_track


def track_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": ["storm-1", "storm-1", "storm-1"],
            "time_utc": pd.to_datetime(
                ["2000-01-01T00:00:00Z", "2000-01-01T03:00:00Z", "2000-01-01T06:00:00Z"]
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


def test_trackset_normalises_and_selects_tracks():
    trackset = TrackSet(track_frame())
    assert isinstance(trackset.tracks, gpd.GeoDataFrame)
    assert trackset.tracks.crs.to_epsg() == 4326
    assert trackset.track_ids.tolist() == ["storm-1"]
    assert str(trackset.tracks.time_utc.dtype) == "datetime64[us, UTC]"
    assert len(trackset.get_track("storm-1")) == 3
    np.testing.assert_allclose(trackset.tracks.geometry.x, trackset.tracks["lon"])
    np.testing.assert_allclose(trackset.tracks.geometry.y, trackset.tracks["lat"])


def test_trackset_repr_summarises_metadata_and_track_statistics():
    frame = pd.concat(
        [track_frame(), track_frame().assign(track_id="storm-2", year=2001)],
        ignore_index=True,
    )
    trackset = TrackSet(frame, metadata={"source": "test", "scenario": "ssp585"})

    assert repr(trackset) == (
        "TrackSet(\n"
        "  metadata={\n"
        "    'source': 'test',\n"
        "    'scenario': 'ssp585',\n"
        "  },\n"
        "  tracks=6 observations,\n"
        "  storms=2,\n"
        "  years=2000-2001,\n"
        ")"
    )


def test_trackset_rejects_duplicate_times():
    frame = track_frame()
    frame.loc[1, "time_utc"] = frame.loc[0, "time_utc"]
    with pytest.raises(ValueError, match="duplicate"):
        TrackSet(frame)


def test_trackset_rebuilds_geometry_from_authoritative_coordinates():
    frame = gpd.GeoDataFrame(
        track_frame(),
        geometry=gpd.points_from_xy([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        crs="EPSG:4326",
    )

    trackset = TrackSet(frame)

    np.testing.assert_allclose(trackset.tracks.geometry.x, frame["lon"])
    np.testing.assert_allclose(trackset.tracks.geometry.y, frame["lat"])


def test_trackset_derives_missing_coordinates_from_geometry():
    frame = track_frame()
    geometry = gpd.points_from_xy(frame["lon"], frame["lat"], crs="EPSG:4326")
    geometry_only = gpd.GeoDataFrame(
        frame.drop(columns=["lat", "lon"]), geometry=geometry, crs="EPSG:4326"
    )

    trackset = TrackSet(geometry_only)

    np.testing.assert_allclose(trackset.tracks["lon"], frame["lon"])
    np.testing.assert_allclose(trackset.tracks["lat"], frame["lat"])


def test_trackset_filter_by_bbox_rejects_antimeridian_box():
    with pytest.raises(ValueError, match="Longitude bounds"):
        TrackSet(track_frame()).filter_by_bbox(
            (170.0, -10.0, -170.0, 10.0), search_radius_deg=1.0
        )


def test_trackset_filter_by_bbox_keeps_intermediate_departure_and_return():
    frame = track_frame().assign(lon=[119.8, 118.0, 120.2])
    trackset = TrackSet(frame)

    selected = trackset.filter_by_bbox(
        (120.0, 9.9, 120.3, 10.3), search_radius_deg=0.3
    )

    assert selected.track_ids.tolist() == ["storm-1"]
    assert selected.tracks["lon"].tolist() == [119.8, 118.0, 120.2]


def test_trackset_filter_by_bbox_returns_empty_when_no_track_can_affect_area():
    selected = TrackSet(track_frame()).filter_by_bbox(
        (0.0, 0.0, 1.0, 1.0), search_radius_deg=0.1
    )

    assert selected.tracks.empty


def test_trackset_read_parquet_reads_and_validates(tmp_path):
    path = tmp_path / "tracks.parquet"
    track_frame().to_parquet(path)

    trackset = TrackSet.read_parquet(path)

    assert isinstance(trackset, TrackSet)
    assert isinstance(trackset.tracks, gpd.GeoDataFrame)
    assert trackset.track_ids.tolist() == ["storm-1"]


def test_trackset_read_parquet_decodes_geoparquet_geometry(tmp_path):
    path = tmp_path / "tracks.geoparquet"
    frame = track_frame()
    gpd.GeoDataFrame(
        frame.drop(columns=["lat", "lon"]),
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    ).to_parquet(path)

    trackset = TrackSet.read_parquet(path)

    np.testing.assert_allclose(trackset.tracks["lon"], frame["lon"])
    np.testing.assert_allclose(trackset.tracks["lat"], frame["lat"])


def test_geodesic_returns_expected_equatorial_distance():
    _, distance_m = bearing_and_great_circle_distance(0.0, 0.0, 1.0, 0.0)
    assert distance_m == pytest.approx(111_195, rel=0.002)


def test_hourly_interpolation_and_motion():
    interpolated = interpolate_track(track_frame())
    assert len(interpolated) == 7
    motion = derive_track_motion(interpolated)
    assert motion.translation_speed_ms.gt(0).all()
    assert motion.translation_heading_deg.notna().all()


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


def multi_trackset() -> TrackSet:
    second = track_frame().copy()
    second["track_id"] = "storm-2"
    second["time_utc"] = second["time_utc"] + pd.DateOffset(years=1)
    second["year"] = 2001
    second["lat"] += 0.2
    return TrackSet(pd.concat([track_frame(), second], ignore_index=True), {"source": "test"})


def test_internal_event_footprint_is_grid_shaped_and_nonzero():
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.1)
    footprint = _compute_event_footprint(track_frame(), grid)
    assert footprint.shape == grid.shape
    assert footprint.dtype == np.float32
    assert footprint.max() > 0
    assert np.isfinite(footprint).all()


def test_gradient_catalogue_has_event_and_year_coordinates():
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    footprints = compute_gradient_winds(multi_trackset(), grid)

    assert isinstance(footprints, WindFootprintSet)
    assert footprints.level == "gradient"
    assert footprints.complete
    assert footprints.data.max_wind_speed_ms.shape == (2, *grid.shape)
    assert footprints.event_ids.tolist() == ["storm-1", "storm-2"]
    assert footprints.data.year.values.tolist() == [2000, 2001]
    assert float(footprints.data.max_wind_speed_ms.max()) > 0


def test_partitioned_zarr_gradient_matches_in_memory_and_rejects_rewrites(tmp_path):
    trackset = multi_trackset()
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    expected = compute_gradient_winds(trackset, grid)
    store = initialize_wind_footprints(tmp_path / "gradient.zarr", trackset, grid, level="gradient")

    first = TrackSet(trackset.tracks.loc[trackset.tracks.track_id == "storm-1"], trackset.metadata)
    second = TrackSet(trackset.tracks.loc[trackset.tracks.track_id == "storm-2"], trackset.metadata)
    compute_gradient_winds(first, grid, output=store)
    assert not store.complete
    compute_gradient_winds(second, grid, output=store)

    assert store.complete
    np.testing.assert_allclose(store.data.max_wind_speed_ms.values, expected.data.max_wind_speed_ms.values)
    with pytest.raises(ValueError, match="already exist"):
        compute_gradient_winds(first, grid, output=store)
    unknown = TrackSet(trackset.tracks.assign(track_id="unknown"), trackset.metadata)
    with pytest.raises(ValueError, match="no events"):
        compute_gradient_winds(unknown, grid, output=store)


def test_downscale_catalogue_to_separate_surface_store(tmp_path):
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
        transform=from_bounds(grid.lons[0] - grid.resolution / 2, grid.lats[0] - grid.resolution / 2, grid.lons[-1] + grid.resolution / 2, grid.lats[-1] + grid.resolution / 2, grid.nlon, grid.nlat),
    ) as dataset:
        dataset.write(data, 1)

    pd.DataFrame({
        "glob_cover_2009_id": [1],
        "roughness_length_m": [0.05],
    }).to_csv(mapping_path, index=False)

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
    written = downscale_winds(gradient, method=lambda _: np.ones(grid.shape), output=surface_store)
    assert written.complete
    np.testing.assert_allclose(
        written.data.max_wind_speed_ms, gradient.data.max_wind_speed_ms
    )


def test_return_period_maps_uses_annual_maxima_and_plotting_position(tmp_path):
    dataset = xr.Dataset(
        {
            "max_wind_speed_ms": (("event", "lat", "lon"), np.array([[[10]], [[20]], [[30]]], dtype=np.float32)),
            "computed": (("event",), [True, True, True]),
        },
        coords={
            "event": ["a", "b", "c"],
            "year": ("event", [2000, 2001, 2002]),
            "lat": [10.0],
            "lon": [120.0],
        },
        attrs={"wind_level": "surface", "grid_resolution_deg": 1.0},
    )
    maps = return_period_maps(WindFootprintSet(dataset=dataset), [2, 4], output=tmp_path / "rp.zarr")

    np.testing.assert_allclose(maps.values[:, 0, 0], [20, 30])
    assert maps.attrs["plotting_position"] == "(n_years + 1) / rank"
    assert (tmp_path / "rp.zarr").exists()


def test_downscale_and_return_period_reject_incomplete_stores(tmp_path):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    store = initialize_wind_footprints(tmp_path / "gradient.zarr", multi_trackset(), grid, level="gradient")

    with pytest.raises(ValueError, match="incomplete"):
        downscale_winds(store, method=lambda _: np.ones(grid.shape))
