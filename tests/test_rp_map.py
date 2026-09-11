from collections.abc import Callable

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr

from tc_wind_lib import (
    RegularGrid,
    ReturnPeriodMapSet,
    TrackSet,
    WindFootprintSet,
    downscale_winds,
    initialize_wind_footprints,
    return_period_maps,
)


def test_return_period_maps_uses_annual_maxima_and_plotting_position(tmp_path):
    dataset = xr.Dataset(
        {
            "max_wind_speed_ms": (
                ("event", "lat", "lon"),
                np.array([[[10]], [[20]], [[30]]], dtype=np.float32),
            ),
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
    maps = return_period_maps(
        WindFootprintSet(dataset=dataset), [2, 4], output=tmp_path / "rp.zarr"
    )
    assert isinstance(maps, ReturnPeriodMapSet)
    assert maps.is_persisted
    np.testing.assert_allclose(maps.data.values[:, 0, 0], [20, 30])
    assert maps.data.attrs["plotting_position"] == "(n_years + 1) / rank"
    assert (tmp_path / "rp.zarr").exists()
    reopened_maps = ReturnPeriodMapSet.open(tmp_path / "rp.zarr")
    np.testing.assert_allclose(reopened_maps.data.values, maps.data.values)


def test_return_period_maps_matches_reopened_surface_store(
    tmp_path, multi_trackset: Callable[[], TrackSet]
):
    trackset = multi_trackset()
    grid = RegularGrid.from_bbox((119.5, 9.5, 120.5, 10.5), 1.0)
    store = initialize_wind_footprints(
        tmp_path / "surface.zarr", trackset, grid, level="surface"
    )
    root = zarr.open_group(store.path, mode="r+")
    root["max_wind_speed_ms"][:] = np.array([[[10.0]], [[20.0]]])
    root["computed"][:] = True
    persisted_maps = return_period_maps(WindFootprintSet.open(store.path), [3])
    in_memory = WindFootprintSet(
        dataset=xr.Dataset(
            {
                "max_wind_speed_ms": (
                    ("event", "lat", "lon"),
                    np.array([[[10.0]], [[20.0]]]),
                ),
                "computed": (("event",), [True, True]),
            },
            coords={
                "event": trackset.track_ids,
                "year": ("event", [2000, 2001]),
                "lat": grid.lats,
                "lon": grid.lons,
            },
            attrs={"wind_level": "surface", "grid_resolution_deg": grid.resolution},
        )
    )
    in_memory_maps = return_period_maps(in_memory, [3])
    np.testing.assert_allclose(persisted_maps.data.values, [[[20.0]]])
    np.testing.assert_allclose(persisted_maps.data.values, in_memory_maps.data.values)


def test_return_period_map_set_writes_north_up_geotiffs_and_overwrites(tmp_path):
    maps = ReturnPeriodMapSet(
        data=xr.DataArray(
            np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            dims=("return_period", "lat", "lon"),
            coords={
                "return_period": [10.0],
                "lat": [10.0, 11.0],
                "lon": [120.0, 121.0],
            },
            name="max_wind_speed_ms",
            attrs={"grid_resolution_deg": 1.0, "model_family": "emanuel"},
        )
    )
    paths = maps.write_geotiffs(tmp_path / "maps")
    assert paths == [tmp_path / "maps" / "return_period_10_year.tif"]
    with rasterio.open(paths[0]) as raster:
        assert raster.crs.to_epsg() == 4326
        assert raster.transform.e < 0
        np.testing.assert_allclose(raster.read(1), [[3.0, 4.0], [1.0, 2.0]])
        assert raster.tags()["return_period_years"] == "10"
        assert raster.tags()["units"] == "m s-1"
        assert raster.tags()["model_family"] == "emanuel"
    maps.data.values[0, 0, 0] = 9.0
    maps.write_geotiffs(tmp_path / "maps")
    with rasterio.open(paths[0]) as raster:
        np.testing.assert_allclose(raster.read(1), [[3.0, 4.0], [9.0, 2.0]])


def test_downscale_and_return_period_reject_incomplete_stores(
    tmp_path, multi_trackset: Callable[[], TrackSet]
):
    grid = RegularGrid.from_bbox((119.5, 9.5, 121.0, 11.0), 0.25)
    store = initialize_wind_footprints(
        tmp_path / "gradient.zarr", multi_trackset(), grid, level="gradient"
    )
    with pytest.raises(ValueError, match="incomplete"):
        downscale_winds(store, method=lambda _: np.ones(grid.shape))
