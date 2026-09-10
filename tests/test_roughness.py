import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

from tc_wind_lib import RegularGrid, surface_roughness_factors
from tc_wind_lib.wind.downscale.roughness import roughness_from_land_cover


def test_land_cover_mapping_is_averaged_onto_grid(tmp_path):
    raster_path = tmp_path / "land-cover.tif"
    mapping_path = tmp_path / "roughness.csv"
    classes = np.array([[11, 11], [210, 210]], dtype=np.uint8)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=classes.dtype,
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 2, 2, 2, 2),
    ) as dataset:
        dataset.write(classes, 1)
    pd.DataFrame(
        {
            "glob_cover_2009_id": [11, 210],
            "roughness_length_m": [0.1, 0.0003],
        }
    ).to_csv(mapping_path, index=False)

    grid = RegularGrid.from_bbox((0, 0, 2, 2), 1.0)
    roughness = roughness_from_land_cover(raster_path, mapping_path, grid)
    factors = surface_roughness_factors(raster_path, mapping_path, grid)

    assert roughness.shape == grid.shape
    assert roughness[0, 0] == 0.0003
    assert roughness[1, 0] == 0.1
    assert np.isfinite(factors).all()
