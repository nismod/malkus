"""Regular geographic grids and arbitrary point collections for wind evaluation."""

from dataclasses import dataclass

import numpy as np

from .geodesic import bearing_and_great_circle_distance


@dataclass(frozen=True)
class PointCloud:
    """An ordered collection of geographic evaluation points in EPSG:4326."""

    lats: np.ndarray
    lons: np.ndarray

    def __post_init__(self) -> None:
        lats = np.asarray(self.lats, dtype=float)
        lons = np.asarray(self.lons, dtype=float)
        if lats.ndim != 1 or lons.ndim != 1 or lats.shape != lons.shape:
            raise ValueError("lats and lons must be one-dimensional arrays of equal length")
        if not np.isfinite(lats).all() or not np.isfinite(lons).all():
            raise ValueError("Point coordinates must be finite")
        object.__setattr__(self, "lats", lats)
        object.__setattr__(self, "lons", lons)

    def indices_within_radius(self, lat: float, lon: float, radius_m: float) -> np.ndarray:
        """Return flat point indices within ``radius_m`` of an eye location."""

        if radius_m <= 0:
            raise ValueError("radius_m must be positive")
        _, distance_m = bearing_and_great_circle_distance(self.lons, self.lats, lon, lat)
        return np.flatnonzero(distance_m <= radius_m)


@dataclass(frozen=True)
class RegularGrid:
    """A cell-centred, regular EPSG:4326 grid."""

    lats: np.ndarray
    lons: np.ndarray
    resolution: float

    def __post_init__(self) -> None:
        lats = np.asarray(self.lats, dtype=float)
        lons = np.asarray(self.lons, dtype=float)
        if lats.ndim != 1 or lons.ndim != 1 or not len(lats) or not len(lons):
            raise ValueError("RegularGrid requires non-empty one-dimensional lat and lon coordinates")
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        if len(lats) > 1 and not np.allclose(np.diff(lats), self.resolution):
            raise ValueError("Latitudes must be regularly spaced at resolution")
        if len(lons) > 1 and not np.allclose(np.diff(lons), self.resolution):
            raise ValueError("Longitudes must be regularly spaced at resolution")
        object.__setattr__(self, "lats", lats)
        object.__setattr__(self, "lons", lons)

    @classmethod
    def from_bbox(
        cls, bounds: tuple[float, float, float, float], resolution: float
    ) -> "RegularGrid":
        """Create centres covering ``(min_lon, min_lat, max_lon, max_lat)``."""

        min_lon, min_lat, max_lon, max_lat = bounds
        if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
            raise ValueError("bounds must be ordered EPSG:4326 coordinates within world limits")
        nlon = int(np.ceil((max_lon - min_lon) / resolution))
        nlat = int(np.ceil((max_lat - min_lat) / resolution))
        lons = min_lon + resolution * (0.5 + np.arange(nlon))
        lats = min_lat + resolution * (0.5 + np.arange(nlat))
        return cls(lats=lats, lons=lons, resolution=resolution)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lats), len(self.lons)

    @property
    def nlat(self) -> int:
        return len(self.lats)

    @property
    def nlon(self) -> int:
        return len(self.lons)

    @property
    def flat_points(self) -> PointCloud:
        lons, lats = np.meshgrid(self.lons, self.lats)
        return PointCloud(lats.ravel(), lons.ravel())

    def indices_within_radius(self, lat: float, lon: float, radius_m: float) -> np.ndarray:
        return self.flat_points.indices_within_radius(lat, lon, radius_m)
