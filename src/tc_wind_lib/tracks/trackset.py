"""The primary catalogue object for canonical tropical-cyclone tracks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import box

from .schema import normalise_track_frame, validate_track_frame


@dataclass(frozen=True)
class TrackSet:
    """Validated GeoDataFrame-backed tracks with lightweight catalogue metadata.

    ``lat`` and ``lon`` are authoritative; ``geometry`` is regenerated from
    them in EPSG:4326 whenever a TrackSet is constructed.
    """

    tracks: gpd.GeoDataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalised = normalise_track_frame(self.tracks)
        validate_track_frame(normalised)
        normalised = normalised.drop(columns="geometry", errors="ignore")
        tracks = gpd.GeoDataFrame(
            normalised,
            geometry=gpd.points_from_xy(normalised["lon"], normalised["lat"]),
            crs="EPSG:4326",
        )
        object.__setattr__(self, "tracks", tracks)

    def __repr__(self) -> str:
        """Return a compact catalogue summary suitable for interactive use."""

        n_observations = len(self.tracks)
        n_storms = len(self.track_ids)
        if n_observations:
            years = self.tracks["year"]
            year_range = f"{int(years.min())}-{int(years.max())}"
        else:
            year_range = "empty"
        if self.metadata:
            metadata = "\n".join(
                f"    {key!r}: {value!r}," for key, value in self.metadata.items()
            )
            metadata_repr = f"{{\n{metadata}\n  }}"
        else:
            metadata_repr = "{}"
        return (
            "TrackSet(\n"
            f"  metadata={metadata_repr},\n"
            f"  storms={n_storms},\n"
            f"  observations={n_observations},\n"
            f"  years={year_range},\n"
            ")"
        )

    @classmethod
    def read_parquet(
        cls, path: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> "TrackSet":
        """Read and validate a processed Parquet or GeoParquet track table.

        The Parquet footer identifies GeoParquet files without reading their
        rows. GeoPandas decodes GeoParquet WKB geometry and retains its CRS;
        ordinary Parquet files continue through the pandas path.
        """

        parquet_file = pq.ParquetFile(path)
        parquet_metadata = parquet_file.metadata.metadata or {}
        is_geoparquet = (
            "geometry" in parquet_file.schema_arrow.names
            and b"geo" in parquet_metadata
        )
        tracks = gpd.read_parquet(path) if is_geoparquet else pd.read_parquet(path)
        return cls(tracks, {} if metadata is None else dict(metadata))

    def with_metadata(self, **metadata: Any) -> "TrackSet":
        """Return a new track set with added or replaced catalogue metadata."""

        return TrackSet(self.tracks, {**self.metadata, **metadata})

    @property
    def track_ids(self) -> pd.Index:
        return pd.Index(self.tracks["track_id"].unique(), name="track_id")

    def get_track(self, track_id: str) -> gpd.GeoDataFrame:
        """Return one track ordered by ``time_utc``."""

        track = self.tracks.loc[self.tracks["track_id"] == track_id].copy()
        if track.empty:
            raise KeyError(f"Unknown track_id: {track_id}")
        return track.reset_index(drop=True)

    def filter_by_bbox(
        self,
        bounds: tuple[float, float, float, float],
        *,
        search_radius_deg: float,
    ) -> "TrackSet":
        """Keep each track's first-to-last potential impact on a bbox.

        An eye observation is considered relevant when it falls within
        ``search_radius_deg`` of the EPSG:4326 bounding box. For each matching
        track, all observations from its first relevant observation through its
        last are retained, including an intervening excursion outside the
        buffered box. This preserves a storm's continuous trajectory when it
        leaves and later returns to the area of interest.

        ``bounds`` are ``(min_lon, min_lat, max_lon, max_lat)``. Version 1
        does not support bounding boxes crossing the antimeridian.
        """

        min_lon, min_lat, max_lon, max_lat = bounds
        if not (-180 <= min_lon <= max_lon <= 180):
            raise ValueError("Longitude bounds must satisfy -180 <= min_lon <= max_lon <= 180")
        if not (-90 <= min_lat <= max_lat <= 90):
            raise ValueError("Latitude bounds must satisfy -90 <= min_lat <= max_lat <= 90")
        if search_radius_deg < 0:
            raise ValueError("search_radius_deg must be non-negative")

        buffered_bbox = box(min_lon, min_lat, max_lon, max_lat).buffer(
            search_radius_deg
        )
        potentially_affecting = self.tracks.geometry.intersects(buffered_bbox)

        slices: list[pd.DataFrame] = []
        for _, track in self.tracks.groupby("track_id", sort=False):
            positions = np.flatnonzero(
                potentially_affecting.loc[track.index].to_numpy()
            )
            if len(positions):
                slices.append(track.iloc[positions[0] : positions[-1] + 1])
        if not slices:
            return TrackSet(self.tracks.iloc[0:0].copy(), self.metadata)
        return TrackSet(pd.concat(slices, ignore_index=True), self.metadata)

    def iter_tracks(self) -> Iterator[tuple[str, gpd.GeoDataFrame]]:
        """Yield ``(track_id, track)`` pairs in stable catalogue order."""

        for track_id, track in self.tracks.groupby("track_id", sort=False):
            yield track_id, track.reset_index(drop=True)
