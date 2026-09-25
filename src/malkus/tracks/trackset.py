"""The primary catalogue object for canonical tropical-cyclone tracks."""

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import box

from .schema import normalise_track_frame, validate_track_frame
from .source import (
    IS_SYNTHETIC_BY_SOURCE,
    WIND_SPEED_REFERENCE_BY_SOURCE,
    TrackSource,
    WindSpeedReference,
)


@dataclass(frozen=True)
class TrackSet:
    """Validated GeoDataFrame-backed tracks with lightweight catalogue metadata.

    ``lat`` and ``lon`` are authoritative; ``geometry`` is regenerated from
    them in EPSG:4326 whenever a TrackSet is constructed.
    """

    tracks: gpd.GeoDataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    source: TrackSource | str | None = None
    wind_speed_reference: WindSpeedReference | str | None = None
    is_synthetic: bool | None = None
    aoi: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if "is_synthetic" in self.metadata:
            raise ValueError("Use the is_synthetic TrackSet field, not metadata")
        normalised = normalise_track_frame(self.tracks)
        validate_track_frame(normalised)
        tracks = gpd.GeoDataFrame(
            normalised,
            geometry=gpd.points_from_xy(normalised["lon"], normalised["lat"]),
            crs="EPSG:4326",
        )
        object.__setattr__(self, "tracks", tracks)
        source = _normalise_source(self.source)
        reference = _normalise_wind_speed_reference(self.wind_speed_reference)
        expected_reference = (
            WIND_SPEED_REFERENCE_BY_SOURCE[source]
            if isinstance(source, TrackSource)
            else None
        )
        if expected_reference is not None:
            if reference is not None and reference != expected_reference:
                raise ValueError(
                    f"{source.value} tracks require {expected_reference.value} wind speeds"
                )
            reference = expected_reference
        expected_is_synthetic = (
            IS_SYNTHETIC_BY_SOURCE[source]
            if isinstance(source, TrackSource)
            else None
        )
        is_synthetic = self.is_synthetic
        if expected_is_synthetic is not None:
            if is_synthetic is not None and is_synthetic != expected_is_synthetic:
                raise ValueError(
                    f"{source.value} tracks require is_synthetic={expected_is_synthetic}"
                )
            is_synthetic = expected_is_synthetic
        elif not isinstance(is_synthetic, bool):
            raise ValueError(
                "Custom or unspecified sources require an explicit boolean is_synthetic"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "wind_speed_reference", reference)
        object.__setattr__(self, "is_synthetic", is_synthetic)

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
            f"  source={self.source},\n"
            f"  metadata={metadata_repr},\n"
            f"  is_synthetic={self.is_synthetic},\n"
            f"  aoi={self.aoi},\n"
            f"  storms={n_storms},\n"
            f"  observations={n_observations},\n"
            f"  years={year_range},\n"
            ")"
        )

    @classmethod
    def read_parquet(
        cls,
        path: str | Path | Sequence[str | Path],
        *,
        metadata: dict[str, Any] | None = None,
        source: TrackSource | str | None = None,
        wind_speed_reference: WindSpeedReference | str | None = None,
        is_synthetic: bool | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        search_radius_deg: float | None = None,
        minimum_max_wind_speed_ms: float | None = None,
        n_years: int | None = None,
    ) -> "TrackSet":
        """Read and filter one or more processed track tables.

        Files are read completely one at a time. When filters are supplied,
        bbox filtering is applied first, followed by peak-wind filtering;
        ``n_years`` is applied after the filtered files are concatenated.
        """
        if bbox is None and search_radius_deg is not None:
            raise ValueError("search_radius_deg requires bbox")
        if bbox is not None and search_radius_deg is None:
            raise ValueError("bbox requires search_radius_deg")

        paths = [path] if isinstance(path, (str, Path)) else list(path)
        if not paths:
            raise ValueError("path must contain at least one path")

        frames: list[pd.DataFrame] = []
        seen_years: set[Any] = set()
        seen_event_ids: set[Any] = set()
        for input_path in paths:
            parquet_file = pq.ParquetFile(input_path)
            parquet_metadata = parquet_file.metadata.metadata or {}
            is_geoparquet = (
                "geometry" in parquet_file.schema_arrow.names
                and b"geo" in parquet_metadata
            )
            raw_tracks = (
                gpd.read_parquet(input_path)
                if is_geoparquet
                else pd.read_parquet(input_path)
            )
            years = set(raw_tracks["year"].dropna().unique().tolist())
            collisions = seen_years.intersection(years)
            if collisions:
                raise ValueError(
                    "Input files contain colliding years: "
                    f"{sorted(collisions)[:5]}"
                )
            seen_years.update(years)
            event_ids = set(raw_tracks["track_id"].dropna().unique().tolist())
            event_collisions = seen_event_ids.intersection(event_ids)
            if event_collisions:
                raise ValueError(
                    "Input files contain colliding event IDs: "
                    f"{sorted(event_collisions)[:5]}"
                )
            seen_event_ids.update(event_ids)

            file_trackset = cls(
                raw_tracks,
                {} if metadata is None else dict(metadata),
                source=source,
                wind_speed_reference=wind_speed_reference,
                is_synthetic=is_synthetic,
                aoi=bbox,
            )
            if bbox is not None:
                file_trackset = file_trackset.filter_by_bbox(
                    bbox, search_radius_deg=search_radius_deg
                )
            if minimum_max_wind_speed_ms is not None:
                file_trackset = file_trackset.filter_by_minimum_max_wind_speed(
                    minimum_max_wind_speed_ms
                )
            frames.append(file_trackset.tracks)

        tracks = pd.concat(frames, ignore_index=True)
        result = cls(
            tracks,
            {} if metadata is None else dict(metadata),
            source=source,
            wind_speed_reference=wind_speed_reference,
            is_synthetic=is_synthetic,
            aoi=bbox,
        )
        if n_years is not None:
            result = result.filter_first_years(n_years)
        return result

    def with_metadata(self, **metadata: Any) -> "TrackSet":
        """Return a new track set with added or replaced catalogue metadata."""

        return self._with_tracks(self.tracks, metadata={**self.metadata, **metadata})

    @property
    def model_family(self) -> str | None:
        """Return the source-model family used for footprint provenance."""

        if self.source is None:
            return None
        return self.source.value if isinstance(self.source, TrackSource) else self.source

    def require_wind_speed_reference(self) -> WindSpeedReference:
        """Return the resolved input convention needed for wind generation."""

        if self.wind_speed_reference is None:
            raise ValueError(
                "Wind generation requires a known source or an explicit wind_speed_reference"
            )
        return self.wind_speed_reference

    @property
    def footprint_metadata(self) -> dict[str, Any]:
        """Return provenance metadata for generated earth-relative footprints."""

        metadata = {**self.metadata, "is_synthetic": self.is_synthetic}
        if self.model_family is not None:
            metadata["model_family"] = self.model_family
        return metadata

    @property
    def track_ids(self) -> pd.Index:
        return pd.Index(self.tracks["track_id"].unique(), name="track_id")

    def get_track(self, track_id: str) -> gpd.GeoDataFrame:
        """Return one track ordered by ``time_utc``."""

        track = self.tracks.loc[self.tracks["track_id"] == track_id].copy()
        if track.empty:
            raise KeyError(f"Unknown track_id: {track_id}")
        return track.reset_index(drop=True)

    def filter_first_tracks(self, n: int) -> "TrackSet":
        """Keep the first ``n`` tracks in canonical track-ID order."""

        _validate_positive_count(n)
        track_ids = self.track_ids[:n]
        tracks = self.tracks.loc[self.tracks["track_id"].isin(track_ids)].copy()
        return self._with_tracks(tracks)

    def filter_first_years(self, n: int) -> "TrackSet":
        """Keep observations with a calendar year less than ``n``."""

        _validate_positive_count(n)
        tracks = self.tracks.loc[self.tracks["year"] < n].copy()
        return self._with_tracks(tracks)

    def filter_by_minimum_max_wind_speed(
        self, minimum_max_wind_speed_ms: float
    ) -> "TrackSet":
        """Keep complete tracks whose peak wind meets a minimum in m/s."""

        if not np.isfinite(minimum_max_wind_speed_ms) or minimum_max_wind_speed_ms < 0:
            raise ValueError("Minimum_max_wind_speed_ms must be finite and non-negative")
        peak_wind_speed = self.tracks.groupby("track_id", sort=False)[
            "max_wind_speed_ms"
        ].max()
        qualifying_track_ids = peak_wind_speed.index[
            peak_wind_speed >= minimum_max_wind_speed_ms
        ]
        tracks = self.tracks.loc[
            self.tracks["track_id"].isin(qualifying_track_ids)
        ].copy()
        return self._with_tracks(tracks)

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
            return self._with_tracks(self.tracks.iloc[0:0].copy())
        return self._with_tracks(pd.concat(slices, ignore_index=True))

    def iter_tracks(self) -> Iterator[tuple[str, gpd.GeoDataFrame]]:
        """Yield ``(track_id, track)`` pairs in stable catalogue order."""

        for track_id, track in self.tracks.groupby("track_id", sort=False):
            yield track_id, track.reset_index(drop=True)

    def _with_tracks(
        self, tracks: pd.DataFrame, *, metadata: dict[str, Any] | None = None
    ) -> "TrackSet":
        return TrackSet(
            tracks,
            self.metadata if metadata is None else metadata,
            source=self.source,
            wind_speed_reference=self.wind_speed_reference,
            is_synthetic=self.is_synthetic,
            aoi=self.aoi,
        )


def _normalise_source(source: TrackSource | str | None) -> TrackSource | str | None:
    if source is None or isinstance(source, TrackSource):
        return source
    try:
        return TrackSource(source.lower())
    except ValueError:
        return source


def _validate_positive_count(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be positive")


def _normalise_wind_speed_reference(
    reference: WindSpeedReference | str | None,
) -> WindSpeedReference | None:
    if reference is None or isinstance(reference, WindSpeedReference):
        return reference
    try:
        return WindSpeedReference(reference.lower())
    except ValueError as error:
        raise ValueError(f"Unknown wind_speed_reference: {reference!r}") from error
