"""Event wind-footprint collections and their serial calculation."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing as mp
from pathlib import Path
from textwrap import indent
from typing import Any, Iterator

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from malkus.hazard.grid.grid import RegularGrid
from malkus.tracks.source import WindSpeedReference
from malkus.tracks.trackset import TrackSet
from malkus.wind.advection import ADVECTIVE_SPEED_FRACTION
from malkus.wind.env_pressure import ENV_PRESSURE
from malkus.wind.evaluate import evaluate_at_points
from malkus.wind.interpolate import derive_track_motion, interpolate_track
from malkus.wind.profiles import WindProfile, holland_1980


WIND_VARIABLE = "max_wind_speed_ms"


def _evaluate_event_batch(
    batch: list[tuple[str, pd.DataFrame]],
    grid: RegularGrid,
    profile: WindProfile,
    evaluation_radius_m: float,
    wind_speed_reference: WindSpeedReference,
) -> list[tuple[str, np.ndarray]]:
    return [
        (
            event_id,
            _evaluate_prepared_event_footprint(
                track,
                grid,
                profile=profile,
                evaluation_radius_m=evaluation_radius_m,
                wind_speed_reference=wind_speed_reference,
            ),
        )
        for event_id, track in batch
    ]


def _event_batches(
    events: list[tuple[str, pd.DataFrame]], batch_size: int
) -> Iterator[list[tuple[str, pd.DataFrame]]]:
    for start in range(0, len(events), batch_size):
        yield events[start : start + batch_size]


def _iter_event_footprints(
    prepared_tracks: list[tuple[str, pd.DataFrame]],
    grid: RegularGrid,
    profile: WindProfile,
    evaluation_radius_m: float,
    wind_speed_reference: WindSpeedReference,
    *,
    n_workers: int,
    batch_size: int,
) -> Iterator[tuple[str, np.ndarray]]:
    batches = _event_batches(prepared_tracks, batch_size)
    if n_workers == 1:
        for batch in batches:
            yield from _evaluate_event_batch(
                batch, grid, profile, evaluation_radius_m, wind_speed_reference
            )
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as executor:
        for result_batch in executor.map(
            _evaluate_event_batch,
            batches,
            (grid for _ in range((len(prepared_tracks) + batch_size - 1) // batch_size)),
            (profile for _ in range((len(prepared_tracks) + batch_size - 1) // batch_size)),
            (evaluation_radius_m for _ in range((len(prepared_tracks) + batch_size - 1) // batch_size)),
            (wind_speed_reference for _ in range((len(prepared_tracks) + batch_size - 1) // batch_size)),
        ):
            yield from result_batch


def _validate_parallel_options(n_workers: int, batch_size: int) -> None:
    if isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError("n_workers must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


@dataclass(frozen=True)
class WindFootprintSet:
    """Maximum wind footprints, either in memory or stored in Zarr.

    The underlying dataset has dimensions ``(event, lat, lon)``. ``event`` is
    the source ``TrackSet.track_id`` and ``computed(event)`` records whether a
    worker has successfully written the event's complete footprint.
    """

    dataset: xr.Dataset | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        if (self.dataset is None) == (self.path is None):
            raise ValueError("Provide exactly one of dataset or path")
        if self.dataset is not None:
            _validate_footprint_dataset(self.dataset)

    def __repr__(self) -> str:
        """Return the backing store location and xarray dataset summary."""

        path = None if self.path is None else str(self.path)
        return (
            "WindFootprintSet(\n"
            f"  path={path!r},\n"
            "  data=\n"
            f"{indent(repr(self.data), '    ')}\n"
            ")"
        )

    @classmethod
    def open(cls, path: str | Path) -> "WindFootprintSet":
        """Open an existing Zarr footprint store without loading wind fields."""

        instance = cls(path=Path(path))
        _validate_footprint_dataset(instance.data)
        return instance

    @property
    def data(self) -> xr.Dataset:
        """Return the underlying xarray dataset."""

        if self.dataset is not None:
            return self.dataset
        assert self.path is not None
        return xr.open_zarr(self.path, consolidated=False, mask_and_scale=False)

    @property
    def is_persisted(self) -> bool:
        return self.path is not None

    @property
    def grid(self) -> RegularGrid:
        data = self.data
        return RegularGrid(
            lats=data["lat"].values,
            lons=data["lon"].values,
            resolution=float(data.attrs["grid_resolution_deg"]),
        )

    @property
    def event_ids(self) -> np.ndarray:
        return self.data["event"].values.astype(str)

    @property
    def complete(self) -> bool:
        return bool(np.asarray(self.data["computed"].values).all())

    def require_complete(self) -> None:
        """Raise when a partial store is used as a downstream input."""

        data = self.data
        missing = data["event"].values[~np.asarray(data["computed"].values)]
        if len(missing):
            preview = ", ".join(map(str, missing[:5]))
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(f"Wind footprints are incomplete; missing events: {preview}{suffix}")


def initialize_wind_footprints(
    path: str | Path,
    trackset: TrackSet,
    grid: RegularGrid,
    *,
    metadata: dict[str, Any] | None = None,
) -> WindFootprintSet:
    """Create an empty event-footprint Zarr store for external worker writes."""

    path = Path(path)

    event_ids, years = _event_metadata(trackset)
    attrs = {**trackset.footprint_metadata, **({} if metadata is None else metadata)}
    attrs.update(
        {
            "grid_resolution_deg": grid.resolution,
        }
    )
    root = zarr.open_group(path, mode="w")
    root.attrs.update(attrs)
    root.create_dataset(
        WIND_VARIABLE,
        shape=(len(event_ids), grid.nlat, grid.nlon),
        chunks=(1, grid.nlat, grid.nlon),
        dtype="float32",
        fill_value=0.0,
    )
    root[WIND_VARIABLE].attrs["_ARRAY_DIMENSIONS"] = ["event", "lat", "lon"]
    root.create_dataset(
        "computed",
        data=np.zeros(len(event_ids), dtype=bool),
    )
    root["computed"].attrs["_ARRAY_DIMENSIONS"] = ["event"]
    _write_coordinate(root, "event", _unicode_array(event_ids), ["event"])
    _write_coordinate(root, "year", years.astype(np.int64), ["event"])
    _write_coordinate(root, "lat", grid.lats, ["lat"])
    _write_coordinate(root, "lon", grid.lons, ["lon"])
    return WindFootprintSet.open(path)


def compute_winds(
    trackset: TrackSet,
    grid: RegularGrid,
    *,
    output: WindFootprintSet | None = None,
    profile: WindProfile = holland_1980,
    interpolation_frequency: str = "1h",
    interpolation_spacing_factor: float | None = None,
    evaluation_radius_m: float = 1_000_000,
    interpolated_tracks_path: str | Path | None = None,
    storm_qc_path: str | Path | None = None,
    n_workers: int = 1,
    batch_size: int = 32,
) -> WindFootprintSet:
    """Compute maximum wind footprints for every track in a TrackSet.

    With no ``output``, return an in-memory collection. With an initialized
    store, write only the supplied tracks by matching ``track_id``.
    This makes independent external event partitions safe to calculate.
    """

    _validate_parallel_options(n_workers, batch_size)
    wind_speed_reference = trackset.require_wind_speed_reference()
    prepared_tracks = [
        (
            event_id,
            _prepare_track_for_wind_evaluation(
                track,
                interpolation_frequency=interpolation_frequency,
                interpolation_spacing_factor=interpolation_spacing_factor,
                wind_speed_reference=wind_speed_reference,
            ),
        )
        for event_id, track in trackset.iter_tracks()
    ]
    _write_qc_outputs(
        prepared_tracks,
        model_family=trackset.model_family,
        interpolated_tracks_path=interpolated_tracks_path,
        storm_qc_path=storm_qc_path,
    )

    if output is None:
        event_ids, years = _event_metadata(trackset)
        fields = [footprint for _, footprint in _iter_event_footprints(
            prepared_tracks, grid, profile, evaluation_radius_m,
            wind_speed_reference, n_workers=n_workers, batch_size=batch_size,
        )]
        array = (
            np.stack(fields).astype(np.float32, copy=False)
            if fields
            else np.empty((0, *grid.shape), dtype=np.float32)
        )
        return WindFootprintSet(
            dataset=_footprint_dataset(
                array,
                event_ids,
                years,
                grid,
                attrs={
                    **trackset.footprint_metadata,
                    "interpolation_frequency": interpolation_frequency,
                    "interpolation_spacing_factor": interpolation_spacing_factor,
                    "evaluation_radius_m": evaluation_radius_m,
                    "wind_profile": getattr(profile, "__name__", type(profile).__name__),
                },
            )
        )

    _validate_store_for_tracks(output, trackset, grid)
    assert output.path is not None
    event_indices = {event_id: index for index, event_id in enumerate(output.event_ids)}
    computed = np.asarray(output.data["computed"].values)
    requested = trackset.track_ids.to_numpy(dtype=str)
    already_computed = [event_id for event_id in requested if computed[event_indices[event_id]]]
    if already_computed:
        raise ValueError(f"Footprints already exist for events: {already_computed[:5]}")

    root = zarr.open_group(output.path, mode="r+")
    _set_wind_provenance(
        root,
        profile=profile,
        interpolation_frequency=interpolation_frequency,
        interpolation_spacing_factor=interpolation_spacing_factor,
        evaluation_radius_m=evaluation_radius_m,
    )
    for event_id, footprint in _iter_event_footprints(
        prepared_tracks, grid, profile, evaluation_radius_m,
        wind_speed_reference, n_workers=n_workers, batch_size=batch_size,
    ):
        event_index = event_indices[event_id]
        root[WIND_VARIABLE][event_index] = footprint
        root["computed"][event_index] = True
    return output


def downscale_winds(
    input_footprints: WindFootprintSet,
    *,
    method: Any,
    output: WindFootprintSet | None = None,
) -> WindFootprintSet:
    """Apply a static grid-aligned downscaling method to wind footprints."""

    input_footprints.require_complete()
    if not callable(method):
        raise TypeError("method must be a callable that returns grid-aligned factors")
    factors = np.asarray(method(input_footprints.grid), dtype=np.float32)
    if factors.shape != input_footprints.grid.shape:
        raise ValueError("Downscaling factors must have the footprint grid shape")
    if not np.isfinite(factors).all() or (factors < 0).any():
        raise ValueError("Downscaling factors must be finite and non-negative")

    if output is None:
        data = input_footprints.data
        fields = np.asarray(data[WIND_VARIABLE].values, dtype=np.float32) * factors
        return WindFootprintSet(
            dataset=_footprint_dataset(
                fields,
                    input_footprints.event_ids,
                    data["year"].values,
                    input_footprints.grid,
                    attrs={
                        **data.attrs,
                        "downscaling_method": type(method).__name__,
                    },
            )
        )

    _validate_store_matches_footprints(output, input_footprints)
    assert output.path is not None
    target_computed = np.asarray(output.data["computed"].values)
    if target_computed.any():
        raise ValueError("Surface output store already contains computed events")
    source = input_footprints.data
    root = zarr.open_group(output.path, mode="r+")
    for event_index in range(len(input_footprints.event_ids)):
        root[WIND_VARIABLE][event_index] = (
            np.asarray(source[WIND_VARIABLE].isel(event=event_index).values) * factors
        ).astype(np.float32)
        root["computed"][event_index] = True
    return output


def _compute_event_footprint(
    track: pd.DataFrame,
    grid: RegularGrid,
    *,
    profile: WindProfile = holland_1980,
    interpolation_frequency: str = "1h",
    interpolation_spacing_factor: float | None = None,
    evaluation_radius_m: float = 1_000_000,
) -> np.ndarray:
    """Compute one earth-relative event footprint from a raw track."""

    prepared = _prepare_track_for_wind_evaluation(
        track,
        interpolation_frequency=interpolation_frequency,
        interpolation_spacing_factor=interpolation_spacing_factor,
        wind_speed_reference=WindSpeedReference.EARTH_RELATIVE,
    )
    return _evaluate_prepared_event_footprint(
        prepared,
        grid,
        profile=profile,
        evaluation_radius_m=evaluation_radius_m,
        wind_speed_reference=WindSpeedReference.EARTH_RELATIVE,
    )


def _evaluate_prepared_event_footprint(
    track: pd.DataFrame,
    grid: RegularGrid,
    *,
    profile: WindProfile,
    evaluation_radius_m: float,
    wind_speed_reference: WindSpeedReference,
) -> np.ndarray:
    """Evaluate one prepared event track on a grid."""

    if evaluation_radius_m <= 0:
        raise ValueError("evaluation_radius_m must be positive")
    if len(track) < 2:
        return np.zeros(grid.shape, dtype=np.float32)

    flat_grid = grid.flat_points
    max_wind = np.zeros(grid.nlat * grid.nlon, dtype=np.float32)

    for observation in track.itertuples(index=False):
        indices = flat_grid.indices_within_radius(
            observation.lat, observation.lon, evaluation_radius_m
        )
        if not len(indices):
            continue
        env_pressure_hpa = getattr(observation, "environmental_pressure_hpa", np.nan)
        if pd.isna(env_pressure_hpa):
            env_pressure_hpa = ENV_PRESSURE[observation.basin_id]
        try:
            speeds = evaluate_at_points(
            flat_grid.lons[indices],
            flat_grid.lats[indices],
            eye_lon=observation.lon,
            eye_lat=observation.lat,
            max_wind_speed_ms=observation.max_wind_speed_ms,
            radius_to_max_winds_m=observation.radius_to_max_winds_km * 1_000,
            min_pressure_pa=observation.min_pressure_hpa * 100,
            env_pressure_pa=float(env_pressure_hpa) * 100,
            track_heading_deg=observation.translation_heading_deg,
            translation_speed_ms=observation.translation_speed_ms,
            wind_speed_reference=wind_speed_reference,
            profile=profile,
            )
        except Exception as error:
            logging.debug(f"\n{track}\n{error}")
            speeds = np.zeros_like(max_wind, dtype=np.float32)
        np.maximum.at(max_wind, indices, np.nan_to_num(speeds, nan=0.0))
    return max_wind.reshape(grid.shape)


def _prepare_track_for_wind_evaluation(
    track: pd.DataFrame,
    *,
    interpolation_frequency: str,
    interpolation_spacing_factor: float | None = None,
    wind_speed_reference: WindSpeedReference,
) -> pd.DataFrame:
    """Interpolate a track and add motion and wind-frame QC fields."""

    interpolated = interpolate_track(
        track,
        interpolation_frequency,
        spacing_factor=interpolation_spacing_factor,
    )
    if len(interpolated) == 1:
        result = interpolated.copy()
        result["translation_heading_deg"] = 0.0
        result["translation_speed_ms"] = 0.0
        result["translation_acceleration_ms2"] = 0.0
    else:
        result = derive_track_motion(interpolated)

    advective_wind_speed = (
        result["translation_speed_ms"].to_numpy(dtype=float)
        * ADVECTIVE_SPEED_FRACTION
    )
    maximum_wind_speed = result["max_wind_speed_ms"].to_numpy(dtype=float)
    if wind_speed_reference == WindSpeedReference.EARTH_RELATIVE:
        rotational_wind_speed = maximum_wind_speed - advective_wind_speed
    elif wind_speed_reference == WindSpeedReference.EYE_RELATIVE:
        rotational_wind_speed = maximum_wind_speed
    else:
        raise ValueError(f"cannot interpret {wind_speed_reference=}")
    rotational_wind_nonpositive = rotational_wind_speed <= 0
    chi = np.divide(
        advective_wind_speed,
        rotational_wind_speed,
        out=np.full(len(result), np.inf, dtype=float),
        where=~rotational_wind_nonpositive,
    )
    result["advective_wind_speed_ms"] = advective_wind_speed
    result["rotational_max_wind_speed_ms"] = rotational_wind_speed
    result["chi"] = chi
    result["rotational_wind_nonpositive"] = rotational_wind_nonpositive
    return result


def _write_qc_outputs(
    prepared_tracks: list[tuple[str, pd.DataFrame]],
    *,
    model_family: str | None,
    interpolated_tracks_path: str | Path | None,
    storm_qc_path: str | Path | None,
) -> None:
    """Optionally persist per-timestep and per-storm wind-generation QC."""

    paths = [
        Path(path)
        for path in (interpolated_tracks_path, storm_qc_path)
        if path is not None
    ]
    if not paths:
        return

    tables = [track for _, track in prepared_tracks]
    if tables:
        interpolated_tracks = pd.concat(tables, ignore_index=True)
    else:
        interpolated_tracks = pd.DataFrame()
    interpolated_tracks = interpolated_tracks.drop(columns="geometry", errors="ignore")
    interpolated_tracks["model_family"] = model_family

    if storm_qc_path is not None:
        storm_qc = _storm_qc_summary(interpolated_tracks)
        storm_qc.to_parquet(storm_qc_path, index=False)
    if interpolated_tracks_path is not None:
        interpolated_tracks.to_parquet(interpolated_tracks_path, index=False)


def _storm_qc_summary(interpolated_tracks: pd.DataFrame) -> pd.DataFrame:
    if interpolated_tracks.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "year",
                "model_family",
                "max_advective_wind_speed_ms",
                "max_rotational_max_wind_speed_ms",
                "max_chi",
                "max_abs_translation_acceleration_ms2",
                "n_nonpositive_rotational_wind_observations",
            ]
        )
    summary = interpolated_tracks.assign(
        abs_translation_acceleration_ms2=interpolated_tracks[
            "translation_acceleration_ms2"
        ].abs()
    ).groupby("track_id", sort=False).agg(
        year=("year", "first"),
        model_family=("model_family", "first"),
        max_advective_wind_speed_ms=("advective_wind_speed_ms", "max"),
        max_rotational_max_wind_speed_ms=("rotational_max_wind_speed_ms", "max"),
        max_chi=("chi", "max"),
        max_abs_translation_acceleration_ms2=(
            "abs_translation_acceleration_ms2",
            "max",
        ),
        n_nonpositive_rotational_wind_observations=(
            "rotational_wind_nonpositive",
            "sum",
        ),
    )
    return summary.reset_index()


def _event_metadata(trackset: TrackSet) -> tuple[np.ndarray, np.ndarray]:
    grouped = trackset.tracks.groupby("track_id", sort=False)["year"]
    unique_years = grouped.nunique()
    invalid = unique_years[unique_years != 1]
    if not invalid.empty:
        raise ValueError(f"Each track must have one year; invalid events: {invalid.index[:5].tolist()}")
    years = grouped.first().to_numpy(dtype=np.int64)
    return trackset.track_ids.to_numpy(dtype=str), years


def _footprint_dataset(
    fields: np.ndarray,
    event_ids: np.ndarray,
    years: np.ndarray,
    grid: RegularGrid,
    *,
    attrs: dict[str, Any],
) -> xr.Dataset:
    return xr.Dataset(
        {
            WIND_VARIABLE: (("event", "lat", "lon"), fields.astype(np.float32)),
            "computed": (("event",), np.ones(len(event_ids), dtype=bool)),
        },
        coords={
            "event": _unicode_array(event_ids),
            "year": ("event", years.astype(np.int64)),
            "lat": grid.lats,
            "lon": grid.lons,
        },
        attrs={
            **attrs,
            "grid_resolution_deg": grid.resolution,
        },
    )


def _validate_footprint_dataset(dataset: xr.Dataset) -> None:
    required = {WIND_VARIABLE, "computed", "event", "year", "lat", "lon"}
    missing = required.difference(dataset.variables)
    if missing:
        raise ValueError(f"Invalid footprint dataset; missing: {sorted(missing)}")
    if dataset[WIND_VARIABLE].dims != ("event", "lat", "lon"):
        raise ValueError(f"{WIND_VARIABLE} must have dimensions (event, lat, lon)")
    if dataset["computed"].dims != ("event",):
        raise ValueError("computed must have dimension (event,)")
    if dataset["year"].dims != ("event",):
        raise ValueError("year must have dimension (event,)")
    if "grid_resolution_deg" not in dataset.attrs:
        raise ValueError("Footprint dataset is missing grid_resolution_deg")


def _validate_store_for_tracks(
    output: WindFootprintSet,
    trackset: TrackSet,
    grid: RegularGrid,
) -> None:
    if not output.is_persisted:
        raise ValueError("output must be an initialized Zarr WindFootprintSet")
    _validate_grid(output.grid, grid)
    available = set(output.event_ids)
    missing = set(trackset.track_ids.astype(str)).difference(available)
    if missing:
        raise ValueError(f"Output store has no events: {sorted(missing)[:5]}")


def _validate_store_matches_footprints(
    output: WindFootprintSet, source: WindFootprintSet
) -> None:
    if not output.is_persisted:
        raise ValueError("output must be an initialized Zarr WindFootprintSet")
    _validate_grid(output.grid, source.grid)
    if not np.array_equal(output.event_ids, source.event_ids):
        raise ValueError("Output store events must exactly match input events")
    if not np.array_equal(output.data["year"].values, source.data["year"].values):
        raise ValueError("Output store years must exactly match input years")


def _validate_grid(actual: RegularGrid, expected: RegularGrid) -> None:
    if (
        actual.shape != expected.shape
        or not np.allclose(actual.lats, expected.lats)
        or not np.allclose(actual.lons, expected.lons)
    ):
        raise ValueError("Footprint store grid does not match the requested grid")


def _set_wind_provenance(
    root: zarr.Group,
    *,
    profile: WindProfile,
    interpolation_frequency: str,
    interpolation_spacing_factor: float | None,
    evaluation_radius_m: float,
) -> None:
    """Record calculation settings, rejecting incompatible resumed writes."""

    provenance = {
        "wind_profile": getattr(profile, "__name__", type(profile).__name__),
        "interpolation_frequency": interpolation_frequency,
        "interpolation_spacing_factor": interpolation_spacing_factor,
        "evaluation_radius_m": evaluation_radius_m,
    }
    for name, value in provenance.items():
        existing = root.attrs.get(name)
        if existing is not None and existing != value:
            raise ValueError(
                f"Footprint store has {name}={existing!r}, not requested {value!r}"
            )
    root.attrs.update(provenance)


def _write_coordinate(root: zarr.Group, name: str, values: np.ndarray, dimensions: list[str]) -> None:
    root.create_dataset(name, data=values)
    root[name].attrs["_ARRAY_DIMENSIONS"] = dimensions


def _unicode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=str)
    width = max((len(value) for value in values), default=1)
    return values.astype(f"<U{width}")
