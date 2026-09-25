#!/usr/bin/env python3
"""Filter input tracksets by a geographic bounding box."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from malkus import TrackSet, TrackSource


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        type=Path,
        help="Parent directory for the filtered trackset folders",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/in/tracks"),
        help="Directory containing one folder per trackset (default: data/in/tracks)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in degrees",
    )
    parser.add_argument(
        "--search-radius-deg",
        type=float,
        default=3.0,
        help="Track search radius in degrees (default: 3)",
    )
    parser.add_argument(
        "--source",
        choices=[source.value for source in TrackSource],
        help="Source for every trackset; otherwise infer it from each folder name",
    )
    return parser


def _infer_source(folder: Path) -> TrackSource:
    name = folder.name.lower()
    for source in TrackSource:
        if name.startswith(source.value):
            return source
    raise ValueError(
        f"Cannot infer TrackSource from {folder.name!r}; pass --source explicitly"
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.search_radius_deg < 0:
        raise SystemExit("--search-radius-deg must be non-negative")

    input_root = args.input_root
    trackset_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not trackset_dirs:
        raise SystemExit(f"No trackset directories found under {input_root}")

    bbox = tuple(args.bbox)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    for trackset_dir in trackset_dirs:
        input_path = trackset_dir / "tracks.geoparquet"
        if not input_path.is_file():
            logging.info("Skipping %s: %s is missing", trackset_dir.name, input_path)
            continue

        output_dir = args.output_root / trackset_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "tracks.geoparquet"
        if output_path.exists():
            logging.info("Skipping %s, already created", trackset_dir.name)
            continue

        source = TrackSource(args.source) if args.source else _infer_source(trackset_dir)
        trackset = TrackSet.read_parquet(
            input_path,
            source=source,
            bbox=bbox,
            search_radius_deg=args.search_radius_deg,
        )
        trackset.tracks.to_parquet(output_path, index=False)
        logging.info(
            "%s: wrote %d observations to %s",
            trackset_dir.name,
            len(trackset.tracks),
            output_path,
        )


if __name__ == "__main__":
    main()
