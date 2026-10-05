#!/usr/bin/env python3
"""Pool return-period Zarr maps and calculate weighted mean/IQR."""

from __future__ import annotations
import argparse
from collections import Counter
import logging
from numbers import Real
from pathlib import Path

from malkus import ReturnPeriodMapSet, pool_return_period_maps

def main() -> None:
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--epoch-before",
        type=int,
        help="Select maps with epoch strictly before this year",
    )
    parser.add_argument(
        "--epoch-after",
        type=int,
        help="Select maps with epoch strictly after this year",
    )
    parser.add_argument(
        "--equal-source-weights",
        action="store_true",
        help="Give each unique model_family a total pool weight of one",
    )
    args = parser.parse_args()
    if (
        args.epoch_after is not None
        and args.epoch_before is not None
        and args.epoch_after >= args.epoch_before
    ):
        parser.error("--epoch-after must be less than --epoch-before")
    selected = []
    for path in sorted(args.input_root.rglob("*.zarr")):
        try:
            maps = ReturnPeriodMapSet.open(path)
        except (KeyError, ValueError, OSError, TypeError) as error:
            logging.info("Skipping %s: not a return-period map (%s)", path, error)
            continue
        if args.epoch_before is not None or args.epoch_after is not None:
            epoch = maps.data.attrs.get("epoch")
            if not isinstance(epoch, Real) or not float(epoch).is_integer():
                logging.warning("Skipping %s: missing or invalid epoch metadata", path)
                continue
            epoch = int(epoch)
            if args.epoch_before is not None and epoch >= args.epoch_before:
                continue
            if args.epoch_after is not None and epoch <= args.epoch_after:
                continue
        selected.append(maps)
    if not selected:
        raise SystemExit("No return-period maps matched the requested filters")
    for map in selected:
        logging.info(map.path)
    weights = None
    if args.equal_source_weights:
        sources = [maps.data.attrs.get("model_family") for maps in selected]
        if any(not isinstance(source, str) or not source for source in sources):
            raise SystemExit(
                "--equal-source-weights requires model_family metadata on every map"
            )
        counts = Counter(sources)
        weights = [1.0 / counts[source] for source in sources]
        logging.info(
            "Per-map weights: %s", {source: round(1.0 / counts[source], 3) for source in sources}
        )
    pool_return_period_maps(selected, weights=weights, output=args.output)
    logging.info("Pooled %d maps into %s", len(selected), args.output)

if __name__ == "__main__":
    main()
