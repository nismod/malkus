#!/usr/bin/env bash

# Process tracksets for every canonical trackset under data/in/tracks.
# Each trackset directory must use the naming scheme
# source-{source}_epoch-{epoch}_scenario-{scenario}_gcm-{gcm}. The script reads
# metadata from that directory name, then computes wind fields, downscales them
# using the configured land-cover data, creates return-period maps, writes
# GeoTIFFs, and generates track and accumulation plots.
#
# Set TRACK_ROOT or OUTPUT_DIR to use different input/output locations.
# LAND_COVER, ROUGHNESS_MAPPING, MAX_CPUS, and INTERP_DIST_FACTOR can also
# be overridden through environment variables.

set -euo pipefail

TRACK_ROOT="${TRACK_ROOT:-data/in/tracks/lesser-antilles}"
LAND_COVER="${LAND_COVER:-data/in/land_cover/glob_cover_2009/GLOBCOVER_L4_200901_200912_V2.3.tif}"
ROUGHNESS_MAPPING="${ROUGHNESS_MAPPING:-data/in/land_cover/land_cover_to_surface_roughness.csv}"
BBOX=(-61.34 13.53 -60.60 14.28)
GRID_RESOLUTION=$(bc -l <<< "30 * 1 / (60 * 60)")  # 30 arc seconds
RETURN_PERIODS=(5 10 20 50 100)
MAX_CPUS="${MAX_CPUS:-48}"
INTERP_DIST_FACTOR="${INTERP_DIST_FACTOR:-0.3}"
OUTPUT_DIR="${OUTPUT_DIR:-data/out}"
NAME="${NAME:-lca}"


while IFS= read -r -d '' tracks; do
    trackset_dir="$(dirname "$tracks")"
    metadata_dir="$trackset_dir"
    if [[ "$(basename "$metadata_dir")" == "0" ]]; then
        metadata_dir="$(dirname "$metadata_dir")"
    fi
    metadata_name="$(basename "$metadata_dir")"
    if [[ "$metadata_name" =~ ^source-([^_]+)_epoch-([0-9]+)_scenario-(.+)_gcm-(.+)$ ]]; then
        stem="${BASH_REMATCH[1]}_${BASH_REMATCH[3]}_${BASH_REMATCH[4]}_${BASH_REMATCH[2]}"
    else
        echo "Trackset folder is not canonical: $metadata_name" >&2
        exit 1
    fi
    wind_fields="$OUTPUT_DIR/wind_fields/$NAME/$stem.zarr"

    if [[ -e "$wind_fields" ]]; then
        echo "Skipping $tracks: wind fields already exist at $wind_fields"
        continue
    fi

    scripts/run_trackset.sh \
        --name "$NAME" \
        --bbox "${BBOX[@]}" \
        --grid-resolution "$GRID_RESOLUTION" \
        --return-periods "$(IFS=,; echo "${RETURN_PERIODS[*]}")" \
        --max-cpus "$MAX_CPUS" \
        --interp-dist-factor "$INTERP_DIST_FACTOR" \
        --output-dir "$OUTPUT_DIR" \
        --tracks "$tracks" \
        --land-cover "$LAND_COVER" \
        --roughness-mapping "$ROUGHNESS_MAPPING"
done < <(find "$TRACK_ROOT" -type f -path '*/tracks.geoparquet' -print0 | sort -z)
