#!/usr/bin/env bash

# Process tracksets for every canonical trackset under data/in/tracks.
# Each trackset directory must use the naming scheme
# source-{source}_epoch-{epoch}_scenario-{scenario}_gcm-{gcm}. The script reads
# metadata from that directory name, then computes wind fields, downscales them
# using the configured land-cover data, creates return-period maps, writes
# GeoTIFFs, generates track and accumulation plots and then pools all maps to
# determine mean and IQR values across provided tracksets at each location and
# return period.
#
# Set TRACK_ROOT or OUTPUT_DIR to use different input/output locations.
# Set BBOX to four whitespace-separated coordinates: west south east north.
# NAME, LAND_COVER, ROUGHNESS_MAPPING, MAX_CPUS, and INTERP_DIST_FACTOR can also
# be overridden through environment variables.

# Example usage:
# $ NAME="mur" TRACK_ROOT="data/in/tracks/mur" BBOX="56.79 -20.84 58.37 -19.59" ./scripts/run_all_tracksets.sh


set -euo pipefail

NAME="${NAME}"
TRACK_ROOT="${TRACK_ROOT}"
BBOX_STRING="${BBOX}"
read -r -a BBOX <<< "$BBOX_STRING"
if [[ ${#BBOX[@]} -ne 4 ]]; then
    echo "BBOX must contain four whitespace-separated coordinates: west south east north" >&2
    exit 2
fi
MAX_CPUS="${MAX_CPUS:-1}"
RETURN_PERIODS=(5 10 20 50 100)
GRID_RESOLUTION=$(bc -l <<< "30 * 1 / (60 * 60)")  # 30 arc seconds
INTERP_DIST_FACTOR="${INTERP_DIST_FACTOR:-0.3}"
LAND_COVER="${LAND_COVER:-data/in/land_cover/glob_cover_2009/GLOBCOVER_L4_200901_200912_V2.3.tif}"
ROUGHNESS_MAPPING="${ROUGHNESS_MAPPING:-data/in/land_cover/land_cover_to_surface_roughness.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/out}"


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

POOLED_RP_MAPS="$OUTPUT_DIR/hazard_maps/$NAME/pooled-contemporary.zarr"
POOLED_RP_PLOTS="$OUTPUT_DIR/hazard_maps/$NAME/pooled-contemporary-plots"
pixi run python scripts/pool_rp_maps.py \
    "$OUTPUT_DIR/hazard_maps/$NAME" \
    "$POOLED_RP_MAPS" \
    --epoch-before 2030 \
    --equal-source-weights
pixi run python scripts/plot_pooled_rp_maps.py \
    "$POOLED_RP_MAPS" \
    "$POOLED_RP_PLOTS"

POOLED_RP_MAPS="$OUTPUT_DIR/hazard_maps/$NAME/pooled-future.zarr"
POOLED_RP_PLOTS="$OUTPUT_DIR/hazard_maps/$NAME/pooled-future-plots"
pixi run python scripts/pool_rp_maps.py \
    "$OUTPUT_DIR/hazard_maps/$NAME" \
    "$POOLED_RP_MAPS" \
    --epoch-after 2030 \
    --equal-source-weights
pixi run python scripts/plot_pooled_rp_maps.py \
    "$POOLED_RP_MAPS" \
    "$POOLED_RP_PLOTS"
