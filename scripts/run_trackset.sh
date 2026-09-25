#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_trackset.sh \
  --name NAME --bbox WEST SOUTH EAST NORTH \
  --grid-resolution RESOLUTION --return-periods PERIODS \
  --max-cpus N --interp-dist-factor FACTOR --output-dir DIR \
  --tracks PATH --land-cover PATH --roughness-mapping PATH

--return-periods accepts a comma-separated list, for example: 5,10,20,50,100.
EOF
}

NAME=""
BBOX=()
GRID_RESOLUTION=""
RETURN_PERIODS_CSV=""
MAX_CPUS=""
INTERP_DIST_FACTOR=""
OUTPUT_DIR=""
TRACKS=""
LAND_COVER=""
ROUGHNESS_MAPPING=""

while (($#)); do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --bbox) BBOX=("$2" "$3" "$4" "$5"); shift 5 ;;
        --grid-resolution) GRID_RESOLUTION="$2"; shift 2 ;;
        --return-periods) RETURN_PERIODS_CSV="$2"; shift 2 ;;
        --max-cpus) MAX_CPUS="$2"; shift 2 ;;
        --interp-dist-factor) INTERP_DIST_FACTOR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tracks) TRACKS="$2"; shift 2 ;;
        --land-cover) LAND_COVER="$2"; shift 2 ;;
        --roughness-mapping) ROUGHNESS_MAPPING="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$NAME" || ${#BBOX[@]} -ne 4 || -z "$GRID_RESOLUTION" || \
      -z "$RETURN_PERIODS_CSV" || -z "$MAX_CPUS" || \
      -z "$INTERP_DIST_FACTOR" || -z "$OUTPUT_DIR" || -z "$TRACKS" || \
      -z "$LAND_COVER" || -z "$ROUGHNESS_MAPPING" ]]; then
    echo "All arguments are required" >&2
    usage >&2
    exit 2
fi

IFS=',' read -r -a RETURN_PERIODS <<< "$RETURN_PERIODS_CSV"
TRACKSET_DIR="$(dirname "$TRACKS")"
if [[ "$(basename "$TRACKSET_DIR")" == "0" ]]; then
    TRACKSET_DIR="$(dirname "$TRACKSET_DIR")"
fi
TRACKSET_NAME="$(basename "$TRACKSET_DIR")"
if [[ "$TRACKSET_NAME" =~ ^source-([^_]+)_epoch-([0-9]+)_scenario-(.+)_gcm-(.+)$ ]]; then
    SOURCE="${BASH_REMATCH[1]}"
    EPOCH="${BASH_REMATCH[2]}"
    SCENARIO="${BASH_REMATCH[3]}"
    GCM="${BASH_REMATCH[4]}"
else
    echo "TRACKS does not point to a canonical trackset: $TRACKS" >&2
    exit 1
fi

# Templated output paths
STEM="${SOURCE}_${SCENARIO}_${GCM}_${EPOCH}"
WIND_FIELDS="$OUTPUT_DIR/wind_fields/$NAME/$STEM.zarr"
RP_STORE="$OUTPUT_DIR/hazard_maps/$NAME/$STEM.zarr"
RP_GEOTIFFS="$OUTPUT_DIR/hazard_maps/$NAME/$STEM"
TRACK_PLOTS="$OUTPUT_DIR/wind_fields/$NAME/${STEM}_track-plots"
RP_ACCUMULATION="$OUTPUT_DIR/hazard_maps/$NAME/$STEM.gif"

pixi run python scripts/trackset_to_rp_maps.py \
    "$TRACKS" \
    --land-cover "$LAND_COVER" \
    --roughness-mapping "$ROUGHNESS_MAPPING" \
    --bbox "${BBOX[@]}" \
    --grid-resolution-deg "$GRID_RESOLUTION" \
    --return-periods "${RETURN_PERIODS[@]}" \
    --source "$SOURCE" \
    --scenario "$SCENARIO" \
    --gcm "$GCM" \
    --epoch "$EPOCH" \
    --wind-fields "$WIND_FIELDS" \
    --rp-maps "$RP_STORE" \
    --rp-geotiffs "$RP_GEOTIFFS" \
    --interp-dist-factor "$INTERP_DIST_FACTOR" \
    --max-cpus "$MAX_CPUS"

pixi run python scripts/plot_track_footprints.py \
    "$TRACKS" \
    "$WIND_FIELDS" \
    "$TRACK_PLOTS" \
    --most-intense \
    --max-tracks 100 \
    --max-cpus "$MAX_CPUS"

pixi run python scripts/plot_rp_map_accumulation.py \
    "$WIND_FIELDS" \
    "$RP_ACCUMULATION" \
    --return-periods "${RETURN_PERIODS[@]}" \
    --max-cpus "$MAX_CPUS"
