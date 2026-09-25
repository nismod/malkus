#!/usr/bin/env bash

set -euo pipefail

# Parameters
TRACKS="data/in/tracks/lesser-antilles/CHAZ_SSP-585_GCM-UKESM1-0-LL_epoch-2010/tracks.geoparquet"
LAND_COVER="data/in/land_cover/glob_cover_2009/GLOBCOVER_L4_200901_200912_V2.3.tif"
ROUGHNESS_MAPPING="data/in/land_cover/land_cover_to_surface_roughness.csv"
BBOX=(-61.34 13.53 -60.60 14.28)
GRID_RESOLUTION=$(bc -l <<< "30 * 1 / (60 * 60)")  # 30 arc second
NAME="lca"
SOURCE="chaz"
SCENARIO="SSP585"
GCM="UKESM1-0-LL"
EPOCH=2010
RETURN_PERIODS=(5 10 20 50 100)
MAX_CPUS=48
INTERP_DIST_FACTOR=0.3
OUTPUT_DIR="data/out"

# Templated output paths
STEM="${NAME}_${SOURCE}_${SCENARIO}_${GCM}_${EPOCH}"
WIND_FIELDS="$OUTPUT_DIR/wind_fields/$STEM.zarr"
RP_STORE="$OUTPUT_DIR/hazard_maps/$STEM.zarr"
RP_GEOTIFFS="$OUTPUT_DIR/hazard_maps/$STEM"
TRACK_PLOTS="$OUTPUT_DIR/wind_fields/${STEM}_track-plots"
RP_ACCUMULATION="$OUTPUT_DIR/hazard_maps/$STEM.gif"

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
