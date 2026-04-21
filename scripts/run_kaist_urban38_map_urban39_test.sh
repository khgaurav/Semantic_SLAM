#!/usr/bin/env bash
# Build a semantic map from KAIST urban38, then test localization on urban39.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAP_SEQUENCE="urban38"
TEST_SEQUENCE="urban39"
CHECK_ONLY=0
SKIP_MAPPING=0
BAG_RATE=""
RUN_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [--check-only] [--skip-mapping] [--rate <rate>] [--run-dir <dir>]

Pipeline:
  1. scripts/run_kaist_mapping_pipeline.sh urban38
  2. scripts/run_kaist_pose_eval.sh --map-sequence urban38 urban39

Options:
  --check-only    Validate available inputs without starting the long run.
  --skip-mapping  Reuse an existing data/hybrid_maps/urban38 map.
  --rate <rate>   Forward playback rate to the urban39 localization test.
  --run-dir <dir> Forward output directory to the urban39 localization test.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

has_map() {
    local map_dir="${HYBRID_MAP_ROOT:-$SCRIPT_DIR/../data/hybrid_maps}/$MAP_SEQUENCE"
    [[ -f "$map_dir/map_index.faiss" \
        && -f "$map_dir/keyframe_poses.npy" \
        && -f "$map_dir/keyframe_ids.npy" ]]
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check-only)
                CHECK_ONLY=1
                ;;
            --skip-mapping)
                SKIP_MAPPING=1
                ;;
            --rate)
                shift
                [[ $# -gt 0 ]] || die "--rate requires a value"
                BAG_RATE="$1"
                ;;
            --run-dir)
                shift
                [[ $# -gt 0 ]] || die "--run-dir requires a path"
                RUN_DIR="$1"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
        shift
    done
}

run_eval() {
    local args=("--map-sequence" "$MAP_SEQUENCE")
    if [[ -n "$BAG_RATE" ]]; then
        args+=("--rate" "$BAG_RATE")
    fi
    if [[ -n "$RUN_DIR" ]]; then
        args+=("--run-dir" "$RUN_DIR")
    fi
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        args+=("--check-only")
    fi
    args+=("$TEST_SEQUENCE")

    "$SCRIPT_DIR/run_kaist_pose_eval.sh" "${args[@]}"
}

parse_args "$@"

echo "Map sequence:  $MAP_SEQUENCE"
echo "Test sequence: $TEST_SEQUENCE"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    "$SCRIPT_DIR/run_kaist_mapping_pipeline.sh" --check-only "$MAP_SEQUENCE"
    if has_map; then
        run_eval
    else
        echo "urban38 map is not present yet; localization test preflight will pass after mapping finishes."
    fi
    echo "Cross-sequence preflight complete."
    exit 0
fi

if [[ "$SKIP_MAPPING" -eq 0 ]]; then
    "$SCRIPT_DIR/run_kaist_mapping_pipeline.sh" "$MAP_SEQUENCE"
else
    has_map || die "Missing urban38 map. Run without --skip-mapping first."
fi

run_eval
