#!/usr/bin/env bash
# Replay a headless KAIST localization run with camera + RViz at high speed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
RVIZ_CONFIG="${RVIZ_CONFIG:-$REPO_ROOT/rviz/kaist_localization.rviz}"
RAW_VIEW_TOPIC="${RAW_VIEW_TOPIC:-/kaist/stereo_left/image_raw}"
RAW_VIEW_ENCODING="${RAW_VIEW_ENCODING:-mono8}"
PLAYBACK_RATE="${PLAYBACK_RATE:-10.0}"
RUN_DIR=""
PIDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--rate <playback_rate>] <run_dir>

Example:
  scripts/replay_kaist_localization_results.sh \\
    data/eval/urban39_from_urban38/20260420-211222
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --rate)
                shift
                [[ $# -gt 0 ]] || die "--rate requires a value"
                PLAYBACK_RATE="$1"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                [[ -z "$RUN_DIR" ]] || die "Only one run directory may be provided."
                RUN_DIR="$1"
                ;;
        esac
        shift
    done

    [[ -n "$RUN_DIR" ]] || { usage; exit 1; }
}

source_setup() {
    [[ -f "$ROS_SETUP" ]] || die "ROS setup not found: $ROS_SETUP"
    set +u
    # shellcheck source=/dev/null
    source "$ROS_SETUP"
    set -u

    [[ -f "$WORKSPACE_SETUP" ]] || die "Workspace setup not found: $WORKSPACE_SETUP"
    set +u
    # shellcheck source=/dev/null
    source "$WORKSPACE_SETUP"
    set -u
}

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "=== Cleaning up background processes ==="
        for pid in "${PIDS[@]}"; do
            kill -INT "$pid" 2>/dev/null || true
        done
        sleep 2
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
    exit "$code"
}

start_bg() {
    "$@" &
    PIDS+=("$!")
}

parse_args "$@"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
[[ -d "$RUN_DIR" ]] || die "Run directory not found: $RUN_DIR"
[[ -f "$RUN_DIR/eval_env.txt" ]] || die "Missing eval_env.txt in $RUN_DIR"
[[ -d "$RUN_DIR/record" ]] || die "Missing record bag in $RUN_DIR"
[[ -f "$RVIZ_CONFIG" ]] || die "RViz config not found: $RVIZ_CONFIG"

source_setup

# shellcheck disable=SC1090
source "$RUN_DIR/eval_env.txt"

[[ -d "$bag_dir" ]] || die "Original bag directory not found: $bag_dir"
[[ -f "$bag_dir/metadata.yaml" ]] || die "Original bag metadata not found: $bag_dir/metadata.yaml"
[[ -f "$RUN_DIR/record/metadata.yaml" ]] || die "Recorded results metadata not found: $RUN_DIR/record/metadata.yaml"

trap cleanup EXIT INT TERM

echo "Run dir:   $RUN_DIR"
echo "Bag:       $bag_dir"
echo "Results:   $RUN_DIR/record"
echo "Rate:      $PLAYBACK_RATE"
echo "Image:     $image_topic -> $RAW_VIEW_TOPIC"
echo "Pose:      $pose_topic"
echo "Path:      $path_topic"

echo "=== Starting camera republisher ==="
start_bg ros2 run hybrid_localization compressed_image_republisher --ros-args \
    -p use_sim_time:=true \
    -p "input_topic:=$image_topic" \
    -p "input_is_compressed:=$image_is_compressed" \
    -p "output_topic:=$RAW_VIEW_TOPIC" \
    -p "output_encoding:=$RAW_VIEW_ENCODING"

echo "=== Starting static map TF anchor for RViz ==="
start_bg ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0 \
    --roll 0 --pitch 0 --yaw 0 \
    --frame-id map \
    --child-frame-id map_anchor

echo "=== Opening camera viewer and RViz ==="
start_bg ros2 run rqt_image_view rqt_image_view "$RAW_VIEW_TOPIC"
start_bg ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG" --ros-args -p use_sim_time:=true

sleep 3

echo "=== Playing original bag ==="
start_bg ros2 bag play "$bag_dir" --clock --rate "$PLAYBACK_RATE" \
    --topics "$image_topic"

echo "=== Playing localization results bag ==="
ros2 bag play "$RUN_DIR/record" --rate "$PLAYBACK_RATE" \
    --topics "$pose_topic" "$path_topic"

echo "=== Results replay finished ==="
