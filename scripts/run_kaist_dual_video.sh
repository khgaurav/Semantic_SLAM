#!/usr/bin/env bash
# Play KAIST urban38 and urban39 camera streams side-by-side.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
KAIST_ROS2_ROOT="${KAIST_ROS2_ROOT:-$REPO_ROOT/data/kaist_ros2}"
SOURCE_TOPIC="${SOURCE_TOPIC:-/kaist/stereo_left/image_raw/compressed}"
URBAN38_COMPRESSED_TOPIC="${URBAN38_COMPRESSED_TOPIC:-/urban38/stereo_left/image_raw/compressed}"
URBAN39_COMPRESSED_TOPIC="${URBAN39_COMPRESSED_TOPIC:-/urban39/stereo_left/image_raw/compressed}"
URBAN38_RAW_TOPIC="${URBAN38_RAW_TOPIC:-/urban38/stereo_left/image_raw}"
URBAN39_RAW_TOPIC="${URBAN39_RAW_TOPIC:-/urban39/stereo_left/image_raw}"
OUTPUT_ENCODING="${OUTPUT_ENCODING:-mono8}"
BAG_RATE="${BAG_RATE:-1.0}"
LOOP=0
PIDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--rate <rate>] [--loop]

Plays only the KAIST stereo-left videos:
  urban38 -> $URBAN38_RAW_TOPIC
  urban39 -> $URBAN39_RAW_TOPIC
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
                BAG_RATE="$1"
                ;;
            --loop)
                LOOP=1
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

source_setup() {
    [[ -f "$ROS_SETUP" ]] || die "ROS setup not found: $ROS_SETUP"
    set +u
    # shellcheck source=/dev/null
    source "$ROS_SETUP"
    set -u

    [[ -f "$WORKSPACE_SETUP" ]] || die "Workspace setup not found: $WORKSPACE_SETUP. Run: colcon build --symlink-install --packages-select hybrid_localization"
    set +u
    # shellcheck source=/dev/null
    source "$WORKSPACE_SETUP"
    set -u
}

require_ros_executable() {
    local pkg="$1"
    local exe="$2"
    ros2 pkg executables "$pkg" 2>/dev/null | grep -Fq "$pkg $exe" \
        || die "ROS executable not found: $pkg/$exe"
}

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "=== Cleaning up video processes ==="
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

play_bag() {
    local sequence="$1"
    local output_topic="$2"
    local bag_dir="$KAIST_ROS2_ROOT/$sequence"
    local loop_args=()
    if [[ "$LOOP" -eq 1 ]]; then
        loop_args=("--loop")
    fi

    ros2 bag play "$bag_dir" \
        --rate "$BAG_RATE" \
        "${loop_args[@]}" \
        --topics "$SOURCE_TOPIC" \
        --remap "$SOURCE_TOPIC:=$output_topic"
}

parse_args "$@"
source_setup

[[ -f "$KAIST_ROS2_ROOT/urban38/metadata.yaml" ]] || die "urban38 bag not found under $KAIST_ROS2_ROOT"
[[ -f "$KAIST_ROS2_ROOT/urban39/metadata.yaml" ]] || die "urban39 bag not found under $KAIST_ROS2_ROOT"
require_ros_executable hybrid_localization compressed_image_republisher
require_ros_executable rqt_image_view rqt_image_view

trap cleanup EXIT INT TERM

echo "Playing camera-only KAIST bags at rate $BAG_RATE"
echo "urban38: $URBAN38_COMPRESSED_TOPIC -> $URBAN38_RAW_TOPIC"
echo "urban39: $URBAN39_COMPRESSED_TOPIC -> $URBAN39_RAW_TOPIC"

start_bg ros2 run hybrid_localization compressed_image_republisher --ros-args \
    -p "input_topic:=$URBAN38_COMPRESSED_TOPIC" \
    -p input_is_compressed:=true \
    -p "output_topic:=$URBAN38_RAW_TOPIC" \
    -p "output_encoding:=$OUTPUT_ENCODING"

start_bg ros2 run hybrid_localization compressed_image_republisher --ros-args \
    -p "input_topic:=$URBAN39_COMPRESSED_TOPIC" \
    -p input_is_compressed:=true \
    -p "output_topic:=$URBAN39_RAW_TOPIC" \
    -p "output_encoding:=$OUTPUT_ENCODING"

start_bg ros2 run rqt_image_view rqt_image_view "$URBAN38_RAW_TOPIC"
start_bg ros2 run rqt_image_view rqt_image_view "$URBAN39_RAW_TOPIC"

sleep 2

start_bg play_bag urban38 "$URBAN38_COMPRESSED_TOPIC"
start_bg play_bag urban39 "$URBAN39_COMPRESSED_TOPIC"

wait "${PIDS[@]}"
