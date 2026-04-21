#!/usr/bin/env bash
# Build a semantic map for a prepared KAIST Complex Urban ROS2 bag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
KAIST_ROS2_ROOT="${KAIST_ROS2_ROOT:-$REPO_ROOT/data/kaist_ros2}"
HYBRID_MAP_ROOT="${HYBRID_MAP_ROOT:-$REPO_ROOT/data/hybrid_maps}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/kaist/stereo_left/image_raw/compressed}"
IMAGE_IS_COMPRESSED="${IMAGE_IS_COMPRESSED:-true}"
ODOM_TOPIC="${ODOM_TOPIC:-/kaist/global_pose/odom}"
MODEL_NAME="${MODEL_NAME:-google/siglip-base-patch16-224}"
ROS_LOG_DIR="${ROS_LOG_DIR:-$REPO_ROOT/log/ros}"
HF_HOME="${HF_HOME:-$REPO_ROOT/data/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
CHECK_ONLY=0
SEQUENCE=""
PIDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--check-only] <urban38|urban39>

Expected prepared bag:
  data/kaist_ros2/<sequence>/metadata.yaml

Create it from the raw KAIST archives with:
  scripts/prepare_kaist_urban_bag.sh <sequence>

Environment overrides:
  KAIST_ROS2_ROOT      default: $KAIST_ROS2_ROOT
  HYBRID_MAP_ROOT      default: $HYBRID_MAP_ROOT
  IMAGE_TOPIC          default: $IMAGE_TOPIC
  ODOM_TOPIC           default: $ODOM_TOPIC
  IMAGE_IS_COMPRESSED  default: $IMAGE_IS_COMPRESSED
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

canonical_sequence() {
    case "$1" in
        urban38|urban39) printf '%s\n' "$1" ;;
        *) return 1 ;;
    esac
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check-only)
                CHECK_ONLY=1
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                [[ -z "$SEQUENCE" ]] || die "Only one sequence may be provided."
                SEQUENCE="$1"
                ;;
        esac
        shift
    done

    [[ -n "$SEQUENCE" ]] || { usage; exit 1; }
    SEQUENCE="$(canonical_sequence "$SEQUENCE")" || die "Unsupported KAIST sequence: $SEQUENCE"
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

require_bag_topic() {
    local topic="$1"
    grep -Fq "$topic" <<<"$BAG_INFO" || die "Bag $BAG_DIR does not contain topic $topic"
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
source_setup
export ROS_LOG_DIR
export HF_HOME
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
mkdir -p "$ROS_LOG_DIR" "$HF_HOME"

BAG_DIR="$KAIST_ROS2_ROOT/$SEQUENCE"
MAP_DIR="$HYBRID_MAP_ROOT/$SEQUENCE"

[[ -d "$BAG_DIR" ]] || die "Bag directory not found: $BAG_DIR. Run scripts/prepare_kaist_urban_bag.sh $SEQUENCE"
[[ -f "$BAG_DIR/metadata.yaml" ]] || die "ROS2 bag metadata not found: $BAG_DIR/metadata.yaml"
mkdir -p "$MAP_DIR"
[[ -w "$MAP_DIR" ]] || die "Map directory is not writable: $MAP_DIR"

require_ros_executable hybrid_localization mapping_node

BAG_INFO="$(ros2 bag info "$BAG_DIR")"
require_bag_topic "$IMAGE_TOPIC"
require_bag_topic "$ODOM_TOPIC"

echo "Sequence: $SEQUENCE"
echo "Bag:      $BAG_DIR"
echo "Map out:  $MAP_DIR"
echo "Camera:   $IMAGE_TOPIC"
echo "Odom:     $ODOM_TOPIC"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "Preflight checks passed."
    exit 0
fi

trap cleanup EXIT INT TERM

echo "=== Starting hybrid mapping node ==="
start_bg ros2 run hybrid_localization mapping_node --ros-args \
    -p use_sim_time:=true \
    -p "map_dir:=$MAP_DIR" \
    -p "odom_topic:=$ODOM_TOPIC" \
    -p "image_topic:=$IMAGE_TOPIC" \
    -p "image_is_compressed:=$IMAGE_IS_COMPRESSED" \
    -p "model_name:=$MODEL_NAME"
sleep 8

echo "=== Playing bag ==="
ros2 bag play "$BAG_DIR" --clock --topics "$ODOM_TOPIC" "$IMAGE_TOPIC"

echo "=== Bag playback finished ==="
sleep 5
