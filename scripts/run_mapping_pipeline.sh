#!/usr/bin/env bash
# Run LIO-SAM plus the hybrid mapping node on a ROS2-converted M2DGR outdoor bag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
M2DGR_ROS2_ROOT="${M2DGR_ROS2_ROOT:-$REPO_ROOT/data/m2dgr_ros2}"
HYBRID_MAP_ROOT="${HYBRID_MAP_ROOT:-$REPO_ROOT/data/hybrid_maps}"
LOCAL_GTSAM_ROOT="${LOCAL_GTSAM_ROOT:-$REPO_ROOT/data/local/gtsam}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/color/image_raw/compressed}"
IMAGE_IS_COMPRESSED="${IMAGE_IS_COMPRESSED:-true}"
ODOM_TOPIC="${ODOM_TOPIC:-/lio_sam/mapping/odometry}"
POINTS_TOPIC="${POINTS_TOPIC:-/velodyne_points}"
IMU_TOPIC="${IMU_TOPIC:-/handsfree/imu}"
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
Usage: $(basename "$0") [--check-only] <outdoor_sequence>

Outdoor sequences:
  gate_01 gate_02 gate_03
  Circle_01 Circle_02
  street_01 street_02 street_03 street_04 street_05 street_06 street_07 street_08 street_09 street_010
  walk_01

Environment overrides:
  ROS_SETUP          default: $ROS_SETUP
  WORKSPACE_SETUP    default: $WORKSPACE_SETUP
  M2DGR_ROS2_ROOT    default: $M2DGR_ROS2_ROOT
  HYBRID_MAP_ROOT    default: $HYBRID_MAP_ROOT
  IMAGE_IS_COMPRESSED default: $IMAGE_IS_COMPRESSED
  LIO_SAM_PARAMS_FILE
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

canonical_sequence() {
    case "$1" in
        gate_01|gate_02|gate_03|walk_01) printf '%s\n' "$1" ;;
        Circle_01|circle_01) printf 'Circle_01\n' ;;
        Circle_02|circle_02) printf 'Circle_02\n' ;;
        street_01|street_02|street_03|street_04|street_05) printf '%s\n' "$1" ;;
        street_06|street_07|street_08|street_09|street_010) printf '%s\n' "$1" ;;
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
    SEQUENCE="$(canonical_sequence "$SEQUENCE")" || die "Unsupported outdoor sequence: $SEQUENCE"
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

use_local_gtsam_if_present() {
    if [[ -d "$LOCAL_GTSAM_ROOT/lib" ]]; then
        export CMAKE_PREFIX_PATH="$LOCAL_GTSAM_ROOT:${CMAKE_PREFIX_PATH:-}"
        export LD_LIBRARY_PATH="$LOCAL_GTSAM_ROOT/lib:${LD_LIBRARY_PATH:-}"
    fi
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

choose_lio_params() {
    if [[ -n "${LIO_SAM_PARAMS_FILE:-}" ]]; then
        PARAMS_FILE="$LIO_SAM_PARAMS_FILE"
        return
    fi

    if [[ -f "$REPO_ROOT/config/m2dgr_lio_sam_params.yaml" ]]; then
        PARAMS_FILE="$REPO_ROOT/config/m2dgr_lio_sam_params.yaml"
        return
    fi

    local lio_prefix
    lio_prefix="$(ros2 pkg prefix lio_sam 2>/dev/null || true)"
    if [[ -n "$lio_prefix" && -f "$lio_prefix/share/lio_sam/config/params.yaml" ]]; then
        PARAMS_FILE="$lio_prefix/share/lio_sam/config/params.yaml"
    else
        PARAMS_FILE="$REPO_ROOT/third_party/M2DGR/my_params_lidar.yaml"
    fi
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
use_local_gtsam_if_present
export ROS_LOG_DIR
export HF_HOME
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
mkdir -p "$ROS_LOG_DIR"
mkdir -p "$HF_HOME"

BAG_DIR="$M2DGR_ROS2_ROOT/$SEQUENCE"
MAP_DIR="$HYBRID_MAP_ROOT/$SEQUENCE"

[[ -d "$BAG_DIR" ]] || die "Bag directory not found: $BAG_DIR"
[[ -f "$BAG_DIR/metadata.yaml" ]] || die "ROS2 bag metadata not found: $BAG_DIR/metadata.yaml"
mkdir -p "$MAP_DIR"
[[ -w "$MAP_DIR" ]] || die "Map directory is not writable: $MAP_DIR"

require_ros_executable tf2_ros static_transform_publisher
require_ros_executable lio_sam lio_sam_imuPreintegration
require_ros_executable lio_sam lio_sam_imageProjection
require_ros_executable lio_sam lio_sam_featureExtraction
require_ros_executable lio_sam lio_sam_mapOptimization
require_ros_executable hybrid_localization mapping_node

choose_lio_params
[[ -f "$PARAMS_FILE" ]] || die "LIO-SAM params file not found: $PARAMS_FILE"

BAG_INFO="$(ros2 bag info "$BAG_DIR")"
require_bag_topic "$POINTS_TOPIC"
require_bag_topic "$IMU_TOPIC"
require_bag_topic "$IMAGE_TOPIC"

echo "Sequence: $SEQUENCE"
echo "Bag:      $BAG_DIR"
echo "Map out:  $MAP_DIR"
echo "Params:   $PARAMS_FILE"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "Preflight checks passed."
    exit 0
fi

trap cleanup EXIT INT TERM

echo "=== Starting static TF publisher ==="
start_bg ros2 run tf2_ros static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 map odom \
    --ros-args -p use_sim_time:=true
sleep 1

echo "=== Starting LIO-SAM nodes ==="
start_bg ros2 run lio_sam lio_sam_imuPreintegration --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 0.5
start_bg ros2 run lio_sam lio_sam_imageProjection --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 0.5
start_bg ros2 run lio_sam lio_sam_featureExtraction --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 0.5
start_bg ros2 run lio_sam lio_sam_mapOptimization --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 2

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
ros2 bag play "$BAG_DIR" --clock --topics "$POINTS_TOPIC" "$IMU_TOPIC" "$IMAGE_TOPIC"

echo "=== Bag playback finished ==="
sleep 5
