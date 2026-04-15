#!/usr/bin/env bash
# Run gate/outdoor localization and record poses for LiDAR-odometry comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
M2DGR_ROS2_ROOT="${M2DGR_ROS2_ROOT:-$REPO_ROOT/data/m2dgr_ros2}"
HYBRID_MAP_ROOT="${HYBRID_MAP_ROOT:-$REPO_ROOT/data/hybrid_maps}"
LOCAL_GTSAM_ROOT="${LOCAL_GTSAM_ROOT:-$REPO_ROOT/data/local/gtsam}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/data/eval}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/color/image_raw/compressed}"
ODOM_TOPIC="${ODOM_TOPIC:-/lio_sam/mapping/odometry}"
POSE_TOPIC="${POSE_TOPIC:-/localized_pose}"
POINTS_TOPIC="${POINTS_TOPIC:-/velodyne_points}"
IMU_TOPIC="${IMU_TOPIC:-/handsfree/imu}"
MODEL_NAME="${MODEL_NAME:-google/siglip-base-patch16-224}"
TOP_K="${TOP_K:-10}"
TEMPORAL_FILTER_ENABLED="${TEMPORAL_FILTER_ENABLED:-true}"
MAX_POSE_JUMP_M="${MAX_POSE_JUMP_M:-4.0}"
MAX_KEYFRAME_JUMP="${MAX_KEYFRAME_JUMP:-8}"
MAX_KEYFRAME_POSE_JUMP_M="${MAX_KEYFRAME_POSE_JUMP_M:-8.0}"
TEMPORAL_SCORE_MARGIN="${TEMPORAL_SCORE_MARGIN:-0.05}"
MIN_GLOBAL_MATCH_MARGIN="${MIN_GLOBAL_MATCH_MARGIN:-0.12}"
MAX_REJECT_HOLD_SEC="${MAX_REJECT_HOLD_SEC:-2.5}"
STALE_GLOBAL_MATCH_MARGIN="${STALE_GLOBAL_MATCH_MARGIN:-0.04}"
HOLD_LAST_POSE_ON_REJECT="${HOLD_LAST_POSE_ON_REJECT:-true}"
DIAGNOSTICS_PREFIX="${DIAGNOSTICS_PREFIX:-/localization}"
ROS_LOG_DIR="${ROS_LOG_DIR:-$REPO_ROOT/log/ros}"
HF_HOME="${HF_HOME:-$REPO_ROOT/data/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-47}"
BAG_RATE="${BAG_RATE:-0.5}"
CHECK_ONLY=0
SEQUENCE=""
RUN_DIR=""
PIDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--check-only] [--rate <rate>] [--run-dir <dir>] <outdoor_sequence>

Records:
  $POSE_TOPIC
  $ODOM_TOPIC
  /clock

Outputs:
  data/eval/<sequence>/<timestamp>/record
  data/eval/<sequence>/<timestamp>/*.log
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

    [[ -f "$WORKSPACE_SETUP" ]] || die "Workspace setup not found: $WORKSPACE_SETUP"
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
    else
        PARAMS_FILE="$REPO_ROOT/config/m2dgr_lio_sam_params.yaml"
    fi
    [[ -f "$PARAMS_FILE" ]] || die "LIO-SAM params file not found: $PARAMS_FILE"
}

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "=== Cleaning up background processes ==="
        for pid in "${PIDS[@]}"; do
            kill -INT "$pid" 2>/dev/null || true
        done
        sleep 3
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
    exit "$code"
}

start_bg() {
    local name="$1"
    shift
    echo "=== Starting $name ==="
    "$@" > "$RUN_DIR/$name.log" 2>&1 &
    PIDS+=("$!")
}

parse_args "$@"
source_setup
use_local_gtsam_if_present
choose_lio_params

export ROS_DOMAIN_ID
export ROS_LOG_DIR
export HF_HOME
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
mkdir -p "$ROS_LOG_DIR" "$HF_HOME"

BAG_DIR="$M2DGR_ROS2_ROOT/$SEQUENCE"
MAP_DIR="$HYBRID_MAP_ROOT/$SEQUENCE"

[[ -d "$BAG_DIR" ]] || die "Bag directory not found: $BAG_DIR"
[[ -f "$BAG_DIR/metadata.yaml" ]] || die "ROS2 bag metadata not found: $BAG_DIR/metadata.yaml"
[[ -f "$MAP_DIR/map_index.faiss" ]] || die "Map index not found: $MAP_DIR/map_index.faiss"
[[ -f "$MAP_DIR/keyframe_poses.npy" ]] || die "Keyframe poses not found: $MAP_DIR/keyframe_poses.npy"
[[ -f "$MAP_DIR/keyframe_ids.npy" ]] || die "Keyframe IDs not found: $MAP_DIR/keyframe_ids.npy"

require_ros_executable tf2_ros static_transform_publisher
require_ros_executable lio_sam lio_sam_imuPreintegration
require_ros_executable lio_sam lio_sam_imageProjection
require_ros_executable lio_sam lio_sam_featureExtraction
require_ros_executable lio_sam lio_sam_mapOptimization
require_ros_executable hybrid_localization localization_node

BAG_INFO="$(ros2 bag info "$BAG_DIR")"
require_bag_topic "$POINTS_TOPIC"
require_bag_topic "$IMU_TOPIC"
require_bag_topic "$IMAGE_TOPIC"

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$EVAL_ROOT/$SEQUENCE/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/eval_env.txt" <<EOF
sequence=$SEQUENCE
bag_dir=$BAG_DIR
map_dir=$MAP_DIR
params_file=$PARAMS_FILE
bag_rate=$BAG_RATE
ros_domain_id=$ROS_DOMAIN_ID
image_topic=$IMAGE_TOPIC
odom_topic=$ODOM_TOPIC
pose_topic=$POSE_TOPIC
diagnostics_prefix=$DIAGNOSTICS_PREFIX
top_k=$TOP_K
max_pose_jump_m=$MAX_POSE_JUMP_M
max_keyframe_jump=$MAX_KEYFRAME_JUMP
max_keyframe_pose_jump_m=$MAX_KEYFRAME_POSE_JUMP_M
temporal_score_margin=$TEMPORAL_SCORE_MARGIN
min_global_match_margin=$MIN_GLOBAL_MATCH_MARGIN
max_reject_hold_sec=$MAX_REJECT_HOLD_SEC
stale_global_match_margin=$STALE_GLOBAL_MATCH_MARGIN
EOF

echo "Sequence: $SEQUENCE"
echo "Bag:      $BAG_DIR"
echo "Map:      $MAP_DIR"
echo "Run dir:  $RUN_DIR"
echo "Rate:     $BAG_RATE"
echo "Domain:   $ROS_DOMAIN_ID"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "Preflight checks passed."
    exit 0
fi

trap cleanup EXIT INT TERM

start_bg static_tf ros2 run tf2_ros static_transform_publisher \
    0.0 0.0 0.0 0.0 0.0 0.0 map odom --ros-args -p use_sim_time:=true
sleep 1

start_bg lio_imu_preintegration ros2 run lio_sam lio_sam_imuPreintegration \
    --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 0.5
start_bg lio_image_projection ros2 run lio_sam lio_sam_imageProjection \
    --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 0.5
start_bg lio_feature_extraction ros2 run lio_sam lio_sam_featureExtraction \
    --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 0.5
start_bg lio_map_optimization ros2 run lio_sam lio_sam_mapOptimization \
    --ros-args --params-file "$PARAMS_FILE" -p use_sim_time:=true
sleep 2

start_bg localization ros2 run hybrid_localization localization_node --ros-args \
    -p use_sim_time:=true \
    -p "map_dir:=$MAP_DIR" \
    -p "image_topic:=$IMAGE_TOPIC" \
    -p "localized_pose_topic:=$POSE_TOPIC" \
    -p "localized_path_topic:=/localized_path" \
    -p "debug_image_topic:=/localization_debug_image" \
    -p "pose_frame_id:=map" \
    -p "model_name:=$MODEL_NAME" \
    -p "top_k:=$TOP_K" \
    -p "temporal_filter_enabled:=$TEMPORAL_FILTER_ENABLED" \
    -p "max_pose_jump_m:=$MAX_POSE_JUMP_M" \
    -p "max_keyframe_jump:=$MAX_KEYFRAME_JUMP" \
    -p "max_keyframe_pose_jump_m:=$MAX_KEYFRAME_POSE_JUMP_M" \
    -p "temporal_score_margin:=$TEMPORAL_SCORE_MARGIN" \
    -p "min_global_match_margin:=$MIN_GLOBAL_MATCH_MARGIN" \
    -p "max_reject_hold_sec:=$MAX_REJECT_HOLD_SEC" \
    -p "stale_global_match_margin:=$STALE_GLOBAL_MATCH_MARGIN" \
    -p "hold_last_pose_on_reject:=$HOLD_LAST_POSE_ON_REJECT" \
    -p "diagnostics_prefix:=$DIAGNOSTICS_PREFIX" \
    -p publish_debug_image:=false

echo "Waiting for model and ROS graph startup..."
sleep 8

start_bg record ros2 bag record -o "$RUN_DIR/record" \
    "$POSE_TOPIC" "$ODOM_TOPIC" /clock \
    "$DIAGNOSTICS_PREFIX/matched_keyframe_id" \
    "$DIAGNOSTICS_PREFIX/top_keyframe_id" \
    "$DIAGNOSTICS_PREFIX/match_score" \
    "$DIAGNOSTICS_PREFIX/top_match_score" \
    "$DIAGNOSTICS_PREFIX/match_margin" \
    "$DIAGNOSTICS_PREFIX/match_pose_jump" \
    "$DIAGNOSTICS_PREFIX/match_rejected" \
    "$DIAGNOSTICS_PREFIX/match_status" \
    "$DIAGNOSTICS_PREFIX/matched_keyframe_stamp"
sleep 2

echo "=== Playing bag ==="
ros2 bag play "$BAG_DIR" --clock --rate "$BAG_RATE" \
    --topics "$POINTS_TOPIC" "$IMU_TOPIC" "$IMAGE_TOPIC" | tee "$RUN_DIR/bag_play.log"

echo "=== Bag playback finished ==="
sleep 5
echo "Evaluation recording: $RUN_DIR/record"
