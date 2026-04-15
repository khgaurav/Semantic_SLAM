#!/usr/bin/env bash
# Run hybrid localization on one ROS2-converted M2DGR outdoor bag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
M2DGR_ROS2_ROOT="${M2DGR_ROS2_ROOT:-$REPO_ROOT/data/m2dgr_ros2}"
HYBRID_MAP_ROOT="${HYBRID_MAP_ROOT:-$REPO_ROOT/data/hybrid_maps}"
RVIZ_CONFIG="${RVIZ_CONFIG:-$REPO_ROOT/rviz/m2dgr_localization.rviz}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/color/image_raw/compressed}"
RAW_VIEW_TOPIC="${RAW_VIEW_TOPIC:-/m2dgr/camera/image_raw}"
DEBUG_TOPIC="${DEBUG_TOPIC:-/localization_debug_image}"
POSE_TOPIC="${POSE_TOPIC:-/localized_pose}"
PATH_TOPIC="${PATH_TOPIC:-/localized_path}"
POSE_FRAME_ID="${POSE_FRAME_ID:-map}"
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
LOCALIZATION_STARTUP_DELAY="${LOCALIZATION_STARTUP_DELAY:-8}"
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

Expected inputs:
  Bag: data/m2dgr_ros2/<sequence>/metadata.yaml
  Map: data/hybrid_maps/<sequence>/{map_index.faiss,keyframe_poses.npy,keyframe_ids.npy}
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

require_python_modules() {
    local missing
    missing="$(python3 - <<'PY'
modules = {
    "faiss": "faiss-cpu",
    "torch": "torch",
    "transformers": "transformers",
}
missing = []
for module_name, package_name in modules.items():
    try:
        __import__(module_name)
    except Exception:
        missing.append(package_name)
print(" ".join(missing))
PY
)"
    [[ -z "$missing" ]] || die "Missing Python modules: $missing. Install with: python3 -m pip install -r requirements.txt"
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
mkdir -p "$ROS_LOG_DIR"
mkdir -p "$HF_HOME"

BAG_DIR="$M2DGR_ROS2_ROOT/$SEQUENCE"
MAP_DIR="$HYBRID_MAP_ROOT/$SEQUENCE"

[[ -d "$BAG_DIR" ]] || die "Bag directory not found: $BAG_DIR"
[[ -f "$BAG_DIR/metadata.yaml" ]] || die "ROS2 bag metadata not found: $BAG_DIR/metadata.yaml"
[[ -d "$MAP_DIR" ]] || die "Map directory not found: $MAP_DIR"
[[ -f "$MAP_DIR/map_index.faiss" ]] || die "Map index not found: $MAP_DIR/map_index.faiss"
[[ -f "$MAP_DIR/keyframe_poses.npy" ]] || die "Keyframe poses not found: $MAP_DIR/keyframe_poses.npy"
[[ -f "$MAP_DIR/keyframe_ids.npy" ]] || die "Keyframe IDs not found: $MAP_DIR/keyframe_ids.npy"
[[ -f "$RVIZ_CONFIG" ]] || die "RViz config not found: $RVIZ_CONFIG"

require_python_modules
require_ros_executable hybrid_localization localization_node
require_ros_executable hybrid_localization compressed_image_republisher
require_ros_executable rqt_image_view rqt_image_view
require_ros_executable rviz2 rviz2

BAG_INFO="$(ros2 bag info "$BAG_DIR")"
require_bag_topic "$IMAGE_TOPIC"

echo "Sequence: $SEQUENCE"
echo "Bag:      $BAG_DIR"
echo "Map:      $MAP_DIR"
echo "Camera:   $IMAGE_TOPIC -> $RAW_VIEW_TOPIC"
echo "Pose:     $POSE_TOPIC"
echo "Path:     $PATH_TOPIC"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "Preflight checks passed."
    exit 0
fi

trap cleanup EXIT INT TERM

echo "=== Starting localization node ==="
start_bg ros2 run hybrid_localization localization_node --ros-args \
    -p use_sim_time:=true \
    -p "map_dir:=$MAP_DIR" \
    -p "image_topic:=$IMAGE_TOPIC" \
    -p "localized_pose_topic:=$POSE_TOPIC" \
    -p "localized_path_topic:=$PATH_TOPIC" \
    -p "debug_image_topic:=$DEBUG_TOPIC" \
    -p "pose_frame_id:=$POSE_FRAME_ID" \
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
    -p "diagnostics_prefix:=$DIAGNOSTICS_PREFIX"

echo "=== Starting camera republisher ==="
start_bg ros2 run hybrid_localization compressed_image_republisher --ros-args \
    -p use_sim_time:=true \
    -p "input_topic:=$IMAGE_TOPIC" \
    -p "output_topic:=$RAW_VIEW_TOPIC"

echo "=== Opening image viewers and RViz ==="
start_bg ros2 run rqt_image_view rqt_image_view "$RAW_VIEW_TOPIC"
start_bg ros2 run rqt_image_view rqt_image_view "$DEBUG_TOPIC"
start_bg ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG" --ros-args -p use_sim_time:=true

echo "Waiting ${LOCALIZATION_STARTUP_DELAY}s for model/map startup..."
sleep "$LOCALIZATION_STARTUP_DELAY"

echo "=== Playing bag ==="
ros2 bag play "$BAG_DIR" --clock --topics "$IMAGE_TOPIC"

echo "=== Bag playback finished ==="
sleep 2
