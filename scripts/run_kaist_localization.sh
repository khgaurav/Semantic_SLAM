#!/usr/bin/env bash
# Run hybrid localization on a prepared KAIST Complex Urban ROS2 bag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
KAIST_ROS2_ROOT="${KAIST_ROS2_ROOT:-$REPO_ROOT/data/kaist_ros2}"
HYBRID_MAP_ROOT="${HYBRID_MAP_ROOT:-$REPO_ROOT/data/hybrid_maps}"
RVIZ_CONFIG="${RVIZ_CONFIG:-$REPO_ROOT/rviz/kaist_localization.rviz}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/kaist/stereo_left/image_raw/compressed}"
IMAGE_IS_COMPRESSED="${IMAGE_IS_COMPRESSED:-true}"
RAW_VIEW_TOPIC="${RAW_VIEW_TOPIC:-/kaist/stereo_left/image_raw}"
RAW_VIEW_ENCODING="${RAW_VIEW_ENCODING:-mono8}"
DEBUG_TOPIC="${DEBUG_TOPIC:-/localization_debug_image}"
POSE_TOPIC="${POSE_TOPIC:-/localized_pose}"
PATH_TOPIC="${PATH_TOPIC:-/localized_path}"
POSE_FRAME_ID="${POSE_FRAME_ID:-map}"
MODEL_NAME="${MODEL_NAME:-google/siglip-base-patch16-224}"
TOP_K="${TOP_K:-25}"
DEBUG_CANDIDATE_COUNT="${DEBUG_CANDIDATE_COUNT:-10}"
USE_MAP_TIME_GATE="${USE_MAP_TIME_GATE:-true}"
MAP_TIME_HISTORY_LEN="${MAP_TIME_HISTORY_LEN:-8}"
MAP_TIME_DIRECTION_MIN_DELTA_SEC="${MAP_TIME_DIRECTION_MIN_DELTA_SEC:-1.0}"
MAP_TIME_DIRECTION_MIN_CONSENSUS="${MAP_TIME_DIRECTION_MIN_CONSENSUS:-0.7}"
MAP_TIME_SWITCH_THRESHOLD_FRAMES="${MAP_TIME_SWITCH_THRESHOLD_FRAMES:-4}"
MAX_MAP_TIME_REWIND_SEC="${MAX_MAP_TIME_REWIND_SEC:-2.0}"
MAP_TIME_REWIND_PENALTY_PER_SEC="${MAP_TIME_REWIND_PENALTY_PER_SEC:-0.004}"
USE_KEYFRAME_JUMP_GATE="${USE_KEYFRAME_JUMP_GATE:-false}"
CANDIDATE_CLUSTER_RADIUS_M="${CANDIDATE_CLUSTER_RADIUS_M:-20.0}"
INITIAL_MIN_GLOBAL_MATCH_MARGIN="${INITIAL_MIN_GLOBAL_MATCH_MARGIN:-0.003}"
INITIAL_MIN_CLUSTER_SIZE="${INITIAL_MIN_CLUSTER_SIZE:-2}"
INITIAL_MIN_CLUSTER_SCORE="${INITIAL_MIN_CLUSTER_SCORE:-1.9}"
RELOCALIZATION_MIN_CLUSTER_SIZE="${RELOCALIZATION_MIN_CLUSTER_SIZE:-2}"
RELOCALIZATION_MIN_CLUSTER_SCORE="${RELOCALIZATION_MIN_CLUSTER_SCORE:-1.9}"
TEMPORAL_FILTER_ENABLED="${TEMPORAL_FILTER_ENABLED:-true}"
MAX_POSE_JUMP_M="${MAX_POSE_JUMP_M:-4.0}"
MAX_KEYFRAME_JUMP="${MAX_KEYFRAME_JUMP:-8}"
MAX_KEYFRAME_POSE_JUMP_M="${MAX_KEYFRAME_POSE_JUMP_M:-8.0}"
TEMPORAL_SCORE_MARGIN="${TEMPORAL_SCORE_MARGIN:-0.05}"
MIN_GLOBAL_MATCH_MARGIN="${MIN_GLOBAL_MATCH_MARGIN:-0.006}"
MAX_REJECT_HOLD_SEC="${MAX_REJECT_HOLD_SEC:-2.5}"
STALE_GLOBAL_MATCH_MARGIN="${STALE_GLOBAL_MATCH_MARGIN:-0.003}"
HOLD_LAST_POSE_ON_REJECT="${HOLD_LAST_POSE_ON_REJECT:-true}"
DIAGNOSTICS_PREFIX="${DIAGNOSTICS_PREFIX:-/localization}"
ROS_LOG_DIR="${ROS_LOG_DIR:-$REPO_ROOT/log/ros}"
HF_HOME="${HF_HOME:-$REPO_ROOT/data/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
LOCALIZATION_STARTUP_DELAY="${LOCALIZATION_STARTUP_DELAY:-8}"
CHECK_ONLY=0
SEQUENCE=""
MAP_SEQUENCE="${MAP_SEQUENCE:-}"
PIDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--check-only] [--map-sequence <urban38|urban39>] <urban38|urban39>

Expected inputs:
  Bag: data/kaist_ros2/<sequence>/metadata.yaml
  Map: data/hybrid_maps/<map-sequence>/{map_index.faiss,keyframe_poses.npy,keyframe_ids.npy}

Create the bag from raw KAIST archives with:
  scripts/prepare_kaist_urban_bag.sh <sequence>

Build the map with:
  scripts/run_kaist_mapping_pipeline.sh <sequence>
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
            --map-sequence)
                shift
                [[ $# -gt 0 ]] || die "--map-sequence requires a value"
                MAP_SEQUENCE="$1"
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
    if [[ -z "$MAP_SEQUENCE" ]]; then
        MAP_SEQUENCE="$SEQUENCE"
    fi
    MAP_SEQUENCE="$(canonical_sequence "$MAP_SEQUENCE")" || die "Unsupported KAIST map sequence: $MAP_SEQUENCE"
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
mkdir -p "$ROS_LOG_DIR" "$HF_HOME"

BAG_DIR="$KAIST_ROS2_ROOT/$SEQUENCE"
MAP_DIR="$HYBRID_MAP_ROOT/$MAP_SEQUENCE"

[[ -d "$BAG_DIR" ]] || die "Bag directory not found: $BAG_DIR. Run scripts/prepare_kaist_urban_bag.sh $SEQUENCE"
[[ -f "$BAG_DIR/metadata.yaml" ]] || die "ROS2 bag metadata not found: $BAG_DIR/metadata.yaml"
[[ -d "$MAP_DIR" ]] || die "Map directory not found: $MAP_DIR. Run scripts/run_kaist_mapping_pipeline.sh $MAP_SEQUENCE"
[[ -f "$MAP_DIR/map_index.faiss" ]] || die "Map index not found: $MAP_DIR/map_index.faiss"
[[ -f "$MAP_DIR/keyframe_poses.npy" ]] || die "Keyframe poses not found: $MAP_DIR/keyframe_poses.npy"
[[ -f "$MAP_DIR/keyframe_ids.npy" ]] || die "Keyframe IDs not found: $MAP_DIR/keyframe_ids.npy"
[[ -f "$RVIZ_CONFIG" ]] || die "RViz config not found: $RVIZ_CONFIG"

require_python_modules
require_ros_executable hybrid_localization localization_node
require_ros_executable hybrid_localization compressed_image_republisher
require_ros_executable rqt_image_view rqt_image_view
require_ros_executable rviz2 rviz2
require_ros_executable tf2_ros static_transform_publisher

BAG_INFO="$(ros2 bag info "$BAG_DIR")"
require_bag_topic "$IMAGE_TOPIC"

echo "Sequence: $SEQUENCE"
echo "Bag:      $BAG_DIR"
echo "Map seq:  $MAP_SEQUENCE"
echo "Map:      $MAP_DIR"
echo "Camera:   $IMAGE_TOPIC -> $RAW_VIEW_TOPIC ($RAW_VIEW_ENCODING)"
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
    -p "image_is_compressed:=$IMAGE_IS_COMPRESSED" \
    -p "localized_pose_topic:=$POSE_TOPIC" \
    -p "localized_path_topic:=$PATH_TOPIC" \
    -p "debug_image_topic:=$DEBUG_TOPIC" \
    -p "pose_frame_id:=$POSE_FRAME_ID" \
    -p "model_name:=$MODEL_NAME" \
    -p "top_k:=$TOP_K" \
    -p "debug_candidate_count:=$DEBUG_CANDIDATE_COUNT" \
    -p "use_map_time_gate:=$USE_MAP_TIME_GATE" \
    -p "map_time_history_len:=$MAP_TIME_HISTORY_LEN" \
    -p "map_time_direction_min_delta_sec:=$MAP_TIME_DIRECTION_MIN_DELTA_SEC" \
    -p "map_time_direction_min_consensus:=$MAP_TIME_DIRECTION_MIN_CONSENSUS" \
    -p "map_time_switch_threshold_frames:=$MAP_TIME_SWITCH_THRESHOLD_FRAMES" \
    -p "max_map_time_rewind_sec:=$MAX_MAP_TIME_REWIND_SEC" \
    -p "map_time_rewind_penalty_per_sec:=$MAP_TIME_REWIND_PENALTY_PER_SEC" \
    -p "use_keyframe_jump_gate:=$USE_KEYFRAME_JUMP_GATE" \
    -p "candidate_cluster_radius_m:=$CANDIDATE_CLUSTER_RADIUS_M" \
    -p "initial_min_global_match_margin:=$INITIAL_MIN_GLOBAL_MATCH_MARGIN" \
    -p "initial_min_cluster_size:=$INITIAL_MIN_CLUSTER_SIZE" \
    -p "initial_min_cluster_score:=$INITIAL_MIN_CLUSTER_SCORE" \
    -p "relocalization_min_cluster_size:=$RELOCALIZATION_MIN_CLUSTER_SIZE" \
    -p "relocalization_min_cluster_score:=$RELOCALIZATION_MIN_CLUSTER_SCORE" \
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
    -p "input_is_compressed:=$IMAGE_IS_COMPRESSED" \
    -p "output_topic:=$RAW_VIEW_TOPIC" \
    -p "output_encoding:=$RAW_VIEW_ENCODING"

echo "=== Starting static map TF anchor for RViz ==="
start_bg ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0 \
    --roll 0 --pitch 0 --yaw 0 \
    --frame-id "$POSE_FRAME_ID" \
    --child-frame-id "${POSE_FRAME_ID}_anchor"

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
