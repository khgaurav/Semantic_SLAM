#!/usr/bin/env bash
# Run KAIST localization and record predictions against the dataset global pose.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
KAIST_ROS2_ROOT="${KAIST_ROS2_ROOT:-$REPO_ROOT/data/kaist_ros2}"
HYBRID_MAP_ROOT="${HYBRID_MAP_ROOT:-$REPO_ROOT/data/hybrid_maps}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/data/eval}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/kaist/stereo_left/image_raw/compressed}"
IMAGE_IS_COMPRESSED="${IMAGE_IS_COMPRESSED:-true}"
ODOM_TOPIC="${ODOM_TOPIC:-/kaist/global_pose/odom}"
POSE_TOPIC="${POSE_TOPIC:-/localized_pose}"
PATH_TOPIC="${PATH_TOPIC:-/localized_path}"
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
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-48}"
BAG_RATE="${BAG_RATE:-1.0}"
CHECK_ONLY=0
SEQUENCE=""
MAP_SEQUENCE="${MAP_SEQUENCE:-}"
RUN_DIR=""
PIDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--check-only] [--rate <rate>] [--run-dir <dir>] [--map-sequence <urban38|urban39>] <urban38|urban39>

Records:
  $POSE_TOPIC
  $ODOM_TOPIC
  /clock

Outputs:
  data/eval/<sequence>[_from_<map-sequence>]/<timestamp>/record
  data/eval/<sequence>[_from_<map-sequence>]/<timestamp>/pose_error.csv
  data/eval/<sequence>[_from_<map-sequence>]/<timestamp>/pose_error_summary.json
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

    [[ -f "$WORKSPACE_SETUP" ]] || die "Workspace setup not found: $WORKSPACE_SETUP"
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
    stop_background_processes
    exit "$code"
}

stop_background_processes() {
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
        PIDS=()
    fi
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

export ROS_DOMAIN_ID
export ROS_LOG_DIR
export HF_HOME
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
mkdir -p "$ROS_LOG_DIR" "$HF_HOME"

BAG_DIR="$KAIST_ROS2_ROOT/$SEQUENCE"
MAP_DIR="$HYBRID_MAP_ROOT/$MAP_SEQUENCE"

[[ -d "$BAG_DIR" ]] || die "Bag directory not found: $BAG_DIR. Run scripts/prepare_kaist_urban_bag.sh $SEQUENCE"
[[ -f "$BAG_DIR/metadata.yaml" ]] || die "ROS2 bag metadata not found: $BAG_DIR/metadata.yaml"
[[ -f "$MAP_DIR/map_index.faiss" ]] || die "Map index not found: $MAP_DIR/map_index.faiss"
[[ -f "$MAP_DIR/keyframe_poses.npy" ]] || die "Keyframe poses not found: $MAP_DIR/keyframe_poses.npy"
[[ -f "$MAP_DIR/keyframe_ids.npy" ]] || die "Keyframe IDs not found: $MAP_DIR/keyframe_ids.npy"

require_ros_executable hybrid_localization localization_node

BAG_INFO="$(ros2 bag info "$BAG_DIR")"
require_bag_topic "$IMAGE_TOPIC"
require_bag_topic "$ODOM_TOPIC"

if [[ -z "$RUN_DIR" ]]; then
    RUN_LABEL="$SEQUENCE"
    if [[ "$MAP_SEQUENCE" != "$SEQUENCE" ]]; then
        RUN_LABEL="${SEQUENCE}_from_${MAP_SEQUENCE}"
    fi
    RUN_DIR="$EVAL_ROOT/$RUN_LABEL/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/eval_env.txt" <<EOF
sequence=$SEQUENCE
map_sequence=$MAP_SEQUENCE
bag_dir=$BAG_DIR
map_dir=$MAP_DIR
bag_rate=$BAG_RATE
ros_domain_id=$ROS_DOMAIN_ID
image_topic=$IMAGE_TOPIC
image_is_compressed=$IMAGE_IS_COMPRESSED
odom_topic=$ODOM_TOPIC
pose_topic=$POSE_TOPIC
path_topic=$PATH_TOPIC
diagnostics_prefix=$DIAGNOSTICS_PREFIX
top_k=$TOP_K
debug_candidate_count=$DEBUG_CANDIDATE_COUNT
use_map_time_gate=$USE_MAP_TIME_GATE
map_time_history_len=$MAP_TIME_HISTORY_LEN
map_time_direction_min_delta_sec=$MAP_TIME_DIRECTION_MIN_DELTA_SEC
map_time_direction_min_consensus=$MAP_TIME_DIRECTION_MIN_CONSENSUS
map_time_switch_threshold_frames=$MAP_TIME_SWITCH_THRESHOLD_FRAMES
max_map_time_rewind_sec=$MAX_MAP_TIME_REWIND_SEC
map_time_rewind_penalty_per_sec=$MAP_TIME_REWIND_PENALTY_PER_SEC
use_keyframe_jump_gate=$USE_KEYFRAME_JUMP_GATE
candidate_cluster_radius_m=$CANDIDATE_CLUSTER_RADIUS_M
initial_min_global_match_margin=$INITIAL_MIN_GLOBAL_MATCH_MARGIN
initial_min_cluster_size=$INITIAL_MIN_CLUSTER_SIZE
initial_min_cluster_score=$INITIAL_MIN_CLUSTER_SCORE
relocalization_min_cluster_size=$RELOCALIZATION_MIN_CLUSTER_SIZE
relocalization_min_cluster_score=$RELOCALIZATION_MIN_CLUSTER_SCORE
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
echo "Map seq:  $MAP_SEQUENCE"
echo "Map:      $MAP_DIR"
echo "Run dir:  $RUN_DIR"
echo "Rate:     $BAG_RATE"
echo "Domain:   $ROS_DOMAIN_ID"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "Preflight checks passed."
    exit 0
fi

trap cleanup EXIT INT TERM

start_bg localization ros2 run hybrid_localization localization_node --ros-args \
    -p use_sim_time:=true \
    -p "map_dir:=$MAP_DIR" \
    -p "image_topic:=$IMAGE_TOPIC" \
    -p "image_is_compressed:=$IMAGE_IS_COMPRESSED" \
    -p "localized_pose_topic:=$POSE_TOPIC" \
    -p "localized_path_topic:=/localized_path" \
    -p "debug_image_topic:=/localization_debug_image" \
    -p "pose_frame_id:=map" \
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
    -p "diagnostics_prefix:=$DIAGNOSTICS_PREFIX" \
    -p publish_debug_image:=false

echo "Waiting for model and ROS graph startup..."
sleep 8

start_bg record ros2 bag record -o "$RUN_DIR/record" \
    "$POSE_TOPIC" "$PATH_TOPIC" "$ODOM_TOPIC" /clock \
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
    --topics "$IMAGE_TOPIC" "$ODOM_TOPIC" | tee "$RUN_DIR/bag_play.log"

echo "=== Bag playback finished ==="
sleep 5

echo "=== Stopping recorder and localization nodes ==="
stop_background_processes

python3 "$REPO_ROOT/scripts/analyze_m2dgr_pose_eval.py" \
    "$RUN_DIR/record" \
    --map-dir "$MAP_DIR" \
    --localized-topic "$POSE_TOPIC" \
    --odom-topic "$ODOM_TOPIC" \
    --csv-out "$RUN_DIR/pose_error.csv" \
    --summary-out "$RUN_DIR/pose_error_summary.json"

echo "Evaluation recording: $RUN_DIR/record"
