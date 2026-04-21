#!/usr/bin/env bash
# Prepare KAIST Complex Urban raw archives as ROS2 bag directories.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

[[ -f "$ROS_SETUP" ]] || {
    echo "ERROR: ROS setup not found: $ROS_SETUP" >&2
    exit 1
}

set +u
# shellcheck source=/dev/null
source "$ROS_SETUP"
set -u

cd "$REPO_ROOT"
python3 "$REPO_ROOT/scripts/prepare_kaist_urban_bag.py" "$@"
