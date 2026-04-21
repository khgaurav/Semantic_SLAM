#!/usr/bin/env bash
# Run KAIST localization headless, recording poses/path for later playback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [args passed through] <urban38|urban39>

This is a convenience wrapper around:
  scripts/run_kaist_pose_eval.sh

It runs localization without RViz/rqt windows, records:
  /localized_pose
  /localized_path
  /kaist/global_pose/odom
  /clock

The resulting run directory can be replayed with:
  scripts/replay_kaist_localization_results.sh <run_dir>
EOF
}

for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        usage
        exit 0
    fi
done

exec "$SCRIPT_DIR/run_kaist_pose_eval.sh" "$@"
