#!/usr/bin/env bash
# Check the local runtime environment for the hybrid M2DGR localization stack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-$REPO_ROOT/install/setup.bash}"
LOCAL_GTSAM_ROOT="${LOCAL_GTSAM_ROOT:-$REPO_ROOT/data/local/gtsam}"

echo "=== System ==="
lsb_release -ds 2>/dev/null || cat /etc/os-release
uname -r

echo
echo "=== ROS ==="
if [[ -f "$ROS_SETUP" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "$ROS_SETUP"
    [[ -f "$WORKSPACE_SETUP" ]] && source "$WORKSPACE_SETUP"
    set -u
    ros2 pkg executables hybrid_localization 2>/dev/null || true
else
    echo "Missing ROS setup: $ROS_SETUP"
fi

echo
echo "=== Local GTSAM ==="
if [[ -d "$LOCAL_GTSAM_ROOT/lib" ]]; then
    echo "prefix: $LOCAL_GTSAM_ROOT"
    find "$LOCAL_GTSAM_ROOT/lib" -maxdepth 1 -name 'libgtsam*.so*' -o -name 'libmetis-gtsam.so*' | sort
else
    echo "missing: $LOCAL_GTSAM_ROOT"
fi

echo
echo "=== Python ==="
python3 - <<'PY'
mods = ["torch", "transformers", "faiss", "cv2", "PIL", "numpy"]
for module_name in mods:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "ok")
        print(f"{module_name}: {version}")
    except Exception as exc:
        print(f"{module_name}: MISSING ({exc})")

try:
    import torch
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch GPU check failed: {exc}")
PY
