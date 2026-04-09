#!/bin/bash
# Hybrid Localization Mapping Pipeline
# Launches LIO-SAM + mapping_node + bag playback

set -e

source /opt/ros/humble/setup.bash
source /home/gauravkh/ros2_ws/install/setup.bash

PARAMS_FILE="/home/gauravkh/ros2_ws/install/lio_sam/share/lio_sam/config/params.yaml"
BAG_DIR="/home/gauravkh/ros2_ws/data/gate_01_mcap"

cleanup() {
    echo "Cleaning up..."
    kill 0 2>/dev/null
    wait
}
trap cleanup EXIT INT TERM

echo "=== Starting Static TF Publisher ==="
ros2 run tf2_ros static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 map odom \
    --ros-args -p use_sim_time:=true &
sleep 1

echo "=== Starting LIO-SAM Nodes ==="
ros2 run lio_sam lio_sam_imuPreintegration --ros-args --params-file "$PARAMS_FILE" &
sleep 0.5
ros2 run lio_sam lio_sam_imageProjection --ros-args --params-file "$PARAMS_FILE" &
sleep 0.5
ros2 run lio_sam lio_sam_featureExtraction --ros-args --params-file "$PARAMS_FILE" &
sleep 0.5
ros2 run lio_sam lio_sam_mapOptimization --ros-args --params-file "$PARAMS_FILE" &
sleep 2

echo "=== Starting Mapping Node ==="
python3 /home/gauravkh/ros2_ws/src/hybrid_localization/hybrid_localization/mapping_node.py &
sleep 8  # Wait for SigLIP model to load

echo "=== Starting Bag Playback ==="
echo "Playing only: /velodyne_points /handsfree/imu /camera/color/image_raw/compressed"
ros2 bag play "$BAG_DIR" --clock \
    --topics /velodyne_points /handsfree/imu /camera/color/image_raw/compressed

echo "=== Bag playback finished ==="
echo "Waiting 5 seconds for final processing..."
sleep 5

echo "=== Pipeline complete ==="
# Kill background processes
cleanup
