# Local Data Layout

Keep large dataset archives, prepared ROS2 bags, and generated semantic maps
here. The contents of this directory are ignored except for this README.

Expected ROS2-converted bag layout:

```text
data/m2dgr_ros2/<sequence>/metadata.yaml
```

Expected KAIST Complex Urban raw archive layout:

```text
data/urban38/urban38-pankyo_calibration.tar.gz
data/urban38/urban38-pankyo_data.tar.gz
data/urban38/urban38-pankyo_img.tar.gz
data/urban38/urban38-pankyo_pose.tar.gz
data/urban39/urban39-pankyo_calibration.tar.gz
data/urban39/urban39-pankyo_data.tar.gz
data/urban39/urban39-pankyo_img.tar.gz
data/urban39/urban39-pankyo_pose.tar.gz
```

Prepare KAIST bags with:

```text
scripts/prepare_kaist_urban_bag.sh urban38
scripts/prepare_kaist_urban_bag.sh urban39
```

Prepared KAIST bag layout:

```text
data/kaist_ros2/<sequence>/metadata.yaml
```

KAIST prepared bag topics:

```text
/kaist/stereo_left/image_raw/compressed
/kaist/global_pose/odom
```

To build the semantic map on `urban38` and test localization on `urban39`,
run:

```text
scripts/run_kaist_urban38_map_urban39_test.sh
```

Expected semantic map layout:

```text
data/hybrid_maps/<sequence>/map_index.faiss
data/hybrid_maps/<sequence>/keyframe_poses.npy
data/hybrid_maps/<sequence>/keyframe_ids.npy
data/hybrid_maps/<sequence>/keyframe_stamps.npy
```

The runner scripts also default Hugging Face model downloads to:

```text
data/huggingface/
```
