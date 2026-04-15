# Local Data Layout

Keep large M2DGR bags and generated semantic maps here. The contents of this
directory are ignored except for this README.

Expected ROS2-converted bag layout:

```text
data/m2dgr_ros2/<sequence>/metadata.yaml
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
