#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import message_filters

import torch
import numpy as np
import faiss
import os
import signal
from PIL import Image as PilImage
from transformers import AutoProcessor, AutoModel


class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')

        default_map_dir = os.path.expanduser(
            os.environ.get("HYBRID_MAP_DIR", "~/data/hybrid_maps/default")
        )
        self.map_dir = self.declare_parameter('map_dir', default_map_dir).value
        self.odom_topic = self.declare_parameter(
            'odom_topic', '/lio_sam/mapping/odometry').value
        self.image_topic = self.declare_parameter(
            'image_topic', '/camera/color/image_raw/compressed').value
        self.model_name = self.declare_parameter(
            'model_name', 'google/siglip-base-patch16-224').value
        self.min_keyframe_dist = self.declare_parameter(
            'min_keyframe_dist', 1.0).value
        self.min_keyframe_angle = self.declare_parameter(
            'min_keyframe_angle', 0.3).value
        self.max_step_dist = self.declare_parameter(
            'max_step_dist', 50.0).value

        # Determine the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Load SigLIP 2
        self.get_logger().info(f"Loading vision model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name).to(self.device).eval()
        self.get_logger().info("Model loaded!")

        self.bridge = CvBridge()

        # Map storage
        self.embedding_dim = self.model.config.vision_config.hidden_size
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        self.keyframe_poses = []
        self.keyframe_ids = []
        self.keyframe_stamps = []
        self.keyframe_count = 0
        self.frame_count = 0  # total frames received (before filtering)

        self.last_keyframe_pose = None

        # Subscribers
        self.odom_sub = message_filters.Subscriber(
            self,
            Odometry,
            self.odom_topic,
            qos_profile=qos_profile_sensor_data
        )
        self.image_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            self.image_topic,
            qos_profile=qos_profile_sensor_data
        )

        # Synchronizer (approximate time synchronization)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.image_sub],
            queue_size=50,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(
            f"Mapping pipeline ready. Odom: {self.odom_topic}, "
            f"image: {self.image_topic}")

        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum, _frame):
        signal_name = signal.Signals(signum).name
        self.get_logger().info(
            f"Received {signal_name}, shutting down and saving final map...")
        raise KeyboardInterrupt

    def _pose_distance(self, pose_a, pose_b):
        """Euclidean distance between two [x,y,z,qx,qy,qz,qw] pose arrays."""
        delta = np.array(pose_a[:3]) - np.array(pose_b[:3])
        return float(np.linalg.norm(delta))

    def _quat_angle(self, q1, q2):
        """Approximate angular distance (radians) between two quaternions."""
        dot = abs(sum(a * b for a, b in zip(q1[3:7], q2[3:7])))
        dot = min(dot, 1.0)
        return 2.0 * np.arccos(dot)

    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def sync_callback(self, odom_msg, img_msg):
        self.frame_count += 1

        # Extract pose from odometry
        pose = odom_msg.pose.pose
        pos = pose.position
        ori = pose.orientation
        pose_array = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

        # Skip degenerate origin poses (before SLAM initializes)
        if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6:
            if self.frame_count <= 10:
                self.get_logger().info(
                    "Skipping pre-init origin pose "
                    f"(frame {self.frame_count})")
                return

        # Keyframe selection: enforce minimum distance/angle from last keyframe
        if self.last_keyframe_pose is not None:
            dist = self._pose_distance(pose_array, self.last_keyframe_pose)
            angle = self._quat_angle(pose_array, self.last_keyframe_pose)

            # Reject SLAM divergence (impossibly large jump)
            if dist > self.max_step_dist:
                self.get_logger().warn(
                    f"Rejecting divergent pose: {dist:.1f}m jump "
                    f"(frame {self.frame_count})")
                return

            # Skip if too close to last keyframe
            if (
                dist < self.min_keyframe_dist
                and angle < self.min_keyframe_angle
            ):
                return

        self.get_logger().info(
            f"New keyframe {self.keyframe_count} (frame {self.frame_count}) "
            f"at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")

        # 1. Image preprocessing
        cv_img = self.bridge.compressed_imgmsg_to_cv2(
            img_msg, desired_encoding='rgb8')
        pil_img = PilImage.fromarray(cv_img)

        # 2. Extract Embedding
        inputs = self.processor(
            images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            # Use pooled output for global embedding
            global_embedding = vision_outputs.pooler_output
            global_embedding = torch.nn.functional.normalize(
                global_embedding, p=2, dim=1)

        embedding_np = global_embedding.cpu().numpy().astype(np.float32)

        # 3. Add to FAISS index
        self.faiss_index.add(embedding_np)

        # 4. Save Pose
        self.keyframe_poses.append(pose_array)
        self.keyframe_stamps.append(self._stamp_to_sec(img_msg.header.stamp))
        self.last_keyframe_pose = pose_array
        self.keyframe_ids.append(self.keyframe_count)
        self.keyframe_count += 1

        # Periodically save map data
        if self.keyframe_count % 10 == 0:
            self.save_map()

    def save_map(self):
        os.makedirs(self.map_dir, exist_ok=True)

        # Save FAISS index
        faiss.write_index(
            self.faiss_index, os.path.join(self.map_dir, "map_index.faiss"))

        # Save poses
        np.save(
            os.path.join(self.map_dir, "keyframe_poses.npy"),
            np.array(self.keyframe_poses)
        )
        np.save(
            os.path.join(self.map_dir, "keyframe_ids.npy"),
            np.array(self.keyframe_ids)
        )
        np.save(
            os.path.join(self.map_dir, "keyframe_stamps.npy"),
            np.array(self.keyframe_stamps)
        )

        self.get_logger().info(
            f"Saved map with {self.keyframe_count} keyframes "
            f"to {self.map_dir}")


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "Keyboard interrupt, shutting down and saving final map...")
    finally:
        node.save_map()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
