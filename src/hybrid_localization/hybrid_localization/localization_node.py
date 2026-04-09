#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import torch
import numpy as np
import faiss
import os
from PIL import Image as PilImage
from transformers import AutoProcessor, AutoModel

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization_node')
        
        # Determine the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load Map
        map_dir = "/home/gauravkh/ros2_ws/data/hybrid_map"
        self.get_logger().info(f"Loading map from {map_dir}...")
        
        index_path = os.path.join(map_dir, "map_index.faiss")
        poses_path = os.path.join(map_dir, "keyframe_poses.npy")
        ids_path = os.path.join(map_dir, "keyframe_ids.npy")
        
        if not os.path.exists(index_path) or not os.path.exists(poses_path):
            self.get_logger().error("Map files not found! Please run the mapping node first.")
            raise RuntimeError("Map files not found.")
            
        self.faiss_index = faiss.read_index(index_path)
        self.keyframe_poses = np.load(poses_path)
        self.keyframe_ids = np.load(ids_path)
        self.get_logger().info(f"Map loaded! Contains {self.faiss_index.ntotal} keyframes.")
        
        # Load SigLIP 2
        self.get_logger().info("Loading SigLIP 2 ViT-B/16...")
        self.model_name = "google/siglip-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.get_logger().info("Model loaded!")
        
        self.bridge = CvBridge()
        
        # Publisher and Subscriber
        self.pose_pub = self.create_publisher(PoseStamped, '/localized_pose', 10)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        
        self.get_logger().info("Localization node ready. Waiting for images...")

    def image_callback(self, img_msg):
        # 1. Image preprocessing
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='rgb8')
            pil_img = PilImage.fromarray(cv_img)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
            
        # 2. Extract Embedding
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            global_embedding = vision_outputs.pooler_output
            global_embedding = torch.nn.functional.normalize(global_embedding, p=2, dim=1)
        
        embedding_np = global_embedding.cpu().numpy().astype(np.float32)
        
        # 3. Query FAISS index for top-1 nearest neighbor
        distances, indices = self.faiss_index.search(embedding_np, 1)
        
        if len(indices) == 0 or len(indices[0]) == 0 or indices[0][0] == -1:
            self.get_logger().warn("No match found in FAISS index.")
            return
            
        best_idx = indices[0][0]
        score = distances[0][0]
        localized_pose_array = self.keyframe_poses[best_idx]
        keyframe_id = self.keyframe_ids[best_idx]
        
        self.get_logger().info(f"Localized! Matched Keyframe ID: {keyframe_id} (Score: {score:.4f})")
        
        # 4. Construct PoseStamped and publish
        pose_msg = PoseStamped()
        pose_msg.header = img_msg.header # Use the image's timestamp and frame_id
        # Default frame_id if empty (should correspond to the map frame LIO-SAM built)
        if not pose_msg.header.frame_id:
            pose_msg.header.frame_id = "map"
            
        pose_msg.pose.position.x = float(localized_pose_array[0])
        pose_msg.pose.position.y = float(localized_pose_array[1])
        pose_msg.pose.position.z = float(localized_pose_array[2])
        
        pose_msg.pose.orientation.x = float(localized_pose_array[3])
        pose_msg.pose.orientation.y = float(localized_pose_array[4])
        pose_msg.pose.orientation.z = float(localized_pose_array[5])
        pose_msg.pose.orientation.w = float(localized_pose_array[6])
        
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
