#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2

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
        self.debug_pub = self.create_publisher(Image, '/localization_debug_image', 10)
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

        # 5. Visual Debug Image
        try:
            # Flatten and normalize embedding
            emb_1d = embedding_np.flatten()
            emb_min, emb_max = emb_1d.min(), emb_1d.max()
            emb_norm = np.clip((emb_1d - emb_min) / (emb_max - emb_min + 1e-6) * 255, 0, 255).astype(np.uint8)
            
            # Create a 50px tall bar
            bar_height = 50
            emb_bar = np.tile(emb_norm, (bar_height, 1))
            
            # Resize strip to match image width
            img_h, img_w = cv_img.shape[:2]
            emb_bar_resized = cv2.resize(emb_bar, (img_w, bar_height), interpolation=cv2.INTER_NEAREST)
            
            # Apply colormap (cv2 expects BGR for colormap usually, but we are producing an RGB heatmap since we will stack it with an RGB image...
            # Actually, applyColorMap returns BGR, so if we are working in RGB, we'll need to convert it)
            emb_heatmap_bgr = cv2.applyColorMap(emb_bar_resized, cv2.COLORMAP_JET)
            emb_heatmap_rgb = cv2.cvtColor(emb_heatmap_bgr, cv2.COLOR_BGR2RGB)
            
            # Append heatmap to the bottom of the image
            debug_img = np.vstack((cv_img, emb_heatmap_rgb))
            
            # Overlay text
            text_lines = [
                f"KF: {keyframe_id} Score: {score:.3f}",
                f"Pose: ({localized_pose_array[0]:.2f}, {localized_pose_array[1]:.2f}, {localized_pose_array[2]:.2f})"
            ]
            
            y0, dy = 30, 30
            for i, line in enumerate(text_lines):
                y = y0 + i * dy
                cv2.putText(debug_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3) # Outline
                cv2.putText(debug_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="rgb8")
            debug_msg.header = img_msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

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
