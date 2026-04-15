#!/usr/bin/env python3
import os

import cv2
import faiss
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from PIL import Image as PilImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32, Float64, Int32, String
from transformers import AutoModel, AutoProcessor


class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization_node')

        default_map_dir = os.path.expanduser(
            os.environ.get("HYBRID_MAP_DIR", "~/data/hybrid_maps/default")
        )
        self.map_dir = self.declare_parameter('map_dir', default_map_dir).value
        self.image_topic = self.declare_parameter(
            'image_topic', '/camera/color/image_raw/compressed').value
        self.localized_pose_topic = self.declare_parameter(
            'localized_pose_topic', '/localized_pose').value
        self.localized_path_topic = self.declare_parameter(
            'localized_path_topic', '/localized_path').value
        self.debug_image_topic = self.declare_parameter(
            'debug_image_topic', '/localization_debug_image').value
        self.pose_frame_id = self.declare_parameter(
            'pose_frame_id', 'map').value
        self.model_name = self.declare_parameter(
            'model_name', 'google/siglip-base-patch16-224').value
        self.publish_debug_image = self.declare_parameter(
            'publish_debug_image', True).value
        self.top_k = max(
            1,
            int(self.declare_parameter('top_k', 10).value)
        )
        self.temporal_filter_enabled = self.declare_parameter(
            'temporal_filter_enabled', True).value
        self.max_pose_jump_m = float(
            self.declare_parameter('max_pose_jump_m', 4.0).value
        )
        self.max_keyframe_jump = max(
            0,
            int(self.declare_parameter('max_keyframe_jump', 8).value)
        )
        self.max_keyframe_pose_jump_m = float(
            self.declare_parameter('max_keyframe_pose_jump_m', 8.0).value
        )
        self.temporal_score_margin = float(
            self.declare_parameter('temporal_score_margin', 0.05).value
        )
        self.min_global_match_margin = float(
            self.declare_parameter('min_global_match_margin', 0.12).value
        )
        self.max_reject_hold_sec = max(
            0.0,
            float(self.declare_parameter('max_reject_hold_sec', 2.5).value)
        )
        self.stale_global_match_margin = float(
            self.declare_parameter('stale_global_match_margin', 0.04).value
        )
        self.hold_last_pose_on_reject = self.declare_parameter(
            'hold_last_pose_on_reject', True).value

        diagnostics_prefix = self.declare_parameter(
            'diagnostics_prefix', '/localization').value.rstrip('/')
        self.matched_keyframe_topic = self._diagnostic_topic(
            diagnostics_prefix, 'matched_keyframe_id')
        self.top_keyframe_topic = self._diagnostic_topic(
            diagnostics_prefix, 'top_keyframe_id')
        self.match_score_topic = self._diagnostic_topic(
            diagnostics_prefix, 'match_score')
        self.top_match_score_topic = self._diagnostic_topic(
            diagnostics_prefix, 'top_match_score')
        self.match_margin_topic = self._diagnostic_topic(
            diagnostics_prefix, 'match_margin')
        self.match_pose_jump_topic = self._diagnostic_topic(
            diagnostics_prefix, 'match_pose_jump')
        self.match_rejected_topic = self._diagnostic_topic(
            diagnostics_prefix, 'match_rejected')
        self.match_status_topic = self._diagnostic_topic(
            diagnostics_prefix, 'match_status')
        self.matched_keyframe_stamp_topic = self._diagnostic_topic(
            diagnostics_prefix, 'matched_keyframe_stamp')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        self.get_logger().info(f"Loading map from {self.map_dir}...")
        self._load_map()

        self.get_logger().info(f"Loading vision model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name).to(self.device).eval()
        self.get_logger().info("Model loaded!")

        self.bridge = CvBridge()

        self.pose_pub = self.create_publisher(
            PoseStamped, self.localized_pose_topic, 10)
        self.path_pub = self.create_publisher(
            Path, self.localized_path_topic, 10)
        self.debug_pub = self.create_publisher(
            Image, self.debug_image_topic, 10)
        self.matched_keyframe_pub = self.create_publisher(
            Int32, self.matched_keyframe_topic, 10)
        self.top_keyframe_pub = self.create_publisher(
            Int32, self.top_keyframe_topic, 10)
        self.match_score_pub = self.create_publisher(
            Float32, self.match_score_topic, 10)
        self.top_match_score_pub = self.create_publisher(
            Float32, self.top_match_score_topic, 10)
        self.match_margin_pub = self.create_publisher(
            Float32, self.match_margin_topic, 10)
        self.match_pose_jump_pub = self.create_publisher(
            Float32, self.match_pose_jump_topic, 10)
        self.match_rejected_pub = self.create_publisher(
            Bool, self.match_rejected_topic, 10)
        self.match_status_pub = self.create_publisher(
            String, self.match_status_topic, 10)
        self.matched_keyframe_stamp_pub = self.create_publisher(
            Float64, self.matched_keyframe_stamp_topic, 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.pose_frame_id

        self.last_accepted_pose_array = None
        self.last_accepted_keyframe_id = None
        self.last_accepted_map_idx = None
        self.last_accepted_score = 0.0
        self.last_accepted_stamp_sec = None
        self.fallback_started_stamp_sec = None
        self.accepted_count = 0
        self.rejected_count = 0

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Localization node ready. Listening on {self.image_topic}. "
            f"top_k={self.top_k}, max_pose_jump_m={self.max_pose_jump_m:.2f}, "
            f"max_keyframe_jump={self.max_keyframe_jump}, "
            f"max_keyframe_pose_jump_m={self.max_keyframe_pose_jump_m:.2f}, "
            f"max_reject_hold_sec={self.max_reject_hold_sec:.2f}, "
            f"stale_global_match_margin={self.stale_global_match_margin:.3f}")

    def _diagnostic_topic(self, prefix, name):
        if not prefix:
            return f'/{name}'
        return f'{prefix}/{name}'

    def _load_map(self):
        index_path = os.path.join(self.map_dir, "map_index.faiss")
        poses_path = os.path.join(self.map_dir, "keyframe_poses.npy")
        ids_path = os.path.join(self.map_dir, "keyframe_ids.npy")
        stamps_path = os.path.join(self.map_dir, "keyframe_stamps.npy")

        required = (index_path, poses_path, ids_path)
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            missing_names = ", ".join(
                os.path.basename(path) for path in missing)
            self.get_logger().error(
                f"Map files not found in {self.map_dir}: {missing_names}")
            raise RuntimeError("Map files not found.")

        self.faiss_index = faiss.read_index(index_path)
        self.keyframe_poses = np.load(poses_path)
        self.keyframe_ids = np.load(ids_path)
        self.keyframe_stamps = None
        if os.path.exists(stamps_path):
            self.keyframe_stamps = np.load(stamps_path)
            if len(self.keyframe_stamps) != self.faiss_index.ntotal:
                self.get_logger().warn(
                    "Ignoring keyframe_stamps.npy because its length does not "
                    "match the FAISS index.")
                self.keyframe_stamps = None
        else:
            self.get_logger().warn(
                "keyframe_stamps.npy not found. Rebuild the map to enable "
                "timestamp diagnostics.")

        if (
            self.faiss_index.ntotal != len(self.keyframe_poses)
            or self.faiss_index.ntotal != len(self.keyframe_ids)
        ):
            raise RuntimeError(
                "Map index, poses, and IDs have inconsistent lengths.")

        self.get_logger().info(
            f"Map loaded! Contains {self.faiss_index.ntotal} keyframes.")

    def image_callback(self, img_msg):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(
                img_msg, desired_encoding='rgb8')
            pil_img = PilImage.fromarray(cv_img)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        inputs = self.processor(
            images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            global_embedding = vision_outputs.pooler_output
            global_embedding = torch.nn.functional.normalize(
                global_embedding, p=2, dim=1)

        embedding_np = global_embedding.cpu().numpy().astype(np.float32)
        candidates = self._query_candidates(embedding_np)
        if not candidates:
            self.get_logger().warn("No match found in FAISS index.")
            return

        stamp_sec = self._stamp_to_sec(img_msg.header.stamp)
        selection = self._select_candidate(candidates, stamp_sec)
        self._update_fallback_timer(selection, stamp_sec)
        self._publish_match_diagnostics(selection)

        if selection['accepted']:
            candidate = selection['accepted_candidate']
            pose_msg = self._make_pose_msg(
                candidate['pose_array'], img_msg.header.stamp)
            self._publish_pose(pose_msg, append_path=True)
            self._remember_accept(candidate, stamp_sec)
            self.get_logger().info(
                f"{selection['status']}: KF {candidate['keyframe_id']} "
                f"score={candidate['score']:.4f} "
                f"top_kf={selection['top_candidate']['keyframe_id']} "
                f"margin={selection['top_margin']:.4f} "
                f"jump={candidate['pose_jump_m']:.2f}m")
        else:
            self.rejected_count += 1
            if (
                self.hold_last_pose_on_reject
                and self.last_accepted_pose_array is not None
            ):
                pose_msg = self._make_pose_msg(
                    self.last_accepted_pose_array, img_msg.header.stamp)
                self._publish_pose(pose_msg, append_path=False)
                self.get_logger().warn(
                    f"{selection['status']}: holding KF "
                    f"{self.last_accepted_keyframe_id}; rejected top KF "
                    f"{selection['top_candidate']['keyframe_id']} "
                    f"score={selection['top_candidate']['score']:.4f} "
                    f"margin={selection['top_margin']:.4f} "
                    f"jump={selection['top_candidate']['pose_jump_m']:.2f}m")
            else:
                self.get_logger().warn(
                    f"{selection['status']}: rejected top KF "
                    f"{selection['top_candidate']['keyframe_id']} and no "
                    "previous pose is available to hold.")

        if self.publish_debug_image:
            self._publish_debug_image(cv_img, embedding_np, img_msg, selection)

    def _query_candidates(self, embedding_np):
        k = min(self.top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(embedding_np, k)
        candidates = []
        for rank, (index, score) in enumerate(zip(indices[0], distances[0])):
            if index < 0:
                continue
            map_idx = int(index)
            keyframe_id = int(self.keyframe_ids[map_idx])
            pose_array = self.keyframe_poses[map_idx]
            pose_jump_m = self._pose_distance(
                pose_array, self.last_accepted_pose_array)
            keyframe_jump = self._keyframe_jump(keyframe_id)
            candidates.append({
                'rank': rank,
                'map_idx': map_idx,
                'keyframe_id': keyframe_id,
                'score': float(score),
                'pose_array': pose_array,
                'pose_jump_m': pose_jump_m,
                'keyframe_jump': keyframe_jump,
            })
        return candidates

    def _select_candidate(self, candidates, stamp_sec):
        top_candidate = candidates[0]
        top_margin = self._score_margin(candidates)
        hold_duration_sec = self._fallback_duration_sec(stamp_sec)
        selection = {
            'accepted': True,
            'accepted_candidate': top_candidate,
            'top_candidate': top_candidate,
            'top_margin': top_margin,
            'hold_duration_sec': hold_duration_sec,
            'status': 'accepted',
        }

        if not self.temporal_filter_enabled:
            selection['status'] = 'accepted_filter_disabled'
            return selection

        if self.last_accepted_pose_array is None:
            selection['status'] = 'accepted_initial'
            return selection

        if self._is_temporally_plausible(top_candidate):
            return selection

        local_candidates = [
            candidate for candidate in candidates
            if self._is_temporally_plausible(candidate)
        ]

        if local_candidates:
            local_candidate = local_candidates[0]
            score_drop = top_candidate['score'] - local_candidate['score']
            if score_drop <= self.temporal_score_margin:
                if self._should_accept_stale_global(top_margin, stamp_sec):
                    selection['status'] = 'accepted_stale_global_relocalization'
                    return selection

                selection['accepted_candidate'] = local_candidate
                selection['status'] = 'temporal_fallback'
                return selection

            if top_margin >= self.min_global_match_margin:
                selection['status'] = 'accepted_global_high_margin'
                return selection

            if self._should_accept_stale_global(top_margin, stamp_sec):
                selection['status'] = 'accepted_stale_global_relocalization'
                return selection

            return {
                'accepted': False,
                'accepted_candidate': None,
                'top_candidate': top_candidate,
                'local_candidate': local_candidate,
                'top_margin': top_margin,
                'hold_duration_sec': hold_duration_sec,
                'status': 'rejected_ambiguous_global_jump',
            }

        if top_margin >= self.min_global_match_margin:
            selection['status'] = 'accepted_global_high_margin_no_local'
            return selection

        if self._should_accept_stale_global(top_margin, stamp_sec):
            selection['status'] = 'accepted_stale_global_relocalization'
            return selection

        return {
            'accepted': False,
            'accepted_candidate': None,
            'top_candidate': top_candidate,
            'top_margin': top_margin,
            'hold_duration_sec': hold_duration_sec,
            'status': 'rejected_global_jump',
        }

    def _is_temporally_plausible(self, candidate):
        if self.last_accepted_pose_array is None:
            return True
        if candidate['pose_jump_m'] <= self.max_pose_jump_m:
            return True
        if (
            candidate['keyframe_jump'] is not None
            and candidate['keyframe_jump'] <= self.max_keyframe_jump
            and candidate['pose_jump_m'] <= self.max_keyframe_pose_jump_m
        ):
            return True
        return False

    def _score_margin(self, candidates):
        if len(candidates) < 2:
            return 0.0
        return float(candidates[0]['score'] - candidates[1]['score'])

    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _fallback_duration_sec(self, stamp_sec):
        if self.fallback_started_stamp_sec is None:
            return 0.0
        return max(0.0, float(stamp_sec) - float(self.fallback_started_stamp_sec))

    def _should_accept_stale_global(self, top_margin, stamp_sec):
        return (
            self._fallback_duration_sec(stamp_sec) >= self.max_reject_hold_sec
            and top_margin >= self.stale_global_match_margin
        )

    def _update_fallback_timer(self, selection, stamp_sec):
        status = selection['status']
        if status in (
            'temporal_fallback',
            'rejected_ambiguous_global_jump',
            'rejected_global_jump',
        ):
            if self.fallback_started_stamp_sec is None:
                self.fallback_started_stamp_sec = stamp_sec
            return
        self.fallback_started_stamp_sec = None

    def _pose_distance(self, pose_a, pose_b):
        if pose_a is None or pose_b is None:
            return 0.0
        delta = np.array(pose_a[:3]) - np.array(pose_b[:3])
        return float(np.linalg.norm(delta))

    def _keyframe_jump(self, keyframe_id):
        if self.last_accepted_keyframe_id is None:
            return None
        return abs(int(keyframe_id) - int(self.last_accepted_keyframe_id))

    def _make_pose_msg(self, pose_array, stamp):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.pose_frame_id

        pose_msg.pose.position.x = float(pose_array[0])
        pose_msg.pose.position.y = float(pose_array[1])
        pose_msg.pose.position.z = float(pose_array[2])
        pose_msg.pose.orientation.x = float(pose_array[3])
        pose_msg.pose.orientation.y = float(pose_array[4])
        pose_msg.pose.orientation.z = float(pose_array[5])
        pose_msg.pose.orientation.w = float(pose_array[6])
        return pose_msg

    def _publish_pose(self, pose_msg, append_path):
        self.pose_pub.publish(pose_msg)
        if append_path:
            self.path_msg.header.stamp = pose_msg.header.stamp
            self.path_msg.poses.append(pose_msg)
            self.path_pub.publish(self.path_msg)

    def _remember_accept(self, candidate, stamp_sec):
        self.last_accepted_pose_array = candidate['pose_array'].copy()
        self.last_accepted_keyframe_id = candidate['keyframe_id']
        self.last_accepted_map_idx = candidate['map_idx']
        self.last_accepted_score = candidate['score']
        self.last_accepted_stamp_sec = stamp_sec
        self.accepted_count += 1

    def _publish_match_diagnostics(self, selection):
        accepted = selection['accepted']
        accepted_candidate = selection.get('accepted_candidate')
        top_candidate = selection['top_candidate']

        matched_id = -1
        match_score = 0.0
        match_jump = top_candidate['pose_jump_m']
        matched_stamp = 0.0

        if accepted and accepted_candidate is not None:
            matched_id = accepted_candidate['keyframe_id']
            match_score = accepted_candidate['score']
            match_jump = accepted_candidate['pose_jump_m']
            matched_stamp = self._keyframe_stamp(accepted_candidate['map_idx'])
        elif self.last_accepted_keyframe_id is not None:
            matched_id = int(self.last_accepted_keyframe_id)
            match_score = float(self.last_accepted_score)
            matched_stamp = self._keyframe_stamp(self.last_accepted_map_idx)

        self.matched_keyframe_pub.publish(Int32(data=int(matched_id)))
        self.top_keyframe_pub.publish(
            Int32(data=int(top_candidate['keyframe_id'])))
        self.match_score_pub.publish(Float32(data=float(match_score)))
        self.top_match_score_pub.publish(
            Float32(data=float(top_candidate['score'])))
        self.match_margin_pub.publish(
            Float32(data=float(selection['top_margin'])))
        self.match_pose_jump_pub.publish(Float32(data=float(match_jump)))
        self.match_rejected_pub.publish(Bool(data=not accepted))
        self.match_status_pub.publish(String(data=selection['status']))
        self.matched_keyframe_stamp_pub.publish(
            Float64(data=float(matched_stamp)))

    def _keyframe_stamp(self, map_idx):
        if self.keyframe_stamps is None or map_idx is None:
            return 0.0
        return float(self.keyframe_stamps[int(map_idx)])

    def _publish_debug_image(self, cv_img, embedding_np, img_msg, selection):
        try:
            emb_1d = embedding_np.flatten()
            emb_min, emb_max = emb_1d.min(), emb_1d.max()
            emb_norm = np.clip(
                (emb_1d - emb_min) / (emb_max - emb_min + 1e-6) * 255,
                0,
                255
            ).astype(np.uint8)

            bar_height = 50
            emb_bar = np.tile(emb_norm, (bar_height, 1))

            img_w = cv_img.shape[1]
            emb_bar_resized = cv2.resize(
                emb_bar,
                (img_w, bar_height),
                interpolation=cv2.INTER_NEAREST
            )

            emb_heatmap_bgr = cv2.applyColorMap(
                emb_bar_resized, cv2.COLORMAP_JET)
            emb_heatmap_rgb = cv2.cvtColor(emb_heatmap_bgr, cv2.COLOR_BGR2RGB)

            debug_img = np.vstack((cv_img, emb_heatmap_rgb))
            top = selection['top_candidate']
            accepted = selection.get('accepted_candidate')
            if accepted is not None:
                pose_array = accepted['pose_array']
                matched_id = accepted['keyframe_id']
                score = accepted['score']
                jump = accepted['pose_jump_m']
            elif self.last_accepted_pose_array is not None:
                pose_array = self.last_accepted_pose_array
                matched_id = self.last_accepted_keyframe_id
                score = self.last_accepted_score
                jump = top['pose_jump_m']
            else:
                pose_array = top['pose_array']
                matched_id = -1
                score = 0.0
                jump = top['pose_jump_m']

            text_lines = [
                f"{selection['status']}",
                (
                    f"KF: {matched_id} Top: {top['keyframe_id']} "
                    f"Score: {score:.3f} Top: {top['score']:.3f}"
                ),
                (
                    f"Margin: {selection['top_margin']:.3f} "
                    f"Jump: {jump:.2f}m "
                    f"Hold: {selection['hold_duration_sec']:.1f}s"
                ),
                (
                    "Pose: "
                    f"({pose_array[0]:.2f}, "
                    f"{pose_array[1]:.2f}, "
                    f"{pose_array[2]:.2f})"
                )
            ]

            y0, dy = 30, 30
            for i, line in enumerate(text_lines):
                y = y0 + i * dy
                cv2.putText(
                    debug_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 3)
                cv2.putText(
                    debug_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

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
