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
        self.image_is_compressed = self.declare_parameter(
            'image_is_compressed', True).value
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
        self.show_matched_preview = self.declare_parameter(
            'show_matched_preview', True).value
        self.preview_subdir = self.declare_parameter(
            'preview_subdir', 'keyframe_previews').value
        self.debug_candidate_count = max(
            1,
            int(self.declare_parameter('debug_candidate_count', 10).value)
        )
        self.top_k = max(
            1,
            int(self.declare_parameter('top_k', 10).value)
        )
        self.use_map_time_gate = self.declare_parameter(
            'use_map_time_gate', False).value
        self.map_time_history_len = max(
            2,
            int(self.declare_parameter('map_time_history_len', 8).value)
        )
        self.map_time_direction_min_delta_sec = max(
            0.0,
            float(
                self.declare_parameter(
                    'map_time_direction_min_delta_sec', 1.0
                ).value
            )
        )
        self.map_time_direction_min_consensus = min(
            1.0,
            max(
                0.5,
                float(
                    self.declare_parameter(
                        'map_time_direction_min_consensus', 0.7
                    ).value
                ),
            ),
        )
        self.map_time_switch_threshold_frames = max(
            1,
            int(
                self.declare_parameter(
                    'map_time_switch_threshold_frames', 4
                ).value
            )
        )
        self.max_map_time_rewind_sec = max(
            0.0,
            float(self.declare_parameter('max_map_time_rewind_sec', 2.0).value)
        )
        self.map_time_rewind_penalty_per_sec = max(
            0.0,
            float(
                self.declare_parameter(
                    'map_time_rewind_penalty_per_sec', 0.004
                ).value
            )
        )
        self.use_keyframe_jump_gate = self.declare_parameter(
            'use_keyframe_jump_gate', True).value
        self.candidate_cluster_radius_m = float(
            self.declare_parameter('candidate_cluster_radius_m', 20.0).value
        )
        self.initial_min_global_match_margin = float(
            self.declare_parameter('initial_min_global_match_margin', 0.01).value
        )
        self.initial_min_cluster_size = max(
            1,
            int(self.declare_parameter('initial_min_cluster_size', 2).value)
        )
        self.initial_min_cluster_score = float(
            self.declare_parameter('initial_min_cluster_score', 1.9).value
        )
        self.relocalization_min_cluster_size = max(
            1,
            int(
                self.declare_parameter(
                    'relocalization_min_cluster_size', 2
                ).value
            )
        )
        self.relocalization_min_cluster_score = float(
            self.declare_parameter(
                'relocalization_min_cluster_score', 1.9
            ).value
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
        if self.use_map_time_gate and self.keyframe_stamps is None:
            self.get_logger().warn(
                "Map-time gating requested, but keyframe_stamps.npy is not "
                "available. Disabling map-time gating.")
            self.use_map_time_gate = False

        self.get_logger().info(f"Loading vision model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name).to(self.device).eval()
        self.get_logger().info("Model loaded!")

        self.bridge = CvBridge()
        self.preview_dir = os.path.join(self.map_dir, self.preview_subdir)
        self.preview_cache = {}
        self.preview_cache_order = []
        self.preview_cache_limit = 32

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
        self.last_accepted_keyframe_stamp = None
        self.last_accepted_map_idx = None
        self.map_time_recent_deltas = []
        self.map_time_countertrend_frames = 0
        self.last_accepted_score = 0.0
        self.last_accepted_stamp_sec = None
        self.fallback_started_stamp_sec = None
        self.accepted_count = 0
        self.rejected_count = 0

        image_msg_type = CompressedImage if self.image_is_compressed else Image
        self.image_sub = self.create_subscription(
            image_msg_type,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Localization node ready. Listening on {self.image_topic}. "
            f"compressed={self.image_is_compressed}, "
            f"show_matched_preview={self.show_matched_preview}, "
            f"top_k={self.top_k}, debug_candidates={self.debug_candidate_count}, "
            f"use_map_time_gate={self.use_map_time_gate}, "
            f"map_time_history_len={self.map_time_history_len}, "
            f"map_time_switch_threshold_frames="
            f"{self.map_time_switch_threshold_frames}, "
            f"max_map_time_rewind_sec={self.max_map_time_rewind_sec:.1f}, "
            f"use_keyframe_jump_gate={self.use_keyframe_jump_gate}, "
            f"cluster_radius_m={self.candidate_cluster_radius_m:.1f}, "
            f"max_pose_jump_m={self.max_pose_jump_m:.2f}, "
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
            if self.image_is_compressed:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(
                    img_msg, desired_encoding='rgb8')
            else:
                cv_img = self.bridge.imgmsg_to_cv2(
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
        self._update_map_time_switch_state(selection)
        self._update_fallback_timer(selection, stamp_sec)
        self._publish_match_diagnostics(selection)

        if selection['accepted']:
            candidate = selection['accepted_candidate']
            pose_msg = self._make_pose_msg(
                candidate['pose_array'], img_msg.header.stamp)
            self._publish_pose(pose_msg, append_path=True)
            self._remember_accept(
                candidate,
                stamp_sec,
                switch_direction=selection.get('switch_direction', False),
            )
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
            keyframe_stamp = self._map_keyframe_stamp(map_idx)
            map_time_delta_sec = self._map_time_delta(keyframe_stamp)
            candidates.append({
                'rank': rank,
                'map_idx': map_idx,
                'keyframe_id': keyframe_id,
                'keyframe_stamp': keyframe_stamp,
                'score': float(score),
                'pose_array': pose_array,
                'pose_jump_m': pose_jump_m,
                'keyframe_jump': keyframe_jump,
                'map_time_delta_sec': map_time_delta_sec,
                'map_time_penalty': self._map_time_penalty(map_time_delta_sec),
            })
        self._annotate_candidate_clusters(candidates)
        return candidates

    def _select_candidate(self, candidates, stamp_sec):
        top_candidate = candidates[0]
        top_margin = self._score_margin(candidates)
        consensus_candidate, consensus_margin = self._best_supported_candidate(
            candidates
        )
        decision_candidate = consensus_candidate
        decision_margin = max(top_margin, consensus_margin)
        hold_duration_sec = self._fallback_duration_sec(stamp_sec)
        map_time_expected_direction = self._map_time_expected_direction()
        selection = {
            'accepted': True,
            'accepted_candidate': decision_candidate,
            'top_candidate': top_candidate,
            'consensus_candidate': consensus_candidate,
            'consensus_margin': consensus_margin,
            'candidates': candidates,
            'top_margin': top_margin,
            'hold_duration_sec': hold_duration_sec,
            'map_time_expected_direction': map_time_expected_direction,
            'map_time_countertrend_frames': self.map_time_countertrend_frames,
            'switch_direction': False,
            'status': 'accepted',
        }

        if not self.temporal_filter_enabled:
            selection['status'] = 'accepted_filter_disabled'
            return selection

        if self.last_accepted_pose_array is None:
            if top_margin >= self.initial_min_global_match_margin:
                selection['accepted_candidate'] = top_candidate
                selection['status'] = 'accepted_initial_high_margin'
                return selection
            if self._is_cluster_confident(consensus_candidate, initial=True):
                selection['status'] = 'accepted_initial_cluster'
                return selection
            return {
                'accepted': False,
                'accepted_candidate': None,
                'top_candidate': top_candidate,
                'consensus_candidate': consensus_candidate,
                'consensus_margin': consensus_margin,
                'candidates': candidates,
                'top_margin': top_margin,
                'hold_duration_sec': hold_duration_sec,
                'status': 'rejected_initial_ambiguous',
            }

        if self._is_temporally_plausible(decision_candidate):
            if decision_candidate['map_idx'] != top_candidate['map_idx']:
                selection['status'] = 'accepted_consensus'
            return selection

        local_candidates = [
            candidate for candidate in candidates
            if self._is_temporally_plausible(candidate)
        ]

        if local_candidates:
            local_candidate, _local_margin = self._best_supported_candidate(
                local_candidates
            )
            score_drop = decision_candidate['score'] - local_candidate['score']
            if self._should_switch_map_time_direction(decision_candidate):
                selection['status'] = 'accepted_direction_switch'
                selection['switch_direction'] = True
                return selection
            if score_drop <= self.temporal_score_margin:
                if self._should_accept_stale_global(
                    decision_margin, decision_candidate, stamp_sec
                ):
                    selection['status'] = 'accepted_stale_global_relocalization'
                    return selection

                selection['accepted_candidate'] = local_candidate
                selection['status'] = 'temporal_fallback'
                return selection

            if self._is_global_relocalization_confident(
                decision_candidate, decision_margin
            ):
                selection['status'] = 'accepted_global_high_margin'
                return selection

            if self._should_accept_stale_global(
                decision_margin, decision_candidate, stamp_sec
            ):
                selection['status'] = 'accepted_stale_global_relocalization'
                return selection

            return {
                'accepted': False,
                'accepted_candidate': None,
                'top_candidate': top_candidate,
                'consensus_candidate': consensus_candidate,
                'consensus_margin': consensus_margin,
                'local_candidate': local_candidate,
                'candidates': candidates,
                'top_margin': top_margin,
                'hold_duration_sec': hold_duration_sec,
                'status': 'rejected_ambiguous_global_jump',
            }

        if self._should_switch_map_time_direction(decision_candidate):
            selection['status'] = 'accepted_direction_switch'
            selection['switch_direction'] = True
            return selection

        if self._is_global_relocalization_confident(
            decision_candidate, decision_margin
        ):
            selection['status'] = 'accepted_global_high_margin_no_local'
            return selection

        if self._should_accept_stale_global(
            decision_margin, decision_candidate, stamp_sec
        ):
            selection['status'] = 'accepted_stale_global_relocalization'
            return selection

        return {
            'accepted': False,
            'accepted_candidate': None,
            'top_candidate': top_candidate,
            'consensus_candidate': consensus_candidate,
            'consensus_margin': consensus_margin,
            'candidates': candidates,
            'top_margin': top_margin,
            'hold_duration_sec': hold_duration_sec,
            'status': 'rejected_global_jump',
        }

    def _annotate_candidate_clusters(self, candidates):
        if not candidates:
            return
        for candidate in candidates:
            cluster_members = [
                other for other in candidates
                if self._pose_distance(
                    candidate['pose_array'], other['pose_array']
                ) <= self.candidate_cluster_radius_m
            ]
            candidate['cluster_size'] = len(cluster_members)
            candidate['cluster_score'] = float(
                sum(other['score'] for other in cluster_members)
            )
            candidate['selection_score'] = self._selection_score(candidate)

    def _best_supported_candidate(self, candidates):
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.get('selection_score', self._selection_score(candidate)),
                candidate.get('cluster_score', candidate['score']),
                candidate.get('cluster_size', 1),
                candidate['score'],
            ),
            reverse=True,
        )
        best = ranked[0]
        second_score = (
            ranked[1].get(
                'selection_score', self._selection_score(ranked[1])
            )
            if len(ranked) > 1 else 0.0
        )
        best_score = best.get('selection_score', self._selection_score(best))
        return best, float(best_score - second_score)

    def _map_time_trend_sign(self, map_time_delta_sec):
        if map_time_delta_sec is None:
            return 0
        if abs(float(map_time_delta_sec)) < self.map_time_direction_min_delta_sec:
            return 0
        return 1 if map_time_delta_sec > 0.0 else -1

    def _map_time_expected_direction(self):
        if not self.use_map_time_gate:
            return 0
        filtered = [
            delta for delta in self.map_time_recent_deltas
            if abs(delta) >= self.map_time_direction_min_delta_sec
        ]
        if len(filtered) < 2:
            return 0
        positive = sum(1 for delta in filtered if delta > 0.0)
        negative = sum(1 for delta in filtered if delta < 0.0)
        dominant = max(positive, negative)
        if dominant / float(len(filtered)) < self.map_time_direction_min_consensus:
            return 0
        return 1 if positive >= negative else -1

    def _is_cluster_confident(self, candidate, initial=False):
        min_size = (
            self.initial_min_cluster_size if initial
            else self.relocalization_min_cluster_size
        )
        min_score = (
            self.initial_min_cluster_score if initial
            else self.relocalization_min_cluster_score
        )
        return (
            candidate.get('cluster_size', 1) >= min_size
            and candidate.get('cluster_score', candidate['score']) >= min_score
        )

    def _is_global_relocalization_confident(self, candidate, margin):
        return (
            margin >= self.min_global_match_margin
            or self._is_cluster_confident(candidate, initial=False)
        )

    def _is_temporally_plausible(self, candidate, allow_direction_switch=False):
        if self.last_accepted_pose_array is None:
            return True
        if not self._is_map_time_plausible(candidate, allow_direction_switch):
            return False
        if candidate['pose_jump_m'] <= self.max_pose_jump_m:
            return True
        if not self.use_keyframe_jump_gate:
            return False
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

    def _map_keyframe_stamp(self, map_idx):
        if self.keyframe_stamps is None or map_idx is None:
            return None
        return float(self.keyframe_stamps[int(map_idx)])

    def _map_time_delta(self, keyframe_stamp):
        if (
            not self.use_map_time_gate
            or keyframe_stamp is None
            or self.last_accepted_keyframe_stamp is None
        ):
            return None
        return float(keyframe_stamp - self.last_accepted_keyframe_stamp)

    def _map_time_countertrend_sec(self, map_time_delta_sec):
        if map_time_delta_sec is None:
            return 0.0
        expected_direction = self._map_time_expected_direction()
        if expected_direction == 0:
            return max(
                0.0,
                -float(map_time_delta_sec) - self.max_map_time_rewind_sec,
            )
        return max(
            0.0,
            -(expected_direction * float(map_time_delta_sec))
            - self.max_map_time_rewind_sec,
        )

    def _map_time_penalty(self, map_time_delta_sec):
        countertrend_sec = self._map_time_countertrend_sec(map_time_delta_sec)
        return countertrend_sec * self.map_time_rewind_penalty_per_sec

    def _selection_score(self, candidate):
        return (
            candidate.get('cluster_score', candidate['score'])
            - candidate.get('map_time_penalty', 0.0)
        )

    def _is_countertrend_candidate(self, candidate):
        return self._map_time_countertrend_sec(
            candidate.get('map_time_delta_sec')
        ) > 0.0

    def _should_switch_map_time_direction(self, candidate):
        if not self.use_map_time_gate:
            return False
        if not self._is_countertrend_candidate(candidate):
            return False
        if self.map_time_countertrend_frames < self.map_time_switch_threshold_frames:
            return False
        return self._is_global_relocalization_confident(
            candidate, candidate.get('selection_score', candidate['score'])
        )

    def _is_map_time_plausible(self, candidate, allow_direction_switch=False):
        if not self.use_map_time_gate:
            return True
        if candidate.get('map_time_delta_sec') is None:
            return True
        if self._map_time_countertrend_sec(candidate.get('map_time_delta_sec')) <= 0.0:
            return True
        return allow_direction_switch and self._should_switch_map_time_direction(
            candidate
        )

    def _should_accept_stale_global(self, top_margin, candidate, stamp_sec):
        return (
            self._fallback_duration_sec(stamp_sec) >= self.max_reject_hold_sec
            and (
                top_margin >= self.stale_global_match_margin
                or self._is_cluster_confident(candidate, initial=False)
            )
        )

    def _update_fallback_timer(self, selection, stamp_sec):
        status = selection['status']
        if status in (
            'temporal_fallback',
            'rejected_initial_ambiguous',
            'rejected_ambiguous_global_jump',
            'rejected_global_jump',
        ):
            if self.fallback_started_stamp_sec is None:
                self.fallback_started_stamp_sec = stamp_sec
            return
        self.fallback_started_stamp_sec = None

    def _update_map_time_switch_state(self, selection):
        if not self.use_map_time_gate:
            self.map_time_countertrend_frames = 0
            return
        probe_candidate = (
            selection.get('consensus_candidate')
            or selection.get('top_candidate')
            or selection.get('accepted_candidate')
        )
        if probe_candidate is None:
            self.map_time_countertrend_frames = 0
            return
        expected_direction = selection.get(
            'map_time_expected_direction',
            self._map_time_expected_direction(),
        )
        if expected_direction == 0:
            self.map_time_countertrend_frames = 0
            return
        if selection.get('switch_direction'):
            self.map_time_countertrend_frames = 0
            return
        if self._is_countertrend_candidate(probe_candidate):
            self.map_time_countertrend_frames += 1
        else:
            self.map_time_countertrend_frames = 0

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

    def _remember_accept(self, candidate, stamp_sec, switch_direction=False):
        self.last_accepted_pose_array = candidate['pose_array'].copy()
        self.last_accepted_keyframe_id = candidate['keyframe_id']
        self.last_accepted_keyframe_stamp = candidate.get('keyframe_stamp')
        self.last_accepted_map_idx = candidate['map_idx']
        map_time_delta_sec = candidate.get('map_time_delta_sec')
        if switch_direction:
            self.map_time_recent_deltas = []
        if map_time_delta_sec is not None:
            self.map_time_recent_deltas.append(float(map_time_delta_sec))
            self.map_time_recent_deltas = self.map_time_recent_deltas[
                -self.map_time_history_len:
            ]
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
        keyframe_stamp = self._map_keyframe_stamp(map_idx)
        if keyframe_stamp is None:
            return 0.0
        return keyframe_stamp

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
            top = selection['top_candidate']
            consensus = selection.get('consensus_candidate', top)
            accepted = selection.get('accepted_candidate')
            matched_candidate = accepted or top
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

            if self.show_matched_preview:
                top_strip = self._build_top_debug_strip(cv_img, matched_candidate)
            else:
                top_strip = cv_img
            emb_heatmap_rgb = cv2.resize(
                emb_heatmap_rgb,
                (top_strip.shape[1], emb_heatmap_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            debug_img = np.vstack((top_strip, emb_heatmap_rgb))

            summary_lines = [
                f"{selection['status']}",
                (
                    f"KF: {matched_id} Top: {top['keyframe_id']} "
                    f"Consensus: {consensus['keyframe_id']} "
                    f"Score: {score:.3f} Top: {top['score']:.3f}"
                ),
                (
                    f"Margin: {selection['top_margin']:.3f} "
                    f"Cluster: {selection.get('consensus_margin', 0.0):.3f} "
                    f"Jump: {jump:.2f}m "
                    f"Trend: {selection.get('map_time_expected_direction', 0):+d} "
                    f"Ctr: {selection.get('map_time_countertrend_frames', 0)}/"
                    f"{self.map_time_switch_threshold_frames} "
                    f"Map dt: "
                    f"{(accepted or top).get('map_time_delta_sec', 0.0) if (accepted or top).get('map_time_delta_sec') is not None else 0.0:+.1f}s "
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
            for i, line in enumerate(summary_lines):
                y = y0 + i * dy
                self._draw_text(debug_img, line, (10, y), 0.8, (0, 255, 0), 2)

            debug_img = self._append_candidate_panel(debug_img, selection)

            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="rgb8")
            debug_msg.header = img_msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

    def _append_candidate_panel(self, debug_img, selection):
        candidates = selection.get('candidates', [])
        if not candidates:
            return debug_img

        count = min(self.debug_candidate_count, len(candidates))
        row_height = 24
        header_height = 30
        footer_height = 8
        panel_h = header_height + row_height * count + footer_height
        panel = np.zeros((panel_h, debug_img.shape[1], 3), dtype=np.uint8)

        accepted = selection.get('accepted_candidate')
        accepted_map_idx = accepted['map_idx'] if accepted is not None else None
        top_map_idx = selection['top_candidate']['map_idx']
        local_map_idx = None
        if selection.get('local_candidate') is not None:
            local_map_idx = selection['local_candidate']['map_idx']

        header = (
            "Top candidates: "
            "rank KF score d_top d_cluster dt_map jump_m grp status"
        )
        self._draw_text(panel, header, (10, 21), 0.58, (255, 255, 255), 1)

        for row, candidate in enumerate(candidates[:count]):
            y = header_height + row * row_height + 18
            margin = float(selection['top_candidate']['score'] - candidate['score'])
            cluster_margin = (
                selection.get('consensus_candidate', candidate).get(
                    'cluster_score', 0.0
                ) - candidate.get('cluster_score', candidate['score'])
            )
            plausible = self._is_temporally_plausible(candidate)
            markers = []
            if candidate['map_idx'] == accepted_map_idx:
                markers.append('ACCEPT')
            if candidate['map_idx'] == top_map_idx:
                markers.append('TOP')
            if (
                selection.get('consensus_candidate') is not None
                and candidate['map_idx']
                == selection['consensus_candidate']['map_idx']
            ):
                markers.append('CONS')
            if candidate['map_idx'] == local_map_idx:
                markers.append('LOCAL')
            if plausible:
                markers.append('OK')
            else:
                markers.append('REJECT')
            map_dt = candidate.get('map_time_delta_sec')
            trend_sign = self._map_time_trend_sign(map_dt)
            if trend_sign > 0:
                markers.append('DT+')
            elif trend_sign < 0:
                markers.append('DT-')
            if self._is_countertrend_candidate(candidate):
                markers.append('CTR')

            map_dt_text = '-' if map_dt is None else f"{map_dt:+5.1f}"
            line = (
                f"{candidate['rank'] + 1:>2} "
                f"KF {candidate['keyframe_id']:<5} "
                f"s={candidate['score']:.4f} "
                f"d={margin:.4f} "
                f"dc={cluster_margin:>6.3f} "
                f"dt={map_dt_text} "
                f"jump={candidate['pose_jump_m']:>6.1f} "
                f"grp={candidate.get('cluster_size', 1):<2}/"
                f"{candidate.get('selection_score', self._selection_score(candidate)):.2f} "
                f"{'/'.join(markers)}"
            )
            color = self._candidate_color(candidate, accepted_map_idx, plausible)
            self._draw_text(panel, line, (10, y), 0.56, color, 1)

        return np.vstack((debug_img, panel))

    def _build_top_debug_strip(self, query_rgb, matched_candidate):
        query_panel = self._annotate_preview_panel(query_rgb.copy(), "Query image")
        matched_panel = self._load_preview_panel(
            matched_candidate.get('map_idx'),
            query_panel.shape[1],
            query_panel.shape[0],
            (
                f"Matched map KF {matched_candidate.get('keyframe_id', -1)} "
                f"s={matched_candidate.get('score', 0.0):.3f}"
            ),
        )
        return np.hstack((query_panel, matched_panel))

    def _annotate_preview_panel(self, image_rgb, label):
        panel = image_rgb.copy()
        overlay_h = 28
        cv2.rectangle(panel, (0, 0), (panel.shape[1], overlay_h), (0, 0, 0), -1)
        self._draw_text(panel, label, (10, 20), 0.6, (255, 255, 255), 1)
        return panel

    def _load_preview_panel(self, map_idx, width, height, label):
        preview_rgb = self._load_keyframe_preview(map_idx)
        if preview_rgb is None:
            preview_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            self._draw_text(
                preview_rgb,
                "No map preview available",
                (20, max(40, height // 2)),
                0.8,
                (255, 255, 255),
                2,
            )
            return self._annotate_preview_panel(preview_rgb, label)

        preview_rgb = self._resize_to_canvas(preview_rgb, width, height)
        return self._annotate_preview_panel(preview_rgb, label)

    def _load_keyframe_preview(self, map_idx):
        if map_idx is None:
            return None
        map_idx = int(map_idx)
        cached = self.preview_cache.get(map_idx)
        if cached is not None:
            return cached.copy()

        preview_path = os.path.join(self.preview_dir, f"{map_idx:06d}.jpg")
        if not os.path.exists(preview_path):
            return None
        preview_bgr = cv2.imread(preview_path, cv2.IMREAD_COLOR)
        if preview_bgr is None:
            return None
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        self.preview_cache[map_idx] = preview_rgb
        self.preview_cache_order.append(map_idx)
        if len(self.preview_cache_order) > self.preview_cache_limit:
            evict_idx = self.preview_cache_order.pop(0)
            self.preview_cache.pop(evict_idx, None)
        return preview_rgb.copy()

    def _resize_to_canvas(self, image_rgb, width, height):
        h, w = image_rgb.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        scale = min(float(width) / float(w), float(height) / float(h))
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y0 = max(0, (height - out_h) // 2)
        x0 = max(0, (width - out_w) // 2)
        canvas[y0:y0 + out_h, x0:x0 + out_w] = resized
        return canvas

    def _candidate_color(self, candidate, accepted_map_idx, plausible):
        if candidate['map_idx'] == accepted_map_idx:
            return (0, 255, 0)
        if candidate['rank'] == 0 and not plausible:
            return (255, 120, 120)
        if plausible:
            return (255, 230, 80)
        return (180, 180, 180)

    def _draw_text(self, image, text, origin, scale, color, thickness):
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
            scale, (0, 0, 0), thickness + 2)
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness)


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
