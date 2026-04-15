#!/usr/bin/env python3
"""Compare hybrid localization poses against a LiDAR odometry reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path
from statistics import median

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_yaw(q) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def percentile(values: np.ndarray, pct: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, pct))


def load_keyframe_lookup(map_dir: Path | None):
    if map_dir is None:
        return None, None
    poses_path = map_dir / "keyframe_poses.npy"
    ids_path = map_dir / "keyframe_ids.npy"
    if not poses_path.exists() or not ids_path.exists():
        return None, None
    return np.load(poses_path), np.load(ids_path)


def nearest_keyframe_id(position: np.ndarray, poses: np.ndarray, ids: np.ndarray):
    deltas = poses[:, :3] - position.reshape(1, 3)
    idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
    return int(ids[idx]), float(np.linalg.norm(deltas[idx]))


def read_pose_topic(bag_dir: Path, topic: str, nested_pose: bool):
    poses = []
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        topics = {conn.topic for conn in reader.connections}
        if topic not in topics:
            raise RuntimeError(
                f"Bag {bag_dir} is missing required topic: {topic}"
            )
        wanted = [conn for conn in reader.connections if conn.topic == topic]
        for conn, _timestamp, rawdata in reader.messages(connections=wanted):
            msg = reader.deserialize(rawdata, conn.msgtype)
            t = stamp_to_sec(msg.header.stamp)
            pose = msg.pose.pose if nested_pose else msg.pose
            poses.append(
                (
                    t,
                    np.array(
                        [
                            float(pose.position.x),
                            float(pose.position.y),
                            float(pose.position.z),
                        ]
                    ),
                    quat_yaw(pose.orientation),
                )
            )
    return poses


def read_pose_streams(
    bag_dir: Path,
    localized_topic: str,
    odom_topic: str,
    reference_bag_dir: Path | None = None,
):
    reference_bag_dir = reference_bag_dir or bag_dir
    localized = read_pose_topic(bag_dir, localized_topic, nested_pose=False)
    odom = read_pose_topic(reference_bag_dir, odom_topic, nested_pose=True)
    return localized, odom


def align_streams(localized, odom, max_dt: float, map_dir: Path | None):
    odom_times = [row[0] for row in odom]
    keyframe_poses, keyframe_ids = load_keyframe_lookup(map_dir)
    rows = []

    for loc_t, loc_pos, loc_yaw in localized:
        insert_at = bisect_left(odom_times, loc_t)
        candidates = []
        if insert_at > 0:
            candidates.append(insert_at - 1)
        if insert_at < len(odom):
            candidates.append(insert_at)
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda idx: abs(odom[idx][0] - loc_t))
        ref_t, ref_pos, ref_yaw = odom[best_idx]
        dt = loc_t - ref_t
        if abs(dt) > max_dt:
            continue

        delta = loc_pos - ref_pos
        keyframe_id = None
        keyframe_dist = None
        if keyframe_poses is not None and keyframe_ids is not None:
            keyframe_id, keyframe_dist = nearest_keyframe_id(
                loc_pos, keyframe_poses, keyframe_ids
            )

        rows.append(
            {
                "t": loc_t,
                "relative_t": 0.0,
                "dt": dt,
                "loc_x": loc_pos[0],
                "loc_y": loc_pos[1],
                "loc_z": loc_pos[2],
                "ref_x": ref_pos[0],
                "ref_y": ref_pos[1],
                "ref_z": ref_pos[2],
                "err_2d": float(np.linalg.norm(delta[:2])),
                "err_3d": float(np.linalg.norm(delta)),
                "yaw_err_deg": abs(math.degrees(wrap_angle(loc_yaw - ref_yaw))),
                "keyframe_id": keyframe_id,
                "keyframe_dist": keyframe_dist,
            }
        )

    if rows:
        t0 = rows[0]["t"]
        for row in rows:
            row["relative_t"] = row["t"] - t0
    return rows


def contiguous_spans(rows, threshold: float):
    spans = []
    active = None
    for row in rows:
        if row["err_3d"] > threshold:
            if active is None:
                active = {
                    "start": row["relative_t"],
                    "end": row["relative_t"],
                    "max_error": row["err_3d"],
                }
            else:
                active["end"] = row["relative_t"]
                active["max_error"] = max(active["max_error"], row["err_3d"])
        elif active is not None:
            spans.append(active)
            active = None
    if active is not None:
        spans.append(active)
    return spans


def compute_summary(localized_count: int, odom_count: int, rows):
    errors_2d = np.array([row["err_2d"] for row in rows], dtype=float)
    errors_3d = np.array([row["err_3d"] for row in rows], dtype=float)
    yaw_errors = np.array([row["yaw_err_deg"] for row in rows], dtype=float)
    dts = np.array([abs(row["dt"]) for row in rows], dtype=float)

    jumps = []
    for prev, cur in zip(rows, rows[1:]):
        dt = cur["relative_t"] - prev["relative_t"]
        step = math.sqrt(
            (cur["loc_x"] - prev["loc_x"]) ** 2
            + (cur["loc_y"] - prev["loc_y"]) ** 2
            + (cur["loc_z"] - prev["loc_z"]) ** 2
        )
        ref_step = math.sqrt(
            (cur["ref_x"] - prev["ref_x"]) ** 2
            + (cur["ref_y"] - prev["ref_y"]) ** 2
            + (cur["ref_z"] - prev["ref_z"]) ** 2
        )
        if step > 3.0 and dt < 2.0:
            jumps.append(
                {
                    "relative_t": cur["relative_t"],
                    "dt": dt,
                    "localization_step": step,
                    "reference_step": ref_step,
                    "from_keyframe_id": prev["keyframe_id"],
                    "to_keyframe_id": cur["keyframe_id"],
                }
            )

    high_error_spans = contiguous_spans(rows, threshold=5.0)
    largest_errors = sorted(rows, key=lambda row: row["err_3d"], reverse=True)[:10]

    return {
        "localized_messages": localized_count,
        "reference_odometry_messages": odom_count,
        "matched_pairs": len(rows),
        "matched_duration_sec": rows[-1]["relative_t"] if rows else 0.0,
        "time_alignment_abs_dt_sec": {
            "median": median(dts) if dts.size else None,
            "p95": percentile(dts, 95) if dts.size else None,
            "max": float(np.max(dts)) if dts.size else None,
        },
        "error_2d_m": {
            "median": percentile(errors_2d, 50),
            "rmse": float(math.sqrt(np.mean(errors_2d * errors_2d)))
            if errors_2d.size
            else None,
            "p90": percentile(errors_2d, 90),
            "p95": percentile(errors_2d, 95),
            "max": float(np.max(errors_2d)) if errors_2d.size else None,
        },
        "error_3d_m": {
            "median": percentile(errors_3d, 50),
            "rmse": float(math.sqrt(np.mean(errors_3d * errors_3d)))
            if errors_3d.size
            else None,
            "p90": percentile(errors_3d, 90),
            "p95": percentile(errors_3d, 95),
            "max": float(np.max(errors_3d)) if errors_3d.size else None,
            "under_1m_fraction": float(np.mean(errors_3d < 1.0))
            if errors_3d.size
            else None,
            "under_2m_fraction": float(np.mean(errors_3d < 2.0))
            if errors_3d.size
            else None,
            "over_5m_fraction": float(np.mean(errors_3d > 5.0))
            if errors_3d.size
            else None,
        },
        "yaw_error_deg": {
            "median": percentile(yaw_errors, 50),
            "p90": percentile(yaw_errors, 90),
            "p95": percentile(yaw_errors, 95),
            "max": float(np.max(yaw_errors)) if yaw_errors.size else None,
        },
        "localization_jumps_over_3m": {
            "count": len(jumps),
            "largest": sorted(
                jumps, key=lambda row: row["localization_step"], reverse=True
            )[:10],
        },
        "error_spans_over_5m": high_error_spans[:20],
        "largest_errors": [
            {
                "relative_t": row["relative_t"],
                "err_3d": row["err_3d"],
                "err_2d": row["err_2d"],
                "yaw_err_deg": row["yaw_err_deg"],
                "keyframe_id": row["keyframe_id"],
            }
            for row in largest_errors
        ],
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare /localized_pose against /lio_sam/mapping/odometry "
            "from a recorded evaluation bag."
        )
    )
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument(
        "--reference-bag-dir",
        type=Path,
        help=(
            "Optional bag directory to read the odometry reference from. "
            "Useful when the current replay's LIO-SAM odometry diverged but "
            "the localization timestamps should be scored against a known-good "
            "reference run."
        ),
    )
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--localized-topic", default="/localized_pose")
    parser.add_argument("--odom-topic", default="/lio_sam/mapping/odometry")
    parser.add_argument("--max-dt", type=float, default=0.15)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    localized, odom = read_pose_streams(
        args.bag_dir,
        args.localized_topic,
        args.odom_topic,
        args.reference_bag_dir,
    )
    rows = align_streams(localized, odom, args.max_dt, args.map_dir)
    if not rows:
        raise RuntimeError(
            "No synchronized pose pairs were found. Try increasing --max-dt."
        )

    summary = compute_summary(len(localized), len(odom), rows)

    csv_out = args.csv_out or args.bag_dir.parent / "pose_error.csv"
    summary_out = args.summary_out or args.bag_dir.parent / "pose_error_summary.json"
    write_csv(csv_out, rows)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {csv_out}")
    print(f"Wrote {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
