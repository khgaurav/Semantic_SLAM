#!/usr/bin/env python3
"""Convert KAIST Complex Urban raw archives into a small ROS2 bag profile.

The converter keeps the framework-facing topics simple:

* /kaist/stereo_left/image_raw/compressed: sensor_msgs/CompressedImage
* /kaist/global_pose/odom: nav_msgs/Odometry

It streams PNG bytes directly from the image tarball, so it does not need to
expand the large image archives onto disk before writing the ROS2 bag.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import tarfile
from pathlib import Path

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import CompressedImage


DEFAULT_IMAGE_TOPIC = "/kaist/stereo_left/image_raw/compressed"
DEFAULT_ODOM_TOPIC = "/kaist/global_pose/odom"
NSEC_PER_SEC = 1_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare urban38/urban39 KAIST archives as ROS2 bags."
    )
    parser.add_argument("sequence", choices=("urban38", "urban39"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data"),
        help="Directory containing data/<sequence>/<sequence>-pankyo_*.tar.gz.",
    )
    parser.add_argument(
        "--bag-root",
        type=Path,
        default=Path("data/kaist_ros2"),
        help="Output root for ROS2 bag directories.",
    )
    parser.add_argument(
        "--image-topic",
        default=DEFAULT_IMAGE_TOPIC,
        help=f"Compressed image topic. Default: {DEFAULT_IMAGE_TOPIC}",
    )
    parser.add_argument(
        "--odom-topic",
        default=DEFAULT_ODOM_TOPIC,
        help=f"Ground-truth odometry topic. Default: {DEFAULT_ODOM_TOPIC}",
    )
    parser.add_argument(
        "--frame-id",
        default="map",
        help="Odometry header frame_id.",
    )
    parser.add_argument(
        "--child-frame-id",
        default="kaist_vehicle",
        help="Odometry child_frame_id.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start offset from the first available timestamp.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Optional duration to convert.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick smoke-test bags.",
    )
    parser.add_argument(
        "--image-rate-divisor",
        type=int,
        default=1,
        help="Keep every Nth image. Default keeps all images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output bag directory.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate archive presence and print the planned output.",
    )
    return parser.parse_args()


def sequence_archives(raw_root: Path, sequence: str) -> dict[str, Path]:
    seq_dir = raw_root / sequence
    prefix = f"{sequence}-pankyo"
    return {
        "calibration": seq_dir / f"{prefix}_calibration.tar.gz",
        "data": seq_dir / f"{prefix}_data.tar.gz",
        "image": seq_dir / f"{prefix}_img.tar.gz",
        "pose": seq_dir / f"{prefix}_pose.tar.gz",
    }


def require_archives(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing KAIST archives:\n  " + "\n  ".join(missing))


def image_member_name(sequence: str) -> str:
    return f"{sequence}-pankyo/image/stereo_left/"


def pose_member_name(sequence: str) -> str:
    return f"{sequence}-pankyo/global_pose.csv"


def stamp_from_image_name(name: str) -> int | None:
    if not name.endswith(".png"):
        return None
    try:
        return int(Path(name).stem)
    except ValueError:
        return None


def first_image_stamp(image_archive: Path, sequence: str) -> int | None:
    prefix = image_member_name(sequence)
    with tarfile.open(image_archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            stamp_ns = stamp_from_image_name(member.name)
            if stamp_ns is not None:
                return stamp_ns
    return None


def matrix_to_quaternion(m: list[float]) -> tuple[float, float, float, float]:
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = m
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def load_global_poses(pose_archive: Path, sequence: str) -> list[dict[str, object]]:
    member_name = pose_member_name(sequence)
    poses: list[dict[str, object]] = []
    with tarfile.open(pose_archive, "r:gz") as tar:
        pose_file = tar.extractfile(member_name)
        if pose_file is None:
            raise SystemExit(f"Pose archive is missing {member_name}")
        lines = (line.decode("utf-8").strip() for line in pose_file)
        for row in csv.reader(line for line in lines if line):
            if len(row) != 13:
                continue
            stamp_ns = int(row[0])
            values = [float(value) for value in row[1:]]
            rotation = [
                values[0],
                values[1],
                values[2],
                values[4],
                values[5],
                values[6],
                values[8],
                values[9],
                values[10],
            ]
            translation = (values[3], values[7], values[11])
            poses.append(
                {
                    "stamp_ns": stamp_ns,
                    "translation": translation,
                    "quaternion": matrix_to_quaternion(rotation),
                }
            )
    poses.sort(key=lambda pose: int(pose["stamp_ns"]))
    if not poses:
        raise SystemExit(f"No poses found in {pose_archive}")
    return poses


def stamp_msg(stamp_ns: int) -> Time:
    msg = Time()
    msg.sec = stamp_ns // NSEC_PER_SEC
    msg.nanosec = stamp_ns % NSEC_PER_SEC
    return msg


def make_topic_metadata(name: str, msg_type: str):
    kwargs = {
        "name": name,
        "type": msg_type,
        "serialization_format": "cdr",
    }
    try:
        return rosbag2_py.TopicMetadata(**kwargs)
    except TypeError:
        kwargs["offered_qos_profiles"] = ""
        return rosbag2_py.TopicMetadata(**kwargs)


def open_writer(output_dir: Path) -> rosbag2_py.SequentialWriter:
    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(output_dir),
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    writer.open(storage_options, converter_options)
    return writer


def make_odom_msg(
    pose: dict[str, object],
    frame_id: str,
    child_frame_id: str,
) -> Odometry:
    stamp_ns = int(pose["stamp_ns"])
    x, y, z = pose["translation"]
    qx, qy, qz, qw = pose["quaternion"]

    msg = Odometry()
    msg.header.stamp = stamp_msg(stamp_ns)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)
    msg.pose.pose.orientation.x = float(qx)
    msg.pose.pose.orientation.y = float(qy)
    msg.pose.pose.orientation.z = float(qz)
    msg.pose.pose.orientation.w = float(qw)
    return msg


def write_poses(
    writer: rosbag2_py.SequentialWriter,
    poses: list[dict[str, object]],
    topic: str,
    frame_id: str,
    child_frame_id: str,
    start_ns: int,
    end_ns: int | None,
) -> int:
    count = 0
    for pose in poses:
        stamp_ns = int(pose["stamp_ns"])
        if stamp_ns < start_ns:
            continue
        if end_ns is not None and stamp_ns > end_ns:
            break
        msg = make_odom_msg(pose, frame_id, child_frame_id)
        writer.write(topic, serialize_message(msg), stamp_ns)
        count += 1
    return count


def write_images(
    writer: rosbag2_py.SequentialWriter,
    image_archive: Path,
    sequence: str,
    topic: str,
    start_ns: int,
    end_ns: int | None,
    max_images: int | None,
    rate_divisor: int,
) -> int:
    prefix = image_member_name(sequence)
    seen = 0
    written = 0
    with tarfile.open(image_archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            stamp_ns = stamp_from_image_name(member.name)
            if stamp_ns is None or stamp_ns < start_ns:
                continue
            if end_ns is not None and stamp_ns > end_ns:
                continue

            seen += 1
            if (seen - 1) % rate_divisor != 0:
                continue

            image_file = tar.extractfile(member)
            if image_file is None:
                continue

            msg = CompressedImage()
            msg.header.stamp = stamp_msg(stamp_ns)
            msg.header.frame_id = "stereo_left"
            msg.format = "png"
            msg.data = image_file.read()

            writer.write(topic, serialize_message(msg), stamp_ns)
            written += 1
            if max_images is not None and written >= max_images:
                break
    return written


def main() -> None:
    args = parse_args()
    if args.image_rate_divisor < 1:
        raise SystemExit("--image-rate-divisor must be >= 1")
    if args.max_images is not None and args.max_images < 1:
        raise SystemExit("--max-images must be >= 1")
    if args.duration_sec is not None and args.duration_sec <= 0.0:
        raise SystemExit("--duration-sec must be positive")

    raw_root = args.raw_root.resolve()
    bag_root = args.bag_root.resolve()
    output_dir = bag_root / args.sequence
    archives = sequence_archives(raw_root, args.sequence)
    require_archives(archives)

    poses = load_global_poses(archives["pose"], args.sequence)
    first_pose_ns = int(poses[0]["stamp_ns"])
    first_img_ns = first_image_stamp(archives["image"], args.sequence)
    if first_img_ns is None:
        raise SystemExit(f"No stereo_left images found in {archives['image']}")

    base_ns = min(first_pose_ns, first_img_ns)
    start_ns = base_ns + int(args.start_sec * NSEC_PER_SEC)
    end_ns = None
    if args.duration_sec is not None:
        end_ns = start_ns + int(args.duration_sec * NSEC_PER_SEC)

    print(f"Sequence:    {args.sequence}")
    print(f"Raw root:    {raw_root}")
    print(f"Output bag:  {output_dir}")
    print(f"Image topic: {args.image_topic}")
    print(f"Odom topic:  {args.odom_topic}")
    print(f"Start ns:    {start_ns}")
    if end_ns is not None:
        print(f"End ns:      {end_ns}")

    if args.check_only:
        print("Preflight checks passed.")
        return

    if output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Output bag already exists: {output_dir}. "
                "Use --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    writer = open_writer(output_dir)
    writer.create_topic(
        make_topic_metadata(
            args.image_topic,
            "sensor_msgs/msg/CompressedImage",
        )
    )
    writer.create_topic(
        make_topic_metadata(
            args.odom_topic,
            "nav_msgs/msg/Odometry",
        )
    )

    image_count = write_images(
        writer,
        archives["image"],
        args.sequence,
        args.image_topic,
        start_ns,
        end_ns,
        args.max_images,
        args.image_rate_divisor,
    )
    pose_count = write_poses(
        writer,
        poses,
        args.odom_topic,
        args.frame_id,
        args.child_frame_id,
        start_ns,
        end_ns,
    )

    print(f"Wrote poses:  {pose_count}")
    print(f"Wrote images: {image_count}")
    print("Done.")


if __name__ == "__main__":
    main()
