#!/usr/bin/env python3
"""Extract lightweight keyframe preview images for an existing map."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def decode_image(msg, compressed: bool):
    if compressed:
        np_arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    encoding = str(msg.encoding).lower()
    height = int(msg.height)
    width = int(msg.width)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        return data.reshape((height, width, 3)).copy()
    if encoding == "bgr8":
        image_bgr = data.reshape((height, width, 3))
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if encoding == "mono8":
        image_gray = data.reshape((height, width))
        return cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    raise RuntimeError(f"Unsupported raw image encoding: {msg.encoding}")


def resize_preview(rgb_img, target_long_edge: int):
    h, w = rgb_img.shape[:2]
    if h <= 0 or w <= 0:
        return rgb_img
    scale = min(1.0, float(target_long_edge) / float(max(h, w)))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    if out_w == w and out_h == h:
        return rgb_img
    return cv2.resize(rgb_img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract preview JPEGs for map keyframes from a ROS2 bag."
    )
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("map_dir", type=Path)
    parser.add_argument(
        "--image-topic",
        default="/kaist/stereo_left/image_raw/compressed",
    )
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument("--preview-size", type=int, default=320)
    parser.add_argument("--max-stamp-delta-sec", type=float, default=1e-6)
    args = parser.parse_args()

    keyframe_stamps_path = args.map_dir / "keyframe_stamps.npy"
    if not keyframe_stamps_path.exists():
        raise RuntimeError(f"Missing {keyframe_stamps_path}")
    keyframe_stamps = np.load(keyframe_stamps_path)
    preview_dir = args.map_dir / "keyframe_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    stamp_to_indices = {}
    for idx, stamp in enumerate(keyframe_stamps):
        stamp_ns = int(round(float(stamp) * 1e9))
        stamp_to_indices.setdefault(stamp_ns, []).append(idx)

    saved = 0
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([args.bag_dir], default_typestore=typestore) as reader:
        topics = {conn.topic for conn in reader.connections}
        if args.image_topic not in topics:
            raise RuntimeError(
                f"Bag {args.bag_dir} is missing required topic: {args.image_topic}"
            )
        wanted = [conn for conn in reader.connections if conn.topic == args.image_topic]
        for conn, _bag_timestamp, rawdata in reader.messages(connections=wanted):
            msg = reader.deserialize(rawdata, conn.msgtype)
            stamp_ns = int(round(stamp_to_sec(msg.header.stamp) * 1e9))
            matches = stamp_to_indices.pop(stamp_ns, [])
            if not matches:
                continue
            rgb_img = decode_image(msg, compressed=args.compressed)
            if rgb_img is None:
                continue
            preview = resize_preview(rgb_img, args.preview_size)
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            for idx in matches:
                out_path = preview_dir / f"{idx:06d}.jpg"
                cv2.imwrite(str(out_path), preview_bgr)
                saved += 1

    missing = sum(len(v) for v in stamp_to_indices.values())
    print(f"saved={saved}")
    print(f"missing={missing}")
    print(f"preview_dir={preview_dir}")
    if missing:
        print(
            "warning=some keyframe timestamps were not matched exactly; "
            "rerun with a looser extraction method if needed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
