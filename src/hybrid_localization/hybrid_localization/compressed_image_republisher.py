#!/usr/bin/env python3
import rclpy
import cv2
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class CompressedImageRepublisher(Node):
    def __init__(self):
        super().__init__('compressed_image_republisher')

        self.input_topic = self.declare_parameter(
            'input_topic', '/camera/color/image_raw/compressed').value
        self.output_topic = self.declare_parameter(
            'output_topic', '/m2dgr/camera/image_raw').value
        self.input_is_compressed = self.declare_parameter(
            'input_is_compressed', True).value
        self.output_encoding = self.declare_parameter(
            'output_encoding', 'bgr8').value

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        input_msg_type = CompressedImage if self.input_is_compressed else Image
        self.subscription = self.create_subscription(
            input_msg_type,
            self.input_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Republishing {self.input_topic} as {self.output_topic} "
            f"(compressed={self.input_is_compressed})")

    def image_callback(self, msg):
        try:
            if self.input_is_compressed:
                cv_image = self._compressed_msg_to_cv2(msg)
            else:
                cv_image = self._image_msg_to_cv2(msg)
            cv_image, encoding = self._convert_output_encoding(cv_image)
            image_msg = self.bridge.cv2_to_imgmsg(
                cv_image, encoding=encoding)
            image_msg.header = msg.header
            self.publisher.publish(image_msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to republish image: {exc}")

    def _compressed_msg_to_cv2(self, msg):
        try:
            return self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding=self.output_encoding)
        except Exception:
            return self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='passthrough')

    def _image_msg_to_cv2(self, msg):
        try:
            return self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=self.output_encoding)
        except Exception:
            return self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')

    def _convert_output_encoding(self, cv_image):
        if self.output_encoding == 'passthrough':
            return cv_image, self._infer_encoding(cv_image)

        if self.output_encoding == 'mono8':
            if len(cv_image.shape) == 2:
                return cv_image, 'mono8'
            if cv_image.shape[2] == 4:
                return cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY), 'mono8'
            return cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), 'mono8'

        if self.output_encoding == 'bgr8':
            if len(cv_image.shape) == 2:
                return cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR), 'bgr8'
            if cv_image.shape[2] == 4:
                return cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR), 'bgr8'
            return cv_image, 'bgr8'

        if self.output_encoding == 'rgb8':
            if len(cv_image.shape) == 2:
                return cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB), 'rgb8'
            if cv_image.shape[2] == 4:
                return cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB), 'rgb8'
            return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), 'rgb8'

        return cv_image, self.output_encoding

    def _infer_encoding(self, cv_image):
        if len(cv_image.shape) == 2:
            return 'mono8'
        if cv_image.shape[2] == 4:
            return 'bgra8'
        return 'bgr8'


def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
