#!/usr/bin/env python3
import rclpy
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
        self.output_encoding = self.declare_parameter(
            'output_encoding', 'bgr8').value

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            self.input_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Republishing {self.input_topic} as {self.output_topic}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding=self.output_encoding)
            image_msg = self.bridge.cv2_to_imgmsg(
                cv_image, encoding=self.output_encoding)
            image_msg.header = msg.header
            self.publisher.publish(image_msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to republish image: {exc}")


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
