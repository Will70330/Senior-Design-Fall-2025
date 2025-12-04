#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import pyrealsense2 as rs
import numpy as np
import sys
import time

class RealSenseNode(Node):
    def __init__(self):
        super().__init__('realsense_publisher')
        
        # Parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value

        # Publishers
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        # Pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.setup_stream()
        
        # Timer
        self.timer = self.create_timer(1.0/self.fps, self.timer_callback)
        self.get_logger().info(f"RealSense Node Started: {self.width}x{self.height} @ {self.fps}fps")

    def setup_stream(self):
        # Robust config attempts
        configs = [
            (self.width, self.height, self.fps, True),       # Requested
            (self.width, self.height, 15, True),             # Lower FPS
            (640, 480, 15, False),                           # USB2/Fallback
            (424, 240, 15, True)                             # Lowest
        ]

        for w, h, f, strict_fmt in configs:
            try:
                self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
                fmt = rs.format.bgr8 if strict_fmt else rs.format.any
                self.config.enable_stream(rs.stream.color, w, h, fmt, f)
                
                self.pipeline.start(self.config)
                self.get_logger().info(f"Stream started with: {w}x{h} @ {f}fps")
                return
            except Exception as e:
                self.get_logger().warn(f"Config failed ({w}x{h}): {e}")
                self.config.disable_all_streams()
        
        self.get_logger().error("Failed to start any stream configuration.")
        sys.exit(1)

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            now = self.get_clock().now().to_msg()

            # Publish Color
            color_msg = Image()
            color_msg.header.stamp = now
            color_msg.header.frame_id = "camera_color_frame"
            color_msg.height = color_image.shape[0]
            color_msg.width = color_image.shape[1]
            color_msg.encoding = "bgr8"
            color_msg.is_bigendian = 0
            color_msg.step = color_image.shape[1] * 3
            color_msg.data = color_image.tobytes()
            self.color_pub.publish(color_msg)

            # Publish Depth
            depth_msg = Image()
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = "camera_depth_frame"
            depth_msg.height = depth_image.shape[0]
            depth_msg.width = depth_image.shape[1]
            depth_msg.encoding = "16UC1"
            depth_msg.is_bigendian = 0
            depth_msg.step = depth_image.shape[1] * 2
            depth_msg.data = depth_image.tobytes()
            self.depth_pub.publish(depth_msg)

            # Publish Info (Basic)
            info_msg = CameraInfo()
            info_msg.header.stamp = now
            info_msg.header.frame_id = "camera_color_frame"
            info_msg.width = color_image.shape[1]
            info_msg.height = color_image.shape[0]
            self.info_pub.publish(info_msg)

        except Exception as e:
            self.get_logger().error(f"Frame error: {e}")

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
