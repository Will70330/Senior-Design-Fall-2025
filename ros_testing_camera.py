#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult
import numpy as np
import sys
import argparse
import time
import json

# Conditional imports to allow running on devices without specific hardware/libs
try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: 'opencv-python' (cv2) not found. Visualization and Compression will not work.")

class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')
        
        # Check for pyrealsense2
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            self.get_logger().error("pyrealsense2 not found! Cannot publish RealSense images.")
            sys.exit(1)

        # Parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        
        self.restart_required = False
        self.frame_count = 0

        # Publishers
        self.publisher_ = self.create_publisher(CompressedImage, '/camera/color/compressed', 10)
        # Capability publisher (Latched-like behavior via periodic or on-demand, but ROS2 latching is QoS)
        self.cap_publisher_ = self.create_publisher(String, '/camera/capabilities', 10)
        
        # Parameter Callback
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Initialize Camera
        self.pipeline = self.rs.pipeline()
        self.config = self.rs.config()
        
        self.setup_stream()
        
        # Publish capabilities once initialized
        self.publish_capabilities()
        
        # Timer
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Publishing Compressed RealSense images to '/camera/color/compressed' at {self.fps} Hz")

    def get_device_capabilities(self):
        caps = []
        try:
            ctx = self.rs.context()
            if len(ctx.devices) > 0:
                dev = ctx.devices[0]
                sensors = dev.query_sensors()
                for s in sensors:
                    # Check if it's a color sensor (usually first or second)
                    # We can check stream profiles
                    profiles = s.get_stream_profiles()
                    for p in profiles:
                        if p.stream_type() == self.rs.stream.color and p.format() == self.rs.format.bgr8:
                            # Extract w, h, fps
                            # p.as_video_stream_profile() gives width/height
                            vsp = p.as_video_stream_profile()
                            w = vsp.width()
                            h = vsp.height()
                            fps = vsp.fps()
                            entry = {'width': w, 'height': h, 'fps': fps}
                            if entry not in caps:
                                caps.append(entry)
        except Exception as e:
            self.get_logger().error(f"Error querying capabilities: {e}")
        
        # Sort nicely
        caps.sort(key=lambda x: (x['width'], x['height'], x['fps']), reverse=True)
        return caps

    def publish_capabilities(self):
        caps = self.get_device_capabilities()
        msg = String()
        msg.data = json.dumps(caps)
        self.cap_publisher_.publish(msg)
        self.get_logger().info(f"Published {len(caps)} capabilities.")

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'width':
                self.width = param.value
                self.restart_required = True
            elif param.name == 'height':
                self.height = param.value
                self.restart_required = True
            elif param.name == 'fps':
                if param.value > 0:
                    self.fps = param.value
                    self.restart_required = True
        return SetParametersResult(successful=True)

    def restart_stream(self):
        self.get_logger().info("Restarting stream with new parameters...")
        try:
            self.pipeline.stop()
        except Exception:
            pass
            
        # Re-configure
        self.config = self.rs.config()
        self.setup_stream()
        
        # Update timer
        if self.timer:
            self.timer.cancel()
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.restart_required = False
        self.get_logger().info(f"Restart complete: {self.width}x{self.height} @ {self.fps}fps")

    def setup_stream(self):
        try:
            # Try to enable the requested stream
            self.config.enable_stream(self.rs.stream.color, self.width, self.height, self.rs.format.bgr8, self.fps)
            self.pipeline.start(self.config)
        except Exception as e:
            self.get_logger().warn(f"Could not start with requested config: {e}")
            self.get_logger().info("Attempting default configuration...")
            try:
                self.pipeline.start()
            except Exception as e2:
                self.get_logger().error(f"Failed to start camera: {e2}")
                sys.exit(1)

    def timer_callback(self):
        if self.restart_required:
            self.restart_stream()
            return

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            if cv2:
                # Compress Image
                success, encoded_image = cv2.imencode('.jpg', color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                
                if success:
                    # Create ROS CompressedImage message
                    msg = CompressedImage()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "camera_link"
                    msg.format = "jpeg"
                    msg.data = encoded_image.tobytes()
                    
                    self.publisher_.publish(msg)
                    self.frame_count += 1
                    self.get_logger().info(f'Published frame {self.frame_count}')
            else:
                self.get_logger().warn("cv2 not found, cannot compress/publish image.")
                
            # No longer publishing capabilities periodically.
            # Capabilities are published once in __init__ and on demand if requested
            # via a dedicated service/topic (not implemented, but for future extension).
            # if int(time.time()) % 5 == 0:
            #      self.publish_capabilities()

        except RuntimeError as e:
             self.get_logger().warn(f"Frame timeout or error: {e}")

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except:
            pass
        super().destroy_node()


class RemoteListener(Node):
    def __init__(self):
        super().__init__('remote_listener')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.show_gui = True
        self.get_logger().info("Listener started. Waiting for images on '/camera/color/image_raw'...")

    def listener_callback(self, msg):
        # Log basic info
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}, encoding: {msg.encoding}')
        
        if cv2 and self.show_gui:
            try:
                # Convert ROS Image message to numpy array
                # Assuming bgr8 encoding for simplicity as per publisher
                dtype = np.uint8
                n_channels = 3
                if "16" in msg.encoding:
                    dtype = np.uint16
                
                if msg.encoding == "mono8":
                    n_channels = 1
                elif msg.encoding == "mono16":
                    n_channels = 1
                
                # Basic buffer reconstruction
                img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
                
                # Visualization
                cv2.imshow("Remote Listener View", img)
                cv2.waitKey(1)
            except cv2.error as e:
                self.get_logger().warn(f"OpenCV GUI error (disabling visualization): {e}")
                self.show_gui = False
            except Exception as e:
                self.get_logger().error(f"Error displaying image: {e}")
        elif not cv2:
             self.get_logger().info("cv2 not installed, cannot display image.")

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description='ROS 2 RealSense Test Node')
    parser.add_argument('mode', choices=['pub', 'sub'], help='Mode: "pub" for Publisher (Camera), "sub" for Subscriber (Listener)')
    
    # We need to filter out ROS args if we want to use argparse properly with rclpy
    # But usually we can just parse known args
    # A quick hack for ROS 2 python nodes with argparse is to slice sys.argv
    # However, rclpy.init(args=args) handles ROS args. 
    # Let's try to parse only the known args.
    
    # To avoid conflict with ros args like --ros-args, we use parse_known_args
    parsed_args, unknown_args = parser.parse_known_args()

    node = None
    if parsed_args.mode == 'pub':
        print("Starting Publisher Node...")
        node = RealSensePublisher()
    elif parsed_args.mode == 'sub':
        print("Starting Subscriber Node...")
        node = RemoteListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()
        if cv2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

if __name__ == '__main__':
    main()
