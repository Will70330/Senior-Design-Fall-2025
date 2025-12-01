import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import time

class RealSenseRecorder:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        ctx = rs.context()
        if len(ctx.devices) > 0:
            dev = ctx.devices[0]
            self.device_name = dev.get_info(rs.camera_info.name)
            print(f"Device Name: {self.device_name}")
        else:
            self.device_name = "Unknown"
            print("No device connected")

        self.align = rs.align(rs.stream.color)
        
        # State
        self.streaming = False
        self.recording = False
        self.record_dir = ""
        self.images_dir = ""
        self.frame_idx = 0
        
        self.target_fps = 30
        
        # Camera intrinsics (set on start)
        self.intrinsics = None

    def start(self):
        try:
            # Configure streams
            width, height = 640, 480
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.target_fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.target_fps)
            
            self.profile = self.pipeline.start(self.config)
            self.configure_camera_for_motion()
            
            # Get intrinsics
            profile = self.profile.get_stream(rs.stream.color)
            self.intrinsics = profile.as_video_stream_profile().get_intrinsics()
            
            self.streaming = True
            print("Camera started.")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.streaming = False
            return False

    def configure_camera_for_motion(self):
        """
        Attempts to configure sensor options for better motion capture
        (Global shutter emulation / reduce motion blur).
        """
        try:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            color_sensor = self.profile.get_device().first_color_sensor()

            # Enable emitter for better depth texture (usually good for SFM if active)
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
                
            # Auto-Exposure Priority:
            # 0 - Maintain exposure/framerate (good for motion)
            # 1 - Priority to exposure (good for low light, bad for motion)
            if color_sensor.supports(rs.option.auto_exposure_priority):
                color_sensor.set_option(rs.option.auto_exposure_priority, 0.0)
                
        except Exception as e:
            print(f"Could not configure some camera options: {e}")

    def stop(self):
        if self.streaming:
            self.pipeline.stop()
            self.streaming = False
            print("Camera stopped.")

    def create_recording_dir(self, base_dir=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_dir:
            self.record_dir = os.path.join(base_dir, f"capture_{timestamp}")
        else:
            self.record_dir = os.path.join(os.getcwd(), f"capture_{timestamp}")
            
        self.images_dir = os.path.join(self.record_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.frame_idx = 0
        print(f"Recording to: {self.record_dir}")
        return self.record_dir

    def save_frame(self, color_image, depth_image):
        """
        Saves color and depth images.
        """
        if not self.recording or not self.record_dir:
            return

        cv2.imwrite(os.path.join(self.images_dir, f"frame_{self.frame_idx:05d}.jpg"), color_image)
        cv2.imwrite(os.path.join(self.images_dir, f"depth_{self.frame_idx:05d}.png"), depth_image)
        self.frame_idx += 1

    def get_frames(self):
        """
        Returns (color_image, depth_image) or (None, None)
        """
        if not self.streaming:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
                
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except RuntimeError:
            return None, None
