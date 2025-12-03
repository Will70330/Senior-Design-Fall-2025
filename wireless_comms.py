import pyrealsense2 as rs
import numpy as np
import cv2
import zmq
import json
import time
import threading
import socket
import struct
import sys

class CameraServer:
    def __init__(self):
        self.ctx = rs.context()
        self.pipeline = None
        self.config = rs.config()
        self.streaming = False
        self.device_name = "Unknown"
        
        # Network (ZMQ)
        self.zmq_context = zmq.Context()
        
        # Video Publisher (PUB)
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5556")
        self.pub_socket.setsockopt(zmq.SNDHWM, 2) # Prevent buffering old frames
        
        # Control Server (REP)
        self.rep_socket = self.zmq_context.socket(zmq.REP)
        self.rep_socket.bind("tcp://*:5555")
        
        # Settings
        self.width = 640
        self.height = 480
        self.fps = 30
        self.lock = threading.Lock()
        
        # Discovery Beacon
        self.beacon_active = True
        self.beacon_thread = threading.Thread(target=self._broadcast_beacon, daemon=True)
        self.beacon_thread.start()

    def _broadcast_beacon(self):
        """Broadcasts UDP beacon so laptop can find IP automatically"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        msg = b"3DGS_CAMERA_SERVER"
        while self.beacon_active:
            try:
                sock.sendto(msg, ('<broadcast>', 5554))
            except Exception:
                pass # Network might not be up yet
            time.sleep(1.0)

    def find_and_connect_camera(self):
        """Constantly attempts to connect to the camera"""
        while True:
            try:
                if len(self.ctx.query_devices()) > 0:
                    if not self.streaming:
                        print("Device found. Initializing...")
                        self.start_streaming()
                else:
                    if self.streaming:
                        print("Device lost. Stopping...")
                        self.stop_streaming()
            except Exception as e:
                print(f"Error in connection loop: {e}")
                self.stop_streaming()
            
            # Check for control messages periodically in this loop if not blocking elsewhere
            # But we handle control in a separate thread or non-blocking check
            time.sleep(1)

    def start_streaming(self):
        with self.lock:
            try:
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                
                self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                
                profile = self.pipeline.start(self.config)
                
                # Optimizations
                device = profile.get_device()
                depth_sensor = device.first_depth_sensor()
                color_sensor = device.first_color_sensor()
                
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
                if color_sensor.supports(rs.option.auto_exposure_priority):
                    color_sensor.set_option(rs.option.auto_exposure_priority, 0.0)
                
                self.device_name = device.get_info(rs.camera_info.name)
                self.streaming = True
                print(f"Streaming started: {self.device_name} @ {self.width}x{self.height} {self.fps}fps")
                
            except Exception as e:
                print(f"Failed to start streaming: {e}")
                self.streaming = False

    def stop_streaming(self):
        with self.lock:
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except:
                    pass
                self.pipeline = None
            self.streaming = False
            print("Streaming stopped.")

    def get_capabilities(self):
        """Return available profiles (simplified)"""
        # In a real scenario, we might query the device. 
        # For now, we return the standard D455 supported modes to the GUI.
        return {
            "name": self.device_name,
            "profiles": [
                {"width": 424, "height": 240, "fps": [15, 30, 60]},
                {"width": 640, "height": 480, "fps": [15, 30, 60]},
                {"width": 848, "height": 480, "fps": [15, 30, 60]},
                {"width": 1280, "height": 720, "fps": [15, 30]}
            ]
        }

    def update_settings(self, width, height, fps):
        print(f"Request to change settings: {width}x{height} @ {fps}")
        with self.lock:
            need_restart = (width != self.width or height != self.height or fps != self.fps)
            self.width = width
            self.height = height
            self.fps = fps
            
            if self.streaming and need_restart:
                print("Restarting stream with new settings...")
                self.stop_streaming()
                self.start_streaming()
        return True

    def run_control_loop(self):
        """Handles incoming requests (Config, Capabilities)"""
        while True:
            try:
                # Wait for next request from client
                message = self.rep_socket.recv_json()
                
                response = {"status": "error", "message": "Unknown command"}
                
                cmd = message.get("command")
                if cmd == "get_capabilities":
                    response = {"status": "ok", "data": self.get_capabilities()}
                
                elif cmd == "set_settings":
                    w = message.get("width", 640)
                    h = message.get("height", 480)
                    f = message.get("fps", 30)
                    self.update_settings(w, h, f)
                    response = {"status": "ok"}
                
                elif cmd == "ping":
                    response = {"status": "ok", "streaming": self.streaming}

                self.rep_socket.send_json(response)
            except Exception as e:
                print(f"Control loop error: {e}")
                # Re-create socket if needed?
                time.sleep(0.1)

    def run_stream_loop(self):
        """Captures frames and publishes them"""
        align = rs.align(rs.stream.color)
        
        while True:
            if not self.streaming or not self.pipeline:
                time.sleep(0.1)
                continue
                
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Compress for network efficiency
                # Color -> JPEG (Quality 90)
                # Depth -> PNG (Lossless 16-bit)
                # Note: PNG compression is CPU heavy. 
                # For performance on Pi, we might want raw or fast compression.
                # Let's try encoding.
                
                ret_c, color_encoded = cv2.imencode('.jpg', color_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                ret_d, depth_encoded = cv2.imencode('.png', depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 1]) # Low compression for speed
                
                if ret_c and ret_d:
                    # Send Multipart: [Header, ColorData, DepthData]
                    header = {
                        "frame_idx": frames.get_frame_number(),
                        "timestamp": frames.get_timestamp()
                    }
                    self.pub_socket.send_json(header, flags=zmq.SNDMORE)
                    self.pub_socket.send(color_encoded, flags=zmq.SNDMORE)
                    self.pub_socket.send(depth_encoded)
                    
            except Exception as e:
                print(f"Streaming error: {e}")
                time.sleep(0.1)

if __name__ == "__main__":
    server = CameraServer()
    
    # Threads
    t_ctrl = threading.Thread(target=server.run_control_loop, daemon=True)
    t_stream = threading.Thread(target=server.run_stream_loop, daemon=True)
    
    t_ctrl.start()
    t_stream.start()
    
    print("Wireless Camera Server Running...")
    print("Listening on ports 5555 (Control) and 5556 (Stream)")
    
    # Main thread handles device connection monitoring
    server.find_and_connect_camera()
