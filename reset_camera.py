import pyrealsense2 as rs
import time

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No devices found to reset.")
else:
    for dev in devices:
        print(f"Resetting device: {dev.get_info(rs.camera_info.name)}")
        dev.hardware_reset()
        print("Reset command sent. Waiting for device to re-enumerate...")
        time.sleep(5)
    print("Done.")
