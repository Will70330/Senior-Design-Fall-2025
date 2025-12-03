import pyrealsense2 as rs
import sys

try:
    ctx = rs.context()
    print("Querying devices...")
    devices = ctx.query_devices()
    print(f"Count: {len(devices)}")
    if len(devices) > 0:
        for dev in devices:
            print(f"  Device: {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")
            # Try to open
            print("  Attempting to open pipe...")
            pipeline = rs.pipeline()
            pipeline.start()
            print("  Success!")
            pipeline.stop()
    else:
        print("No devices found.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
