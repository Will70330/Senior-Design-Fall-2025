import os
import fcntl
import subprocess
import sys

# Common RealSense Vendor ID
VENDOR_ID = "8086"

def reset_usb_device():
    # Find device via lsusb
    try:
        lsusb_out = subprocess.check_output("lsusb", shell=True).decode("utf-8")
    except Exception as e:
        print(f"Error running lsusb: {e}")
        return

    bus = None
    device = None
    
    for line in lsusb_out.splitlines():
        if VENDOR_ID in line:
            print(f"Found RealSense: {line}")
            parts = line.split()
            bus = parts[1]
            device = parts[3].rstrip(':')
            break
    
    if not bus or not device:
        print("No RealSense device found in lsusb.")
        return

    path = f"/dev/bus/usb/{bus}/{device}"
    print(f"Resetting {path}...")

    try:
        fd = os.open(path, os.O_WRONLY)
        try:
            # USBDEVFS_RESET = 21780
            fcntl.ioctl(fd, 21780, 0)
            print("Reset successful! Wait 5 seconds for re-enumeration.")
        finally:
            os.close(fd)
    except Exception as e:
        print(f"Failed to reset: {e}")
        print("Try running with sudo.")

if __name__ == "__main__":
    reset_usb_device()
