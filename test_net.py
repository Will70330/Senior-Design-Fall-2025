import zmq
import sys
import socket
import time

def test_connection(ip):
    print(f"Testing connection to {ip}...")
    
    # 1. TCP Ping (Socket check)
    print(f"[1/3] Checking TCP reachability on port 5555...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((ip, 5555))
        s.close()
        print("  SUCCESS: Port 5555 is open and reachable.")
    except Exception as e:
        print(f"  FAILURE: Could not reach port 5555. Reason: {e}")
        print("  Hint: Check Firewalls (ufw) on the Pi or network isolation.")
        return

    # 2. ZMQ Control Ping
    print(f"[2/3] Testing ZMQ Control Protocol (Ping)...")
    ctx = zmq.Context()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 3000) # 3s timeout
    req.setsockopt(zmq.LINGER, 0)
    
    try:
        req.connect(f"tcp://{ip}:5555")
        print("  Sending 'ping' command...")
        req.send_json({"command": "ping"})
        resp = req.recv_json()
        print(f"  SUCCESS: Received response: {resp}")
    except zmq.Again:
        print("  FAILURE: Request timed out (Resource temporarily unavailable).")
        print("  The port is open, but the script on the Pi isn't responding to the handshake.")
    except Exception as e:
        print(f"  FAILURE: ZMQ Error: {e}")
    finally:
        req.close()

    # 3. ZMQ Stream Check
    print(f"[3/3] Testing Video Stream (Port 5556)...")
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, 3000)
    sub.connect(f"tcp://{ip}:5556")
    sub.subscribe(b"")
    
    try:
        print("  Waiting for a frame (max 3s)...")
        msg = sub.recv_json()
        print("  SUCCESS: Received a frame header!")
    except zmq.Again:
        print("  FAILURE: Stream timed out.")
        print("  The Pi might not be streaming data.")
    finally:
        sub.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        ip = input("Enter Raspberry Pi IP Address: ").strip()
    else:
        ip = sys.argv[1]
    
    test_connection(ip)
