"""
manual_receiver.py — Training data recorder for wesDroneRL2.

Receives per-timestep flight packets from manual_controller and appends
records to training_data.jsonl (one JSON object per line).

Packet format (4136 bytes, little-endian):
  [4096 bytes] uint8   grayscale image, flat row-major (0-255)
  [   4 bytes] int32   timestep counter
  [   4 bytes] float32 sim timestamp (seconds)
  [   1 byte]  uint8   forward command (0/1)
  [   1 byte]  uint8   yaw_increase command (0/1)
  [   1 byte]  uint8   yaw_decrease command (0/1)
  [   1 byte]  uint8   reset command (0/1)
  [   4 bytes] float32 forward_desired (m/s)
  [   4 bytes] float32 yaw_desired (rad/s)
  [   4 bytes] float32 m1 motor velocity
  [   4 bytes] float32 m2 motor velocity
  [   4 bytes] float32 m3 motor velocity
  [   4 bytes] float32 m4 motor velocity
  [   4 bytes] float32 speed_scalar

Output record format (one JSON line per timestep):
  {
    "timestep":           <int>,
    "timestamp":          <float>,
    "matrix":             [<4096 ints, 0-255, flat row-major>],
    "binary_commands":    [forward, yaw_increase, yaw_decrease, reset],
    "continuous_commands":[forward_desired, yaw_desired],
    "motor_velocities":   [m1, m2, m3, m4],
    "speed_scalar":       <float>
  }

Usage:
  1. Start this script first.
  2. Load manual_controller in Webots and run the simulation.
  3. Fly the drone. Data appends to training_data.jsonl automatically.
  4. Press Ctrl+C or stop Webots to end the session.
  5. Restart to collect more data — all sessions append to the same file.
"""

import os
import socket
import struct
import json
import sys
import signal

PORT        = 8080
PACKET_SIZE = 4136
PACKET_FMT  = '<4096BifBBBBfffffff'
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.jsonl')

assert struct.calcsize(PACKET_FMT) == PACKET_SIZE, \
    f"Format size mismatch: {struct.calcsize(PACKET_FMT)} != {PACKET_SIZE}"


def recv_exact(sock, n):
    """Read exactly n bytes, blocking until all arrive or connection closes."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def parse_packet(data):
    u = struct.unpack(PACKET_FMT, data)
    # Indices: 0-4095 image | 4096 timestep | 4097 timestamp |
    #          4098-4101 binary cmds | 4102-4103 continuous | 4104-4107 motors | 4108 speed
    return {
        'timestep':            int(u[4096]),
        'timestamp':           round(float(u[4097]), 4),
        'matrix':              list(u[:4096]),
        'binary_commands':     [int(u[4098]), int(u[4099]), int(u[4100]), int(u[4101])],
        'continuous_commands': [round(float(u[4102]), 4), round(float(u[4103]), 4)],
        'motor_velocities':    [round(float(u[4104]), 4), round(float(u[4105]), 4),
                                round(float(u[4106]), 4), round(float(u[4107]), 4)],
        'speed_scalar':        round(float(u[4108]), 4),
    }


def count_existing_records():
    if not os.path.exists(OUTPUT_FILE):
        return 0
    with open(OUTPUT_FILE, 'r') as f:
        return sum(1 for line in f if line.strip())


def main():
    total_existing = count_existing_records()

    print(f"\n=== MANUAL RECEIVER ===")
    print(f"  Output file : {OUTPUT_FILE}")
    if total_existing:
        print(f"  Existing records: {total_existing} — new data will be appended.")
    else:
        print(f"  No existing data — starting fresh.")
    print(f"  Listening on port {PORT}...")
    print(f"  Press Ctrl+C to stop.\n")

    # Line-buffered append — each record is flushed immediately
    out = open(OUTPUT_FILE, 'a', buffering=1)
    total_records = total_existing

    def shutdown(sig=None, frame=None):
        print(f"\nShutting down.  Total records in file: {total_records}")
        out.flush()
        out.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', PORT))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print(f"Controller connected: {addr}")
        print(f"  Waiting for first keystroke to begin recording...")
        session_records = 0
        recording = False

        try:
            while True:
                data = recv_exact(conn, PACKET_SIZE)
                if data is None:
                    break

                record = parse_packet(data)
                fwd, yaw_inc, yaw_dec, rst = record['binary_commands']

                # Stop immediately on reset — don't record the reset frame
                if rst:
                    print(f"  Reset received — stopping recording.")
                    break

                # Start recording on first pilot keystroke
                if not recording:
                    if fwd or yaw_inc or yaw_dec:
                        recording = True
                        print(f"  First keystroke — recording started  "
                              f"(speed={record['speed_scalar']:.1f}  "
                              f"t={record['timestamp']:.0f}s)")
                    else:
                        continue

                out.write(json.dumps(record) + '\n')
                session_records += 1
                total_records   += 1

                if session_records % 500 == 0:
                    print(f"  {session_records:5d} new this session  "
                          f"({total_records} total)  "
                          f"speed={record['speed_scalar']:.1f}  "
                          f"t={record['timestamp']:.0f}s")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
            out.flush()
            print(f"Session ended: {session_records} new records  "
                  f"({total_records} total in file)")


if __name__ == '__main__':
    main()
