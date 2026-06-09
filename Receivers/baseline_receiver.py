# Baseline receiver for the Webots drone controller.
# Receives a 64x64 image array + metadata from the controller, applies Gaussian
# blur, runs the rule-based line-following algorithm, sends back a JSON command,
# and logs performance data to a CSV file.
# By Wesley Junkins

import socket
import struct
import time
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import json
import csv

# ── ADJUSTABLE TEST PARAMETERS ────────────────────────────────────────────────
# TEST_SPEEDS and TEST_BLUR_LEVELS define the full automated test sequence.
# The receiver cycles through all combinations automatically.

TEST_SPEEDS      = [0.2, 0.4, 0.6, 0.8, 1.0]  # 5 speeds at 0.2 increments across full range
TEST_BLUR_LEVELS = [0, 1, 2, 3, 4]
TEST_REPETITIONS = 3  # repeat the full 25-test matrix this many times
# Full sequence: 3 complete passes through all 25 combinations = 75 tests total (375 min)
TEST_SEQUENCE    = [(spd, blr) for spd in TEST_SPEEDS for blr in TEST_BLUR_LEVELS] * TEST_REPETITIONS

TIME_LIMIT_SECONDS = 300  # 5 minutes per test run

# Gaussian sigma for each blur level (applied to the 64x64 integer array).
BLUR_SIGMAS = [0, 1.0, 2.0, 3.0, 4.0]

# Vision / control thresholds (normally no need to change these)
YAW_THRESHOLD = 3.5   # pixels — deadband before a yaw correction fires
YAW_OFF       = 1.0   # pixels — hysteresis: stop correcting when |combined| drops below this
LATERAL_CAP   = 10.0  # pixels — max lateral contribution to combined steering signal
CRASH_TIMEOUT = 5.0   # seconds — declare crashed if no data received for this long
# ─────────────────────────────────────────────────────────────────────────────

# Runtime state — updated by the main loop at the start of each test.
# apply_blur() and the summary printers read BLUR_LEVEL as a module-level variable.
# send_command() reads _current_test_speed to embed the speed in every JSON command,
# so the controller can update itself after each simulation reset.
BLUR_LEVEL           = TEST_SEQUENCE[0][1]
_current_test_speed  = TEST_SEQUENCE[0][0]
_current_yaw_scalar  = 1.55   # speed-proportional: 1.55 at speed 0.1, 0.5 at speed 1.0
_yaw_state           = None   # hysteresis: 'decrease', 'increase', or None


def apply_blur(array_2d):
    """
    Apply Gaussian blur at the configured BLUR_LEVEL.
    Returns a float32 array in the 0-255 value range.
    """
    sigma = BLUR_SIGMAS[BLUR_LEVEL]
    if sigma > 0:
        return gaussian_filter(array_2d.astype(np.float32), sigma=sigma)
    return array_2d.astype(np.float32)


def process_array(array_2d, client_socket):
    """
    Rule-based line-following using computer vision.
    array_2d is already blurred and in float32.

    Returns (is_centered: bool, error: float|None, command_values: list)
    error is the absolute lateral offset of the track from the image centre (pixels).
    """
    command_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    img = array_2d  # already float32

    # 1. LINE DETECTION — adaptive threshold: pixels brighter than mean + 0.5*std
    mean_val  = np.mean(img)
    std_val   = np.std(img)
    threshold = mean_val + 0.5 * std_val
    line_mask = (img >= threshold).astype(np.uint8)

    if np.sum(line_mask) < 50:
        print("  [LOST] No clear line detected - continuing forward")
        command_values[0] = 1
        send_command(client_socket, command_values)
        return False, None, command_values

    # 2. ROW CENTERING — find horizontal centroid of line pixels in each row
    rows_with_line = []
    line_centers   = []
    for row in range(64):
        row_pixels = line_mask[row, :]
        if np.sum(row_pixels) > 0:
            col_indices = np.where(row_pixels)[0]
            rows_with_line.append(row)
            line_centers.append(np.mean(col_indices))

    if len(line_centers) < 5:
        print("  [LOST] Insufficient line segments - continuing forward")
        command_values[0] = 1
        send_command(client_socket, command_values)
        return False, None, command_values

    # 3. LINE FITTING — least-squares linear regression: center = slope*row + intercept
    rows_arr    = np.array(rows_with_line)
    centers_arr = np.array(line_centers)
    A = np.vstack([rows_arr, np.ones(len(rows_arr))]).T
    slope, intercept = np.linalg.lstsq(A, centers_arr, rcond=None)[0]

    # 4. CONTROL DECISION
    image_center   = 31.5
    line_at_middle = slope * 32 + intercept   # where track IS (row 32, image centre)
    line_at_top    = slope * 10 + intercept   # where track is GOING (row 10, ahead)

    # error: absolute lateral offset of the track from drone centre (pixels, 0-32 range)
    error_now = line_at_middle - image_center   # signed
    error     = abs(error_now)                  # logged

    # heading signal: how much the track angles relative to current heading
    # positive → track curves right ahead → yaw_decrease (turn right)
    # negative → track curves left  ahead → yaw_increase (turn left)
    heading_error = line_at_top - line_at_middle   # = -22 * slope

    # combined steering: heading (60%) + capped lateral offset (40%)
    lateral_capped = max(-LATERAL_CAP, min(LATERAL_CAP, error_now))
    combined = 0.6 * heading_error + 0.4 * lateral_capped

    global _yaw_state
    command_values[0] = 1  # always forward
    # Hysteresis: latch on at YAW_THRESHOLD, release at YAW_OFF
    if combined > YAW_THRESHOLD:
        _yaw_state = 'decrease'
    elif combined < -YAW_THRESHOLD:
        _yaw_state = 'increase'
    elif abs(combined) < YAW_OFF:
        _yaw_state = None
    if _yaw_state == 'decrease':
        command_values[5] = 1  # yaw_decrease (turn right)
    elif _yaw_state == 'increase':
        command_values[4] = 1  # yaw_increase (turn left)

    command_names = ["fwd", "bwd", "L", "R", "yaw+", "yaw-", "up", "dn", "rst"]
    active = "+".join(n for i, n in enumerate(command_names) if command_values[i] == 1)
    print(f"  err={error:5.1f}px  hdg={heading_error:+5.1f}px  combined={combined:+5.1f}  cmd=[{active}]")

    send_command(client_socket, command_values)

    is_centered = command_values == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    return is_centered, error, command_values


def send_command(client_socket, command_values):
    # Include new_speed_scalar in every command so the controller picks up the
    # correct speed on the first response after a simulation reset.
    command_json = {
        "forward":               command_values[0],
        "backward":              command_values[1],
        "left":                  command_values[2],
        "right":                 command_values[3],
        "yaw_increase":          command_values[4],
        "yaw_decrease":          command_values[5],
        "height_diff_increase":  command_values[6],
        "height_diff_decrease":  command_values[7],
        "reset_simulation":      0,
        "new_speed_scalar":      round(_current_test_speed, 4),
        "new_yaw_scalar":        round(_current_yaw_scalar, 4),
    }
    client_socket.sendall(json.dumps(command_json).encode('utf-8'))


def send_transition_command(client_socket, next_speed_scalar):
    """Send a reset command carrying the next test's speed scalar."""
    cmd = {
        "forward": 0, "backward": 0, "left": 0, "right": 0,
        "yaw_increase": 0, "yaw_decrease": 0,
        "height_diff_increase": 0, "height_diff_decrease": 0,
        "reset_simulation": 1,
        "new_speed_scalar": round(next_speed_scalar, 4),
        "new_yaw_scalar":   round(_current_yaw_scalar, 4),
    }
    client_socket.sendall(json.dumps(cmd).encode('utf-8'))


def print_lap_summary(lap_num, lap_time, errors, cmd_counts, oscillations, track_losses, speed_scalar):
    total_steps = sum(cmd_counts.values())
    if total_steps == 0 or lap_time < 2.0:
        return
    print(f"\n{'='*54}")
    print(f"  LAP {lap_num} COMPLETE  |  speed={speed_scalar:.2f}  blur={BLUR_LEVEL}")
    print(f"{'='*54}")
    print(f"  Lap time:       {lap_time:.2f}s")
    if errors:
        print(f"  Lateral error:  avg={np.mean(errors):.1f}px  "
              f"std={np.std(errors):.1f}px  max={np.max(errors):.1f}px")
    else:
        print(f"  Lateral error:  no data")
    fwd_pct   = 100 * cmd_counts['fwd']      / total_steps
    yaw_pct   = 100 * cmd_counts['yaw']      / total_steps
    lost_pct  = 100 * cmd_counts['lost']     / total_steps
    print(f"  Commands:       fwd={fwd_pct:.0f}%  yaw={yaw_pct:.0f}%  lost={lost_pct:.0f}%")
    print(f"  Oscillations:   {oscillations}  (L<->R reversals)")
    print(f"  Track losses:   {track_losses}")
    print(f"{'='*54}\n")


def print_session_summary(lap_records, speed_scalar):
    print(f"\n{'#'*54}")
    print(f"  SESSION SUMMARY  |  speed={speed_scalar:.2f}  blur={BLUR_LEVEL}")
    print(f"{'#'*54}")
    if not lap_records:
        print("  No laps completed.")
        print(f"{'#'*54}\n")
        return
    valid = [r for r in lap_records if r['lap_time'] >= 2.0]
    if not valid:
        print("  No valid laps completed.")
        print(f"{'#'*54}\n")
        return
    lap_times  = [r['lap_time'] for r in valid]
    all_errors = [e for r in valid for e in r['errors']]
    total_osc  = sum(r['oscillations'] for r in valid)
    total_lost = sum(r['track_losses'] for r in valid)
    print(f"  Total laps:     {len(valid)}")
    print(f"  Lap times:      {'  '.join(f'{t:.1f}s' for t in lap_times)}")
    print(f"  Avg lap time:   {np.mean(lap_times):.2f}s  (best={np.min(lap_times):.2f}s)")
    if all_errors:
        print(f"  Overall error:  avg={np.mean(all_errors):.1f}px  max={np.max(all_errors):.1f}px")
    print(f"  Total osc:      {total_osc}  (L<->R reversals across all laps)")
    print(f"  Total losses:   {total_lost}")
    print(f"{'#'*54}\n")


# ── SERVER SETUP ──────────────────────────────────────────────────────────────
hostname = 'localhost'
port     = 8080

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((hostname, port))
server_socket.listen(1)
server_socket.setblocking(False)

total_minutes = len(TEST_SEQUENCE) * TIME_LIMIT_SECONDS // 60
print(f"Server listening on port {port}  "
      f"[{len(TEST_SEQUENCE)} tests × {TIME_LIMIT_SECONDS}s = {total_minutes} min total]")

# ── CSV SETUP — append mode so all 50 runs accumulate in one file ─────────────
csv_filename = "error_log.csv"
csv_is_new   = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0
csv_file     = open(csv_filename, 'a', newline='')
csv_writer   = csv.writer(csv_file)
if csv_is_new:
    csv_writer.writerow(['timestep', 'timestamp', 'error', 'lap_number', 'speed_scalar', 'blur_level'])

# ── PACKET FORMAT (from controller) ───────────────────────────────────────────
#   [4096 bytes] uint8_t 64x64 grayscale array (0-255)
#   [   4 bytes] int32   lap count (little-endian)
#   [   4 bytes] float32 speed scalar (little-endian)
EXPECTED_BYTES = 4104

# ── MAIN LOOP — iterates over TEST_SEQUENCE ────────────────────────────────────
timestep   = 0
test_index = 0

while test_index < len(TEST_SEQUENCE):
    current_speed, current_blur = TEST_SEQUENCE[test_index]

    # Update module-level globals read by apply_blur(), process_array(), and send_command()
    BLUR_LEVEL          = current_blur
    _current_test_speed = current_speed
    # Linear yaw scalar: 1.55 at speed 0.1, 0.5 at speed 1.0
    _current_yaw_scalar = max(0.5, 1.55 - 1.05 * (current_speed - 0.1) / 0.9)
    _yaw_state          = None  # reset hysteresis at the start of each test

    remaining_min = (len(TEST_SEQUENCE) - test_index) * TIME_LIMIT_SECONDS // 60
    print(f"\n{'='*60}")
    print(f"  TEST {test_index + 1}/{len(TEST_SEQUENCE)}:  "
          f"speed={current_speed:.1f}  blur={current_blur}  "
          f"({remaining_min} min remaining)")
    print(f"  Waiting for controller connection...")
    print(f"{'='*60}")

    # ── Accept connection ──────────────────────────────────────────────────────
    client_socket = None
    try:
        while True:
            try:
                client_socket, client_address = server_socket.accept()
                print(f"Controller connected from {client_address}")
                client_socket.setblocking(False)
                break
            except socket.error:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        break

    # ── Per-connection state ───────────────────────────────────────────────────
    buffer               = b''
    start_time           = None
    first_image_received = False
    last_data_time       = time.time()
    current_speed_scalar = current_speed

    prev_lap_count   = 0
    lap_start_time   = None
    lap_errors       = []
    lap_cmd_counts   = {'fwd': 0, 'left': 0, 'right': 0,
                        'yaw': 0, 'backward': 0, 'lost': 0}
    lap_oscillations = 0
    lap_track_losses = 0
    last_lateral     = None
    lap_records      = []

    test_done = False  # set True when 5-min timer fires (normal end)
    aborted   = False  # set True on unexpected disconnect or crash

    try:
        while True:
            try:
                chunk = client_socket.recv(EXPECTED_BYTES)

                if len(chunk) > 0:
                    last_data_time = time.time()
                    buffer += chunk

                    while len(buffer) >= EXPECTED_BYTES:
                        # ── Parse packet ──────────────────────────────────────────
                        array_data         = buffer[:4096]
                        lap_count_bytes    = buffer[4096:4100]
                        speed_scalar_bytes = buffer[4100:4104]
                        buffer             = buffer[EXPECTED_BYTES:]

                        raw_array            = np.frombuffer(array_data, dtype=np.uint8).reshape((64, 64))
                        lap_count            = int.from_bytes(lap_count_bytes, byteorder='little', signed=True)
                        current_speed_scalar = struct.unpack('<f', speed_scalar_bytes)[0]

                        # ── Start timer on first image ─────────────────────────────
                        if not first_image_received:
                            start_time           = time.time()
                            lap_start_time       = start_time
                            first_image_received = True
                            print(f"First image received — starting {TIME_LIMIT_SECONDS}s timer  "
                                  f"[speed={current_speed_scalar:.2f}  blur={BLUR_LEVEL}]")

                        # ── Check time limit BEFORE processing ─────────────────────
                        elapsed_seconds = time.time() - start_time
                        if elapsed_seconds >= TIME_LIMIT_SECONDS:
                            print(f"\n{TIME_LIMIT_SECONDS}s elapsed — test {test_index + 1} complete")
                            print_session_summary(lap_records, current_speed_scalar)
                            test_index += 1
                            if test_index < len(TEST_SEQUENCE):
                                next_speed = TEST_SEQUENCE[test_index][0]
                                send_transition_command(client_socket, next_speed)
                                print(f"Sent reset  →  next: speed={next_speed:.1f}  "
                                      f"blur={TEST_SEQUENCE[test_index][1]}")
                            else:
                                send_transition_command(client_socket, TEST_SPEEDS[0])
                                print("All tests complete — sent final reset")
                            test_done = True
                            break  # exit inner buffer-parse loop; recv loop exits below

                        # ── Lap-change detection ───────────────────────────────────
                        if lap_count > prev_lap_count:
                            now      = time.time()
                            lap_time = now - lap_start_time
                            print_lap_summary(
                                lap_num=lap_count, lap_time=lap_time,
                                errors=lap_errors, cmd_counts=lap_cmd_counts,
                                oscillations=lap_oscillations, track_losses=lap_track_losses,
                                speed_scalar=current_speed_scalar,
                            )
                            lap_records.append({
                                'lap_num':      lap_count,
                                'lap_time':     lap_time,
                                'errors':       list(lap_errors),
                                'oscillations': lap_oscillations,
                                'track_losses': lap_track_losses,
                            })
                            lap_start_time   = now
                            lap_errors       = []
                            lap_cmd_counts   = {'fwd': 0, 'left': 0, 'right': 0,
                                                'yaw': 0, 'backward': 0, 'lost': 0}
                            lap_oscillations = 0
                            lap_track_losses = 0
                            last_lateral     = None
                            prev_lap_count   = lap_count

                        # ── Apply blur then process ────────────────────────────────
                        blurred_array = apply_blur(raw_array)
                        is_centered, error, cmd = process_array(blurred_array, client_socket)

                        # ── Update per-lap stats ───────────────────────────────────
                        if error is None:
                            lap_cmd_counts['lost'] += 1
                            lap_track_losses += 1
                        else:
                            lap_errors.append(error)
                            has_yaw   = cmd[4] == 1 or cmd[5] == 1
                            has_left  = cmd[2] == 1
                            has_right = cmd[3] == 1
                            has_bwd   = cmd[1] == 1
                            if has_left:
                                lap_cmd_counts['left'] += 1
                                if last_lateral == 'right':
                                    lap_oscillations += 1
                                last_lateral = 'left'
                            elif has_right:
                                lap_cmd_counts['right'] += 1
                                if last_lateral == 'left':
                                    lap_oscillations += 1
                                last_lateral = 'right'
                            else:
                                last_lateral = None
                            if has_yaw:
                                lap_cmd_counts['yaw'] += 1
                            if has_bwd:
                                lap_cmd_counts['backward'] += 1
                            if cmd[0] == 1 and not has_left and not has_right:
                                lap_cmd_counts['fwd'] += 1

                        # ── Log to CSV ─────────────────────────────────────────────
                        timestep += 1
                        csv_writer.writerow([
                            timestep,
                            round(elapsed_seconds, 4),
                            round(error, 4) if error is not None else None,
                            lap_count,
                            round(current_speed_scalar, 4),
                            BLUR_LEVEL,
                        ])
                        csv_file.flush()

                    if test_done:
                        break  # exit outer recv loop

                elif len(chunk) == 0:
                    print("Controller disconnected unexpectedly")
                    print_session_summary(lap_records, current_speed_scalar)
                    aborted = True
                    break

                if first_image_received and (time.time() - last_data_time) > CRASH_TIMEOUT:
                    print(f"No data for {CRASH_TIMEOUT}s — drone may have crashed")
                    print_session_summary(lap_records, current_speed_scalar)
                    aborted = True
                    break

            except socket.error:
                if first_image_received and (time.time() - last_data_time) > CRASH_TIMEOUT:
                    print(f"No data for {CRASH_TIMEOUT}s — drone may have crashed")
                    print_session_summary(lap_records, current_speed_scalar)
                    aborted = True
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nShutting down server...")
        print_session_summary(lap_records, current_speed_scalar)
        try:
            client_socket.close()
        except Exception:
            pass
        break

    try:
        client_socket.close()
    except Exception:
        pass

    if aborted:
        print("Test aborted — stopping sequence")
        break
    # If test_done: test_index was already advanced inside the loop; outer while
    # re-evaluates and either starts the next test or exits naturally.

csv_file.close()
server_socket.close()
if test_index >= len(TEST_SEQUENCE):
    print(f"\nAll {len(TEST_SEQUENCE)} tests complete. Results saved to {csv_filename}")
else:
    print(f"\nStopped at test {test_index + 1}/{len(TEST_SEQUENCE)}. "
          f"Partial results saved to {csv_filename}")
