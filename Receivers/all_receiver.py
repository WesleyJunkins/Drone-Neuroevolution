#!/usr/bin/env python3
# Unified receiver for all test sessions.
# Runs 19 sessions (baseline + 18 NE/ML models) sequentially through the same
# 75-test matrix (5 speeds x 5 blur levels x 3 reps x 300s each).
# Extended CSV: 14 columns including GPS XYZ position and raw model output.
#
# Requires all_controller.c (4116-byte packet with GPS coords appended).
#
# Usage:
#   python3 receivers/all_receiver.py                     # run all 19 sessions
#   python3 receivers/all_receiver.py --session baseline  # run one named session
#
# Auto-resume: each session reads its own CSV on startup and resumes from the
# last incomplete test.  Kill and restart at any time.
#
# By Wesley Junkins

import argparse
import socket
import struct
import time
import os
import sys
import csv
import json
import pickle
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

# == SESSION INVENTORY (19 sessions, ymotors excluded everywhere) ==============

SESSIONS = [
    {
        'name': 'baseline',
        'family': 'baseline',
        'output_type': 'ycommand',
        'model_path': '',
        'neat_config': '',
    },
    {
        'name': 'train_ycommand',
        'family': 'main_cnn',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/main_phase/train_ycommand/best_model/best_model_train_ycommand.pth',
        'neat_config': '',
    },
    {
        'name': 'train_ycontinuous',
        'family': 'main_cnn',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/main_phase/train_ycontinuous/best_model/best_model_train_ycontinuous.pth',
        'neat_config': '',
    },
    {
        'name': 'notrain_ycommand',
        'family': 'main_cnn',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/main_phase/notrain_ycommand/best_model/best_model_notrain_ycommand.pth',
        'neat_config': '',
    },
    {
        'name': 'notrain_ycontinuous',
        'family': 'main_cnn',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/main_phase/notrain_ycontinuous/best_model/best_model_notrain_ycontinuous.pth',
        'neat_config': '',
    },
    {
        'name': 'mlp_256_128_ycommand',
        'family': 'small_mlp',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_256_128_ycommand/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'mlp_256_128_ycontinuous',
        'family': 'small_mlp',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_256_128_ycontinuous/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'mlp_64_32_ycommand',
        'family': 'small_mlp',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_64_32_ycommand/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'mlp_64_32_ycontinuous',
        'family': 'small_mlp',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_64_32_ycontinuous/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'mlp_16_8_ycommand',
        'family': 'small_mlp',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_16_8_ycommand/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'mlp_16_8_ycontinuous',
        'family': 'small_mlp',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_16_8_ycontinuous/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'neat_ycommand',
        'family': 'neat',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/extra_observation/exp_b_neat/neat_ycommand/best_genome.pkl',
        'neat_config': 'Data/model_data/extra_observation/exp_b_neat/neat_ycommand/neat_config.ini',
    },
    {
        'name': 'neat_ycontinuous',
        'family': 'neat',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/extra_observation/exp_b_neat/neat_ycontinuous/best_genome.pkl',
        'neat_config': 'Data/model_data/extra_observation/exp_b_neat/neat_ycontinuous/neat_config.ini',
    },
    {
        'name': 'features_ycommand',
        'family': 'feature_mlp',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/extra_observation/exp_c_features/features_ycommand/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'features_ycontinuous',
        'family': 'feature_mlp',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/extra_observation/exp_c_features/features_ycontinuous/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'exp_e_ycommand',
        'family': 'main_cnn',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/extra_observation/exp_e_standard_backprop/ycommand/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'exp_e_ycontinuous',
        'family': 'main_cnn',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/extra_observation/exp_e_standard_backprop/ycontinuous/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'exp_f_ycommand',
        'family': 'main_cnn',
        'output_type': 'ycommand',
        'model_path': 'Data/model_data/extra_observation/exp_f_extended_notrain/ycommand/best_model.pth',
        'neat_config': '',
    },
    {
        'name': 'exp_f_ycontinuous',
        'family': 'main_cnn',
        'output_type': 'ycontinuous',
        'model_path': 'Data/model_data/extra_observation/exp_f_extended_notrain/ycontinuous/best_model.pth',
        'neat_config': '',
    },
]

# == TEST MATRIX ===============================================================

TEST_SPEEDS      = [0.2, 0.4, 0.6, 0.8, 1.0]
TEST_BLUR_LEVELS = [0, 1, 2, 3, 4]
BLUR_SIGMAS      = [0, 1.0, 2.0, 3.0, 4.0]
TEST_REPETITIONS = 3
TEST_SEQUENCE    = [(spd, blr)
                    for spd in TEST_SPEEDS
                    for blr in TEST_BLUR_LEVELS] * TEST_REPETITIONS

TIME_LIMIT_SECONDS = 300
CRASH_TIMEOUT      = 5.0
TRACK_LOSS_TIMEOUT = 625

# == THRESHOLDS ================================================================

YCOMMAND_FWD_THRESHOLD = 0.5
YCOMMAND_YAW_THRESHOLD = 0.07

YAW_THRESHOLD = 3.5
YAW_OFF       = 1.0
LATERAL_CAP   = 10.0

# == TRACK GEOMETRY ============================================================

TRACK_CENTER_X = 0.000018
TRACK_CENTER_Y = -2.849774

# == PACKET / CSV ==============================================================

EXPECTED_BYTES = 4116   # 4096 image + 4 lap + 4 speed + 4 pos_x + 4 pos_y + 4 pos_z

CSV_HEADER = [
    'test_index', 'timestep', 'timestamp', 'error',
    'lap_number', 'speed_scalar', 'blur_level',
    'pos_x', 'pos_y', 'pos_z', 'dist_from_center',
    'out_0', 'out_1', 'out_2',
]

OUTPUT_DIMS = {'ycommand': 3, 'ycontinuous': 2}

# == PATHS =====================================================================

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


# == MODEL CLASSES =============================================================

class CreateCNNModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4097, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_features)

    def forward(self, x):
        img   = x[:, :4096].view(-1, 1, 64, 64)
        speed = x[:, 4096:4097]
        img = self.pool1(torch.relu(self.conv1(img)))
        img = self.pool2(torch.relu(self.conv2(img)))
        img = self.pool3(torch.relu(self.conv3(img)))
        img = img.view(img.size(0), -1)
        combined = torch.cat([img, speed], dim=1)
        out = torch.relu(self.fc1(combined))
        out = torch.relu(self.fc2(out))
        return self.fc3(out)


class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# == MODEL LOADING =============================================================

def load_session_model(session):
    """Load model for a session; returns None for baseline (no model needed)."""
    family     = session['family']
    out_type   = session['output_type']
    model_path = os.path.join(_PROJECT_ROOT, session['model_path'])
    neat_cfg   = (os.path.join(_PROJECT_ROOT, session['neat_config'])
                  if session['neat_config'] else '')

    if family == 'baseline':
        print('  baseline -- no model to load')
        return None

    if family == 'main_cnn':
        out_dim = OUTPUT_DIMS[out_type]
        model   = CreateCNNModel(out_dim)
        state   = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print(f'  main_cnn  output_dim={out_dim}')
        return model

    if family == 'small_mlp':
        ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
        model = SmallMLP(4097, ckpt['hidden_dims'], ckpt['output_dim'])
        model.load_state_dict(ckpt['weights'])
        model.eval()
        print(f"  small_mlp  hidden={ckpt['hidden_dims']}  output_dim={ckpt['output_dim']}")
        return model

    if family == 'neat':
        try:
            import neat as neat_lib
        except ImportError:
            print('ERROR: neat-python not installed.  pip install neat-python')
            sys.exit(1)
        if not neat_cfg:
            print('ERROR: neat_config path required for family=neat')
            sys.exit(1)
        with open(model_path, 'rb') as f:
            genome = pickle.load(f)
        config = neat_lib.Config(
            neat_lib.DefaultGenome,
            neat_lib.DefaultReproduction,
            neat_lib.DefaultSpeciesSet,
            neat_lib.DefaultStagnation,
            neat_cfg,
        )
        net = neat_lib.nn.FeedForwardNetwork.create(genome, config)
        print(f'  neat  output_dim={OUTPUT_DIMS[out_type]}')
        return net

    if family == 'feature_mlp':
        ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
        model = SmallMLP(3, ckpt['hidden_dims'], ckpt['output_dim'])
        model.load_state_dict(ckpt['weights'])
        model.eval()
        print(f"  feature_mlp  hidden={ckpt['hidden_dims']}  output_dim={ckpt['output_dim']}")
        return model

    print(f"ERROR: Unknown family '{family}'")
    sys.exit(1)


# == VISION ====================================================================

def compute_error(array_2d):
    """
    Baseline vision logic for the CSV error column.  Uses >= threshold and
    50-pixel guard, matching baseline_receiver.py exactly.
    Returns abs(line_at_middle - 31.5), or 31.5 if the track is not detected.
    """
    mean_val  = np.mean(array_2d)
    std_val   = np.std(array_2d)
    threshold = mean_val + 0.5 * std_val
    line_mask = (array_2d >= threshold).astype(np.uint8)

    if np.sum(line_mask) < 50:
        return 31.5

    rows_with_line, line_centers = [], []
    for row in range(64):
        row_pixels = line_mask[row, :]
        if np.sum(row_pixels) > 0:
            col_indices = np.where(row_pixels)[0]
            rows_with_line.append(row)
            line_centers.append(np.mean(col_indices))

    if len(line_centers) < 5:
        return 31.5

    rows_arr    = np.array(rows_with_line)
    centers_arr = np.array(line_centers)
    A = np.vstack([rows_arr, np.ones(len(rows_arr))]).T
    slope, intercept = np.linalg.lstsq(A, centers_arr, rcond=None)[0]
    return abs(slope * 32 + intercept - 31.5)


def extract_features(array_2d):
    """
    Feature extraction for neat/feature_mlp.  Uses strict > threshold,
    no 50-pixel guard.  Returns (slope, line_at_middle) or (0.0, 31.5).
    """
    mean_val  = np.mean(array_2d)
    std_val   = np.std(array_2d)
    threshold = mean_val + 0.5 * std_val
    line_mask = (array_2d > threshold).astype(np.uint8)

    rows_with_line, line_centers = [], []
    for row in range(64):
        row_pixels = line_mask[row, :]
        if np.sum(row_pixels) > 0:
            col_indices = np.where(row_pixels)[0]
            rows_with_line.append(row)
            line_centers.append(np.mean(col_indices))

    if len(line_centers) < 5:
        return 0.0, 31.5

    rows_arr    = np.array(rows_with_line)
    centers_arr = np.array(line_centers)
    A = np.vstack([rows_arr, np.ones(len(rows_arr))]).T
    slope, intercept = np.linalg.lstsq(A, centers_arr, rcond=None)[0]
    return float(slope), float(slope * 32 + intercept)


def process_baseline(array_2d, yaw_state):
    """
    Rule-based control (pure computation, no send).
    Returns (fwd, yi, yd, new_yaw_state) as ints and updated state string.
    """
    mean_val  = np.mean(array_2d)
    std_val   = np.std(array_2d)
    threshold = mean_val + 0.5 * std_val
    line_mask = (array_2d >= threshold).astype(np.uint8)

    if np.sum(line_mask) < 50:
        return 1, 0, 0, yaw_state

    rows_with_line, line_centers = [], []
    for row in range(64):
        row_pixels = line_mask[row, :]
        if np.sum(row_pixels) > 0:
            col_indices = np.where(row_pixels)[0]
            rows_with_line.append(row)
            line_centers.append(np.mean(col_indices))

    if len(line_centers) < 5:
        return 1, 0, 0, yaw_state

    rows_arr    = np.array(rows_with_line)
    centers_arr = np.array(line_centers)
    A = np.vstack([rows_arr, np.ones(len(rows_arr))]).T
    slope, intercept = np.linalg.lstsq(A, centers_arr, rcond=None)[0]

    line_at_middle = slope * 32 + intercept
    line_at_top    = slope * 10 + intercept
    error_now      = line_at_middle - 31.5
    heading_error  = line_at_top - line_at_middle
    lateral_capped = max(-LATERAL_CAP, min(LATERAL_CAP, error_now))
    combined       = 0.6 * heading_error + 0.4 * lateral_capped

    if combined > YAW_THRESHOLD:
        yaw_state = 'decrease'
    elif combined < -YAW_THRESHOLD:
        yaw_state = 'increase'
    elif abs(combined) < YAW_OFF:
        yaw_state = None

    yi = 1 if yaw_state == 'increase' else 0
    yd = 1 if yaw_state == 'decrease' else 0
    return 1, yi, yd, yaw_state


# == INFERENCE =================================================================

def run_inference(model, family, out_type, blurred, speed_scalar, slope, line_at_middle):
    """Return raw model output as float32 numpy array."""
    if family in ('main_cnn', 'small_mlp'):
        flat = blurred.flatten().astype(np.float32) / 255.0
        x    = np.concatenate([flat, [float(speed_scalar)]]).astype(np.float32)
        with torch.no_grad():
            out = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
        return out

    if family == 'neat':
        return np.array(model.activate([slope, line_at_middle, float(speed_scalar)]),
                        dtype=np.float32)

    if family == 'feature_mlp':
        x = np.array([slope, line_at_middle, float(speed_scalar)], dtype=np.float32)
        with torch.no_grad():
            out = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
        return out

    return np.zeros(OUTPUT_DIMS[out_type], dtype=np.float32)


# == COMMAND BUILDING ==========================================================

def build_command(session, raw_output, current_speed, yaw_scalar, reset=False):
    """Build unified JSON command dict from session config and raw output."""
    out_type = session['output_type']
    mode     = 1 if out_type == 'ycontinuous' else 0

    cmd = {
        'command_mode': mode,
        'forward': 0, 'backward': 0, 'left': 0, 'right': 0,
        'yaw_increase': 0, 'yaw_decrease': 0,
        'height_diff_increase': 0, 'height_diff_decrease': 0,
        'forward_desired': 0.0, 'yaw_desired': 0.0,
        'm1': 0.0, 'm2': 0.0, 'm3': 0.0, 'm4': 0.0,
        'reset_simulation': 1 if reset else 0,
        'new_speed_scalar': round(float(current_speed), 4),
        'new_yaw_scalar':   round(float(yaw_scalar), 4),
    }
    if reset:
        return cmd

    if out_type == 'ycommand':
        fwd = 1 if raw_output[0] > YCOMMAND_FWD_THRESHOLD else 0
        yi  = 1 if raw_output[1] > YCOMMAND_YAW_THRESHOLD else 0
        yd  = 1 if raw_output[2] > YCOMMAND_YAW_THRESHOLD else 0
        if yi and yd:
            if raw_output[1] >= raw_output[2]:
                yd = 0
            else:
                yi = 0
        cmd['forward']      = fwd
        cmd['yaw_increase'] = yi
        cmd['yaw_decrease'] = yd

    elif out_type == 'ycontinuous':
        cmd['forward_desired'] = float(raw_output[0])
        cmd['yaw_desired']     = float(raw_output[1])

    return cmd


# == RESUME DETECTION ==========================================================

def get_resume_state(csv_path):
    """
    Read an existing CSV to find where to resume.
    Returns (start_test_index_0based, start_timestep).
    Re-runs the last test_index found (may be incomplete).
    Returns (len(TEST_SEQUENCE), 0) if the session is already complete.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0, 0
    max_ti, max_ts = 0, 0
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ti = int(row.get('test_index', 0))
                ts = int(row.get('timestep', 0))
                if ti > max_ti:
                    max_ti = ti
                if ts > max_ts:
                    max_ts = ts
    except Exception:
        return 0, 0
    if max_ti >= len(TEST_SEQUENCE):
        return len(TEST_SEQUENCE), max_ts
    return max(0, max_ti - 1), max_ts


# == PACKET PARSING ============================================================

def parse_packet(buf):
    """
    Parse a 4116-byte packet from all_controller.c.
    Returns (image_uint8_64x64, lap_count, speed_scalar, pos_x, pos_y, pos_z).
    """
    image        = np.frombuffer(buf[:4096], dtype=np.uint8).reshape((64, 64))
    lap_count    = int.from_bytes(buf[4096:4100], byteorder='little', signed=True)
    speed_scalar = struct.unpack('<f', buf[4100:4104])[0]
    pos_x        = struct.unpack('<f', buf[4104:4108])[0]
    pos_y        = struct.unpack('<f', buf[4108:4112])[0]
    pos_z        = struct.unpack('<f', buf[4112:4116])[0]
    return image, lap_count, speed_scalar, pos_x, pos_y, pos_z


# == SINGLE SESSION RUNNER =====================================================

def run_session(session, server_socket):
    """
    Run all 75 tests for one session.  Handles auto-resume, reconnects,
    5-min timer, CSV writing.
    Returns False if the user interrupted (KeyboardInterrupt).
    """
    name     = session['name']
    family   = session['family']
    out_type = session['output_type']

    csv_dir  = os.path.join(_PROJECT_ROOT, 'Data', 'ALL_FINAL_ERROR_RESULTS')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'{name}_error_log.csv')

    start_ti, start_ts = get_resume_state(csv_path)
    if start_ti >= len(TEST_SEQUENCE):
        print(f'  {name} -- already complete, skipping')
        return True

    if start_ti > 0:
        print(f'  Resuming from test {start_ti + 1}/{len(TEST_SEQUENCE)}  '
              f'(global timestep continues from {start_ts})')

    print('Loading model ...')
    model = load_session_model(session)
    print('Model ready.\n')

    csv_is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file   = open(csv_path, 'a', newline='')
    writer     = csv.writer(csv_file)
    if csv_is_new:
        writer.writerow(CSV_HEADER)

    timestep   = start_ts
    test_index = start_ti

    while test_index < len(TEST_SEQUENCE):
        current_speed, current_blur = TEST_SEQUENCE[test_index]
        current_yaw_scalar = max(0.5, 1.55 - 1.05 * (current_speed - 0.1) / 0.9)

        remaining_min = (len(TEST_SEQUENCE) - test_index) * TIME_LIMIT_SECONDS // 60
        print(f"\n{'='*60}")
        print(f'  TEST {test_index + 1}/{len(TEST_SEQUENCE)}:  '
              f'speed={current_speed:.1f}  blur={current_blur}  '
              f'({remaining_min} min remaining this session)')
        print(f"{'='*60}")

        # Per-test state that persists across mid-test reconnects
        start_time           = None
        first_image_received = False
        test_done            = False
        lap_errors           = []
        yaw_state            = None   # baseline hysteresis, resets each test

        # Reconnect loop
        while not test_done:
            print('  Waiting for controller connection ...')
            client_socket = None

            try:
                while True:
                    try:
                        client_socket, addr = server_socket.accept()
                        print(f'  Controller connected from {addr}')
                        client_socket.setblocking(False)
                        break
                    except socket.error:
                        if start_time and time.time() - start_time >= TIME_LIMIT_SECONDS:
                            print(f'\n{TIME_LIMIT_SECONDS}s elapsed (waiting for reconnect) -- '
                                  f'test {test_index + 1} complete')
                            test_index += 1
                            test_done  = True
                            break
                        time.sleep(0.1)
            except KeyboardInterrupt:
                print('\nShutting down ...')
                csv_file.close()
                try:
                    client_socket.close()
                except Exception:
                    pass
                return False

            if test_done:
                break

            buffer                   = b''
            last_data_time           = time.time()
            prev_lap_count           = 0
            lap_start_time           = None
            consecutive_track_losses = 0
            warned_track_loss        = False

            try:
                while True:
                    try:
                        chunk = client_socket.recv(EXPECTED_BYTES)

                        if len(chunk) > 0:
                            last_data_time = time.time()
                            buffer += chunk

                            while len(buffer) >= EXPECTED_BYTES:
                                image, lap_count, speed_scalar, pos_x, pos_y, pos_z = \
                                    parse_packet(buffer[:EXPECTED_BYTES])
                                buffer = buffer[EXPECTED_BYTES:]

                                if not first_image_received:
                                    start_time           = time.time()
                                    lap_start_time       = start_time
                                    first_image_received = True
                                    print(f'  First image received -- starting {TIME_LIMIT_SECONDS}s timer  '
                                          f'[speed={speed_scalar:.2f}  blur={current_blur}]')
                                elif lap_start_time is None:
                                    lap_start_time = time.time()

                                elapsed = time.time() - start_time

                                # 5-min timer
                                if elapsed >= TIME_LIMIT_SECONDS:
                                    print(f'\n{TIME_LIMIT_SECONDS}s elapsed -- '
                                          f'test {test_index + 1} complete')
                                    if lap_errors:
                                        print(f'  avg_err={np.mean(lap_errors):.1f}px  '
                                              f'laps={lap_count}')
                                    test_index += 1
                                    next_speed = (TEST_SEQUENCE[test_index][0]
                                                  if test_index < len(TEST_SEQUENCE)
                                                  else TEST_SPEEDS[0])
                                    next_yaw = max(0.5, 1.55 - 1.05 * (next_speed - 0.1) / 0.9)
                                    raw_zero = np.zeros(OUTPUT_DIMS.get(out_type, 3),
                                                        dtype=np.float32)
                                    reset_cmd = build_command(session, raw_zero,
                                                              next_speed, next_yaw,
                                                              reset=True)
                                    client_socket.sendall(json.dumps(reset_cmd).encode('utf-8'))
                                    print(f'  Sent reset -- next speed={next_speed:.1f}')
                                    test_done = True
                                    break

                                # Lap tracking
                                if lap_count > prev_lap_count:
                                    now      = time.time()
                                    lap_time = now - lap_start_time
                                    if lap_errors and lap_time >= 2.0:
                                        print(f'  LAP {lap_count}  time={lap_time:.1f}s  '
                                              f'avg_err={np.mean(lap_errors):.1f}px')
                                    lap_start_time = now
                                    lap_errors     = []
                                    prev_lap_count = lap_count

                                # Apply blur
                                sigma   = BLUR_SIGMAS[current_blur]
                                blurred = (gaussian_filter(image.astype(np.float32), sigma=sigma)
                                           if sigma > 0 else image.astype(np.float32))

                                # Error for CSV (31.5 = track not detected)
                                error = compute_error(blurred)
                                lap_errors.append(error)
                                if error < 31.5:
                                    consecutive_track_losses = 0
                                    warned_track_loss        = False
                                else:
                                    consecutive_track_losses += 1
                                    if (TRACK_LOSS_TIMEOUT > 0
                                            and consecutive_track_losses == TRACK_LOSS_TIMEOUT
                                            and not warned_track_loss):
                                        print(f'  [TRACK WARNING] No track for '
                                              f'{TRACK_LOSS_TIMEOUT * 32 / 1000:.0f}s -- '
                                              f'test continues')
                                        warned_track_loss = True

                                # Inference / command
                                if family == 'baseline':
                                    fwd, yi, yd, yaw_state = process_baseline(blurred, yaw_state)
                                    raw_output = np.array([float(fwd), float(yi), float(yd)],
                                                          dtype=np.float32)
                                    cmd = build_command(session, raw_output,
                                                        current_speed, current_yaw_scalar)
                                else:
                                    if family in ('neat', 'feature_mlp'):
                                        slope, line_at_mid = extract_features(blurred)
                                    else:
                                        slope, line_at_mid = None, None
                                    raw_output = run_inference(model, family, out_type,
                                                               blurred, speed_scalar,
                                                               slope, line_at_mid)
                                    cmd = build_command(session, raw_output,
                                                        current_speed, current_yaw_scalar)

                                client_socket.sendall(json.dumps(cmd).encode('utf-8'))

                                # Heartbeat every ~500 frames
                                if timestep % 500 == 0:
                                    if out_type == 'ycommand':
                                        fwd_d = int(raw_output[0] > YCOMMAND_FWD_THRESHOLD)
                                        yi_d  = int(raw_output[1] > YCOMMAND_YAW_THRESHOLD)
                                        yd_d  = int(raw_output[2] > YCOMMAND_YAW_THRESHOLD)
                                        print(f'  [step={timestep}  t={elapsed:.0f}s] '
                                              f'out={np.round(raw_output, 3)}  '
                                              f'-> fwd={fwd_d} yi={yi_d} yd={yd_d}  '
                                              f'err={error:.1f}px')
                                    else:
                                        print(f'  [step={timestep}  t={elapsed:.0f}s] '
                                              f'out={np.round(raw_output, 3)}  '
                                              f'err={error:.1f}px')

                                # dist_from_center
                                dist = math.sqrt((pos_x - TRACK_CENTER_X) ** 2
                                                 + (pos_y - TRACK_CENTER_Y) ** 2)

                                # out_0/1/2 — ycontinuous has only 2 outputs, out_2 = ''
                                out_0 = round(float(raw_output[0]), 6) if len(raw_output) > 0 else ''
                                out_1 = round(float(raw_output[1]), 6) if len(raw_output) > 1 else ''
                                out_2 = (round(float(raw_output[2]), 6)
                                         if len(raw_output) > 2 and out_type != 'ycontinuous'
                                         else '')

                                timestep += 1
                                writer.writerow([
                                    test_index + 1,
                                    timestep,
                                    round(elapsed, 4),
                                    round(error, 4),
                                    lap_count,
                                    round(speed_scalar, 4),
                                    current_blur,
                                    round(pos_x, 6),
                                    round(pos_y, 6),
                                    round(pos_z, 6),
                                    round(dist, 6),
                                    out_0,
                                    out_1,
                                    out_2,
                                ])
                                csv_file.flush()

                            if test_done:
                                break

                        elif len(chunk) == 0:
                            elapsed = round(time.time() - start_time, 1) if start_time else 0.0
                            print(f'  Controller disconnected at {elapsed}s -- '
                                  f'waiting for reconnect')
                            break

                        if first_image_received and (time.time() - last_data_time) > CRASH_TIMEOUT:
                            elapsed = round(time.time() - start_time, 1)
                            print(f'  No data for {CRASH_TIMEOUT}s at {elapsed}s -- '
                                  f'waiting for reconnect')
                            break

                    except socket.error:
                        if first_image_received and (time.time() - last_data_time) > CRASH_TIMEOUT:
                            elapsed = round(time.time() - start_time, 1)
                            print(f'  No data for {CRASH_TIMEOUT}s at {elapsed}s -- '
                                  f'waiting for reconnect')
                            break

                    time.sleep(0.001)

            except KeyboardInterrupt:
                print('\nShutting down ...')
                csv_file.close()
                try:
                    client_socket.close()
                except Exception:
                    pass
                return False

            try:
                client_socket.close()
            except Exception:
                pass
        # End reconnect loop

    # End test loop
    csv_file.close()
    if test_index >= len(TEST_SEQUENCE):
        print(f'\n  {name} COMPLETE -- all {len(TEST_SEQUENCE)} tests done.')
        print(f'  CSV: {csv_path}')
    else:
        print(f'\n  {name} stopped at test {test_index}/{len(TEST_SEQUENCE)}.')
    return True


# == MAIN ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified receiver -- runs all sessions through the 75-test matrix.')
    parser.add_argument('--session', metavar='NAME',
                        help='Run only this named session (must match a name in SESSIONS)')
    parser.add_argument('--port', type=int, default=8080,
                        help='TCP port to listen on (default 8080)')
    args = parser.parse_args()

    if args.session:
        sessions_to_run = [s for s in SESSIONS if s['name'] == args.session]
        if not sessions_to_run:
            valid = [s['name'] for s in SESSIONS]
            print(f"ERROR: No session named '{args.session}'.  Valid names: {valid}")
            sys.exit(1)
    else:
        sessions_to_run = SESSIONS

    total_sessions = len(sessions_to_run)
    total_tests    = total_sessions * len(TEST_SEQUENCE)
    total_min      = total_tests * TIME_LIMIT_SECONDS // 60

    print(f'\nall_receiver.py')
    print(f'  {total_sessions} session(s) x {len(TEST_SEQUENCE)} tests = '
          f'{total_tests} total tests  (~{total_min} min / ~{total_min // 60 // 24} days)')
    print(f'  Output: Data/ALL_FINAL_ERROR_RESULTS/{{session_name}}/{{session_name}}_error_log.csv')
    print(f'  Auto-resume: completed sessions skipped on restart')
    print()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', args.port))
    server_socket.listen(1)
    server_socket.setblocking(False)
    print(f'Listening on port {args.port}\n')

    try:
        for i, session in enumerate(sessions_to_run):
            sessions_left = total_sessions - i
            min_left      = sessions_left * len(TEST_SEQUENCE) * TIME_LIMIT_SECONDS // 60
            print(f"\n{'#'*62}")
            print(f'  SESSION {i + 1}/{total_sessions}: {session["name"]}')
            print(f'  family={session["family"]}  output={session["output_type"]}')
            print(f'  ~{min_left} min remaining across {sessions_left} session(s)')
            print(f"{'#'*62}\n")

            ok = run_session(session, server_socket)
            if not ok:
                print('\nStopped by user.  Restart to resume.\n')
                break
    finally:
        server_socket.close()

    print(f"\n{'#'*62}")
    print(f'  ALL {total_sessions} SESSION(S) COMPLETE')
    print(f'  Results in: {os.path.join(_PROJECT_ROOT, "Data", "ALL_FINAL_ERROR_RESULTS")}/')
    print(f"{'#'*62}\n")


if __name__ == '__main__':
    main()
