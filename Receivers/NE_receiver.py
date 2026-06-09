# NE receiver for Phase 4 model testing.
# Loads each of the 21 trained NE models in sequence and drives the drone through
# the same 75-test automated sequence as baseline_receiver.py for each one.
# Fully unattended: all 21 models x 75 tests run automatically with no intervention.
#
# HOW TO USE (production):
#   python3 receivers/NE_receiver.py
#   Start Webots with NE_controller.  Walk away.
#   All 21 models x 75 tests (~5.8 days) run automatically.
#   If interrupted, restart the same command -- completed models are auto-skipped
#   and the last incomplete model resumes from where it left off.
#
# HOW TO USE (test mode):
#   Set TEST_MODE = True, configure TEST_MODE_MODEL and TEST_MODE_CONFIGS.
#   Runs a single model at specific (speed, blur) pairs.
#   Output goes to {MODEL_NAME}_testmode_error_log.csv -- production data untouched.
#
# By Wesley Junkins

import socket
import struct
import time
import os
import sys
import csv
import json
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

# == ALL 21 MODEL CONFIGS ======================================================
# Complete list run in order. To skip a model, comment out its entry.
# Auto-resume reads existing CSVs on startup -- no need to edit this list
# after an interruption.

ALL_MODEL_CONFIGS = [
    # ── Main Phase CNN (6 models) ───────────────────────────────────────────────
    {'MODEL_FAMILY': 'main_cnn',    'OUTPUT_TYPE': 'ycommand',
     'MODEL_NAME': 'train_ycommand',
     'MODEL_PATH': 'Data/model_data/main_phase/train_ycommand/best_model/best_model_train_ycommand.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'main_phase'},

    {'MODEL_FAMILY': 'main_cnn',    'OUTPUT_TYPE': 'ycontinuous',
     'MODEL_NAME': 'train_ycontinuous',
     'MODEL_PATH': 'Data/model_data/main_phase/train_ycontinuous/best_model/best_model_train_ycontinuous.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'main_phase'},

    {'MODEL_FAMILY': 'main_cnn',    'OUTPUT_TYPE': 'ymotors',
     'MODEL_NAME': 'train_ymotors',
     'MODEL_PATH': 'Data/model_data/main_phase/train_ymotors/best_model/best_model_train_ymotors.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'main_phase'},

    {'MODEL_FAMILY': 'main_cnn',    'OUTPUT_TYPE': 'ycommand',
     'MODEL_NAME': 'notrain_ycommand',
     'MODEL_PATH': 'Data/model_data/main_phase/notrain_ycommand/best_model/best_model_notrain_ycommand.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'main_phase'},

    {'MODEL_FAMILY': 'main_cnn',    'OUTPUT_TYPE': 'ycontinuous',
     'MODEL_NAME': 'notrain_ycontinuous',
     'MODEL_PATH': 'Data/model_data/main_phase/notrain_ycontinuous/best_model/best_model_notrain_ycontinuous.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'main_phase'},

    {'MODEL_FAMILY': 'main_cnn',    'OUTPUT_TYPE': 'ymotors',
     'MODEL_NAME': 'notrain_ymotors',
     'MODEL_PATH': 'Data/model_data/main_phase/notrain_ymotors/best_model/best_model_notrain_ymotors.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'main_phase'},

    # ── Exp A: SmallMLP (9 models) ──────────────────────────────────────────────
    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ycommand',
     'MODEL_NAME': 'mlp_256_128_ycommand',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_256_128_ycommand/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ycontinuous',
     'MODEL_NAME': 'mlp_256_128_ycontinuous',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_256_128_ycontinuous/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ymotors',
     'MODEL_NAME': 'mlp_256_128_ymotors',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_256_128_ymotors/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ycommand',
     'MODEL_NAME': 'mlp_64_32_ycommand',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_64_32_ycommand/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ycontinuous',
     'MODEL_NAME': 'mlp_64_32_ycontinuous',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_64_32_ycontinuous/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ymotors',
     'MODEL_NAME': 'mlp_64_32_ymotors',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_64_32_ymotors/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ycommand',
     'MODEL_NAME': 'mlp_16_8_ycommand',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_16_8_ycommand/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ycontinuous',
     'MODEL_NAME': 'mlp_16_8_ycontinuous',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_16_8_ycontinuous/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    {'MODEL_FAMILY': 'small_mlp',   'OUTPUT_TYPE': 'ymotors',
     'MODEL_NAME': 'mlp_16_8_ymotors',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_a_small_mlp/mlp_16_8_ymotors/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_a_small_mlp'},

    # ── Exp B: NEAT (3 models) ──────────────────────────────────────────────────
    {'MODEL_FAMILY': 'neat',        'OUTPUT_TYPE': 'ycommand',
     'MODEL_NAME': 'neat_ycommand',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_b_neat/neat_ycommand/best_genome.pkl',
     'NEAT_CONFIG_PATH': 'Data/model_data/extra_observation/exp_b_neat/neat_ycommand/neat_config.ini',
     'CSV_SUBFOLDER': 'exp_b_neat'},

    {'MODEL_FAMILY': 'neat',        'OUTPUT_TYPE': 'ycontinuous',
     'MODEL_NAME': 'neat_ycontinuous',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_b_neat/neat_ycontinuous/best_genome.pkl',
     'NEAT_CONFIG_PATH': 'Data/model_data/extra_observation/exp_b_neat/neat_ycontinuous/neat_config.ini',
     'CSV_SUBFOLDER': 'exp_b_neat'},

    {'MODEL_FAMILY': 'neat',        'OUTPUT_TYPE': 'ymotors',
     'MODEL_NAME': 'neat_ymotors',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_b_neat/neat_ymotors/best_genome.pkl',
     'NEAT_CONFIG_PATH': 'Data/model_data/extra_observation/exp_b_neat/neat_ymotors/neat_config.ini',
     'CSV_SUBFOLDER': 'exp_b_neat'},

    # ── Exp C: FeatureMLP (3 models) ────────────────────────────────────────────
    {'MODEL_FAMILY': 'feature_mlp', 'OUTPUT_TYPE': 'ycommand',
     'MODEL_NAME': 'features_ycommand',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_c_features/features_ycommand/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_c_features'},

    {'MODEL_FAMILY': 'feature_mlp', 'OUTPUT_TYPE': 'ycontinuous',
     'MODEL_NAME': 'features_ycontinuous',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_c_features/features_ycontinuous/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_c_features'},

    {'MODEL_FAMILY': 'feature_mlp', 'OUTPUT_TYPE': 'ymotors',
     'MODEL_NAME': 'features_ymotors',
     'MODEL_PATH': 'Data/model_data/extra_observation/exp_c_features/features_ymotors/best_model.pth',
     'NEAT_CONFIG_PATH': '', 'CSV_SUBFOLDER': 'exp_c_features'},
]

# == TEST MODE =================================================================
# Set TEST_MODE = True to run a quick spot-check on a single model/config.
# Full 300s timer and full startup counter -- results are directly comparable
# to real test data.  Output goes to {MODEL_NAME}_testmode_error_log.csv.
# Set TEST_MODE = False (default) to run all 21 models fully automatically.

TEST_MODE = False

TEST_MODE_MODEL = {
    'MODEL_FAMILY':     'main_cnn',
    'OUTPUT_TYPE':      'ycontinuous',
    'MODEL_NAME':       'train_ycontinuous',
    'MODEL_PATH':       'Data/model_data/main_phase/train_ycontinuous/best_model/best_model_train_ycontinuous.pth',
    'NEAT_CONFIG_PATH': '',
    'CSV_SUBFOLDER':    'main_phase',
}
TEST_MODE_CONFIGS = [(0.2, 0)]

# == TEST MATRIX ===============================================================
TEST_SPEEDS      = [0.2, 0.4, 0.6, 0.8, 1.0]
TEST_BLUR_LEVELS = [0, 1, 2, 3, 4]
BLUR_SIGMAS      = [0, 1.0, 2.0, 3.0, 4.0]
TEST_REPETITIONS = 3

TIME_LIMIT_SECONDS = 300
CRASH_TIMEOUT      = 5.0
TRACK_LOSS_TIMEOUT = 625   # consecutive no-track frames before printing a warning (~20s)

YCOMMAND_FWD_THRESHOLD = 0.5
YCOMMAND_YAW_THRESHOLD = 0.07

OUTPUT_DIMS = {'ycommand': 3, 'ycontinuous': 2, 'ymotors': 4}

if TEST_MODE:
    RUN_SEQUENCE  = [TEST_MODE_MODEL]
    TEST_SEQUENCE = list(TEST_MODE_CONFIGS)
else:
    RUN_SEQUENCE  = ALL_MODEL_CONFIGS
    TEST_SEQUENCE = [(spd, blr)
                     for spd in TEST_SPEEDS
                     for blr in TEST_BLUR_LEVELS] * TEST_REPETITIONS

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
EXPECTED_BYTES = 4104

# == MODEL CLASSES =============================================================

class CreateCNNModel(nn.Module):
    """Main-phase CNN -- identical architecture to Neuroevolution/Model.py."""
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
    """Generic MLP -- used for Exp A (4097-dim) and Exp C (3-dim) models."""
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

def load_model(family, out_type, model_path, neat_cfg_path):
    """Load and return the model for the given config."""
    out_dim = OUTPUT_DIMS[out_type]

    if family == 'main_cnn':
        model = CreateCNNModel(out_dim)
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print(f"  main_cnn  output_dim={out_dim}")
        return model

    elif family == 'small_mlp':
        ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
        model = SmallMLP(4097, ckpt['hidden_dims'], ckpt['output_dim'])
        model.load_state_dict(ckpt['weights'])
        model.eval()
        print(f"  small_mlp  hidden={ckpt['hidden_dims']}  output_dim={ckpt['output_dim']}")
        return model

    elif family == 'neat':
        try:
            import neat as neat_lib
        except ImportError:
            print("ERROR: neat-python not installed. Run: pip install neat-python")
            sys.exit(1)
        if not neat_cfg_path:
            print("ERROR: NEAT_CONFIG_PATH must be set for MODEL_FAMILY='neat'")
            sys.exit(1)
        with open(model_path, 'rb') as f:
            genome = pickle.load(f)
        config = neat_lib.Config(
            neat_lib.DefaultGenome,
            neat_lib.DefaultReproduction,
            neat_lib.DefaultSpeciesSet,
            neat_lib.DefaultStagnation,
            neat_cfg_path,
        )
        net = neat_lib.nn.FeedForwardNetwork.create(genome, config)
        print(f"  neat  output_dim={out_dim}")
        return net

    elif family == 'feature_mlp':
        ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
        model = SmallMLP(3, ckpt['hidden_dims'], ckpt['output_dim'])
        model.load_state_dict(ckpt['weights'])
        model.eval()
        print(f"  feature_mlp  hidden={ckpt['hidden_dims']}  output_dim={ckpt['output_dim']}")
        return model

    else:
        print(f"ERROR: Unknown MODEL_FAMILY '{family}'")
        sys.exit(1)


# == VISION / ERROR COMPUTATION ================================================

def compute_error(array_2d):
    """
    Identical to baseline_receiver.py process_array() vision logic.
    Used ONLY for CSV error logging so NE and baseline error values are
    directly comparable.

    Returns abs(line_at_middle - 31.5) when track is detected, or 31.5 (maximum
    possible error) when track is not detected.  Never returns None.
    Matches baseline vision logic exactly: >= threshold, 50-pixel guard, 5-row guard.
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


def extract_model_features(array_2d):
    """
    Vision feature extraction for 'neat' and 'feature_mlp' model families.
    Matches exp_b_neat.py and exp_c_features.py training logic exactly:
    strict > threshold, no 50-pixel guard, 5-row guard only.
    Returns (slope, line_at_middle) or (0.0, 31.5) defaults if no track.
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


# == INFERENCE =================================================================

def run_inference(model, family, out_type, blurred_array, speed_scalar, slope, line_at_middle):
    """Build model input and return raw output as a numpy float32 array."""
    if family in ('main_cnn', 'small_mlp'):
        flat = blurred_array.flatten().astype(np.float32) / 255.0
        x    = np.concatenate([flat, [float(speed_scalar)]]).astype(np.float32)
        with torch.no_grad():
            out = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
        return out

    elif family == 'neat':
        out = np.array(model.activate([slope, line_at_middle, float(speed_scalar)]),
                       dtype=np.float32)
        return out

    elif family == 'feature_mlp':
        x = np.array([slope, line_at_middle, float(speed_scalar)], dtype=np.float32)
        with torch.no_grad():
            out = model(torch.tensor(x).unsqueeze(0)).squeeze(0).numpy()
        return out

    return np.zeros(OUTPUT_DIMS[out_type], dtype=np.float32)


# == COMMAND BUILDING ==========================================================

def build_command(out_type, raw_output, current_speed, yaw_scalar, reset=False):
    """Convert raw model output to unified JSON command dict."""
    _MODE_MAP = {'ycommand': 0, 'ycontinuous': 1, 'ymotors': 2}
    cmd = {
        "command_mode": _MODE_MAP[out_type],
        "forward": 0, "backward": 0, "left": 0, "right": 0,
        "yaw_increase": 0, "yaw_decrease": 0,
        "height_diff_increase": 0, "height_diff_decrease": 0,
        "forward_desired": 0.0, "yaw_desired": 0.0,
        "m1": 0.0, "m2": 0.0, "m3": 0.0, "m4": 0.0,
        "reset_simulation": 1 if reset else 0,
        "new_speed_scalar": round(float(current_speed), 4),
        "new_yaw_scalar":   round(float(yaw_scalar), 4),
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
        cmd["forward"]      = fwd
        cmd["yaw_increase"] = yi
        cmd["yaw_decrease"] = yd

    elif out_type == 'ycontinuous':
        cmd["forward_desired"] = float(raw_output[0])
        cmd["yaw_desired"]     = float(raw_output[1])

    elif out_type == 'ymotors':
        cmd["m1"] = float(raw_output[0])
        cmd["m2"] = float(raw_output[1])
        cmd["m3"] = float(raw_output[2])
        cmd["m4"] = float(raw_output[3])

    return cmd


def send_command(sock, cmd):
    sock.sendall(json.dumps(cmd).encode('utf-8'))


# == RESUME DETECTION ==========================================================

def get_resume_state(csv_path, total_tests):
    """
    Read an existing CSV to find where to resume.
    Returns (start_test_index_0based, start_timestep).
    If the model is already complete (all 75 tests present), returns
    (total_tests, max_ts) so the outer loop skips it.
    Re-runs the last test_index found (it may be incomplete).
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
    if max_ti >= total_tests:
        return total_tests, max_ts   # model fully complete
    # Re-run the last test seen (it may be partial); max_ti is 1-based.
    return max(0, max_ti - 1), max_ts


# == SERVER SETUP ==============================================================

total_models = len(RUN_SEQUENCE)
total_tests  = total_models * len(TEST_SEQUENCE)
total_min    = total_tests * TIME_LIMIT_SECONDS // 60

print(f"\nNE Receiver -- Phase 4{'  [TEST MODE]' if TEST_MODE else ''}")
if TEST_MODE:
    print(f"  Model : {TEST_MODE_MODEL['MODEL_NAME']}  ({TEST_MODE_MODEL['OUTPUT_TYPE']})")
    print(f"  Configs: {TEST_MODE_CONFIGS}")
else:
    print(f"  {total_models} models x {len(TEST_SEQUENCE)} tests = "
          f"{total_tests} total tests  (~{total_min} min / ~{total_min//60//24} days)")
    print(f"  Auto-resume: completed models are skipped on restart")
print()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('localhost', 8080))
server_socket.listen(1)
server_socket.setblocking(False)
print(f"Listening on port 8080\n")

shutdown = False

# == OUTER MODEL LOOP ==========================================================

for model_num, model_cfg in enumerate(RUN_SEQUENCE):
    if shutdown:
        break

    # Resolve paths for this model
    _MODEL_FAMILY = model_cfg['MODEL_FAMILY']
    _OUTPUT_TYPE  = model_cfg['OUTPUT_TYPE']
    _MODEL_NAME   = model_cfg['MODEL_NAME']
    _MODEL_PATH   = os.path.join(_PROJECT_ROOT, model_cfg['MODEL_PATH'])
    _NEAT_CFG     = (os.path.join(_PROJECT_ROOT, model_cfg['NEAT_CONFIG_PATH'])
                     if model_cfg['NEAT_CONFIG_PATH'] else '')
    _CSV_SUBFOLDER = model_cfg['CSV_SUBFOLDER']

    _CSV_DIR  = os.path.join(_PROJECT_ROOT, 'Data', 'NE_results', _CSV_SUBFOLDER)
    os.makedirs(_CSV_DIR, exist_ok=True)
    _csv_stem = f'{_MODEL_NAME}_testmode' if TEST_MODE else _MODEL_NAME
    _CSV_PATH = os.path.join(_CSV_DIR, f'{_csv_stem}_error_log.csv')

    # Check resume state -- skip if already complete
    start_ti, start_ts = get_resume_state(_CSV_PATH, len(TEST_SEQUENCE))
    if start_ti >= len(TEST_SEQUENCE):
        print(f"[{model_num+1}/{total_models}] {_MODEL_NAME} -- already complete, skipping")
        continue

    models_left = total_models - model_num
    min_left    = models_left * len(TEST_SEQUENCE) * TIME_LIMIT_SECONDS // 60
    print(f"\n{'#'*62}")
    print(f"  MODEL {model_num+1}/{total_models}: {_MODEL_NAME}")
    print(f"  Family={_MODEL_FAMILY}  Output={_OUTPUT_TYPE}")
    if start_ti > 0:
        print(f"  Resuming from test {start_ti + 1}/{len(TEST_SEQUENCE)}  "
              f"(timestep continues from {start_ts})")
    print(f"  ~{min_left} min remaining across {models_left} model(s)")
    print(f"  CSV: {_CSV_PATH}")
    print(f"{'#'*62}\n")

    print("Loading model ...")
    _model = load_model(_MODEL_FAMILY, _OUTPUT_TYPE, _MODEL_PATH, _NEAT_CFG)
    print("Model ready.\n")

    # CSV -- append mode; write header only for new files
    csv_is_new = not os.path.exists(_CSV_PATH) or os.path.getsize(_CSV_PATH) == 0
    csv_file   = open(_CSV_PATH, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(['test_index', 'timestep', 'timestamp', 'error',
                             'lap_number', 'speed_scalar', 'blur_level'])

    timestep   = start_ts
    test_index = start_ti

    # == INNER TEST LOOP =========================================================
    # Runs all 75 tests for this model (or resumes from start_ti).
    # Each test runs for exactly TIME_LIMIT_SECONDS regardless of drone behaviour.
    # On disconnect, re-accepts within the same test's 5-min timer.

    while test_index < len(TEST_SEQUENCE) and not shutdown:
        current_speed, current_blur = TEST_SEQUENCE[test_index]
        current_yaw_scalar = max(0.5, 1.55 - 1.05 * (current_speed - 0.1) / 0.9)

        remaining_min = (len(TEST_SEQUENCE) - test_index) * TIME_LIMIT_SECONDS // 60
        print(f"\n{'='*60}")
        print(f"  TEST {test_index + 1}/{len(TEST_SEQUENCE)}:  "
              f"speed={current_speed:.1f}  blur={current_blur}  "
              f"({remaining_min} min remaining this model)")
        print(f"{'='*60}")

        # Test-level state -- persists across mid-test reconnects
        start_time           = None
        first_image_received = False
        lap_errors           = []
        lap_records          = []
        test_done            = False

        # Reconnect loop -- re-accepts socket each time controller drops
        while not test_done and not shutdown:
            print(f"  Waiting for controller connection ...")
            client_socket = None
            try:
                while True:
                    try:
                        client_socket, addr = server_socket.accept()
                        print(f"  Controller connected from {addr}")
                        client_socket.setblocking(False)
                        break
                    except socket.error:
                        if start_time and time.time() - start_time >= TIME_LIMIT_SECONDS:
                            print(f"\n{TIME_LIMIT_SECONDS}s elapsed (during reconnect wait) -- "
                                  f"test {test_index + 1} complete")
                            test_index += 1
                            test_done = True
                            break
                        time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nShutting down ...")
                shutdown = True
                break

            if test_done or shutdown:
                break

            # Per-connection state -- reset on every (re)connect
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
                                # Parse packet
                                raw_array    = np.frombuffer(buffer[:4096], dtype=np.uint8).reshape((64, 64))
                                lap_count    = int.from_bytes(buffer[4096:4100], byteorder='little', signed=True)
                                speed_scalar = struct.unpack('<f', buffer[4100:4104])[0]
                                buffer       = buffer[EXPECTED_BYTES:]

                                # Start timer on the very first image of this test
                                if not first_image_received:
                                    start_time           = time.time()
                                    lap_start_time       = start_time
                                    first_image_received = True
                                    print(f"  First image received -- starting {TIME_LIMIT_SECONDS}s timer  "
                                          f"[speed={speed_scalar:.2f}  blur={current_blur}]")
                                elif lap_start_time is None:
                                    lap_start_time = time.time()

                                elapsed_seconds = time.time() - start_time

                                # 5-min timer expired
                                if elapsed_seconds >= TIME_LIMIT_SECONDS:
                                    print(f"\n{TIME_LIMIT_SECONDS}s elapsed -- test {test_index + 1} complete")
                                    if lap_errors:
                                        print(f"  avg_err={np.mean(lap_errors):.1f}px  laps={lap_count}")
                                    test_index += 1
                                    next_speed = (TEST_SEQUENCE[test_index][0]
                                                  if test_index < len(TEST_SEQUENCE)
                                                  else TEST_SPEEDS[0])
                                    reset_cmd = build_command(
                                        _OUTPUT_TYPE,
                                        np.zeros(OUTPUT_DIMS[_OUTPUT_TYPE]),
                                        next_speed,
                                        max(0.5, 1.55 - 1.05 * (next_speed - 0.1) / 0.9),
                                        reset=True,
                                    )
                                    send_command(client_socket, reset_cmd)
                                    print(f"  Sent reset -- next speed={next_speed:.1f}")
                                    test_done = True
                                    break

                                # Lap change
                                if lap_count > prev_lap_count:
                                    now      = time.time()
                                    lap_time = now - lap_start_time
                                    if lap_errors and lap_time >= 2.0:
                                        print(f"  LAP {lap_count}  time={lap_time:.1f}s  "
                                              f"avg_err={np.mean(lap_errors):.1f}px")
                                    lap_records.append({
                                        'lap_num':  lap_count,
                                        'lap_time': lap_time,
                                        'errors':   list(lap_errors),
                                    })
                                    lap_start_time = now
                                    lap_errors     = []
                                    prev_lap_count = lap_count

                                # Apply blur
                                sigma   = BLUR_SIGMAS[current_blur]
                                blurred = (gaussian_filter(raw_array.astype(np.float32), sigma=sigma)
                                           if sigma > 0 else raw_array.astype(np.float32))

                                # Error for CSV (identical baseline logic)
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
                                        print(f"  [TRACK WARNING] No track for "
                                              f"{TRACK_LOSS_TIMEOUT * 32 / 1000:.0f}s -- "
                                              f"drone may be off-track (test continues)")
                                        warned_track_loss = True

                                # Features for neat/feature_mlp
                                if _MODEL_FAMILY in ('neat', 'feature_mlp'):
                                    slope, line_at_middle = extract_model_features(blurred)
                                else:
                                    slope, line_at_middle = None, None

                                # Model inference
                                raw_out = run_inference(
                                    _model, _MODEL_FAMILY, _OUTPUT_TYPE,
                                    blurred, speed_scalar, slope, line_at_middle,
                                )

                                # Periodic heartbeat -- once per ~500 frames (~16s)
                                if timestep % 500 == 0:
                                    if _OUTPUT_TYPE == 'ycommand':
                                        fwd_d = int(raw_out[0] > YCOMMAND_FWD_THRESHOLD)
                                        yi_d  = int(raw_out[1] > YCOMMAND_YAW_THRESHOLD)
                                        yd_d  = int(raw_out[2] > YCOMMAND_YAW_THRESHOLD)
                                        print(f"  [step={timestep}  t={elapsed_seconds:.0f}s] "
                                              f"out={np.round(raw_out, 3)}  -> fwd={fwd_d} yi={yi_d} yd={yd_d}  "
                                              f"err={error:.1f}px")
                                    else:
                                        print(f"  [step={timestep}  t={elapsed_seconds:.0f}s] "
                                              f"out={np.round(raw_out, 3)}  err={error:.1f}px")

                                # Build and send command
                                cmd = build_command(_OUTPUT_TYPE, raw_out,
                                                    current_speed, current_yaw_scalar)
                                send_command(client_socket, cmd)

                                # Log to CSV
                                timestep += 1
                                csv_writer.writerow([
                                    test_index + 1,   # 1-based
                                    timestep,
                                    round(elapsed_seconds, 4),
                                    round(error, 4),
                                    lap_count,
                                    round(speed_scalar, 4),
                                    current_blur,
                                ])
                                csv_file.flush()

                            if test_done:
                                break

                        elif len(chunk) == 0:
                            elapsed = round(time.time() - start_time, 1) if start_time else 0.0
                            print(f"  Controller disconnected at {elapsed}s -- waiting for reconnect")
                            break

                        if first_image_received and (time.time() - last_data_time) > CRASH_TIMEOUT:
                            elapsed = round(time.time() - start_time, 1)
                            print(f"  No data for {CRASH_TIMEOUT}s at {elapsed}s -- waiting for reconnect")
                            break

                    except socket.error:
                        if first_image_received and (time.time() - last_data_time) > CRASH_TIMEOUT:
                            elapsed = round(time.time() - start_time, 1)
                            print(f"  No data for {CRASH_TIMEOUT}s at {elapsed}s -- waiting for reconnect")
                            break

                    time.sleep(0.001)

            except KeyboardInterrupt:
                print("\nShutting down ...")
                shutdown = True

            try:
                client_socket.close()
            except Exception:
                pass
        # End reconnect loop

    # End inner test loop
    csv_file.close()
    if test_index >= len(TEST_SEQUENCE):
        print(f"\n  {_MODEL_NAME} COMPLETE -- all {len(TEST_SEQUENCE)} tests done.")
    else:
        print(f"\n  {_MODEL_NAME} stopped at test {test_index}/{len(TEST_SEQUENCE)}.")

# End outer model loop
server_socket.close()

if not shutdown:
    print(f"\n{'#'*62}")
    print(f"  ALL {total_models} MODELS COMPLETE")
    print(f"  Results in: {os.path.join(_PROJECT_ROOT, 'Data', 'NE_results')}/")
    print(f"{'#'*62}\n")
else:
    print(f"\nStopped by user. Restart to resume from where it left off.\n")
