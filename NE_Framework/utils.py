import torch
import torch.nn as nn
import json
import numpy as np
import csv
import os


def load_json(json_file_path, subsample=None):
    """
    Load training/testing data from a JSON Lines file produced by prepare_data.py.

    Each line is one JSON object with fields:
        matrix           : list of 4096 ints (0-255), flat 64x64 grayscale image
        binary_commands  : [forward, yaw_increase, yaw_decrease, reset]  (4 ints)
        continuous_commands: [forward_desired, yaw_desired]               (2 floats)
        motor_velocities : [m1, m2, m3, m4]                              (4 floats)
        speed_scalar     : float
        blur_level       : int 0-4 (ignored here — augmentation already baked in)

    Returns (all float32 tensors):
        X            : (N, 4097)  image pixels normalised to [0,1] + speed_scalar
        y_command    : (N, 3)     binary_commands[:3] — forward, yaw_increase, yaw_decrease
        y_continuous : (N, 2)     continuous_commands
        y_motors     : (N, 4)     motor_velocities

    Args:
        json_file_path : path to training.json or testing.json
        subsample      : if an int, randomly sample that many records (without replacement)
    """
    records = []
    with open(json_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if subsample is not None and subsample < len(records):
        indices = np.random.choice(len(records), subsample, replace=False)
        records = [records[i] for i in indices]

    n = len(records)
    X_arr     = np.empty((n, 4097), dtype=np.float32)
    y_cmd_arr = np.empty((n, 3),    dtype=np.float32)
    y_cont_arr= np.empty((n, 2),    dtype=np.float32)
    y_mot_arr = np.empty((n, 4),    dtype=np.float32)

    for i, r in enumerate(records):
        # Image: normalise from [0,255] to [0,1]
        img = np.array(r['matrix'], dtype=np.float32) / 255.0
        X_arr[i, :4096] = img
        X_arr[i, 4096]  = float(r['speed_scalar'])

        cmds = r['binary_commands']
        y_cmd_arr[i]  = [float(cmds[0]), float(cmds[1]), float(cmds[2])]
        y_cont_arr[i] = [float(v) for v in r['continuous_commands']]
        y_mot_arr[i]  = [float(v) for v in r['motor_velocities']]

    X            = torch.tensor(X_arr,      dtype=torch.float32)
    y_command    = torch.tensor(y_cmd_arr,  dtype=torch.float32)
    y_continuous = torch.tensor(y_cont_arr, dtype=torch.float32)
    y_motors     = torch.tensor(y_mot_arr,  dtype=torch.float32)

    return X, y_command, y_continuous, y_motors


def load_json_stratified(json_file_path, n_per_cell):
    """
    Load a deterministic stratified subsample from a JSON Lines file.

    Groups records by (speed_scalar, blur_level), takes the first n_per_cell
    records from each group in file order (no randomness), then assembles tensors.
    With 25 cells (5 speeds × 5 blur levels) and n_per_cell=800: 20,000 records total.

    Args:
        json_file_path : path to all_data.json or any balanced JSON Lines file
        n_per_cell     : records to take from each (speed_scalar, blur_level) group

    Returns same 4 tensors as load_json: X, y_command, y_continuous, y_motors
    """
    from collections import defaultdict as _dd
    cells = _dd(list)
    with open(json_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r   = json.loads(line)
            key = (round(float(r['speed_scalar']), 1), int(r['blur_level']))
            if len(cells[key]) < n_per_cell:
                cells[key].append(r)

    records = []
    for key in sorted(cells):
        records.extend(cells[key])

    n          = len(records)
    X_arr      = np.empty((n, 4097), dtype=np.float32)
    y_cmd_arr  = np.empty((n, 3),    dtype=np.float32)
    y_cont_arr = np.empty((n, 2),    dtype=np.float32)
    y_mot_arr  = np.empty((n, 4),    dtype=np.float32)

    for i, r in enumerate(records):
        img = np.array(r['matrix'], dtype=np.float32) / 255.0
        X_arr[i, :4096] = img
        X_arr[i, 4096]  = float(r['speed_scalar'])

        cmds = r['binary_commands']
        y_cmd_arr[i]  = [float(cmds[0]), float(cmds[1]), float(cmds[2])]
        y_cont_arr[i] = [float(v) for v in r['continuous_commands']]
        y_mot_arr[i]  = [float(v) for v in r['motor_velocities']]

    X            = torch.tensor(X_arr,      dtype=torch.float32)
    y_command    = torch.tensor(y_cmd_arr,  dtype=torch.float32)
    y_continuous = torch.tensor(y_cont_arr, dtype=torch.float32)
    y_motors     = torch.tensor(y_mot_arr,  dtype=torch.float32)

    return X, y_command, y_continuous, y_motors


def load_json_combined(train_path, test_path, subsample=None):
    """
    Load and concatenate training.json + testing.json into a single dataset.
    Used by notrain scripts, which have no overfitting risk and benefit from
    the full ~100k records for a more representative fitness signal.

    Args:
        train_path : path to training.json
        test_path  : path to testing.json
        subsample  : if an int, randomly sample that many records after combining

    Returns same 4 tensors as load_json: X, y_command, y_continuous, y_motors
    """
    records = []
    for path in (train_path, test_path):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if subsample is not None and subsample < len(records):
        indices = np.random.choice(len(records), subsample, replace=False)
        records = [records[i] for i in indices]

    n = len(records)
    X_arr      = np.empty((n, 4097), dtype=np.float32)
    y_cmd_arr  = np.empty((n, 3),    dtype=np.float32)
    y_cont_arr = np.empty((n, 2),    dtype=np.float32)
    y_mot_arr  = np.empty((n, 4),    dtype=np.float32)

    for i, r in enumerate(records):
        img = np.array(r['matrix'], dtype=np.float32) / 255.0
        X_arr[i, :4096] = img
        X_arr[i, 4096]  = float(r['speed_scalar'])

        cmds = r['binary_commands']
        y_cmd_arr[i]  = [float(cmds[0]), float(cmds[1]), float(cmds[2])]
        y_cont_arr[i] = [float(v) for v in r['continuous_commands']]
        y_mot_arr[i]  = [float(v) for v in r['motor_velocities']]

    X            = torch.tensor(X_arr,      dtype=torch.float32)
    y_command    = torch.tensor(y_cmd_arr,  dtype=torch.float32)
    y_continuous = torch.tensor(y_cont_arr, dtype=torch.float32)
    y_motors     = torch.tensor(y_mot_arr,  dtype=torch.float32)

    return X, y_command, y_continuous, y_motors


def load_csv(csv_file_path, return_tensor=True, dtype=np.float32):
    """Load a CSV file into a numpy array or PyTorch tensor."""
    data = []
    with open(csv_file_path, 'r') as f:
        for row in csv.reader(f):
            data.append([float(v) for v in row])
    array = np.array(data, dtype=dtype)
    return torch.tensor(array, dtype=torch.float32) if return_tensor else array


def save_to_csv(data, file_path):
    """Save a list, numpy array, or tensor to a CSV file."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


def loss(predictions, targets, loss_type='mse'):
    """
    Compute scalar loss between predictions and targets.

    Args:
        loss_type: 'mse' | 'mae' | 'rmse'
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

    if loss_type == 'mse':
        return nn.MSELoss()(predictions, targets).item()
    elif loss_type == 'mae':
        return nn.L1Loss()(predictions, targets).item()
    elif loss_type == 'rmse':
        return torch.sqrt(nn.MSELoss()(predictions, targets)).item()
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose mse, mae, or rmse.")


def json_to_csv(input_data_path, output_data_path):
    """
    Convert a JSON Lines data file to CSV files for inspection or legacy use.

    Creates 4 CSV files in output_data_path:
        matrix_speed.csv       — 4097 columns: 4096 normalised pixel values + speed_scalar
        command.csv            — 3 columns: forward, yaw_increase, yaw_decrease
        continuous_command.csv — 2 columns: forward_desired, yaw_desired
        motor_velocities.csv   — 4 columns: m1, m2, m3, m4

    Args:
        input_data_path  : path to training.json or testing.json (JSON Lines)
        output_data_path : directory where CSV files are written
    """
    os.makedirs(output_data_path, exist_ok=True)

    records = []
    with open(input_data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    matrix_path   = os.path.join(output_data_path, 'matrix_speed.csv')
    command_path  = os.path.join(output_data_path, 'command.csv')
    cont_path     = os.path.join(output_data_path, 'continuous_command.csv')
    motors_path   = os.path.join(output_data_path, 'motor_velocities.csv')

    with (open(matrix_path,  'w', newline='') as mf,
          open(command_path, 'w', newline='') as cf,
          open(cont_path,    'w', newline='') as nf,
          open(motors_path,  'w', newline='') as mo):

        mw = csv.writer(mf)
        cw = csv.writer(cf)
        nw = csv.writer(nf)
        ow = csv.writer(mo)

        for r in records:
            img   = [v / 255.0 for v in r['matrix']]
            speed = [float(r['speed_scalar'])]
            mw.writerow(img + speed)

            cmds = r['binary_commands']
            cw.writerow([cmds[0], cmds[1], cmds[2]])

            nw.writerow(r['continuous_commands'])
            ow.writerow(r['motor_velocities'])
