#!/usr/bin/env python3
"""
Experiment E — Standard Backprop CNN Training

Trains the same CreateCNNModel architecture used in the main NE experiments
using only standard supervised learning (Adam + MSELoss, no genetic algorithm).

Comparison context:
  NE trained experiments  — 20 epochs per individual, 5,000-record subsample
  This experiment          — up to 1,000 epochs (convergence-stopped), 16,000 training records

Three output types run sequentially:
  ycommand    — binary commands [forward, yaw_increase, yaw_decrease]  (output_dim=3)
  ycontinuous — continuous values [forward_desired, yaw_desired]        (output_dim=2)
  ymotors     — motor velocities [m1, m2, m3, m4]                       (output_dim=4)

Data: all_data.json (20,000 records — same single file used by all other supplementary
experiments). Split internally: first 640 of each (speed × blur) cell → training (16,000
total), last 160 → validation (4,000 total). This is the same 80/20 stratified split
used to produce training.json and testing.json in prepare_data.py.

Output format:
  best_model.pth — plain state_dict, compatible with NE_receiver.py MODEL_FAMILY='main_cnn'
  metadata.json  — best_val_loss, best_epoch, stopped_epoch, record counts
  training_curve.csv — epoch, train_loss, val_loss, lr, elapsed_s

--- Google Colab setup ---
1. Ensure all_data.json is on Google Drive:
       My Drive/WesDroneRL2/receivers/all_data.json
2. Run this script in Colab (paste into a code cell or upload as .py).
3. Results are saved to Google Drive under:
       My Drive/WesDroneRL2/Neuroevolution2/exp_e_standard_backprop/{output_type}/

--- Simulation testing (after training) ---
Load models with MODEL_FAMILY='main_cnn' in NE_receiver.py, pointing MODEL_PATH to
the new best_model.pth files. No code changes needed — the plain state_dict format
is identical to the main-phase NE models.
"""

import sys, os

# ── Colab: mount Google Drive ─────────────────────────────────────────────────
COLAB = 'google.colab' in sys.modules
if not COLAB:
    try:
        from google.colab import drive as _d
        COLAB = True
    except ImportError:
        pass

if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

# ── Paths ─────────────────────────────────────────────────────────────────────
if COLAB:
    DATA_DIR   = '/content/drive/MyDrive/WesDroneRL2/receivers'
    OUTPUT_DIR = '/content/drive/MyDrive/WesDroneRL2/Neuroevolution2/exp_e_standard_backprop'
else:
    _HERE      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(_HERE, '..', 'Data', 'datasets')
    OUTPUT_DIR = os.path.join(_HERE, 'exp_e_standard_backprop')

ALL_DATA_FILE = os.path.join(DATA_DIR, 'all_data.json')

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS_MAX            = 1000  # safety cap; convergence stops the run much sooner
LEARNING_RATE         = 0.001
BATCH_SIZE            = 512
CONVERGENCE_WINDOW    = 25    # match existing NE experiments exactly
CONVERGENCE_THRESHOLD = 0.001 # match existing NE experiments exactly
LR_PATIENCE           = 10    # ReduceLROnPlateau: halve LR when stalled for this many epochs
LR_FACTOR             = 0.5
LR_MIN                = 1e-6
TRAIN_RATIO           = 0.8   # 80/20 split per (speed × blur) cell — same as prepare_data.py

# Output types: (name, output_dim)
# ymotors is excluded: motor velocities in training data are PID controller outputs that
# depend on IMU feedback unavailable at inference time. This architectural mismatch causes
# all ymotors models to crash in simulation regardless of training quality.
OUTPUT_TYPES = [
    ('ycommand',    3),   # binary [forward, yaw_increase, yaw_decrease]
    ('ycontinuous', 2),   # continuous [forward_desired, yaw_desired]
]

# ── Imports ───────────────────────────────────────────────────────────────────
import json, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── CNN Architecture (identical to Model.py CreateCNNModel) ────
class CreateCNNModel(nn.Module):
    """
    Input: flat tensor (N, 4097) — 4096 normalized pixels (0-1) + 1 speed scalar.
    Architecture: 3× Conv2d+MaxPool → flatten + concat speed → 3× Linear.
    ~1.1M parameters.
    """
    def __init__(self, output_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # -> 16x64x64
        self.pool1 = nn.MaxPool2d(2, 2)                            # -> 16x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> 32x32x32
        self.pool2 = nn.MaxPool2d(2, 2)                            # -> 32x16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64x16x16
        self.pool3 = nn.MaxPool2d(2, 2)                            # -> 64x8x8 = 4096
        self.fc1   = nn.Linear(4097, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, output_features)
        self.output_features = output_features

    def forward(self, x):
        img   = x[:, :4096].view(-1, 1, 64, 64)
        speed = x[:, 4096:4097]
        img   = self.pool1(F.relu(self.conv1(img)))
        img   = self.pool2(F.relu(self.conv2(img)))
        img   = self.pool3(F.relu(self.conv3(img)))
        img   = img.view(img.size(0), -1)                  # (N, 4096)
        combined = torch.cat([img, speed], dim=1)          # (N, 4097)
        out = F.relu(self.fc1(combined))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Data loading ──────────────────────────────────────────────────────────────
def _records_to_tensors(records):
    n = len(records)
    X      = np.empty((n, 4097), dtype=np.float32)
    y_cmd  = np.empty((n, 3),    dtype=np.float32)
    y_cont = np.empty((n, 2),    dtype=np.float32)
    y_mot  = np.empty((n, 4),    dtype=np.float32)
    for i, r in enumerate(records):
        X[i, :4096] = np.array(r['matrix'], dtype=np.float32) / 255.0
        X[i, 4096]  = float(r['speed_scalar'])
        cmds = r['binary_commands']
        y_cmd[i]  = [cmds[0], cmds[1], cmds[2]]
        y_cont[i] = r['continuous_commands']
        y_mot[i]  = r['motor_velocities']
    return (torch.tensor(X,      device=DEVICE),
            torch.tensor(y_cmd,  device=DEVICE),
            torch.tensor(y_cont, device=DEVICE),
            torch.tensor(y_mot,  device=DEVICE))


def load_and_split(path):
    """Load all_data.json and split 80/20 stratified by (speed_scalar, blur_level) cell.

    Replicates the prepare_data.py split: first TRAIN_RATIO fraction of each cell goes
    to training, the remainder to validation. With 800 records/cell this yields
    640 train + 160 val per cell → 16,000 train / 4,000 val total.
    """
    from collections import defaultdict
    print(f"  Reading {os.path.basename(path)}...", end='', flush=True)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f" {len(records):,} records loaded.")

    cells = defaultdict(list)
    for r in records:
        cells[(r['speed_scalar'], r['blur_level'])].append(r)

    train_recs, val_recs = [], []
    for key in sorted(cells):
        cell = cells[key]
        n_tr = int(len(cell) * TRAIN_RATIO)
        train_recs.extend(cell[:n_tr])
        val_recs.extend(cell[n_tr:])

    print(f"  Split: {len(train_recs):,} train  /  {len(val_recs):,} val  "
          f"({len(cells)} cells, {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%})")
    return _records_to_tensors(train_recs), _records_to_tensors(val_recs)


def fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(out_name, output_dim, X_train, y_train, X_val, y_val, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, 'training_curve.csv')
    best_path = os.path.join(out_dir, 'best_model.pth')
    meta_path = os.path.join(out_dir, 'metadata.json')

    model     = CreateCNNModel(output_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN)

    n_train       = X_train.size(0)
    best_val_loss = float('inf')
    best_epoch    = 0
    val_history   = []   # rolling window for convergence check
    run_start     = time.time()
    stopped_epoch = EPOCHS_MAX

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'lr',
                                'elapsed_s', 'converged'])

    for epoch in range(1, EPOCHS_MAX + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        indices = torch.randperm(n_train, device=DEVICE)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            idx  = indices[start:start + BATCH_SIZE]
            out  = model(X_train[idx])
            loss = criterion(out, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        train_loss = total_loss / n_batches

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            total_v, count_v = 0.0, 0
            for start in range(0, X_val.size(0), BATCH_SIZE):
                xb  = X_val[start:start + BATCH_SIZE]
                yb  = y_val[start:start + BATCH_SIZE]
                total_v += criterion(model(xb), yb).item() * xb.size(0)
                count_v += xb.size(0)
            val_loss = total_v / count_v

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - run_start
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), best_path)

        # Convergence: improvement over last CONVERGENCE_WINDOW epochs < threshold
        # Same criterion as all existing NE experiments (window=25, threshold=0.001)
        converged = False
        if len(val_history) > CONVERGENCE_WINDOW:
            window_improvement = (val_history[-(CONVERGENCE_WINDOW + 1)]
                                  - min(val_history[-CONVERGENCE_WINDOW:]))
            if window_improvement < CONVERGENCE_THRESHOLD:
                converged = True

        print(f"  Epoch {epoch:>4}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={current_lr:.2e}  [{fmt_time(elapsed)}]"
              + ("  ◆ converged" if converged else ""))

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, round(train_loss, 8),
                                    round(val_loss, 8), current_lr,
                                    round(elapsed, 1), 1 if converged else 0])

        if converged:
            stopped_epoch = epoch
            print(f"\n  ◆ Converged at epoch {epoch}  "
                  f"(val_loss improvement < {CONVERGENCE_THRESHOLD} "
                  f"over last {CONVERGENCE_WINDOW} epochs)")
            break

    with open(meta_path, 'w') as f:
        json.dump({
            'output_dim':          output_dim,
            'best_val_loss':       best_val_loss,
            'best_epoch':          best_epoch,
            'stopped_epoch':        stopped_epoch,
            'converged':            stopped_epoch < EPOCHS_MAX,
            'total_train_records':  int(n_train),
            'total_val_records':    int(X_val.size(0)),
            'epochs_max':           EPOCHS_MAX,
            'learning_rate':        LEARNING_RATE,
            'batch_size':           BATCH_SIZE,
            'convergence_window':   CONVERGENCE_WINDOW,
            'convergence_threshold': CONVERGENCE_THRESHOLD,
        }, f, indent=2)

    print(f"\n  Best val_loss = {best_val_loss:.6f}  at epoch {best_epoch}")
    print(f"  Model saved  -> {best_path}")
    return best_val_loss, best_epoch, stopped_epoch


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(ALL_DATA_FILE):
        print(f"ERROR: {ALL_DATA_FILE} not found.")
        print(f"  Upload all_data.json to Google Drive at the path configured in DATA_DIR.")
        return

    print(f"\nLoading and splitting all_data.json ...")
    (X_tr, y_cmd_tr, y_cont_tr, y_mot_tr), \
    (X_val, y_cmd_val, y_cont_val, y_mot_val) = load_and_split(ALL_DATA_FILE)

    probe = CreateCNNModel(2)
    print(f"\nCNN parameters: {count_params(probe):,}")
    del probe

    y_train_map = {'ycommand': y_cmd_tr,  'ycontinuous': y_cont_tr,  'ymotors': y_mot_tr}
    y_val_map   = {'ycommand': y_cmd_val, 'ycontinuous': y_cont_val, 'ymotors': y_mot_val}

    results = []
    for out_name, out_dim in OUTPUT_TYPES:
        print(f"\n{'='*70}")
        print(f"  Training: {out_name}  (output_dim={out_dim})")
        print(f"  Epochs max={EPOCHS_MAX}  batch={BATCH_SIZE}  "
              f"lr0={LEARNING_RATE}  conv_window={CONVERGENCE_WINDOW}  "
              f"conv_threshold={CONVERGENCE_THRESHOLD}")
        print(f"  Train records={X_tr.size(0):,}  Val records={X_val.size(0):,}")
        print(f"{'='*70}")
        out_dir = os.path.join(OUTPUT_DIR, out_name)
        best_loss, best_epoch, stopped_epoch = train_model(
            out_name, out_dim,
            X_tr, y_train_map[out_name],
            X_val, y_val_map[out_name],
            out_dir)
        results.append((out_name, out_dim, best_loss, best_epoch, stopped_epoch))

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT E — FINAL SUMMARY")
    print(f"  Standard Backprop CNN  (epochs_max={EPOCHS_MAX}, batch={BATCH_SIZE})")
    print(f"{'='*70}")
    print(f"  {'Output Type':<16} {'Dim':>3}  {'Best Val Loss':>14}  {'Best Epoch':>10}  {'Converged At':>13}")
    print(f"  {'-'*16} {'-'*3}  {'-'*14}  {'-'*10}  {'-'*13}")
    for out_name, out_dim, best_loss, best_epoch, stopped_epoch in results:
        converged_str = f"ep {stopped_epoch}" if stopped_epoch < EPOCHS_MAX else f"ep {EPOCHS_MAX} (cap)"
        print(f"  {out_name:<16} {out_dim:>3}  {best_loss:>14.6f}  {best_epoch:>10}  {converged_str:>13}")
    print(f"\n  Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
