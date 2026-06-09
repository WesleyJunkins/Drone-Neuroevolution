#!/usr/bin/env python3
"""
run_all.py — Configuration hub and launcher for all NE experiments.

Shared hyperparameters are imported by every run_*.py script.
Run this file directly to launch all 6 experiments in separate terminal windows.

To run: python run_all.py # LAUNCH_MODE = 'notrain'
To run: python run_all.py # LAUNCH_MODE = 'train'
To run: python run_all.py # LAUNCH_MODE = 'all'
"""

import os
import glob
import subprocess
import sys
import time
from pathlib import Path
import torch

# ── Device ───────────────────────────────────────────────────────────────────
# Auto-select: MPS (Apple Silicon) → CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Data ─────────────────────────────────────────────────────────────────────
# Paths to cleaned/split data produced by Data/prepare_data.py
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE     = os.path.join(_BASE, '..', 'Data', 'datasets', 'training.json')
TESTING_FILE  = os.path.join(_BASE, '..', 'Data', 'datasets', 'testing.json')
ALL_DATA_FILE = os.path.join(_BASE, '..', 'Data', 'datasets', 'all_data.json')

# All three data files are pre-sized by Data/prepare_data.py (800/cell × 25 cells pool):
#   training.json  = 16,000 records (640/cell — 80% of pool)
#   testing.json   =  4,000 records (160/cell — 20% of pool; loaded in full for fitness eval)
#   all_data.json  = 20,000 records (800/cell — train+test; loaded in full by notrain scripts)
#
# Trained NE: randomly subsamples training.json each run to keep per-individual backprop feasible.
# Notrain NE: loads all_data.json in full (no in-script trimming; already the right size).
TRAINING_SUBSAMPLE = 5000    # random subsample from training.json (16k) — ~31% of the file
NOTRAIN_SUBSAMPLE  = 20000   # informational — matches all_data.json record count

# ── Genetic Algorithm ─────────────────────────────────────────────────────────
# Trained NE (backprop per individual): fewer generations, each gen is expensive.
# Untrained NE (pure evolution): more generations and larger population, each gen is cheap.
GENERATIONS_TRAIN    = 200   # hybrid: backprop converges within-individual; 200 gens is ample
GENERATIONS_NOTRAIN  = 5000  # pure evolution — large budget without gradient signal

# Trained NE population: kept small because each individual requires full backprop training.
POPULATION_SIZE       = 20
BEST_INDIVIDUALS_SIZE = 5    # elites kept per generation (25% of 20)

# Untrained NE population: larger for better selection quality and search coverage.
# Reduced from 100→50 and 10k→5k gens to fit 3 simultaneous MPS processes without
# GPU command buffer errors on M2 Ultra.
POPULATION_SIZE_NOTRAIN       = 50
BEST_INDIVIDUALS_SIZE_NOTRAIN = 12   # elites kept per generation (~24% of 50)

CROSSOVER_RATE    = 0.5   # probability of inheriting weights from second parent
MUTATION_RATE     = 0.05  # probability of perturbing each weight
MUTATION_STRENGTH = 0.05  # std-dev of Gaussian noise added to mutated weights

# ── Training (used by run_train_*.py only) ────────────────────────────────────
TRAINING_EPOCHS        = 20
TRAINING_LEARNING_RATE = 0.001
TRAINING_VERBOSE       = True   # True → print per-epoch loss for every individual

# ── Mini-batch ────────────────────────────────────────────────────────────────
# Batch size for both training and fitness evaluation.
# Keeps peak GPU/CPU memory manageable with large datasets.
BATCH_SIZE = 512

# ── Model ─────────────────────────────────────────────────────────────────────
# 'cnn'  → CreateCNNModel  (recommended — exploits spatial structure of images)
# 'mlp'  → CreateModel     (fully-connected baseline)
MODEL_TYPE = 'cnn'

# ── Output ────────────────────────────────────────────────────────────────────
SAVE_MODELS         = False  # True = save every individual's model (slow, large)
SAVE_CSV_RESULTS    = True   # True = save per-generation fitness CSVs
SAVE_CHECKPOINTS    = True   # True = save best model every CHECKPOINT_INTERVAL gens
CHECKPOINT_INTERVAL = 25     # save a checkpoint every N generations

# ── Convergence ───────────────────────────────────────────────────────────────
# A run is considered converged when the best fitness improves by less than
# CONVERGENCE_THRESHOLD (relative) over the last CONVERGENCE_WINDOW generations.
# A notice is printed once when convergence is first detected.
# Set EARLY_STOPPING = True to automatically halt the run at that point.
CONVERGENCE_WINDOW    = 25     # rolling window length (generations)
CONVERGENCE_THRESHOLD = 0.001  # min relative improvement to count as progress (0.1%)
EARLY_STOPPING        = False  # halt the run when convergence is detected

# ─────────────────────────────────────────────────────────────────────────────


def get_run_files():
    current_dir = Path(__file__).parent
    current_script = Path(__file__).name
    run_files = glob.glob(str(current_dir / 'run_*.py'))
    return sorted(
        f for f in run_files
        if os.path.dirname(f) == str(current_dir)
        and os.path.basename(f) != current_script
    )


def run_in_terminal_macos(script_path):
    script_path = os.path.abspath(script_path)
    working_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    applescript = f'''
    tell application "Terminal"
        activate
        do script "cd '{working_dir}' && python3 {script_name}"
    end tell
    '''
    subprocess.run(['osascript', '-e', applescript], check=False)


def run_in_terminal_linux(script_path):
    script_path = os.path.abspath(script_path)
    working_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    for term in ['gnome-terminal', 'xterm', 'konsole', 'terminator']:
        try:
            if term == 'gnome-terminal':
                subprocess.Popen([term, '--', 'bash', '-c',
                                  f'cd "{working_dir}" && python3 {script_name}; exec bash'])
            else:
                subprocess.Popen([term, '-e',
                                  f'bash -c "cd \\"{working_dir}\\" && python3 {script_name}; exec bash"'])
            return
        except FileNotFoundError:
            continue
    print(f"No terminal emulator found. Run manually: python3 {script_path}")


def run_in_terminal_windows(script_path):
    script_path = os.path.abspath(script_path)
    working_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    subprocess.Popen(f'start cmd /k "cd /d "{working_dir}" && python {script_name}"', shell=True)


def _launch(script_path):
    if sys.platform == 'darwin':
        run_in_terminal_macos(script_path)
    elif sys.platform.startswith('linux'):
        run_in_terminal_linux(script_path)
    elif sys.platform == 'win32':
        run_in_terminal_windows(script_path)
    else:
        subprocess.Popen([sys.executable, script_path])


def main():
    # LAUNCH_MODE controls experiment launch strategy:
    #   'train_only' → run each trained script one at a time in this terminal (blocks until
    #                  each finishes before starting the next — guaranteed sequential, no GPU errors)
    #   'notrain'    → launch all 3 notrain scripts in parallel in separate Terminal windows,
    #                  then auto-launch the 3 trained scripts sequentially when notrain completes
    #   'all'        → launch all 6 notrain scripts at once in separate Terminal windows
    #                  (not recommended — may cause MPS GPU errors)
    LAUNCH_MODE = 'train_only'

    # Stagger delay used when opening multiple Terminal windows at once (notrain / all modes).
    LAUNCH_DELAY_SECONDS = 30

    all_files = get_run_files()
    if not all_files:
        print("No run_*.py files found.")
        return

    trained_files = [f for f in all_files if 'notrain' not in os.path.basename(f)]
    notrain_files = [f for f in all_files if 'notrain' in os.path.basename(f)]

    def launch_batch(files, label):
        """Open each script in a new Terminal window (non-blocking)."""
        if not files:
            print(f"No {label} files found.")
            return
        print(f"\nLaunching {len(files)} {label} experiment(s)  ({LAUNCH_DELAY_SECONDS}s stagger)...")
        for i, run_file in enumerate(files):
            print(f"  → {os.path.basename(run_file)}")
            _launch(run_file)
            if i < len(files) - 1:
                print(f"     (waiting {LAUNCH_DELAY_SECONDS}s before next launch...)")
                time.sleep(LAUNCH_DELAY_SECONDS)
        print(f"✓ {len(files)} {label} experiment(s) launched.")

    def run_sequential(files, label):
        """Run each script directly in this process, one at a time (blocking)."""
        print(f"\nRunning {len(files)} {label} experiment(s) sequentially in this terminal ...")
        for i, run_file in enumerate(files, 1):
            name = os.path.basename(run_file)
            print(f"\n{'='*60}")
            print(f"  [{i}/{len(files)}]  Starting {name}")
            print(f"{'='*60}")
            subprocess.run([sys.executable, run_file], cwd=os.path.dirname(run_file))
            print(f"\n  [{i}/{len(files)}]  {name} finished.")
        print(f"\n✓ All {len(files)} {label} experiment(s) complete.")

    if not os.path.exists(DATA_FILE):
        print(f"\nWARNING: DATA_FILE not found: {DATA_FILE}")
        print("  Run Data/prepare_data.py first to generate training.json.")
    else:
        print(f"Trained data : {os.path.basename(DATA_FILE)}  (subsample={TRAINING_SUBSAMPLE:,})")
        print(f"Notrain data : {os.path.basename(ALL_DATA_FILE)}  ({NOTRAIN_SUBSAMPLE:,} records — loaded in full)")

    if LAUNCH_MODE == 'train_only':
        print(f"\nLAUNCH_MODE = 'train_only'  — 3 trained experiments, one at a time, in this terminal")
        print(f"  Order : {[os.path.basename(f) for f in trained_files]}")
        run_sequential(trained_files, 'trained')

    elif LAUNCH_MODE == 'notrain':
        print(f"\nLAUNCH_MODE = 'notrain'  — 3 notrain experiments in parallel Terminal windows")
        print(f"  Notrain : {[os.path.basename(f) for f in notrain_files]}")
        launch_batch(notrain_files, 'notrain')
        print(f"\n  Notrain experiments running.  Start 'train_only' mode separately when ready.")

    elif LAUNCH_MODE == 'all':
        print("WARNING: launching all 6 simultaneously may cause MPS GPU errors.")
        for i, f in enumerate(all_files, 1):
            print(f"  {i}. {os.path.basename(f)}")
        launch_batch(all_files, 'all')


if __name__ == '__main__':
    main()
