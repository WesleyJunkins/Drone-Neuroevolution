#!/usr/bin/env python3
"""
Supplementary Experiment C — Untrained NE with baseline vision features.

Replaces raw 64x64 pixel input (4,097 dimensions) with 3 low-dimensional
features extracted by the baseline controller's vision algorithm:
    [slope, line_at_middle, speed_scalar]

This reduces the search space from ~1.1M parameters to ~100-600 parameters,
making pure evolution tractable. It directly tests whether the bottleneck
in the main untrained NE experiments was the representation (raw pixels) or
the evolution process itself.

A small MLP (3 → 32 → 16 → output_dim) is used — ~60-700 parameters depending
on output dimension. Evolution alone shapes this network, no backprop.

Three output types tested sequentially: binary commands (3), continuous
values (2), motor velocities (4).

This is the direct counterpart to Experiment B (NEAT with same features):
compare standard GA vs. NEAT on the same 3-feature input.

--- Google Colab setup ---
1. Upload all_data.json to Google Drive at:
       My Drive/WesDroneRL2/receivers/all_data.json
2. Run this script — no additional packages needed (torch + numpy only)
3. Results saved to:  My Drive/WesDroneRL2/Neuroevolution2/exp_c_features/
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

# ── Config ────────────────────────────────────────────────────────────────────
if COLAB:
    DATA_DIR   = '/content/drive/MyDrive/WesDroneRL2/receivers'
    OUTPUT_DIR = '/content/drive/MyDrive/WesDroneRL2/Neuroevolution2/exp_c_features'
else:
    _HERE      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(_HERE, '..', 'Data', 'datasets')
    OUTPUT_DIR = os.path.join(_HERE, 'exp_c_features')

ALL_DATA_FILE = os.path.join(DATA_DIR, 'all_data.json')

# GA hyperparameters — identical to main notrain experiments
POPULATION_SIZE   = 50
N_ELITES          = 12       # ~24% of 50 — matches main notrain experiments
GENERATIONS       = 5000
CROSSOVER_RATE    = 0.5
MUTATION_RATE     = 0.05
MUTATION_STRENGTH = 0.05
BATCH_SIZE          = 512
CHECKPOINT_INTERVAL = 25     # matches main notrain checkpoint interval

# MLP architecture: input_dim → hidden1 → hidden2 → output_dim
# With 3 inputs, this is ~300-700 parameters total — tractable for evolution.
HIDDEN_DIMS = [32, 16]

# Output types: (name, output_dim)
OUTPUT_TYPES = [
    ('ycommand',    3),
    ('ycontinuous', 2),
    ('ymotors',     4),
]

# ── Imports ───────────────────────────────────────────────────────────────────
import json, csv, copy, random, time
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_records(path, subsample=None):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if subsample and subsample < len(records):
        records = random.sample(records, subsample)
    return records

# ── Baseline vision feature extraction ───────────────────────────────────────
def extract_features(records):
    """
    Extract 3 baseline vision features from each record:
        [slope, line_at_middle, speed_scalar]

    Mirrors the baseline controller's line-detection algorithm:
      1. Adaptive threshold = mean + 0.5 * std (same as baseline_receiver.py)
      2. Binary mask of bright (above-threshold) pixels
      3. Row-wise horizontal centroid of bright pixels
      4. Least-squares line fit: centroid = slope * row + intercept
      5. line_at_middle = slope * 32 + intercept  (track position at drone center)

    Records where the line is not detectable (< 5 valid rows) are given
    default values: slope=0, line_at_middle=31.5 (image center).
    """
    N       = len(records)
    X_feat  = np.zeros((N, 3), dtype=np.float32)
    col_idx = np.arange(64, dtype=np.float32)

    print(f"  Extracting features from {N:,} records ...")
    for i, r in enumerate(records):
        img   = np.array(r['matrix'], dtype=np.float32).reshape(64, 64)
        speed = float(r['speed_scalar'])

        thresh = img.mean() + 0.5 * img.std()
        bright = img > thresh

        rows_list, cents_list = [], []
        for row in range(64):
            pix = bright[row]
            if pix.any():
                rows_list.append(float(row))
                cents_list.append((col_idx * pix).sum() / pix.sum())

        if len(rows_list) >= 5:
            ra = np.array(rows_list)
            ca = np.array(cents_list)
            A  = np.column_stack([ra, np.ones(len(ra))])
            slope, intercept = np.linalg.lstsq(A, ca, rcond=None)[0]
            line_at_middle   = float(slope * 32 + intercept)
        else:
            slope, line_at_middle = 0.0, 31.5

        X_feat[i] = [slope, line_at_middle, speed]

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{N} records processed")

    print(f"  Feature extraction complete.  X_feat shape: {X_feat.shape}")
    return X_feat

def build_targets(records):
    n      = len(records)
    y_cmd  = np.empty((n, 3), dtype=np.float32)
    y_cont = np.empty((n, 2), dtype=np.float32)
    y_mot  = np.empty((n, 4), dtype=np.float32)
    for i, r in enumerate(records):
        cmds      = r['binary_commands']
        y_cmd[i]  = [cmds[0], cmds[1], cmds[2]]
        y_cont[i] = r['continuous_commands']
        y_mot[i]  = r['motor_velocities']
    return y_cmd, y_cont, y_mot

# ── Model ─────────────────────────────────────────────────────────────────────
class FeatureMLP(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, HIDDEN_DIMS[0]), nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]), nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[1], output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)

def make_model(output_dim):
    return FeatureMLP(output_dim).to(DEVICE)

def clone_model(m):
    c = FeatureMLP(m.output_dim).to(DEVICE)
    c.load_state_dict(copy.deepcopy(m.state_dict()))
    return c

def param_count(output_dim):
    sizes = [3] + HIDDEN_DIMS + [output_dim]
    return sum(sizes[i] * sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))

# ── GA operations ─────────────────────────────────────────────────────────────
def evaluate(model, X, y):
    model.eval()
    loss_fn = nn.MSELoss()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.size(0), BATCH_SIZE):
            xb, yb = X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            total += loss_fn(model(xb), yb).item() * xb.size(0)
            count += xb.size(0)
    return -(total / count)   # negative MSE — higher is better

def mutate(model):
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand_like(p) < MUTATION_RATE
            p[mask] += (torch.randn_like(p) * MUTATION_STRENGTH)[mask]

def crossover(p1, p2):
    child = clone_model(p1)
    with torch.no_grad():
        for pc, pp2 in zip(child.parameters(), p2.parameters()):
            mask = torch.rand_like(pc) < CROSSOVER_RATE
            pc.data[mask] = pp2.data[mask].clone()
    return child

def fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

# ── GA loop ───────────────────────────────────────────────────────────────────
def run_ga(exp_name, output_dim, X, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    analytics_path = os.path.join(out_dir, 'analytics.csv')
    ckpt_path      = os.path.join(out_dir, 'checkpoint.pt')

    start_gen  = 0
    prev_best  = None
    population = [(make_model(output_dim), None) for _ in range(POPULATION_SIZE)]

    if os.path.exists(ckpt_path):
        ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        start_gen = ckpt['generation'] + 1
        prev_best = ckpt['prev_best']
        for j, (m, _) in enumerate(population):
            m.load_state_dict(ckpt['states'][j]['weights'])
        population = [(m, ckpt['states'][j]['fitness'])
                      for j, (m, _) in enumerate(population)]
        print(f"  Resumed from checkpoint at gen {start_gen - 1}")

    write_header = not os.path.exists(analytics_path) or start_gen == 0
    run_start    = time.time()
    gen_times    = []

    for gen in range(start_gen, GENERATIONS):
        t0 = time.time()

        # Evaluate any unevaluated individuals
        population = [(m, evaluate(m, X, y) if fit is None else fit)
                      for m, fit in population]

        population.sort(key=lambda x: x[1], reverse=True)
        fitnesses = [f for _, f in population]
        best  = fitnesses[0]
        avg   = sum(fitnesses) / len(fitnesses)
        worst = fitnesses[-1]
        delta = best - prev_best if prev_best is not None else 0.0
        prev_best = best

        gen_t = time.time() - t0
        gen_times.append(gen_t)
        if len(gen_times) > 20:
            gen_times.pop(0)
        eta = (sum(gen_times) / len(gen_times)) * (GENERATIONS - gen - 1)

        print(f"  [{exp_name}]  Gen {gen:>4}  "
              f"best={best:.6f}  avg={avg:.6f}  worst={worst:.6f}  "
              f"Δ={delta:+.6f}  [{fmt_time(int(time.time()-run_start))} elapsed  ETA {fmt_time(eta)}]")

        with open(analytics_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['generation', 'best', 'avg', 'worst', 'delta',
                            'elapsed_s', 'eta_s'])
                write_header = False
            w.writerow([gen, best, avg, worst, delta,
                        round(time.time() - run_start, 1), round(eta, 1)])

        if gen % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'generation': gen,
                'prev_best':  prev_best,
                'states': [{'weights': m.state_dict(), 'fitness': fit}
                           for m, fit in population],
            }, ckpt_path)

        elites   = population[:N_ELITES]
        offspring = []
        for _ in range(POPULATION_SIZE - N_ELITES):
            p1 = random.choice(elites)[0]
            p2 = random.choice(elites)[0]
            child = crossover(p1, p2)
            mutate(child)
            offspring.append((child, None))
        population = elites + offspring

    best_model, best_fit = population[0]
    best_path = os.path.join(out_dir, 'best_model.pth')
    torch.save({'weights':     best_model.state_dict(),
                'hidden_dims': HIDDEN_DIMS,
                'output_dim':  output_dim,
                'fitness':     best_fit}, best_path)
    print(f"\n  ✓ Best model saved → {best_path}  (fitness={best_fit:.6f})")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return best_fit

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(ALL_DATA_FILE):
        print(f"ERROR: {ALL_DATA_FILE} not found.")
        print("Upload all_data.json to Google Drive at the path set in DATA_DIR.")
        return

    print(f"\nLoading all_data.json (20,000 records — same dataset as main notrain experiments) ...")
    records = load_records(ALL_DATA_FILE)
    print(f"  Loaded {len(records):,} records.")

    X_feat         = extract_features(records)
    y_cmd, y_cont, y_mot = build_targets(records)

    # Move to device as tensors
    X  = torch.tensor(X_feat, device=DEVICE)
    ys = {
        'ycommand':    torch.tensor(y_cmd,  device=DEVICE),
        'ycontinuous': torch.tensor(y_cont, device=DEVICE),
        'ymotors':     torch.tensor(y_mot,  device=DEVICE),
    }

    print(f"\nArchitecture  : 3 → {HIDDEN_DIMS[0]} → {HIDDEN_DIMS[1]} → output_dim")
    print(f"GA config     : pop={POPULATION_SIZE}, elites={N_ELITES}, "
          f"gens={GENERATIONS}, eval_records={len(records):,}")

    results = []
    for out_name, out_dim in OUTPUT_TYPES:
        n_params = param_count(out_dim)
        exp_name = f"features_{out_name}"
        out_dir  = os.path.join(OUTPUT_DIR, exp_name)
        y        = ys[out_name]

        print(f"\n{'='*70}")
        print(f"  Experiment : {exp_name}")
        print(f"  Params     : {n_params:,}  "
              f"(3 → {HIDDEN_DIMS[0]} → {HIDDEN_DIMS[1]} → {out_dim})")
        print(f"  Pop={POPULATION_SIZE}  Elites={N_ELITES}  Gens={GENERATIONS}  "
              f"EvalRecords={len(records):,}")
        print(f"{'='*70}")

        best_fit = run_ga(exp_name, out_dim, X, y, out_dir)
        results.append((exp_name, n_params, best_fit))

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT C — FINAL SUMMARY  (Feature inputs, untrained NE)")
    print(f"  Architecture: 3 → {HIDDEN_DIMS[0]} → {HIDDEN_DIMS[1]} → output_dim")
    print(f"  ({GENERATIONS} gens, pop={POPULATION_SIZE}, eval on {len(records):,} records)")
    print(f"{'='*70}")
    print(f"  {'Experiment':<28} {'Params':>8}  {'Best Fitness':>14}")
    print(f"  {'-'*28} {'-'*8}  {'-'*14}")
    for exp_name, params, fit in results:
        print(f"  {exp_name:<28} {params:>8,}  {fit:>14.6f}")

if __name__ == '__main__':
    main()
