#!/usr/bin/env python3
"""
Supplementary Experiment A — Untrained NE with smaller MLP architectures.

Hypothesis: pure evolution may perform better on networks with fewer parameters,
where the high-dimensional search space is more tractable.

Three MLP sizes are tested against three output types (binary commands,
continuous values, motor velocities) = 9 experiments, run sequentially.

Input: 4097 features (4096 normalised pixel values + speed_scalar)
Output: 3 (binary commands), 2 (continuous values), or 4 (motor velocities)

--- Google Colab setup ---
1. Upload all_data.json to Google Drive at:
       My Drive/WesDroneRL2/receivers/all_data.json
2. Open this file in Colab (File > Upload notebook, then paste this script
   into a code cell, or upload as .py and run with: !python3 exp_a_small_mlp.py)
3. Results are saved back to Google Drive under:
       My Drive/WesDroneRL2/Neuroevolution2/exp_a_small_mlp/
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
    OUTPUT_DIR = '/content/drive/MyDrive/WesDroneRL2/Neuroevolution2/exp_a_small_mlp'
else:
    _HERE      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(_HERE, '..', 'Data', 'datasets')
    OUTPUT_DIR = os.path.join(_HERE, 'exp_a_small_mlp')

ALL_DATA_FILE = os.path.join(DATA_DIR, 'all_data.json')

# GA hyperparameters
POPULATION_SIZE   = 50
N_ELITES          = 12      # ~24% of 50, consistent with main notrain experiments
GENERATIONS       = 5000    # matches main notrain experiment budget for fair comparison
CROSSOVER_RATE    = 0.5
MUTATION_RATE     = 0.05
MUTATION_STRENGTH = 0.05
BATCH_SIZE          = 512
CHECKPOINT_INTERVAL = 25    # matches main notrain checkpoint interval

# MLP architectures to compare: (name, hidden_layer_sizes)
# All share the same 4097-dimensional input (pixels + speed_scalar)
ARCHITECTURES = [
    ('mlp_256_128', [256, 128]),
    ('mlp_64_32',   [64,  32]),
    ('mlp_16_8',    [16,   8]),
]

# Output types: (name, output_dim)
OUTPUT_TYPES = [
    ('ycommand',    3),   # binary [forward, yaw_increase, yaw_decrease]
    ('ycontinuous', 2),   # continuous [forward_desired, yaw_desired]
    ('ymotors',     4),   # motor velocities [m1, m2, m3, m4]
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
def load_data(path, subsample=None):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if subsample and subsample < len(records):
        records = random.sample(records, subsample)

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

# ── Model ─────────────────────────────────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net        = nn.Sequential(*layers)
        self.input_dim  = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)

def make_model(hidden_dims, output_dim):
    return SmallMLP(4097, hidden_dims, output_dim).to(DEVICE)

def clone_model(m):
    c = SmallMLP(m.input_dim, m.hidden_dims, m.output_dim).to(DEVICE)
    c.load_state_dict(copy.deepcopy(m.state_dict()))
    return c

def param_count(hidden_dims, output_dim):
    sizes = [4097] + hidden_dims + [output_dim]
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
def run_ga(exp_name, hidden_dims, output_dim, X, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    analytics_path = os.path.join(out_dir, 'analytics.csv')
    ckpt_path      = os.path.join(out_dir, 'checkpoint.pt')

    # Initialise or resume population: list of (model, fitness_or_None)
    start_gen  = 0
    prev_best  = None
    population = [(make_model(hidden_dims, output_dim), None)
                  for _ in range(POPULATION_SIZE)]

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

        # Evaluate any unevaluated individuals (new offspring have fitness=None)
        population = [(m, evaluate(m, X, y) if fit is None else fit)
                      for m, fit in population]

        # Sort descending (higher fitness = better)
        population.sort(key=lambda x: x[1], reverse=True)
        fitnesses = [f for _, f in population]
        best      = fitnesses[0]
        avg       = sum(fitnesses) / len(fitnesses)
        worst     = fitnesses[-1]
        delta     = best - prev_best if prev_best is not None else 0.0
        prev_best = best

        gen_t = time.time() - t0
        gen_times.append(gen_t)
        if len(gen_times) > 20:
            gen_times.pop(0)
        eta = (sum(gen_times) / len(gen_times)) * (GENERATIONS - gen - 1)

        print(f"  [{exp_name}]  Gen {gen:>4}  "
              f"best={best:.6f}  avg={avg:.6f}  worst={worst:.6f}  "
              f"Δ={delta:+.6f}  [{fmt_time(int(time.time()-run_start))} elapsed  ETA {fmt_time(eta)}]")

        # Analytics CSV
        with open(analytics_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['generation', 'best', 'avg', 'worst', 'delta',
                            'elapsed_s', 'eta_s'])
                write_header = False
            w.writerow([gen, best, avg, worst, delta,
                        round(time.time() - run_start, 1), round(eta, 1)])

        # Checkpoint
        if gen % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'generation': gen,
                'prev_best':  prev_best,
                'states': [{'weights': m.state_dict(), 'fitness': fit}
                           for m, fit in population],
            }, ckpt_path)

        # Evolve: keep elites, breed offspring
        elites   = population[:N_ELITES]
        offspring = []
        for _ in range(POPULATION_SIZE - N_ELITES):
            p1 = random.choice(elites)[0]
            p2 = random.choice(elites)[0]
            child = crossover(p1, p2)
            mutate(child)
            offspring.append((child, None))
        population = elites + offspring

    # Save best model
    best_model, best_fit = population[0]
    best_path = os.path.join(out_dir, 'best_model.pth')
    torch.save({'weights':     best_model.state_dict(),
                'hidden_dims': hidden_dims,
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
    X, y_cmd, y_cont, y_mot = load_data(ALL_DATA_FILE)
    print(f"  X={X.shape}  y_cmd={y_cmd.shape}  "
          f"y_cont={y_cont.shape}  y_mot={y_mot.shape}")

    y_map = {'ycommand': y_cmd, 'ycontinuous': y_cont, 'ymotors': y_mot}

    results = []
    for arch_name, hidden_dims in ARCHITECTURES:
        for out_name, out_dim in OUTPUT_TYPES:
            n_params  = param_count(hidden_dims, out_dim)
            exp_name  = f"{arch_name}_{out_name}"
            out_dir   = os.path.join(OUTPUT_DIR, exp_name)
            y         = y_map[out_name]

            print(f"\n{'='*70}")
            print(f"  Experiment : {exp_name}")
            print(f"  Params     : {n_params:,}")
            print(f"  Pop={POPULATION_SIZE}  Elites={N_ELITES}  Gens={GENERATIONS}  "
                  f"EvalRecords={X.shape[0]:,}")
            print(f"{'='*70}")

            best_fit = run_ga(exp_name, hidden_dims, out_dim, X, y, out_dir)
            results.append((exp_name, n_params, best_fit))

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT A — FINAL SUMMARY")
    print(f"  ({GENERATIONS} gens, pop={POPULATION_SIZE}, eval on {X.shape[0]:,} records)")
    print(f"{'='*70}")
    print(f"  {'Experiment':<32} {'Params':>8}  {'Best Fitness':>14}")
    print(f"  {'-'*32} {'-'*8}  {'-'*14}")
    for exp_name, params, fit in results:
        print(f"  {exp_name:<32} {params:>8,}  {fit:>14.6f}")

if __name__ == '__main__':
    main()
