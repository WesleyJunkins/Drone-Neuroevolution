#!/usr/bin/env python3
"""
Experiment F — Extended Untrained NE (larger population, more generations)

Runs the same untrained NE GA as the main notrain experiments (NE_Framework/),
scaled up to test whether pure neuroevolution closes the gap with trained models
when given substantially more compute budget.

Comparison to main notrain experiments:
  Main notrain  — pop=50,  elites=12, gens=5,000  → ~190,000 evaluations
  This script   — pop=200, elites=50, gens=10,000 → ~1,500,000 evaluations  (~8×)

Three output types run sequentially:
  ycommand    — binary commands [forward, yaw_increase, yaw_decrease]  (output_dim=3)
  ycontinuous — continuous values [forward_desired, yaw_desired]        (output_dim=2)
  ymotors     — motor velocities [m1, m2, m3, m4]                       (output_dim=4)

Data: all_data.json (20,000 records — same as main notrain experiments).

GA parameters unchanged from main notrain: mutation_rate=0.05, mutation_strength=0.05,
crossover_rate=0.5. Only population size and generation count are scaled up.

Checkpointing: every 50 gens, saves elite models only (~220MB vs ~880MB for full pop).
Resume: on re-run, reloads elites and breeds fresh offspring — seamless across Colab
sessions. Analytics CSV is opened in append mode, so progress accumulates correctly.

Output format:
  best_model.pth — plain state_dict, compatible with NE_receiver.py MODEL_FAMILY='main_cnn'
  analytics.csv  — generation, best, avg, worst, delta, diversity, elapsed_s, eta_s
  checkpoint.pt  — elite weights for resume (deleted after run completes)

--- Google Colab setup ---
1. Upload all_data.json to Google Drive at:
       My Drive/WesDroneRL2/receivers/all_data.json
2. Run this script in Colab. Re-run the same cell to resume from the latest checkpoint.
3. Results are saved to:
       My Drive/WesDroneRL2/Neuroevolution2/exp_f_extended_notrain/{output_type}/

--- Runtime estimate ---
Colab A100: ~8-15s per generation for pop=200 → 10,000 gens ≈ 22-42 hours total per
output type. Requires multiple Colab sessions; use checkpointing to resume.
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
    OUTPUT_DIR = '/content/drive/MyDrive/WesDroneRL2/Neuroevolution2/exp_f_extended_notrain'
else:
    _HERE      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(_HERE, '..', 'Data', 'datasets')
    OUTPUT_DIR = os.path.join(_HERE, 'exp_f_extended_notrain')

ALL_DATA_FILE = os.path.join(DATA_DIR, 'all_data.json')

# ── GA hyperparameters ────────────────────────────────────────────────────────
POPULATION_SIZE   = 200
N_ELITES          = 50      # ~25%, consistent with existing experiments
GENERATIONS       = 10_000
CROSSOVER_RATE    = 0.5
MUTATION_RATE     = 0.05
MUTATION_STRENGTH = 0.05
BATCH_SIZE            = 512
CHECKPOINT_INTERVAL   = 50    # checkpoint every 50 gens (saves elites only, ~220MB)
CONVERGENCE_WINDOW    = 25    # match existing NE experiments exactly
CONVERGENCE_THRESHOLD = 0.001 # match existing NE experiments exactly

# Output types: (name, output_dim)
# ymotors is excluded: all existing ymotors models crash in simulation because motor
# velocities were recorded as PID outputs with IMU feedback unavailable at inference time.
# More training does not fix this architectural mismatch.
OUTPUT_TYPES = [
    ('ycommand',    3),
    ('ycontinuous', 2),
]

# ── Imports ───────────────────────────────────────────────────────────────────
import json, csv, copy, random, time
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


def make_model(output_dim):
    return CreateCNNModel(output_dim).to(DEVICE)


def clone_model(m):
    c = CreateCNNModel(m.output_features).to(DEVICE)
    c.load_state_dict(copy.deepcopy(m.state_dict()))
    return c


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path):
    print(f"  Reading {os.path.basename(path)}...", end='', flush=True)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
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
    print(f" {n:,} records loaded.")
    return (torch.tensor(X,      device=DEVICE),
            torch.tensor(y_cmd,  device=DEVICE),
            torch.tensor(y_cont, device=DEVICE),
            torch.tensor(y_mot,  device=DEVICE))


# ── GA operations ─────────────────────────────────────────────────────────────
def evaluate(model, X, y):
    """Returns negative MSE (higher is better)."""
    model.eval()
    loss_fn = nn.MSELoss()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.size(0), BATCH_SIZE):
            xb, yb = X[i:i + BATCH_SIZE], y[i:i + BATCH_SIZE]
            total += loss_fn(model(xb), yb).item() * xb.size(0)
            count += xb.size(0)
    return -(total / count)


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


def breed_offspring(elites, count):
    offspring = []
    for _ in range(count):
        p1    = random.choice(elites)[0]
        p2    = random.choice(elites)[0]
        child = crossover(p1, p2)
        mutate(child)
        offspring.append((child, None))
    return offspring


def diversity(elites):
    """Mean pairwise L2 distance between elite weight vectors (first 10 elites), normalised."""
    sample = elites[:10]
    if len(sample) < 2:
        return 0.0
    params = [torch.cat([p.data.view(-1) for p in m.parameters()]).cpu()
              for m, _ in sample]
    dists = []
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            dists.append((params[i] - params[j]).norm().item())
    mean_norm = sum(p.norm().item() for p in params) / len(params)
    return (sum(dists) / len(dists)) / mean_norm if mean_norm > 0 else 0.0


def fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ── GA loop ───────────────────────────────────────────────────────────────────
def run_ga(out_name, output_dim, X, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    analytics_path = os.path.join(out_dir, 'analytics.csv')
    ckpt_path      = os.path.join(out_dir, 'checkpoint.pt')
    best_path      = os.path.join(out_dir, 'best_model.pth')

    start_gen  = 0
    prev_best  = None
    population = [(make_model(output_dim), None) for _ in range(POPULATION_SIZE)]

    if os.path.exists(ckpt_path):
        print(f"  Checkpoint found — resuming from checkpoint...")
        ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        start_gen = ckpt['generation'] + 1
        prev_best = ckpt['prev_best']
        # Reload elites from checkpoint, breed fresh offspring for resumed generation
        elites = [(make_model(output_dim), None) for _ in range(N_ELITES)]
        for i, (m, _) in enumerate(elites):
            m.load_state_dict(ckpt['elites'][i]['weights'])
        elites_with_fit = [(m, ckpt['elites'][i]['fitness'])
                           for i, (m, _) in enumerate(elites)]
        offspring  = breed_offspring(elites_with_fit, POPULATION_SIZE - N_ELITES)
        population = elites_with_fit + offspring
        print(f"  Resumed from gen {start_gen - 1}  (prev_best={prev_best:.6f})")

    write_header = not os.path.exists(analytics_path) or start_gen == 0
    run_start    = time.time()
    gen_times    = []
    best_history = []  # rolling window for convergence check

    for gen in range(start_gen, GENERATIONS):
        t0 = time.time()

        # Evaluate unevaluated individuals (elites already have fitness; new offspring do not)
        population = [(m, evaluate(m, X, y) if fit is None else fit)
                      for m, fit in population]

        population.sort(key=lambda x: x[1], reverse=True)
        fitnesses = [f for _, f in population]
        best  = fitnesses[0]
        avg   = sum(fitnesses) / len(fitnesses)
        worst = fitnesses[-1]
        delta = best - prev_best if prev_best is not None else 0.0
        prev_best = best
        div   = diversity(population[:N_ELITES])
        best_history.append(best)

        # Convergence check: improvement over last CONVERGENCE_WINDOW gens < threshold
        # Identical criterion to all existing NE experiments (window=25, threshold=0.001)
        converged = False
        if len(best_history) > CONVERGENCE_WINDOW:
            window_improvement = best_history[-1] - best_history[-(CONVERGENCE_WINDOW + 1)]
            if window_improvement < CONVERGENCE_THRESHOLD:
                converged = True

        gen_t = time.time() - t0
        gen_times.append(gen_t)
        if len(gen_times) > 20:
            gen_times.pop(0)
        eta = (sum(gen_times) / len(gen_times)) * (GENERATIONS - gen - 1)

        print(f"  [{out_name}]  Gen {gen:>5}  "
              f"best={best:.6f}  avg={avg:.6f}  worst={worst:.6f}  "
              f"Δ={delta:+.6f}  div={div:.4f}  "
              f"[{fmt_time(int(time.time() - run_start))} elapsed  ETA {fmt_time(eta)}]"
              + ("  ◆ converged" if converged else ""))

        with open(analytics_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['generation', 'best', 'avg', 'worst', 'delta',
                            'diversity', 'elapsed_s', 'eta_s', 'converged'])
                write_header = False
            w.writerow([gen, best, avg, worst, delta, round(div, 6),
                        round(time.time() - run_start, 1), round(eta, 1),
                        1 if converged else 0])

        if converged:
            print(f"\n  ◆ Converged at generation {gen}  "
                  f"(improvement < {CONVERGENCE_THRESHOLD} "
                  f"over last {CONVERGENCE_WINDOW} gens)")
            break

        if gen % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'generation': gen,
                'prev_best':  prev_best,
                'elites': [{'weights': m.state_dict(), 'fitness': fit}
                           for m, fit in population[:N_ELITES]],
            }, ckpt_path)

        elites    = population[:N_ELITES]
        offspring = breed_offspring(elites, POPULATION_SIZE - N_ELITES)
        population = elites + offspring

    best_model, best_fit = population[0]
    # Save as plain state_dict — compatible with NE_receiver.py MODEL_FAMILY='main_cnn'
    torch.save(best_model.state_dict(), best_path)
    converged_gen = gen  # gen is the loop variable at the point of break or end
    print(f"\n  Best model saved -> {best_path}  (fitness={best_fit:.6f})")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return best_fit, converged_gen


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(ALL_DATA_FILE):
        print(f"ERROR: {ALL_DATA_FILE} not found.")
        print("  Upload all_data.json to Google Drive at the path configured in DATA_DIR.")
        return

    print(f"\nLoading all_data.json...")
    X, y_cmd, y_cont, y_mot = load_data(ALL_DATA_FILE)
    print(f"  X={X.shape}  y_cmd={y_cmd.shape}  "
          f"y_cont={y_cont.shape}  y_mot={y_mot.shape}")

    probe = make_model(2)
    print(f"\nCNN parameters: {count_params(probe):,}")
    del probe

    n_offspring = POPULATION_SIZE - N_ELITES
    total_evals = n_offspring * GENERATIONS
    print(f"\nExtended NE configuration:")
    print(f"  Population = {POPULATION_SIZE}  |  Elites = {N_ELITES}  |  Generations = {GENERATIONS:,}")
    print(f"  Offspring per generation = {n_offspring}  |  Total evaluations ≈ {total_evals:,}")
    print(f"  Mutation rate/strength = {MUTATION_RATE}/{MUTATION_STRENGTH}  "
          f"|  Crossover rate = {CROSSOVER_RATE}")
    print(f"  Checkpoint every {CHECKPOINT_INTERVAL} gens (elites only, ~{N_ELITES * 4:.0f}MB)")

    y_map = {'ycommand': y_cmd, 'ycontinuous': y_cont, 'ymotors': y_mot}

    results = []
    for out_name, out_dim in OUTPUT_TYPES:
        out_dir = os.path.join(OUTPUT_DIR, out_name)
        print(f"\n{'='*70}")
        print(f"  Experiment F: {out_name}  (output_dim={out_dim})")
        print(f"  Pop={POPULATION_SIZE}  Elites={N_ELITES}  Gens={GENERATIONS:,}  "
              f"Records={X.shape[0]:,}")
        print(f"  Output -> {out_dir}")
        print(f"{'='*70}")
        best_fit, converged_gen = run_ga(out_name, out_dim, X, y_map[out_name], out_dir)
        results.append((out_name, out_dim, best_fit, converged_gen))

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT F — FINAL SUMMARY")
    print(f"  Extended Untrained NE  "
          f"(pop={POPULATION_SIZE}, elites={N_ELITES}, gens={GENERATIONS:,})")
    print(f"{'='*70}")
    print(f"  {'Output Type':<16} {'Dim':>3}  {'Best Fitness':>14}  {'Converged Gen':>13}")
    print(f"  {'-'*16} {'-'*3}  {'-'*14}  {'-'*13}")
    for out_name, out_dim, best_fit, converged_gen in results:
        print(f"  {out_name:<16} {out_dim:>3}  {best_fit:>14.6f}  {converged_gen:>13}")
    print(f"\n  Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
