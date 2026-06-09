#!/usr/bin/env python3
"""
Supplementary Experiment B — NEAT with baseline vision features.

Uses NeuroEvolution of Augmenting Topologies (neat-python). NEAT starts with
minimal networks (0 hidden nodes, direct input→output connections) and grows
complexity only when evolution finds it beneficial.

Input: 3 low-dimensional features extracted by the baseline vision algorithm
  [slope, line_at_middle, speed_scalar]
This keeps the initial genome tractable (3 × output_dim connections) and lets
topology search do meaningful work — using raw 4097-pixel inputs with NEAT
would make the search space as large as the standard CNN experiments.

Three output types tested sequentially: binary commands (3), continuous
values (2), motor velocities (4).

Fitness: negative MSE (higher = better), evaluated on all 20,000 records from
all_data.json — the same dataset used by the main notrain experiments.

--- Google Colab setup ---
1. Upload all_data.json to Google Drive at:
       My Drive/WesDroneRL2/receivers/all_data.json
2. In Colab, run:  !pip install neat-python
   (or let the script install it automatically)
3. Results saved to:  My Drive/WesDroneRL2/Neuroevolution2/exp_b_neat/
"""

import sys, os

# ── Install neat-python if missing ───────────────────────────────────────────
try:
    import neat
except ImportError:
    import subprocess
    print("Installing neat-python ...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'neat-python'],
                   check=True)
    import neat

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
    OUTPUT_DIR = '/content/drive/MyDrive/WesDroneRL2/Neuroevolution2/exp_b_neat'
else:
    _HERE      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(_HERE, '..', 'Data', 'datasets')
    OUTPUT_DIR = os.path.join(_HERE, 'exp_b_neat')

ALL_DATA_FILE = os.path.join(DATA_DIR, 'all_data.json')

# NEAT hyperparameters
POPULATION_SIZE     = 50
GENERATIONS         = 5000
CHECKPOINT_INTERVAL = 25     # matches main notrain checkpoint interval

# Output types: (name, output_dim)
OUTPUT_TYPES = [
    ('ycommand',    3),
    ('ycontinuous', 2),
    ('ymotors',     4),
]

# ── Imports ───────────────────────────────────────────────────────────────────
import json, csv, random, time, pickle
import numpy as np

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

    slope          : gradient of the track line across the image (rise/run)
    line_at_middle : horizontal pixel position of the track at row 32
    speed_scalar   : drone speed setting (already in the record)

    Algorithm mirrors the baseline controller:
      1. Adaptive threshold = mean + 0.5 * std
      2. Binary mask of bright pixels
      3. Row-wise horizontal centroid of bright pixels
      4. Least-squares line fit: centroid = slope * row + intercept
      5. line_at_middle = slope * 32 + intercept
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

# ── Vectorised NEAT genome evaluation ────────────────────────────────────────
def eval_genome_vectorized(genome, config, X_feat, y_target):
    """
    Evaluate a NEAT genome against all samples in X_feat using numpy.

    Converts the neat-python FeedForwardNetwork node_evals into a sequence
    of vectorised numpy operations so the entire batch runs in one pass
    rather than looping over individual samples.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    n_samples = X_feat.shape[0]
    # Map input node keys → columns of X_feat
    node_vals = {k: X_feat[:, i] for i, k in enumerate(net.input_nodes)}

    for node_key, act_func, _agg_func, bias, response, links in net.node_evals:
        if links:
            aggregated = sum(
                (node_vals[in_key] if in_key in node_vals else np.zeros(n_samples)) * weight
                for in_key, weight in links
            )
        else:
            aggregated = np.zeros(n_samples)

        z   = bias + response * aggregated
        name = getattr(act_func, '__name__', '')
        if 'relu' in name:
            node_vals[node_key] = np.maximum(0.0, z)
        elif 'sigmoid' in name:
            node_vals[node_key] = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        elif 'tanh' in name:
            node_vals[node_key] = np.tanh(z)
        else:
            # identity / linear / unknown — pass through
            node_vals[node_key] = z

    # Collect outputs in the order neat-python expects
    outputs = np.column_stack([
        node_vals.get(k, np.zeros(n_samples)) for k in net.output_nodes
    ])   # (n_samples, output_dim)

    mse = float(np.mean((outputs - y_target) ** 2))
    return -mse   # higher = better

# ── NEAT config template ──────────────────────────────────────────────────────
NEAT_CONFIG_TEMPLATE = """
[NEAT]
fitness_criterion        = max
fitness_threshold        = 0.0
no_fitness_termination   = True
pop_size                 = {pop_size}
reset_on_extinction      = False

[DefaultGenome]
activation_default      = identity
activation_mutate_rate  = 0.1
activation_options      = identity relu tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.3
conn_delete_prob        = 0.3

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

node_add_prob           = 0.2
node_delete_prob        = 0.2

num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func    = max
max_stagnation          = 20
species_elitism         = 2

[DefaultReproduction]
elitism                 = 2
survival_threshold      = 0.24
"""

def fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

# ── NEAT run ──────────────────────────────────────────────────────────────────
def run_neat(exp_name, output_dim, X_feat, y_target, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    analytics_path = os.path.join(out_dir, 'analytics.csv')

    # Write NEAT config to a temp file (neat-python requires a file path)
    config_str = NEAT_CONFIG_TEMPLATE.format(
        pop_size=POPULATION_SIZE,
        num_inputs=3,
        num_outputs=output_dim,
    )
    config_path = os.path.join(out_dir, 'neat_config.ini')
    with open(config_path, 'w') as f:
        f.write(config_str)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Resume from NEAT checkpoint if available
    ckpt_prefix = os.path.join(out_dir, 'neat-checkpoint-')
    existing_ckpts = [f for f in os.listdir(out_dir)
                      if f.startswith('neat-checkpoint-')]
    if existing_ckpts:
        latest = max(existing_ckpts, key=lambda x: int(x.split('-')[-1]))
        print(f"  Resuming from {latest}")
        pop = neat.Checkpointer.restore_checkpoint(os.path.join(out_dir, latest))
        start_gen = int(latest.split('-')[-1]) + 1
    else:
        pop = neat.Population(config)
        start_gen = 0

    # Reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        generation_interval=CHECKPOINT_INTERVAL,
        filename_prefix=ckpt_prefix,
    ))

    # Fitness function closed over the data
    write_header = not os.path.exists(analytics_path) or start_gen == 0
    run_start    = time.time()
    gen_counter  = [start_gen]
    best_genome_holder = [None]

    def eval_genomes(genomes, cfg):
        gen = gen_counter[0]
        gen_t0 = time.time()
        best_fit = -float('inf')
        for gid, genome in genomes:
            genome.fitness = eval_genome_vectorized(genome, cfg, X_feat, y_target)
            if genome.fitness > best_fit:
                best_fit = genome.fitness
                best_genome_holder[0] = genome

        all_fits = [g.fitness for _, g in genomes if g.fitness is not None]
        avg   = sum(all_fits) / len(all_fits)
        worst = min(all_fits)

        gen_elapsed = time.time() - gen_t0
        elapsed     = time.time() - run_start
        gens_left   = GENERATIONS - gen - 1
        eta         = gen_elapsed * gens_left

        print(f"  [{exp_name}]  Gen {gen:>4}  "
              f"best={best_fit:.6f}  avg={avg:.6f}  worst={worst:.6f}  "
              f"[{fmt_time(int(elapsed))} elapsed  ETA {fmt_time(eta)}]")

        with open(analytics_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header and gen == start_gen:
                w.writerow(['generation', 'best', 'avg', 'worst',
                            'n_species', 'elapsed_s', 'eta_s'])
            n_species = len(pop.species.species) if hasattr(pop, 'species') else 0
            w.writerow([gen, best_fit, avg, worst, n_species,
                        round(elapsed, 1), round(eta, 1)])

        gen_counter[0] += 1

    # Run NEAT
    gens_to_run = GENERATIONS - start_gen
    winner = pop.run(eval_genomes, gens_to_run)

    # Save best genome
    best_path = os.path.join(out_dir, 'best_genome.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump(winner, f)
    best_fit = winner.fitness if winner.fitness is not None else best_genome_holder[0].fitness
    print(f"\n  ✓ Best genome saved → {best_path}  (fitness={best_fit:.6f})")

    # Save best genome's network structure summary
    summary_path = os.path.join(out_dir, 'best_genome_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Experiment : {exp_name}\n")
        f.write(f"Fitness    : {best_fit:.6f}\n")
        f.write(f"Nodes      : {len(winner.nodes)}\n")
        f.write(f"Connections: {len(winner.connections)}\n")
        f.write(f"Enabled    : {sum(1 for c in winner.connections.values() if c.enabled)}\n")

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

    y_map = {'ycommand': y_cmd, 'ycontinuous': y_cont, 'ymotors': y_mot}

    print(f"\nNEAT config: pop={POPULATION_SIZE}, gens={GENERATIONS}, "
          f"inputs=3 (features), eval_records={len(records):,}")

    results = []
    for out_name, out_dim in OUTPUT_TYPES:
        exp_name = f"neat_{out_name}"
        out_dir  = os.path.join(OUTPUT_DIR, exp_name)
        y        = y_map[out_name]

        print(f"\n{'='*70}")
        print(f"  Experiment : {exp_name}  (output_dim={out_dim})")
        print(f"  Inputs     : 3 baseline vision features")
        print(f"  Pop={POPULATION_SIZE}  Gens={GENERATIONS}  EvalRecords={len(records):,}")
        print(f"{'='*70}")

        best_fit = run_neat(exp_name, out_dim, X_feat, y, out_dir)
        results.append((exp_name, out_dim, best_fit))

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT B — FINAL SUMMARY  (NEAT, feature inputs)")
    print(f"  ({GENERATIONS} gens, pop={POPULATION_SIZE}, eval on {len(records):,} records)")
    print(f"{'='*70}")
    print(f"  {'Experiment':<25} {'Out Dim':>7}  {'Best Fitness':>14}")
    print(f"  {'-'*25} {'-'*7}  {'-'*14}")
    for exp_name, out_dim, fit in results:
        print(f"  {exp_name:<25} {out_dim:>7}  {fit:>14.6f}")

if __name__ == '__main__':
    main()
