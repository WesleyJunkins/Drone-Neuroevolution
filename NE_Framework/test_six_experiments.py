"""
test_six_experiments.py — Functional test + runtime estimator for all 6 NE experiments.

Runs a small number of timed generations for each experiment, verifies correctness,
then extrapolates to estimate the wall-clock time for the full production run.

Run from NE_Framework/ directory:
    python3 test_six_experiments.py
"""

import os
import sys
import random
import tempfile
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model import Individual, ModelTrainer
from utils import load_json
from run_all import GENERATIONS_TRAIN, GENERATIONS_NOTRAIN, CHECKPOINT_INTERVAL

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'Data', 'datasets', 'training.json')
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SUBSAMPLE     = 2000   # large enough to give realistic per-batch MPS throughput
TEST_GENS     = 5      # more generations = tighter per-gen average
POP_SIZE      = 4      # individuals per generation in the test
N_ELITES      = 2
EPOCHS        = 5      # training epochs per individual (test); production uses 20
LR            = 0.001
BATCH_SIZE    = 512    # match production batch size for accurate timing
MODEL_TYPE    = 'cnn'

# Production settings (for extrapolation)
PROD_POP_SIZE      = 20
PROD_EPOCHS_TRAIN  = 20
PROD_EPOCHS_NOTRAIN = 0  # no backprop

# Cross-machine scaling factor: M4 Max (40-core GPU, ~54.7 TFLOPS) vs
# M2 Ultra (60-core GPU, ~27.2 TFLOPS).  The M4 Max has ~2× the FP32
# throughput of the M2 Ultra for MPS workloads, so times on the M2 Ultra
# are estimated at ~2× the times measured here on the M4 Max.
M2_ULTRA_SCALE = 2.0

# ── Output helpers ────────────────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results   = []
_timings   = {}   # exp_name -> seconds per generation (at production pop size equivalent)


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((name, condition))
    return condition


def fmt_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


# ── Single experiment runner ──────────────────────────────────────────────────

def run_experiment(exp_name, X, y_target, train_mode, output_dim,
                   full_generations):
    """
    Run TEST_GENS timed generations, verify correctness, extrapolate full runtime.
    Returns seconds-per-generation at production scale (POP_SIZE=20).
    """
    gen_times = []

    try:
        individuals = []

        for gen in range(TEST_GENS):
            t_gen_start = time.perf_counter()

            if gen == 0:
                for i in range(POP_SIZE):
                    ind = Individual(model_type=MODEL_TYPE,
                                     output_features=output_dim,
                                     individual_id=f"0_{i}",
                                     device=DEVICE)
                    if train_mode == 'train':
                        trainer = ModelTrainer(ind.model, learning_rate=LR)
                        trainer.train(X, y_target, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE)
                    ind.evaluate_fitness(X, y_target, batch_size=BATCH_SIZE)
                    individuals.append(ind)
            else:
                individuals.sort()
                elites   = individuals[:N_ELITES]
                offspring = []
                for i in range(POP_SIZE - N_ELITES):
                    child = random.choice(elites).crossover(
                        random.choice(elites))
                    child.individual_id = f"{gen}_{i}"
                    child.mutate(mutation_rate=0.05, mutation_strength=0.05)
                    if train_mode == 'train':
                        trainer = ModelTrainer(child.model, learning_rate=LR)
                        trainer.train(X, y_target, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE)
                    child.evaluate_fitness(X, y_target, batch_size=BATCH_SIZE)
                    offspring.append(child)
                individuals = elites + offspring
                for ind in individuals:
                    if ind.fitness is None:
                        ind.evaluate_fitness(X, y_target, batch_size=BATCH_SIZE)

            gen_times.append(time.perf_counter() - t_gen_start)

        individuals.sort()
        best = individuals[0]
        all_fit = [ind.fitness for ind in individuals]

        check(f"{exp_name}: {TEST_GENS} gens run without error",
              True,
              f"best_fitness={best.fitness:.4f}")
        check(f"{exp_name}: all individuals have fitness",
              all(f is not None for f in all_fit))

        # Save / load round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'best')
            best.save(path)
            loaded = Individual(model_type=MODEL_TYPE,
                                output_features=output_dim,
                                device=DEVICE)
            loaded.load(path + '.pth')
            out_orig   = best.test(X[:2])
            out_loaded = loaded.test(X[:2])
            match = torch.allclose(out_orig, out_loaded)
        check(f"{exp_name}: save/load round-trip identical", match)

        # ── Timing extrapolation ──────────────────────────────────────────────
        # Average per-gen time in the test (test uses POP_SIZE=4, EPOCHS=3).
        # Scale up to production (POP_SIZE=20, EPOCHS=20 or 0).
        avg_test_gen = sum(gen_times) / len(gen_times)

        # Per-individual cost scales linearly with epochs; pop size scales linearly.
        # For notrain: epoch ratio = 1 (no training in either case).
        if train_mode == 'train':
            epoch_scale = PROD_EPOCHS_TRAIN / EPOCHS
        else:
            epoch_scale = 1.0
        pop_scale = PROD_POP_SIZE / POP_SIZE

        sec_per_prod_gen = avg_test_gen * pop_scale * epoch_scale
        total_sec        = sec_per_prod_gen * full_generations

        return sec_per_prod_gen, total_sec, gen_times

    except Exception as exc:
        check(f"{exp_name}: {TEST_GENS} gens run without error",   False, str(exc))
        check(f"{exp_name}: all individuals have fitness",         False, "skipped")
        check(f"{exp_name}: save/load round-trip identical",       False, "skipped")
        return None, None, []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: {DATA_FILE} not found.")
        print("  Run Data/prepare_data.py first.")
        sys.exit(1)

    print(f"\nDevice : {DEVICE}")
    print(f"Loading {SUBSAMPLE} records from real training.json ...")
    X, y_command, y_continuous, y_motors = load_json(DATA_FILE,
                                                      subsample=SUBSAMPLE)
    X            = X.to(DEVICE)
    y_command    = y_command.to(DEVICE)
    y_continuous = y_continuous.to(DEVICE)
    y_motors     = y_motors.to(DEVICE)
    print(f"  X={X.shape}  y_cmd={y_command.shape}  "
          f"y_cont={y_continuous.shape}  y_mot={y_motors.shape}")
    print(f"\nTest config  : {TEST_GENS} gens × {POP_SIZE} individuals × "
          f"{EPOCHS} epochs (train) per experiment")
    print(f"Prod config  : trained={GENERATIONS_TRAIN} gens × {PROD_POP_SIZE} "
          f"individuals × {PROD_EPOCHS_TRAIN} epochs")
    print(f"               notrain={GENERATIONS_NOTRAIN} gens × "
          f"{PROD_POP_SIZE} individuals")
    print(f"Checkpoints  : every {CHECKPOINT_INTERVAL} generations → "
          f"{{BASE_DIR}}/checkpoints/checkpoint_gen_N.pth + latest_best.pth")

    experiments = [
        ("notrain_ycommand",    y_command,    'notrain', 3,
         GENERATIONS_NOTRAIN),
        ("notrain_ycontinuous", y_continuous, 'notrain', 2,
         GENERATIONS_NOTRAIN),
        ("notrain_ymotors",     y_motors,     'notrain', 4,
         GENERATIONS_NOTRAIN),
        ("train_ycommand",      y_command,    'train',   3,
         GENERATIONS_TRAIN),
        ("train_ycontinuous",   y_continuous, 'train',   2,
         GENERATIONS_TRAIN),
        ("train_ymotors",       y_motors,     'train',   4,
         GENERATIONS_TRAIN),
    ]

    print(f"\nRunning {len(experiments)} experiments ...\n")
    timing_rows = []

    for exp_name, y_tgt, mode, out_dim, full_gens in experiments:
        print(f"{'─'*60}")
        print(f"  {exp_name}  [mode={mode}  output_dim={out_dim}]")
        print(f"{'─'*60}")
        sec_per_gen, total_sec, gen_times = run_experiment(
            exp_name, X, y_tgt, mode, out_dim, full_gens)
        if sec_per_gen is not None:
            avg_actual  = sum(gen_times) / len(gen_times)
            total_m2    = total_sec * M2_ULTRA_SCALE
            print(f"  Timing : {avg_actual:.2f}s/gen (test)  →  "
                  f"~{sec_per_gen:.1f}s/gen (M4 Max prod)  →  "
                  f"~{fmt_duration(total_sec)} on M4 Max  |  "
                  f"~{fmt_duration(total_m2)} on M2 Ultra")
            timing_rows.append((exp_name, mode, full_gens,
                                 sec_per_gen, total_sec, total_m2))
        print()

    # ── Correctness summary ───────────────────────────────────────────────────
    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(f"{'='*60}")
    print(f"  CORRECTNESS: {passed}/{total} checks passed")
    if passed < total:
        print("  Failed:")
        for name, ok in _results:
            if not ok:
                print(f"    ✗  {name}")

    # ── Runtime estimate table ────────────────────────────────────────────────
    if timing_rows:
        col = 14
        print(f"\n{'='*75}")
        print(f"  ESTIMATED PRODUCTION RUNTIMES")
        print(f"  Measured on: M4 Max (40-core GPU)  |  "
              f"M2 Ultra estimate: ×{M2_ULTRA_SCALE:.1f} scaling factor")
        print(f"{'='*75}")
        print(f"  {'Experiment':<25} {'Mode':<8} {'Gens':>5}  "
              f"{'s/gen':>6}  {'M4 Max':>{col}}  {'M2 Ultra (est)':>{col}}")
        print(f"  {'-'*25} {'-'*8} {'-'*5}  {'-'*6}  {'-'*col}  {'-'*col}")
        m4_totals, m2_totals = [], []
        for exp_name, mode, gens, spg, tot_m4, tot_m2 in timing_rows:
            print(f"  {exp_name:<25} {mode:<8} {gens:>5}  "
                  f"{spg:>6.1f}  {fmt_duration(tot_m4):>{col}}  "
                  f"{fmt_duration(tot_m2):>{col}}")
            m4_totals.append(tot_m4)
            m2_totals.append(tot_m2)
        print(f"  {'-'*25} {'-'*8} {'-'*5}  {'-'*6}  {'-'*col}  {'-'*col}")
        print(f"  {'Sequential total':<25} {'':8} {'':>5}  {'':>6}  "
              f"{fmt_duration(sum(m4_totals)):>{col}}  "
              f"{fmt_duration(sum(m2_totals)):>{col}}")
        print(f"  {'Parallel total (6x)':<25} {'':8} {'':>5}  {'':>6}  "
              f"{fmt_duration(max(m4_totals)):>{col}}  "
              f"{fmt_duration(max(m2_totals)):>{col}}")
        print()
        print(f"  Note: M2 Ultra estimate assumes ~{M2_ULTRA_SCALE:.0f}× slower than M4 Max")
        print(f"        (M4 Max ~54.7 TFLOPS vs M2 Ultra ~27.2 TFLOPS on MPS).")
        print(f"        Actual speedup varies with batch size and memory bandwidth.")
        print()

    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
