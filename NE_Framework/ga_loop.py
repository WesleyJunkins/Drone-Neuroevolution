"""
ga_loop.py — Shared genetic algorithm loop used by all six run_*.py scripts.

Provides:
  run_ga()   — main GA loop with checkpoint save/resume, rolling analytics CSV, ETA
"""

import os
import csv
import time
import random
import torch

from Model import Individual, ModelTrainer


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _pop_checkpoint_path(base_directory):
    return os.path.join(base_directory, 'checkpoints', 'population_checkpoint.pt')


def save_population_checkpoint(base_directory, generation, individuals,
                                best_history, prev_best, convergence_announced):
    """Save full population state so the run can be resumed after a crash."""
    os.makedirs(os.path.join(base_directory, 'checkpoints'), exist_ok=True)
    checkpoint = {
        'generation': generation,
        'best_history': best_history,
        'prev_best': prev_best,
        'convergence_announced': convergence_announced,
        'individuals': [
            {
                'model_state_dict': ind.model.state_dict(),
                'fitness': ind.fitness,
                'individual_id': ind.individual_id,
                'generation': ind.generation,
            }
            for ind in individuals
        ],
    }
    path = _pop_checkpoint_path(base_directory)
    torch.save(checkpoint, path)


def load_population_checkpoint(base_directory, output_dim, model_type, device):
    """
    Load a saved population checkpoint.  Returns (start_generation, individuals,
    best_history, prev_best, convergence_announced) or None if no checkpoint exists.
    """
    path = _pop_checkpoint_path(base_directory)
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, weights_only=False, map_location=device)
    individuals = []
    for d in checkpoint['individuals']:
        ind = Individual(model_type=model_type, output_features=output_dim,
                         device=device)
        ind.model.load_state_dict(d['model_state_dict'])
        ind.fitness       = d['fitness']
        ind.individual_id = d['individual_id']
        ind.generation    = d['generation']
        individuals.append(ind)

    return (
        checkpoint['generation'] + 1,
        individuals,
        checkpoint['best_history'],
        checkpoint['prev_best'],
        checkpoint['convergence_announced'],
    )


# ── Analytics CSV helper ──────────────────────────────────────────────────────

def _analytics_path(base_directory):
    return os.path.join(base_directory, 'results', 'analytics.csv')


def _append_analytics(base_directory, row_dict):
    """Append one generation's stats to the rolling analytics CSV."""
    path = _analytics_path(base_directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'generation', 'best_fitness', 'avg_fitness', 'worst_fitness',
            'delta', 'diversity', 'elapsed_seconds', 'eta_seconds',
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


# ── Main GA loop ──────────────────────────────────────────────────────────────

def run_ga(
    base_directory,
    model_name,
    output_dim,
    X,
    y_target,
    generations,
    is_trained,
    *,
    X_eval=None,
    y_eval=None,
    model_type,
    population_size,
    best_individuals_size,
    crossover_rate,
    mutation_rate,
    mutation_strength,
    training_epochs,
    training_learning_rate,
    training_verbose,
    batch_size,
    save_models,
    save_csv_results,
    save_checkpoints,
    checkpoint_interval,
    convergence_window,
    convergence_threshold,
    early_stopping,
    device,
):
    """
    Run the genetic algorithm.

    Args:
        base_directory : output directory for this experiment (e.g. 'train_ycommand')
        model_name     : used in the saved best-model filename
        output_dim     : number of network outputs
        X, y_target    : training tensors (already on device); used for backprop in trained mode
        generations    : total generations to run
        is_trained     : if True, backprop-train each individual before fitness eval
        X_eval, y_eval : optional held-out eval tensors; if provided, fitness is measured
                         on these instead of X/y_target (trained mode only — separates
                         the training signal from the selection signal)
        All other args : hyperparameters imported from run_all.py

    Returns:
        best_individual : the Individual with the highest fitness
    """

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    resume = load_population_checkpoint(base_directory, output_dim, model_type, device)
    if resume is not None:
        start_gen, individuals, best_history, prev_best, convergence_announced = resume
        print(f"  ↺ Resuming from checkpoint at generation {start_gen - 1} "
              f"(best so far: {max(ind.fitness for ind in individuals if ind.fitness is not None):.6f})")
    else:
        start_gen           = 0
        individuals         = []
        best_history        = []
        prev_best           = None
        convergence_announced = False

    def fmt_time(s):
        s = int(s)
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    # For trained mode: train on X/y_target, evaluate fitness on X_eval/y_eval (testing set).
    # For notrain mode: X_eval/y_eval are None; fitness is evaluated on X/y_target directly.
    X_fit = X_eval if (is_trained and X_eval is not None) else X
    y_fit = y_eval if (is_trained and y_eval is not None) else y_target

    run_start = time.time()
    gen_times = []   # rolling per-generation wall times for ETA

    for generation in range(start_gen, generations):
        gen_start = time.time()

        # ── Initialise or evolve population ───────────────────────────────────
        if generation == 0:
            if is_trained:
                print(f"\n  Gen 0 — initializing {population_size} individuals "
                      f"({training_epochs} epochs each) ...")
            else:
                print(f"\n  Gen 0 — initializing {population_size} individuals ...")

            for i in range(population_size):
                ind = Individual(model_type=model_type, output_features=output_dim,
                                 individual_id=f"0_{i}", device=device)
                if is_trained:
                    print(f"\n  Ind {i+1:>2}/{population_size}  training ({training_epochs} epochs) ...")
                    trainer = ModelTrainer(ind.model, learning_rate=training_learning_rate,
                                           verbose=training_verbose)
                    trainer.train(X, y_target, epochs=training_epochs, batch_size=batch_size)
                ind.evaluate_fitness(X_fit, y_fit, batch_size=batch_size)
                if is_trained:
                    print(f"  Ind {i+1:>2}/{population_size}  fitness={ind.fitness:.6f}")
                individuals.append(ind)
                if save_models:
                    ind.save(f'{base_directory}/models/model_{ind.individual_id}')
        else:
            individuals.sort()
            elites   = individuals[:best_individuals_size]
            offspring = []
            for i in range(population_size - best_individuals_size):
                child = random.choice(elites).crossover(random.choice(elites),
                                                        crossover_rate=crossover_rate)
                child.individual_id = f"{generation}_{i}"
                child.generation    = generation
                child.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
                offspring.append(child)
            individuals = elites + offspring

            if is_trained:
                n_new = sum(1 for ind in individuals if ind.fitness is None)
                print(f"\n  Gen {generation} — {best_individuals_size} elites kept, "
                      f"{n_new} new offspring to train ...")
                for e in elites:
                    print(f"    [elite]  {e.individual_id}  fitness={e.fitness:.6f}")

        # ── Evaluate unevaluated individuals ─────────────────────────────────
        fitnesses = []
        n_new     = sum(1 for ind in individuals if ind.fitness is None)
        new_count = 0
        for ind in individuals:
            if ind.fitness is None:
                new_count += 1
                if is_trained:
                    print(f"\n  Offspring {new_count:>2}/{n_new}  training ({training_epochs} epochs) ...")
                    trainer = ModelTrainer(ind.model, learning_rate=training_learning_rate,
                                           verbose=training_verbose)
                    trainer.train(X, y_target, epochs=training_epochs, batch_size=batch_size)
                ind.evaluate_fitness(X_fit, y_fit, batch_size=batch_size)
                if is_trained:
                    print(f"  Offspring {new_count:>2}/{n_new}  fitness={ind.fitness:.6f}")
            fitnesses.append(ind.fitness)
            if save_models:
                ind.save(f'{base_directory}/models/model_{ind.individual_id}')

        # ── Per-generation statistics ─────────────────────────────────────────
        best  = max(fitnesses)
        worst = min(fitnesses)
        avg   = sum(fitnesses) / len(fitnesses)

        delta     = best - prev_best if prev_best is not None else 0.0
        delta_str = f"{delta:+.6f}"  if prev_best is not None else "       ---"
        prev_best = best
        diversity = (sum((f - avg) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
        best_history.append(best)

        # ETA
        gen_elapsed  = time.time() - gen_start
        gen_times.append(gen_elapsed)
        if len(gen_times) > 20:
            gen_times.pop(0)
        avg_gen_time = sum(gen_times) / len(gen_times)
        elapsed      = time.time() - run_start
        gens_left    = generations - generation - 1
        eta          = avg_gen_time * gens_left

        elapsed_str = fmt_time(elapsed)
        eta_str     = fmt_time(eta)

        # Convergence check
        conv_tag = ""
        if len(best_history) >= convergence_window:
            window_gain = best_history[-1] - best_history[-convergence_window]
            ref         = best_history[-convergence_window]
            rel_gain    = (window_gain / abs(ref)) if ref != 0 else 0.0
            if rel_gain < convergence_threshold:
                conv_tag = "  ◆ converged"
                if not convergence_announced:
                    print(f"\n  *** CONVERGENCE at generation {generation}: "
                          f"improvement < {convergence_threshold*100:.1f}% "
                          f"over last {convergence_window} gens ***\n")
                    convergence_announced = True

        print(f"Gen {generation:>4}  best={best:.6f}  avg={avg:.6f}  "
              f"worst={worst:.6f}  Δ={delta_str}  div={diversity:.4f}  "
              f"[{elapsed_str} elapsed  ETA {eta_str}]{conv_tag}")

        # ── Persist ───────────────────────────────────────────────────────────
        if save_csv_results:
            _append_analytics(base_directory, {
                'generation':    generation,
                'best_fitness':  best,
                'avg_fitness':   avg,
                'worst_fitness': worst,
                'delta':         delta,
                'diversity':     diversity,
                'elapsed_seconds': round(elapsed, 1),
                'eta_seconds':     round(eta, 1),
            })

        if save_checkpoints and generation % checkpoint_interval == 0:
            individuals.sort()
            chk_dir = f'{base_directory}/checkpoints'
            os.makedirs(chk_dir, exist_ok=True)
            individuals[0].save(f'{chk_dir}/checkpoint_gen_{generation}')
            individuals[0].save(f'{chk_dir}/latest_best')
            save_population_checkpoint(
                base_directory, generation, individuals,
                best_history, prev_best, convergence_announced,
            )

        if early_stopping and convergence_announced:
            print(f'  Early stopping at generation {generation}.')
            break

    # ── Save best model ───────────────────────────────────────────────────────
    individuals.sort()
    best_individual = individuals[0]
    save_path = f'{base_directory}/best_model/best_model_{model_name}'
    best_individual.save(save_path)
    print(f"\n✓ Best model saved → {save_path}.pth")
    print(f"  fitness={best_individual.fitness:.6f}  id={best_individual.individual_id}"
          f"  gen={best_individual.generation}")
    print(f"  Total wall time: {fmt_time(time.time() - run_start)}")

    return best_individual
