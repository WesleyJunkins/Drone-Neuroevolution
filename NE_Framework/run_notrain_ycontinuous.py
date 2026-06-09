"""
run_notrain_ycontinuous.py — Untrained NE, continuous output [forward_desired, yaw_desired].

Individuals start with random weights; evolution pressure alone shapes the population.
No backprop training is applied — fitness is evaluated on raw (random → evolved) weights.
Loads all_data.json in full — it is pre-built with exactly 20,000 records (800/cell × 25 cells)
by Data/prepare_data.py, so no in-script trimming is needed.
Output model: notrain_ycontinuous/best_model/best_model_notrain_ycontinuous.pth
"""
from utils import load_json
from ga_loop import run_ga
from run_all import (ALL_DATA_FILE, NOTRAIN_SUBSAMPLE,
                     GENERATIONS_NOTRAIN,
                     POPULATION_SIZE_NOTRAIN, BEST_INDIVIDUALS_SIZE_NOTRAIN,
                     CROSSOVER_RATE, MUTATION_RATE, MUTATION_STRENGTH,
                     TRAINING_EPOCHS, TRAINING_LEARNING_RATE, TRAINING_VERBOSE,
                     BATCH_SIZE, MODEL_TYPE, SAVE_MODELS, SAVE_CSV_RESULTS,
                     SAVE_CHECKPOINTS, CHECKPOINT_INTERVAL,
                     CONVERGENCE_WINDOW, CONVERGENCE_THRESHOLD,
                     EARLY_STOPPING, DEVICE)

BASE_DIRECTORY = 'notrain_ycontinuous'
MODEL_NAME     = 'notrain_ycontinuous'
OUTPUT_DIM     = 2   # [forward_desired, yaw_desired]

print(f"Loading dataset (all_data.json — {NOTRAIN_SUBSAMPLE:,} pre-balanced records) ...")
X, y_command, y_continuous, y_motors = load_json(ALL_DATA_FILE)
y_target = y_continuous.to(DEVICE)
X        = X.to(DEVICE)
print(f"  Device     : {DEVICE}")
print(f"  Records    : {X.size(0):,}  |  X: {X.shape}  y: {y_target.shape}")
print(f"  Population : {POPULATION_SIZE_NOTRAIN}  |  Elites: {BEST_INDIVIDUALS_SIZE_NOTRAIN}  |  Generations: {GENERATIONS_NOTRAIN:,}")

run_ga(
    base_directory        = BASE_DIRECTORY,
    model_name            = MODEL_NAME,
    output_dim            = OUTPUT_DIM,
    X                     = X,
    y_target              = y_target,
    generations           = GENERATIONS_NOTRAIN,
    is_trained            = False,
    model_type            = MODEL_TYPE,
    population_size       = POPULATION_SIZE_NOTRAIN,
    best_individuals_size = BEST_INDIVIDUALS_SIZE_NOTRAIN,
    crossover_rate        = CROSSOVER_RATE,
    mutation_rate         = MUTATION_RATE,
    mutation_strength     = MUTATION_STRENGTH,
    training_epochs       = TRAINING_EPOCHS,
    training_learning_rate= TRAINING_LEARNING_RATE,
    training_verbose      = TRAINING_VERBOSE,
    batch_size            = BATCH_SIZE,
    save_models           = SAVE_MODELS,
    save_csv_results      = SAVE_CSV_RESULTS,
    save_checkpoints      = SAVE_CHECKPOINTS,
    checkpoint_interval   = CHECKPOINT_INTERVAL,
    convergence_window    = CONVERGENCE_WINDOW,
    convergence_threshold = CONVERGENCE_THRESHOLD,
    early_stopping        = EARLY_STOPPING,
    device                = DEVICE,
)
