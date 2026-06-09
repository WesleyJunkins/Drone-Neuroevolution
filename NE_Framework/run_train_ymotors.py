"""
run_train_ymotors.py — Trained NE, motor velocity output [m1, m2, m3, m4].

Each individual is trained on pilot data via backprop before fitness is evaluated.
Motor velocities are in approximately [-150, +150] rad/s.
Output model: train_ymotors/best_model/best_model_train_ymotors.pth
"""
from utils import load_json
from ga_loop import run_ga
from run_all import (DATA_FILE, TESTING_FILE, TRAINING_SUBSAMPLE,
                     GENERATIONS_TRAIN, POPULATION_SIZE, BEST_INDIVIDUALS_SIZE,
                     CROSSOVER_RATE, MUTATION_RATE, MUTATION_STRENGTH,
                     TRAINING_EPOCHS, TRAINING_LEARNING_RATE, TRAINING_VERBOSE,
                     BATCH_SIZE, MODEL_TYPE, SAVE_MODELS, SAVE_CSV_RESULTS,
                     SAVE_CHECKPOINTS, CHECKPOINT_INTERVAL,
                     CONVERGENCE_WINDOW, CONVERGENCE_THRESHOLD,
                     EARLY_STOPPING, DEVICE)

BASE_DIRECTORY = 'train_ymotors'
MODEL_NAME     = 'train_ymotors'
OUTPUT_DIM     = 4   # [m1, m2, m3, m4]

print(f"Loading training data (subsample={TRAINING_SUBSAMPLE:,}) ...")
X, y_command, y_continuous, y_motors = load_json(DATA_FILE, subsample=TRAINING_SUBSAMPLE)
X        = X.to(DEVICE)
y_target = y_motors.to(DEVICE)

print(f"Loading eval data (testing.json) ...")
X_eval, _, _, y_mot_eval = load_json(TESTING_FILE)
X_eval = X_eval.to(DEVICE)
y_eval = y_mot_eval.to(DEVICE)

print(f"  Device : {DEVICE}")
print(f"  Train  : {X.size(0):,} records  |  X: {X.shape}  y: {y_target.shape}")
print(f"  Eval   : {X_eval.size(0):,} records  |  X: {X_eval.shape}  y: {y_eval.shape}")

run_ga(
    base_directory        = BASE_DIRECTORY,
    X_eval                = X_eval,
    y_eval                = y_eval,
    model_name            = MODEL_NAME,
    output_dim            = OUTPUT_DIM,
    X                     = X,
    y_target              = y_target,
    generations           = GENERATIONS_TRAIN,
    is_trained            = True,
    model_type            = MODEL_TYPE,
    population_size       = POPULATION_SIZE,
    best_individuals_size = BEST_INDIVIDUALS_SIZE,
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
