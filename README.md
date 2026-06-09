# Drone Neuroevolution — Neuroevolutionary vs. Baseline Drone Control

## Overview

This project compares a **neuroevolutionary (NE) drone controller** trained on human pilot data against a **traditional "carrot-chasing" rule-based baseline** for the task of autonomous circular track-following. A Bitcraze Crazyflie quadrotor is simulated in [Webots](https://cyberbotics.com/); it hovers about one metre above a white circular track painted on the ground and must stay centred on that track across a range of forward speeds and camera-blur levels. Each controller runs in Webots (written in C) and talks over a local TCP socket to a Python "receiver" that processes the downward camera image, decides on a command, sends it back, and logs an error value every timestep.

The full study is organised as a single, reproducible pipeline. A human pilot first flies the drone manually to collect a training dataset. That raw dataset is cleaned, temporally shifted to model visuomotor reaction delay, balanced across speeds, split into train/test sets, and augmented with five levels of Gaussian blur. The prepared data then drives a neuroevolutionary framework that evolves convolutional neural-network controllers, using three output representations (binary commands, continuous values, motor velocities) under both a *trained* (backprop-per-individual) and an *untrained* (pure evolution) regime, alongside several supplementary experiments (smaller MLPs, NEAT, geometric-feature inputs, standard backprop, and an extended-budget run). Every resulting model — plus the rule-based baseline — is then tested in simulation over a matrix of 5 speeds × 5 blur levels × 3 repetitions, producing raw per-timestep error logs. Those logs are cleaned (restart-artifact removal, GPS-explosion filtering, three-repetition averaging into a 3D hovering-circle error metric) and finally fed into one comprehensive analysis program that produces every statistic, table, chart, heatmap, path plot, and convergence figure used in the thesis.

Three folders are too large for GitHub — the pilot dataset, the trained models, and the raw simulation logs — so they are hosted separately on Box (see below). Download them into `Data/` to reproduce the thesis exactly, and you can then enter the pipeline at any stage: re-run the analysis only, re-test the existing models in simulation, retrain the models, or start over from raw pilot-data collection. If you would rather build everything yourself, you can skip the download and regenerate all three folders by following the pipeline from Step 1.

---

## Reproducing the thesis vs. running from scratch

The repository contains all of the **code**, the Webots world, the track texture, and the
analysis program. The three **large data folders** are not in the repository; they live on Box.

**Box download:** `<ADD BOX LINK HERE>`

Download the folders from Box and place them inside `Data/` so the layout becomes
`Data/datasets/`, `Data/model_data/`, and `Data/ALL_FINAL_ERROR_RESULTS/`.

| Box folder | Put it at | Needed for | Regenerate yourself with |
| --- | --- | --- | --- |
| `datasets/` | `Data/datasets/` | Steps 2–3 (training data) | Step 1 (fly manually) → Step 2 |
| `model_data/` | `Data/model_data/` | Steps 4, 6 (testing + training-curve figure) | Step 3 (train), then assemble into the `Data/model_data/` layout |
| `ALL_FINAL_ERROR_RESULTS/` | `Data/ALL_FINAL_ERROR_RESULTS/` | Steps 5–6 (cleaning + analysis) | Step 4 (simulate) |

- **To reproduce the thesis exactly:** download all three from Box. The human pilot recording
  (`datasets/training_data.jsonl`) cannot be re-derived by code, and both NE training and the
  simulation runs are stochastic — so a from-scratch run will produce valid but non-identical
  results.
- **To run from scratch:** skip the download. Every step below recreates its own outputs (each
  script makes the folders it writes to), and the two analysis steps fail gracefully with a clear
  message if their input folder is missing. The one manual sub-step is assembling trained models
  into the `Data/model_data/` layout (see Step 3).

---

## Directory map

```
.
├── README.md                     This file.
├── requirements.txt              All Python dependencies for the whole pipeline.
│
├── Worlds/
│   └── wesDroneRL2.wbt           Webots world (R2025a): Crazyflie + downward camera +
│                                 circular track. Its robot controller field selects which
│                                 C controller runs (default: all_controller).
│
├── assets/
│   └── circle.png                Track texture for the world's ground plane. The world
│                                 references it as ../assets/circle.png, so this folder
│                                 must stay at the repository root, beside Worlds/.
│
├── Controllers/                  Webots controllers (C). Each folder has its .c source,
│   │                             pid_controller.c/.h (flight PID library), and a Makefile.
│   ├── baseline_controller/      Rule-based carrot-chasing flight controller.
│   ├── manual_controller/        Keyboard-driven flight; streams pilot data to the receiver.
│   ├── NE_controller/            Runs one NE model; output mode chosen at runtime via JSON.
│   ├── all_controller/           Production controller for batch testing. Reads
│   │                             WEBOTS_CONTROLLER_PORT so many instances run at once;
│   │                             sends GPS position in its packet.
│   └── NE_centering/             One-shot run used to fit the track circle geometry.
│
├── Receivers/                    Python side of the controller–receiver TCP protocol.
│   ├── manual_receiver.py        Records pilot keystrokes + images → training_data.jsonl.
│   ├── baseline_receiver.py      Implements the rule-based vision algorithm + logging.
│   ├── NE_receiver.py            Loads a single model (legacy single-session workflow).
│   ├── all_receiver.py           Unified runner: loads any of the 19 sessions' models,
│   │                             runs the 75-test matrix, logs results.
│   └── run_all.py                Concurrent test orchestrator: spawns N Webots+receiver
│                                 worker pairs and pulls sessions from a queue.
│
├── Data/                         Data folders + the two data-preparation programs.
│   ├── prepare_data.py           Builds the NE training data from raw pilot data.
│   ├── clean_data.py             Cleans the raw simulation error logs.
│   ├── datasets/                 (Box / regenerable) training_data.jsonl (raw pilot data)
│   │                             and the prepared training.json / testing.json / all_data.json.
│   ├── model_data/               (Box / regenerable) trained models (main_phase/ +
│   │                             extra_observation/): .pth weights and NEAT .pkl genomes.
│   ├── ALL_FINAL_ERROR_RESULTS/          (Box / regenerable) raw per-session test logs.
│   └── ALL_FINAL_ERROR_RESULTS_CLEANED/  (produced by clean_data.py) FOR_PLOTTING/
│                                          and FOR_ANALYSIS/ cleaned datasets.
│
├── NE_Framework/                 Neuroevolution training framework + experiments.
│   ├── Model.py                  CNN/MLP models, ModelTrainer, GA Individual.
│   ├── ga_loop.py                Shared genetic-algorithm loop + checkpointing.
│   ├── utils.py                  Data loading (load_json) and CSV helpers.
│   ├── nn.py                     Flexible feedforward NN utility class.
│   ├── run_all.py                Config hub (hyperparameters) + multi-terminal launcher.
│   ├── run_train_ycommand.py     Trained NE, binary-command output.
│   ├── run_train_ycontinuous.py  Trained NE, continuous output.
│   ├── run_train_ymotors.py      Trained NE, motor-velocity output.
│   ├── run_notrain_ycommand.py   Untrained NE, binary-command output.
│   ├── run_notrain_ycontinuous.py Untrained NE, continuous output.
│   ├── run_notrain_ymotors.py    Untrained NE, motor-velocity output.
│   ├── exp_a_small_mlp.py        Supplementary: untrained NE on smaller MLPs.
│   ├── exp_b_neat.py             Supplementary: NEAT (evolving topology).
│   ├── exp_c_features.py         Supplementary: NE on 3 geometric features.
│   ├── exp_e_standard_backprop.py Supplementary: standard backprop CNN (no GA).
│   ├── exp_f_extended_notrain.py  Supplementary: extended-budget untrained NE.
│   ├── visualize_neat.py         Renders the best NEAT genome topology to PNG.
│   ├── test_framework.py         Unit tests for the framework.
│   ├── test_six_experiments.py   Runtime estimator + correctness check.
│   └── clear.py                  Utility: wipes experiment output directories.
│
├── Analysis/                     The single comprehensive analysis program.
│   ├── analyze.py                Sections 01–16: rankings, heatmaps, statistics
│   │                             (ANOVA/Kruskal–Wallis/Wilcoxon), overview charts,
│   │                             convergence, training curves, and per-session path
│   │                             plots. Reads from Data/, writes only to Analysis_Results/.
│   └── Analysis_Results/         All analysis output lands here (created on first run).
│
└── Extras/
    └── centering_run/            Centering-run GPS log + fitted-circle figure used to
                                  derive the track centre and radius constants.
```

---

## Running the full pipeline

The steps below go from a clean machine through to the final analysis. If you download the three data folders from Box (see *Reproducing the thesis vs. running from scratch* above), you can stop after any step or jump straight to the cleaning/analysis steps (5–6) to reproduce the results. If you do not download them, start at Step 1 and the pipeline will build each folder for you.

> All commands are run from the repository root unless noted. On a case-sensitive filesystem (Linux), use the exact capitalisation shown.

### Step 0 — Prerequisites and environment setup

**Inputs:** a machine with Python 3.10 and (for the simulation steps) Webots installed.
**Outputs:** an activated virtual environment with all dependencies installed.

1. Install **Python 3.10** (the version this project was developed and tested against).
2. Install **Webots R2025a** from [https://cyberbotics.com/](https://cyberbotics.com/) — only needed for the simulation steps (1, 4). On macOS the default path is `/Applications/Webots.app`; if you install it elsewhere or on another OS, update `WEBOTS_EXE` near the top of `Receivers/run_all.py`. The world loads its track texture from `assets/circle.png`, so keep the `assets/` folder at the repository root.
3. Create and activate a virtual environment, then install the dependencies:
  ```bash
   python3.10 -m venv venv
   source venv/bin/activate          # macOS/Linux  (Windows: venv\Scripts\activate)
   python -m pip install --upgrade pip
   pip install -r requirements.txt
  ```
   On Apple Silicon, PyTorch automatically uses the MPS GPU backend for training and inference.
4. **(To reproduce the thesis)** download the three data folders from Box and place them inside `Data/` as `Data/datasets/`, `Data/model_data/`, and `Data/ALL_FINAL_ERROR_RESULTS/`. Skip this if you intend to regenerate everything from Step 1.

### Step 1 — (Optional) Collect pilot data by flying manually

Only needed if you want to build a *new* training dataset. The original pilot recording used for the thesis is on Box — download `datasets/` and place it at `Data/datasets/training_data.jsonl` to reuse it instead of flying.

**Inputs:** human keyboard flight in Webots.
**Outputs:** `Receivers/training_data.jsonl` (then move it to `Data/datasets/`).

1. Start the receiver first so it is listening on the socket:
  ```bash
   python Receivers/manual_receiver.py
  ```
2. In Webots, open `Worlds/wesDroneRL2.wbt`, set the robot's `controller` field to `manual_controller`, and run the simulation.
3. Fly the drone with the keyboard (W = forward, A/D = yaw, Q/E = up/down, +/- = speed, R = reset). Frames are appended to `training_data.jsonl` once you begin steering.
4. Move the result into place: `mv Receivers/training_data.jsonl Data/datasets/`.

### Step 2 — Prepare the training data

**Inputs:** `Data/datasets/training_data.jsonl`.
**Outputs:** `Data/datasets/training.json`, `testing.json`, `all_data.json`, and `Data/datasets/visualizations/` (7 PNGs).

```bash
python Data/prepare_data.py
```

This analyzes, temporally shifts (≈224 ms reaction delay), cleans, balances by speed, splits 80/20, and applies five blur levels, then writes the three dataset files and verification visualizations. (This reads the full 1.5 GB raw file, so run it on a machine with ample RAM.)

### Step 3 — Train the neuroevolution models

**Inputs:** `Data/datasets/{training,testing,all_data}.json`.
**Outputs:** each experiment writes its best model into its **own** folder under `NE_Framework/` (e.g. `NE_Framework/train_ycommand/best_model/best_model_train_ycommand.pth`, `NE_Framework/exp_a_small_mlp/<arch>/best_model.pth`).

- Main NE experiments (config and launch modes are set in `NE_Framework/run_all.py`):
  ```bash
  python NE_Framework/run_all.py
  ```
- Supplementary experiments run individually, e.g.:
  ```bash
  python NE_Framework/exp_a_small_mlp.py        # smaller MLPs (untrained NE)
  python NE_Framework/exp_b_neat.py             # NEAT
  python NE_Framework/exp_c_features.py         # geometric-feature input
  python NE_Framework/exp_e_standard_backprop.py # standard backprop CNN
  python NE_Framework/exp_f_extended_notrain.py  # extended untrained NE
  ```

Training is the most time-consuming stage (hours to days depending on hardware).

**Assembling `Data/model_data/`:** Step 4 loads models from a fixed layout under `Data/model_data/` (`main_phase/` for the four main CNN sessions, `extra_observation/` for the supplementary ones). The training scripts above do **not** write into that layout, so to test your own models you must copy each experiment's best model (and, for NEAT, its `best_genome.pkl` + `neat_config.ini`) into the paths listed in the `SESSIONS` block of `Receivers/all_receiver.py`. Because training is stochastic, your models will differ from the thesis — download `model_data/` from Box to reproduce the thesis results exactly, and you can then skip this step entirely.

### Step 4 — Test every controller in simulation

**Inputs:** trained models in `Data/model_data/` (from Box, or assembled from Step 3) and the Webots world.
**Outputs:** raw per-session logs in `Data/ALL_FINAL_ERROR_RESULTS/{session}_error_log.csv` (the runner creates this folder automatically).

Make sure `Worlds/wesDroneRL2.wbt`'s robot `controller` field is set to `all_controller`, then launch the concurrent runner (it starts the Webots instances and receivers for you):

```bash
python Receivers/run_all.py --workers 4
```

Each worker runs one session's full 5 speeds × 5 blur × 3 repetitions test matrix; the runner refills workers from the session queue until all 19 sessions are done. (For a single model the legacy `Receivers/NE_receiver.py` workflow also exists, and `baseline_receiver.py` / `NE_centering` cover the baseline and track-geometry runs.) If you want the thesis's raw logs instead of running the simulation, download `ALL_FINAL_ERROR_RESULTS/` from Box, place it inside `Data/`, and skip to Step 5.

### Step 5 — Clean the raw results

**Inputs:** `Data/ALL_FINAL_ERROR_RESULTS/`.
**Outputs:** `Data/ALL_FINAL_ERROR_RESULTS_CLEANED/FOR_PLOTTING/` and `FOR_ANALYSIS/` (19 CSVs each).

```bash
python Data/clean_data.py
```

This removes simulation-restart artifacts, NaN-filters GPS explosions, builds the rep-1 raw XY paths (`FOR_PLOTTING/`), and computes the three-repetition-averaged 3D hovering-circle error (`FOR_ANALYSIS/`).

### Step 6 — Run the analysis

**Inputs:** `Data/ALL_FINAL_ERROR_RESULTS_CLEANED/`, `Data/ALL_FINAL_ERROR_RESULTS/`, and `Data/model_data/` (the last is used only for the standard-backprop training-curve figure; if it is absent the analysis still runs and just skips that one figure with a warning).
**Outputs:** everything under `Analysis/Analysis_Results/` — rankings and per-session summaries, 5×5 heatmaps, speed/blur trends, ANOVA / Kruskal–Wallis / Wilcoxon statistics, composite performance scores, lap and spatial analyses, convergence figures, overview charts, and one path-plot PNG per session.

```bash
python Analysis/analyze.py                 # all sections (01–16)
python Analysis/analyze.py --sections 01 09 16   # or a subset
```

Nothing outside `Analysis/Analysis_Results/` is written, so this step never touches the input data.

---

### Pipeline at a glance


| Step | Program(s)                                           | Input                      | Output                             |
| ---- | ---------------------------------------------------- | -------------------------- | ---------------------------------- |
| 1    | `manual_controller` + `Receivers/manual_receiver.py` | manual flight              | `training_data.jsonl`              |
| 2    | `Data/prepare_data.py`                               | `training_data.jsonl`      | `training/testing/all_data.json`   |
| 3    | `NE_Framework/run_all.py` + `exp_*.py`               | the `*.json` datasets      | `NE_Framework/<exp>/` → assemble into `Data/model_data/` |
| 4    | `Receivers/run_all.py` + `all_controller` (Webots)   | trained models             | `Data/ALL_FINAL_ERROR_RESULTS/`    |
| 5    | `Data/clean_data.py`                                 | `ALL_FINAL_ERROR_RESULTS/` | `ALL_FINAL_ERROR_RESULTS_CLEANED/` |
| 6    | `Analysis/analyze.py`                                | cleaned results + models   | `Analysis/Analysis_Results/`       |


