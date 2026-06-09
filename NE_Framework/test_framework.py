"""
test_framework.py — Comprehensive tests for the NE framework.

Tests cover:
  1.  CNN forward pass shape
  2.  MLP forward pass shape
  3.  CNN input reshaping (image + speed concatenation)
  4.  Mini-batch training decreases loss
  5.  Full-batch vs mini-batch loss equivalence
  6.  Individual clone preserves weights exactly
  7.  Individual mutate changes weights
  8.  Individual crossover produces valid offspring within parent bounds
  9.  Architecture preservation through clone / crossover
  10. Fitness = negative MSE, batched eval matches full eval
  11. Sorting: higher fitness ranks first
  12. Save / load round-trip (identical outputs)
  13. GA loop: best fitness monotonically improves or stays (trained NE)
  14. GA loop: untrained NE makes progress through evolution only
  15. load_json with real data (skipped gracefully if file absent)
  16. load_json subsample works
  17. save_to_csv / load_csv round-trip
  18. utils.loss() all three variants
  19. ModelTrainer.evaluate() sample-weighted batching matches full eval
  20. Full mini-pipeline smoke test (2 generations, 4 individuals)

Run from NE_Framework/ directory:
    python3 test_framework.py
"""

import sys
import os
import math
import tempfile
import json
import random
import torch
import numpy as np

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model import CreateCNNModel, CreateModel, ModelTrainer, Individual
from utils import load_json, save_to_csv, load_csv, loss as util_loss

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((name, condition))
    return condition


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def synthetic_batch(n=64, output_dim=3):
    """Random (N, 4097) input and (N, output_dim) target tensors."""
    X = torch.rand(n, 4097)
    y = torch.rand(n, output_dim)
    return X, y


def params_snapshot(model):
    """Flat tensor of all parameters for comparison."""
    return torch.cat([p.data.flatten() for p in model.parameters()])


# ── Test 1: CNN forward pass shape ────────────────────────────────────────────
section("1. Model Forward Pass Shapes")

for out_dim in [2, 3, 4]:
    X, _ = synthetic_batch(8, out_dim)
    cnn = CreateCNNModel(output_features=out_dim)
    out = cnn(X)
    check(f"CNN output_features={out_dim} → shape (8,{out_dim})",
          out.shape == (8, out_dim), str(out.shape))

    mlp = CreateModel(4097, 512, 256, 128, out_dim)
    out_m = mlp(X)
    check(f"MLP output_features={out_dim} → shape (8,{out_dim})",
          out_m.shape == (8, out_dim), str(out_m.shape))

# ── Test 2: CNN image/speed split ─────────────────────────────────────────────
section("2. CNN Image + Speed Concatenation")

cnn2 = CreateCNNModel(output_features=3)
X_fixed = torch.zeros(4, 4097)
X_fixed[:, 4096] = 0.5           # speed = 0.5
out_a = cnn2(X_fixed)
X_fixed2 = X_fixed.clone()
X_fixed2[:, 4096] = 0.9          # different speed
out_b = cnn2(X_fixed2)
# Same image, different speed should produce different outputs
check("Different speed → different output (speed is used)",
      not torch.allclose(out_a, out_b))

# ── Test 3: Single sample inference ───────────────────────────────────────────
section("3. Single-Sample Inference")

cnn3 = CreateCNNModel(output_features=3)
X1 = torch.rand(1, 4097)
out = cnn3(X1)
check("CNN handles batch_size=1", out.shape == (1, 3))
out_t = cnn3.test(X1)
check("CNN.test() returns same shape as forward()", out_t.shape == (1, 3))

# ── Test 4: Mini-batch training decreases loss ────────────────────────────────
section("4. Mini-batch Training Reduces Loss")

for model_type in ['cnn', 'mlp']:
    for out_dim in [2, 3, 4]:
        X_tr, y_tr = synthetic_batch(200, out_dim)
        ind = Individual(model_type=model_type, output_features=out_dim)
        loss_before = -ind.evaluate_fitness(X_tr, y_tr, batch_size=64)
        trainer = ModelTrainer(ind.model, learning_rate=0.005)
        trainer.train(X_tr, y_tr, epochs=30, batch_size=64)
        loss_after = -ind.evaluate_fitness(X_tr, y_tr, batch_size=64)
        improved = loss_after < loss_before
        check(f"Training reduces loss ({model_type}, out={out_dim})",
              improved, f"{loss_before:.4f} → {loss_after:.4f}")

# ── Test 5: Full-batch vs mini-batch loss equivalence ─────────────────────────
section("5. Full-batch vs Mini-batch Loss Equivalence")

X5, y5 = synthetic_batch(500, 3)
cnn5 = CreateCNNModel(output_features=3)
trainer5 = ModelTrainer(cnn5)
full_loss = trainer5.evaluate(X5, y5, batch_size=None)
batch_loss = trainer5.evaluate(X5, y5, batch_size=64)
check("Full-batch and mini-batch evaluate() agree (tol 1e-4)",
      abs(full_loss - batch_loss) < 1e-4,
      f"full={full_loss:.6f}  batch={batch_loss:.6f}")

# Also check evaluate_fitness
ind5 = Individual(model_type='cnn', output_features=3)
ind5.model = cnn5
fit_full  = ind5.evaluate_fitness(X5, y5, batch_size=None)
fit_batch = ind5.evaluate_fitness(X5, y5, batch_size=64)
check("evaluate_fitness() full vs batched agree (tol 1e-4)",
      abs(fit_full - fit_batch) < 1e-4,
      f"full={fit_full:.6f}  batch={fit_batch:.6f}")

# ── Test 6: Clone preserves weights exactly ───────────────────────────────────
section("6. Clone Preserves Weights")

orig = Individual(model_type='cnn', output_features=3, individual_id='orig')
cloned = orig.clone(new_id='clone')
p_orig   = params_snapshot(orig.model)
p_cloned = params_snapshot(cloned.model)
check("Clone has identical weights to original",
      torch.allclose(p_orig, p_cloned))
check("Clone is a different object (deep copy)",
      orig.model is not cloned.model)
check("Clone fitness is None (not copied)",
      cloned.fitness is None)
check("Clone parent_ids contains original id",
      cloned.parent_ids == ['orig'])

# Check that modifying clone doesn't affect original
with torch.no_grad():
    for p in cloned.model.parameters():
        p.fill_(99.0)
p_after = params_snapshot(orig.model)
check("Modifying clone does not affect original",
      torch.allclose(p_orig, p_after))

# ── Test 7: Mutate changes weights ────────────────────────────────────────────
section("7. Mutation Changes Weights")

ind7 = Individual(model_type='cnn', output_features=3)
before = params_snapshot(ind7.model).clone()
ind7.mutate(mutation_rate=0.5, mutation_strength=1.0)  # aggressive to guarantee changes
after = params_snapshot(ind7.model)
changed = (before != after).sum().item()
check("Mutate changes some weights", changed > 0, f"{changed} weights changed")
check("Mutate resets fitness to None", ind7.fitness is None)
check("Mutate increments mutations_applied counter",
      ind7.metadata['mutations_applied'] == 1)

# Mutation rate 0 should change nothing
ind7b = Individual(model_type='cnn', output_features=3)
before_b = params_snapshot(ind7b.model).clone()
ind7b.mutate(mutation_rate=0.0, mutation_strength=1.0)
after_b = params_snapshot(ind7b.model)
check("mutation_rate=0 changes no weights",
      torch.allclose(before_b, after_b))

# ── Test 8: Crossover produces valid offspring ────────────────────────────────
section("8. Crossover")

p1 = Individual(model_type='cnn', output_features=3, individual_id='p1')
p2 = Individual(model_type='cnn', output_features=3, individual_id='p2')
child = p1.crossover(p2, crossover_rate=0.5)

pp1 = params_snapshot(p1.model)
pp2 = params_snapshot(p2.model)
pc  = params_snapshot(child.model)

# Each child weight must come from either p1 or p2 exactly
from_p1 = torch.isclose(pc, pp1)
from_p2 = torch.isclose(pc, pp2)
all_sourced = (from_p1 | from_p2).all().item()
check("Every child weight comes from either parent",
      all_sourced, f"mismatched={int((~(from_p1|from_p2)).sum())}")
check("Child fitness is None", child.fitness is None)
check("Child parent_ids contain both parents",
      'p1' in child.parent_ids and 'p2' in child.parent_ids)
check("Child generation = max(parents)+1",
      child.generation == max(p1.generation, p2.generation) + 1)

# ── Test 9: Architecture preservation ────────────────────────────────────────
section("9. Architecture Preservation Through Clone / Crossover")

for mt in ['cnn', 'mlp']:
    for od in [2, 3, 4]:
        orig9 = Individual(model_type=mt, output_features=od)
        clone9 = orig9.clone()
        X9, _ = synthetic_batch(4, od)
        out_orig  = orig9.model(X9)
        out_clone = clone9.model(X9)
        check(f"Clone preserves arch ({mt}, out={od}) — output shape matches",
              out_clone.shape == (4, od))

        p9b = Individual(model_type=mt, output_features=od)
        child9 = orig9.crossover(p9b)
        out_child = child9.model(X9)
        check(f"Crossover child preserves arch ({mt}, out={od})",
              out_child.shape == (4, od))

# ── Test 10: Fitness sign and batched match ───────────────────────────────────
section("10. Fitness = Negative MSE")

X10, y10 = synthetic_batch(100, 3)
ind10 = Individual(model_type='cnn', output_features=3)
manual_loss = torch.nn.MSELoss()(ind10.model.test(X10), y10).item()
fitness = ind10.evaluate_fitness(X10, y10)
check("fitness == -MSELoss",
      abs(fitness - (-manual_loss)) < 1e-6,
      f"fitness={fitness:.6f}  -mse={-manual_loss:.6f}")
check("metadata['evaluated'] = True after evaluate_fitness",
      ind10.metadata['evaluated'])

# ── Test 11: Sorting (higher fitness = better) ────────────────────────────────
section("11. Sorting — Higher Fitness Ranks First")

inds11 = []
fitnesses_set = [-0.5, -0.1, -0.8, -0.3]
for i, f in enumerate(fitnesses_set):
    ind = Individual(model_type='cnn', output_features=3, individual_id=str(i))
    ind.fitness = f
    inds11.append(ind)
inds11.sort()
sorted_fitnesses = [i.fitness for i in inds11]
check("After sort(), highest fitness (least negative) is first",
      sorted_fitnesses[0] == -0.1,
      str(sorted_fitnesses))
check("Sorted in descending fitness order",
      sorted_fitnesses == sorted(fitnesses_set, reverse=True),
      str(sorted_fitnesses))

# ── Test 12: Save / load round-trip ───────────────────────────────────────────
section("12. Save / Load Round-trip")

with tempfile.TemporaryDirectory() as tmpdir:
    for mt in ['cnn', 'mlp']:
        ind12 = Individual(model_type=mt, output_features=3, individual_id='t12')
        X12, _ = synthetic_batch(8, 3)
        out_before = ind12.model.test(X12)

        save_path = os.path.join(tmpdir, f'model_{mt}')
        ind12.save(save_path)

        ind12b = Individual(model_type=mt, output_features=3)
        ind12b.load(save_path + '.pth')
        out_after = ind12b.model.test(X12)
        check(f"Save/load round-trip ({mt}) — identical outputs",
              torch.allclose(out_before, out_after))

# ── Test 13: GA loop — trained NE improves ───────────────────────────────────
section("13. Trained GA Loop — Best Fitness Improves")

torch.manual_seed(42)
random.seed(42)
X13, y13 = synthetic_batch(300, 3)
POP = 6; ELITES = 2; GENS = 4

population = []
for i in range(POP):
    ind = Individual(model_type='cnn', output_features=3, individual_id=f'0_{i}')
    trainer = ModelTrainer(ind.model, learning_rate=0.01)
    trainer.train(X13, y13, epochs=20, batch_size=64)
    ind.evaluate_fitness(X13, y13, batch_size=64)
    population.append(ind)

first_best = max(p.fitness for p in population)
prev_best = first_best

for gen in range(1, GENS):
    population.sort()
    elites = population[:ELITES]
    offspring = []
    for i in range(POP - ELITES):
        child = random.choice(elites).crossover(random.choice(elites))
        child.individual_id = f'{gen}_{i}'
        child.generation = gen
        child.mutate(mutation_rate=0.1, mutation_strength=0.05)
        offspring.append(child)
    population = elites + offspring
    for ind in population:
        if ind.fitness is None:
            tr = ModelTrainer(ind.model, learning_rate=0.01)
            tr.train(X13, y13, epochs=20, batch_size=64)
            ind.evaluate_fitness(X13, y13, batch_size=64)

gen_best = max(p.fitness for p in population)
check(f"Trained GA: best fitness after {GENS} gens ≥ initial best (elitism holds)",
      gen_best >= first_best,
      f"initial={first_best:.4f}  final={gen_best:.4f}")

# Verify training actually converged: final loss must be far below an untrained baseline
untrained = Individual(model_type='cnn', output_features=3)
untrained_loss = -untrained.evaluate_fitness(X13, y13, batch_size=64)
final_loss = -gen_best
check("Trained GA: final loss is well below untrained random baseline",
      final_loss < untrained_loss * 0.5,
      f"untrained_loss={untrained_loss:.4f}  final_trained_loss={final_loss:.4f}")

# ── Test 14: GA loop — untrained NE makes progress ───────────────────────────
section("14. Untrained GA Loop — Evolution Reduces Loss")

torch.manual_seed(0)
random.seed(0)
X14, y14 = synthetic_batch(300, 3)

population14 = []
for i in range(POP):
    ind = Individual(model_type='cnn', output_features=3, individual_id=f'0_{i}')
    ind.evaluate_fitness(X14, y14, batch_size=64)
    population14.append(ind)

first_best14 = max(p.fitness for p in population14)

for gen in range(1, 8):  # more gens since no training — pure selection pressure
    population14.sort()
    elites = population14[:ELITES]
    offspring = []
    for i in range(POP - ELITES):
        child = random.choice(elites).crossover(random.choice(elites))
        child.individual_id = f'{gen}_{i}'
        child.generation = gen
        child.mutate(mutation_rate=0.2, mutation_strength=0.15)
        offspring.append(child)
    population14 = elites + offspring
    for ind in population14:
        if ind.fitness is None:
            ind.evaluate_fitness(X14, y14, batch_size=64)

final_best14 = max(p.fitness for p in population14)
check("Untrained GA: best fitness does not degrade (elitism holds)",
      final_best14 >= first_best14,
      f"initial={first_best14:.4f}  final={final_best14:.4f}")

# ── Test 15: load_json with real data ────────────────────────────────────────
section("15. load_json — Real Data File")

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'Data', 'datasets', 'training.json')
if os.path.exists(DATA_FILE):
    X_real, y_cmd, y_cont, y_mot = load_json(DATA_FILE, subsample=500)
    check("Real data: X shape (500, 4097)",   X_real.shape == (500, 4097), str(X_real.shape))
    check("Real data: y_command shape (500,3)", y_cmd.shape == (500, 3),   str(y_cmd.shape))
    check("Real data: y_continuous shape (500,2)", y_cont.shape == (500, 2), str(y_cont.shape))
    check("Real data: y_motors shape (500,4)", y_mot.shape == (500, 4),   str(y_mot.shape))
    check("Real data: image pixels in [0,1]",
          float(X_real[:, :4096].min()) >= 0.0 and float(X_real[:, :4096].max()) <= 1.0,
          f"min={float(X_real[:,:4096].min()):.4f}  max={float(X_real[:,:4096].max()):.4f}")
    check("Real data: speed_scalar in [0.1, 1.0]",
          float(X_real[:, 4096].min()) >= 0.09 and float(X_real[:, 4096].max()) <= 1.01,
          f"min={float(X_real[:,4096].min()):.3f}  max={float(X_real[:,4096].max()):.3f}")
    check("Real data: binary commands are 0 or 1",
          ((y_cmd == 0) | (y_cmd == 1)).all().item())
    check("Real data: motor velocities have realistic range",
          float(y_mot.abs().max()) < 200.0,
          f"max_abs={float(y_mot.abs().max()):.1f}")
else:
    print(f"  [SKIP] training.json not found at {DATA_FILE}")
    print(f"         Run Data/prepare_data.py first to enable this test.")

# ── Test 16: load_json subsample ──────────────────────────────────────────────
section("16. load_json Subsample")

# Create a tiny synthetic JSON Lines file for subsample testing
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
    fake_path = tf.name
    for i in range(200):
        rec = {
            'matrix': [i % 256] * 4096,
            'binary_commands': [1, 0, 0, 0],
            'continuous_commands': [0.4, 0.0],
            'motor_velocities': [-50.0, 50.0, -50.0, 50.0],
            'speed_scalar': 0.4,
            'blur_level': 0,
        }
        tf.write(json.dumps(rec) + '\n')

X16_full, _, _, _ = load_json(fake_path)
X16_sub,  _, _, _ = load_json(fake_path, subsample=50)
os.unlink(fake_path)

check("load_json: full load returns all 200 records", X16_full.shape[0] == 200)
check("load_json: subsample=50 returns exactly 50 records", X16_sub.shape[0] == 50)
check("load_json: subsample tensor is (50, 4097)", X16_sub.shape == (50, 4097))

# ── Test 17: save_to_csv / load_csv round-trip ───────────────────────────────
section("17. save_to_csv / load_csv Round-trip")

with tempfile.TemporaryDirectory() as tmpdir:
    data17 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path17 = os.path.join(tmpdir, 'test.csv')
    save_to_csv(data17, path17)
    loaded17 = load_csv(path17)
    check("CSV round-trip: values match", torch.allclose(data17, loaded17))
    check("CSV round-trip: shape preserved", data17.shape == loaded17.shape)

    # 1D list
    data17b = [0.1, 0.2, 0.3, 0.4]
    path17b = os.path.join(tmpdir, 'test1d.csv')
    save_to_csv(data17b, path17b)
    loaded17b = load_csv(path17b)
    check("CSV 1D list: reshaped to column",
          loaded17b.shape == (4, 1),
          str(loaded17b.shape))

# ── Test 18: utils.loss() variants ───────────────────────────────────────────
section("18. utils.loss() Variants")

p18 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
t18 = torch.tensor([[2.0, 2.0], [3.0, 5.0]])
# MSE: ((1-2)^2 + (3-3)^2 + (4-5)^2) / 4 = (1+0+0+1)/4 = 0.5  wait:
# MSE = mean((p-t)^2) = (1+0+0+1)/4 = 0.5
mse_val = util_loss(p18, t18, 'mse')
check("utils.loss mse", abs(mse_val - 0.5) < 1e-6, f"got {mse_val}")

mae_val = util_loss(p18, t18, 'mae')
# MAE = mean(|p-t|) = (1+0+0+1)/4 = 0.5
check("utils.loss mae", abs(mae_val - 0.5) < 1e-6, f"got {mae_val}")

rmse_val = util_loss(p18, t18, 'rmse')
check("utils.loss rmse = sqrt(mse)", abs(rmse_val - math.sqrt(0.5)) < 1e-6, f"got {rmse_val}")

try:
    util_loss(p18, t18, 'bad')
    check("utils.loss bad type raises ValueError", False)
except ValueError:
    check("utils.loss bad type raises ValueError", True)

# ── Test 19: Sample-weighted batch evaluation ─────────────────────────────────
section("19. Sample-Weighted Batch Evaluation Correctness")

# Build a dataset where we know the exact MSE
torch.manual_seed(7)
N19 = 130  # not divisible by batch_size=32 to create a partial last batch
X19 = torch.rand(N19, 4097)
y19 = torch.rand(N19, 3)
cnn19 = CreateCNNModel(output_features=3)
cnn19.eval()
with torch.no_grad():
    preds_full = cnn19(X19)
true_mse = torch.nn.MSELoss()(preds_full, y19).item()

tr19 = ModelTrainer(cnn19)
eval_batch = tr19.evaluate(X19, y19, batch_size=32)
check("Sample-weighted evaluate() matches full MSE (tol 1e-5)",
      abs(eval_batch - true_mse) < 1e-5,
      f"full={true_mse:.8f}  batched={eval_batch:.8f}")

# Same for Individual.evaluate_fitness
ind19 = Individual(model_type='cnn', output_features=3)
ind19.model = cnn19
fit_full19  = ind19.evaluate_fitness(X19, y19, batch_size=None)
fit_batch19 = ind19.evaluate_fitness(X19, y19, batch_size=32)
check("Sample-weighted evaluate_fitness() matches full (tol 1e-5)",
      abs(fit_full19 - fit_batch19) < 1e-5,
      f"full={fit_full19:.8f}  batched={fit_batch19:.8f}")

# ── Test 20: Full mini-pipeline smoke test ────────────────────────────────────
section("20. Full Mini-Pipeline Smoke Test (2 gens, 4 individuals, 3 output types)")

with tempfile.TemporaryDirectory() as tmpdir:
    torch.manual_seed(1)
    random.seed(1)
    X20, y_cmd20 = synthetic_batch(256, 3)
    _, y_cont20 = synthetic_batch(256, 2)
    _, y_mot20  = synthetic_batch(256, 4)

    for label, y20, out_dim in [
        ('ycommand',    y_cmd20,  3),
        ('ycontinuous', y_cont20, 2),
        ('ymotors',     y_mot20,  4),
    ]:
        pop = []
        for i in range(4):
            ind = Individual(model_type='cnn', output_features=out_dim,
                             individual_id=f'0_{i}')
            tr = ModelTrainer(ind.model, learning_rate=0.01)
            tr.train(X20, y20, epochs=10, batch_size=64)
            ind.evaluate_fitness(X20, y20, batch_size=64)
            pop.append(ind)

        gen0_best = max(p.fitness for p in pop)

        pop.sort()
        elites = pop[:2]
        offspring = []
        for i in range(2):
            child = random.choice(elites).crossover(random.choice(elites))
            child.individual_id = f'1_{i}'
            child.generation = 1
            child.mutate(mutation_rate=0.1, mutation_strength=0.05)
            offspring.append(child)
        pop = elites + offspring
        for ind in pop:
            if ind.fitness is None:
                tr = ModelTrainer(ind.model, learning_rate=0.01)
                tr.train(X20, y20, epochs=10, batch_size=64)
                ind.evaluate_fitness(X20, y20, batch_size=64)

        gen1_best = max(p.fitness for p in pop)

        pop.sort()
        best = pop[0]
        save_path = os.path.join(tmpdir, f'best_{label}')
        best.save(save_path)

        loaded = Individual(model_type='cnn', output_features=out_dim)
        loaded.load(save_path + '.pth')
        out_saved  = best.model.test(X20[:8])
        out_loaded = loaded.model.test(X20[:8])
        check(f"Pipeline {label}: gen1 best ≥ gen0 best",
              gen1_best >= gen0_best,
              f"gen0={gen0_best:.4f}  gen1={gen1_best:.4f}")
        check(f"Pipeline {label}: save/load produces identical outputs",
              torch.allclose(out_saved, out_loaded))

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
total  = len(_results)
passed = sum(1 for _, ok in _results if ok)
failed = total - passed
print(f"\n  {passed}/{total} tests passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    print("\n  Failed tests:")
    for name, ok in _results:
        if not ok:
            print(f"    ✗  {name}")
else:
    print("  — all OK")
print()

sys.exit(0 if failed == 0 else 1)
