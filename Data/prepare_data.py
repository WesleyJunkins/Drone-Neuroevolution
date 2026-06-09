"""
prepare_data.py — Full data preparation pipeline for wesDroneRL2.

Steps:
  1.  Analyze raw training_data.jsonl (record counts, timing, speed/command distribution)
  2.  Temporal shift, clean, balance, split 80/20, apply Gaussian blur
      → training.json, testing.json, all_data.json
  3.  Verify output file correctness (structural checks + blur sanity)
  4a. Visualize split distribution → split_distribution.png (7-panel figure)
  4b. Generate dataset visualizations:
        dataset_heatmaps.png     command_distribution.png
        blur_level_samples.png   speed_distribution.png
        motor_velocity_dist.png  blur_speed_stacked.png

Run from any directory:
  python3 Data/prepare_data.py
Inputs and outputs live in Data/datasets/ (training_data.jsonl in, the three
JSON datasets + visualizations/ out).
"""

import json
import os
import sys
import statistics
from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────
# This script lives in <repo root>/Data/. The raw pilot data and all outputs
# live in Data/datasets/. Put training_data.jsonl in Data/datasets/ and run;
# training.json, testing.json, all_data.json (+ visualizations/) are written there.
DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(DIR, 'datasets')
RAW_FILE   = os.path.join(DATA_DIR, 'training_data.jsonl')
TRAIN_FILE = os.path.join(DATA_DIR, 'training.json')
TEST_FILE  = os.path.join(DATA_DIR, 'testing.json')
ALL_FILE   = os.path.join(DATA_DIR, 'all_data.json')
VIZ_DIR    = os.path.join(DATA_DIR, 'visualizations')

# ── Config ────────────────────────────────────────────────────────────────────
BLUR_SIGMAS  = [0, 1, 2, 3, 4]
N_BLUR       = len(BLUR_SIGMAS)
TRAIN_RATIO  = 0.8
MAX_MOTOR    = 150.0
TEMPORAL_LAG = 7            # 7 steps × 32 ms/step = 224 ms visuomotor reaction time

# all_data.json is purpose-built for the notrain NE experiments.
# Stratified: ALL_DATA_PER_CELL records per (speed × blur) cell, in file order.
# 800/cell × 5 speeds × 5 blur levels = 20,000 total — balanced, deterministic,
# small enough for 3 simultaneous MPS processes without GPU command buffer errors.
ALL_DATA_PER_CELL = 800

SPEEDS       = [0.2, 0.4, 0.6, 0.8, 1.0]
BLUR_LEVELS  = [0, 1, 2, 3, 4]

PASS_STR     = "\033[92mPASS\033[0m"
FAIL_STR     = "\033[91mFAIL\033[0m"
TRAIN_COLOR  = '#4C8BF5'
TEST_COLOR   = '#F5A623'
BLUR_LABELS  = ['0\n(none)', '1\n(light)', '2\n(mod)', '3\n(heavy)', '4\n(max)']
SPEED_LABELS = ['0.2', '0.4', '0.6', '0.8', '1.0']


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    with open(path, 'w', buffering=1) as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"  Wrote {len(records):,} records → {path}")


def split_into_sessions(records):
    if not records:
        return []
    sessions, current = [], [records[0]]
    for i in range(1, len(records)):
        if records[i]['timestep'] != records[i - 1]['timestep'] + 1:
            sessions.append(current)
            current = [records[i]]
        else:
            current.append(records[i])
    sessions.append(current)
    return sessions


def find_sessions_by_timestamp(records):
    starts = [0]
    for i in range(1, len(records)):
        if records[i]['timestamp'] < records[i - 1]['timestamp']:
            starts.append(i)
    ends = starts[1:] + [len(records)]
    return list(zip(starts, ends))


def apply_temporal_shift(records, lag):
    if lag == 0:
        return records
    sessions = split_into_sessions(records)
    shifted  = []
    for session in sessions:
        if len(session) <= lag:
            continue
        for i in range(len(session) - lag):
            img_rec = session[i]
            cmd_rec = session[i + lag]
            new_r   = dict(img_rec)
            new_r['binary_commands']     = cmd_rec['binary_commands']
            new_r['continuous_commands'] = cmd_rec['continuous_commands']
            new_r['motor_velocities']    = cmd_rec['motor_velocities']
            shifted.append(new_r)
    return shifted


def is_clean(r):
    m = r['matrix']
    if min(m) == max(m):
        return False
    if any(abs(v) > MAX_MOTOR for v in r['motor_velocities']):
        return False
    return True


def apply_blur(matrix_flat, sigma):
    if sigma == 0:
        return matrix_flat
    img     = np.array(matrix_flat, dtype=np.float32).reshape(64, 64)
    blurred = gaussian_filter(img, sigma=sigma)
    return [int(round(float(v))) for v in blurred.flatten()]


def assign_and_blur(records_for_speed, n_per_blur):
    out = []
    for i, r in enumerate(records_for_speed):
        blur_idx     = i // n_per_blur
        sigma        = BLUR_SIGMAS[blur_idx]
        new_r        = dict(r)
        new_r['matrix']     = apply_blur(r['matrix'], sigma)
        new_r['blur_level'] = blur_idx
        out.append(new_r)
    return out


def speed_blur_grid(records):
    g = defaultdict(int)
    for r in records:
        g[(r['speed_scalar'], r['blur_level'])] += 1
    return g


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Analyze raw data
# ─────────────────────────────────────────────────────────────────────────────

def step1_analyze(records):
    section("STEP 1 — ANALYZE RAW DATA  (training_data.jsonl)")
    sessions = find_sessions_by_timestamp(records)
    total    = len(records)

    print(f"  File             : {RAW_FILE}")
    print(f"  Total records    : {total:,}")
    print(f"  Total sessions   : {len(sessions)}")
    total_dur = sum(
        records[e - 1]['timestamp'] - records[s]['timestamp']
        for s, e in sessions
    )
    print(f"  Total flight time: {total_dur:.1f}s  ({total_dur / 60:.1f} min)")

    dts = [records[i + 1]['timestamp'] - records[i]['timestamp']
           for i in range(total - 1)
           if records[i + 1]['timestamp'] > records[i]['timestamp']]
    if dts:
        print(f"\n  dt per record  : min={min(dts)*1000:.1f}ms  "
              f"max={max(dts)*1000:.1f}ms  "
              f"mean={statistics.mean(dts)*1000:.1f}ms  (expected 32ms)")

    ts_gaps  = [records[i + 1]['timestep'] - records[i]['timestep']
                for i in range(total - 1)
                if records[i + 1]['timestamp'] > records[i]['timestamp']]
    bad_gaps = [g for g in ts_gaps if g != 1]
    print(f"  Dropped packets  : {len(bad_gaps)}  (timestep gaps != 1)")

    first_ts = [records[s]['timestep'] for s, _ in sessions]
    gate_ok  = all(ts >= 500 for ts in first_ts)
    print(f"  First timestep per session: {first_ts}")
    print(f"  Startup gate (>=500): {'YES' if gate_ok else 'NO'}")

    print()
    speed_counts = Counter(r['speed_scalar'] for r in records)
    max_count    = max(speed_counts.values())
    for spd in sorted(speed_counts):
        count = speed_counts[spd]
        bar   = '█' * int(40 * count / max_count)
        pct   = 100 * count / total
        print(f"  {spd:.1f}  {count:6,} records ({pct:5.1f}%)  {bar}")
    if len(speed_counts) > 1:
        counts    = list(speed_counts.values())
        imbalance = max(counts) / min(counts)
        print(f"  Balance ratio (max/min): {imbalance:.2f}x  "
              f"({'OK' if imbalance < 2 else 'UNBALANCED'})")

    fwd   = sum(1 for r in records if r['binary_commands'][0])
    yl    = sum(1 for r in records if r['binary_commands'][1])
    yr    = sum(1 for r in records if r['binary_commands'][2])
    idle  = sum(1 for r in records if r['binary_commands'] == [0, 0, 0, 0])
    multi = sum(1 for r in records if sum(r['binary_commands'][:3]) > 1)
    print(f"\n  Commands: fwd={fwd:,}  yaw-left={yl:,}  yaw-right={yr:,}  "
          f"idle={idle:,}  multi={multi:,}")
    print(f"  (Track is clockwise — yaw right dominance expected)")

    uniform   = sum(1 for r in records if min(r['matrix']) == max(r['matrix']))
    sample_px = [v for r in records[:500] for v in r['matrix']]
    print(f"\n  Uniform images   : {uniform}/{total}")
    print(f"  Pixel range (500-record sample): {min(sample_px)} – {max(sample_px)}")

    all_motors = [v for r in records for v in r['motor_velocities']]
    extreme    = sum(1 for v in all_motors if abs(v) > 150)
    print(f"  Motor range      : {min(all_motors):.2f} to {max(all_motors):.2f}  "
          f"mean={statistics.mean(all_motors):.2f}  spikes={extreme}")

    print(f"\n  {'#':>3}  {'records':>7}  {'dur(s)':>7}  {'ts range':>14}  "
          f"{'speed':>6}  {'fwd':>5}  {'yl':>5}  {'yr':>5}")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*14}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for idx, (s, e) in enumerate(sessions):
        recs = records[s:e]
        dur  = recs[-1]['timestamp'] - recs[0]['timestamp']
        sfwd = sum(1 for r in recs if r['binary_commands'][0])
        syl  = sum(1 for r in recs if r['binary_commands'][1])
        syr  = sum(1 for r in recs if r['binary_commands'][2])
        spds = sorted(set(r['speed_scalar'] for r in recs))
        ts0  = recs[0]['timestep']
        ts1  = recs[-1]['timestep']
        print(f"  {idx + 1:>3}  {len(recs):>7,}  {dur:>7.1f}  "
              f"{ts0:>6}..{ts1:<6}  {str(spds):>6}  "
              f"{sfwd:>5}  {syl:>5}  {syr:>5}")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — Clean, balance, split, blur, write
# ─────────────────────────────────────────────────────────────────────────────

def step2_clean_and_build(raw):
    section("STEP 2 — TEMPORAL SHIFT  (per-session, per-speed)")
    sessions_raw = split_into_sessions(raw)
    n_sessions   = len(sessions_raw)

    print(f"  Sessions detected : {n_sessions}")
    for i, s in enumerate(sessions_raw):
        spds = sorted(set(r['speed_scalar'] for r in s))
        print(f"    Session {i + 1:2d}: {len(s):>6,} records  speed={spds}"
              f"  timestep {s[0]['timestep']}–{s[-1]['timestep']}")
    print(f"\n  Temporal lag      : {TEMPORAL_LAG} steps × 32 ms = "
          f"{TEMPORAL_LAG * 32} ms  (human choice reaction time lower bound)")
    print(f"  Records dropped   : {n_sessions * TEMPORAL_LAG} total "
          f"({TEMPORAL_LAG} per session)")

    raw = apply_temporal_shift(raw, TEMPORAL_LAG)
    print(f"  Records after shift: {len(raw):,}")

    mixed = sum(1 for s in sessions_raw if len(set(r['speed_scalar'] for r in s)) > 1)
    if mixed:
        print(f"  WARNING: {mixed} session(s) contain multiple speeds.")
    else:
        print(f"  Verified: all {n_sessions} sessions are single-speed.")

    section("STEP 3 — CLEAN")
    records = [r for r in raw if is_clean(r)]
    print(f"  Removed (uniform images / motor spikes): {len(raw) - len(records):,}")
    print(f"  Clean records: {len(records):,}")

    section("STEP 4 — BALANCE BY SPEED")
    by_speed = defaultdict(list)
    for r in records:
        by_speed[r['speed_scalar']].append(r)
    speeds = sorted(by_speed)
    print(f"  Speeds found: {speeds}")
    for spd in speeds:
        print(f"    {spd:.1f}: {len(by_speed[spd]):>8,} records")

    chunk          = N_BLUR * 10
    raw_min        = min(len(by_speed[s]) for s in speeds)
    balanced_count = (raw_min // chunk) * chunk
    if balanced_count == 0:
        print(f"\n  ERROR: min speed count ({raw_min}) too small (need >= {chunk}).")
        sys.exit(1)
    trimmed = sum(len(by_speed[s]) - balanced_count for s in speeds)
    print(f"\n  Min raw count   : {raw_min:,}")
    print(f"  Balanced count  : {balanced_count:,} per speed  (rounded to multiple of {chunk})")
    print(f"  Records trimmed : {trimmed:,}")
    for spd in speeds:
        by_speed[spd] = by_speed[spd][:balanced_count]

    section("STEP 5 — TRAIN / TEST SPLIT  (80 / 20 of ALL_DATA_PER_CELL per speed)")
    # All three output files share the same ALL_DATA_PER_CELL-record pool per (speed × blur) cell.
    # training + testing = all_data (invariant preserved; sizes stay manageable on GPU).
    n_per_blur_all   = ALL_DATA_PER_CELL                        # 800
    n_per_blur_train = int(ALL_DATA_PER_CELL * TRAIN_RATIO)     # 640
    n_per_blur_test  = ALL_DATA_PER_CELL - n_per_blur_train     # 160
    n_all_per_speed  = N_BLUR * n_per_blur_all                  # 4,000
    n_train_per_speed = N_BLUR * n_per_blur_train               # 3,200
    n_test_per_speed  = N_BLUR * n_per_blur_test                # 800

    if n_all_per_speed > balanced_count:
        print(f"\n  ERROR: ALL_DATA_PER_CELL={ALL_DATA_PER_CELL} requires {n_all_per_speed} "
              f"records/speed but only {balanced_count} available.")
        sys.exit(1)

    print(f"  ALL_DATA_PER_CELL = {ALL_DATA_PER_CELL}  "
          f"(train: {n_per_blur_train}/cell  test: {n_per_blur_test}/cell  all: {n_per_blur_all}/cell)")
    print(f"  Per speed  — train: {n_train_per_speed:,}   test: {n_test_per_speed:,}   all_data: {n_all_per_speed:,}")
    print(f"  Total      — train: {n_train_per_speed * len(speeds):,}   "
          f"test: {n_test_per_speed * len(speeds):,}   "
          f"all_data: {n_all_per_speed * len(speeds):,}  (train+test=all ✓)")

    section("STEP 6 — APPLY GAUSSIAN BLUR")
    print(f"  Blur sigmas: {BLUR_SIGMAS}")
    train_out, test_out, all_out = [], [], []
    for spd in speeds:
        recs = by_speed[spd]
        print(f"  Blurring speed {spd:.1f} ...", end=' ', flush=True)
        train_out.extend(assign_and_blur(recs[:n_train_per_speed],                         n_per_blur_train))
        test_out.extend(assign_and_blur(recs[n_train_per_speed:n_all_per_speed],           n_per_blur_test))
        all_out.extend(assign_and_blur(recs[:n_all_per_speed],                             n_per_blur_all))
        print("done")

    section("STEP 7 — WRITE OUTPUT FILES")
    write_jsonl(TRAIN_FILE, train_out)
    write_jsonl(TEST_FILE,  test_out)
    write_jsonl(ALL_FILE,   all_out)
    print(f"\n  SUMMARY")
    print(f"    training.json : {len(train_out):,}  ({n_per_blur_train:,}/cell × "
          f"{len(speeds)} speeds × {N_BLUR} blur levels)")
    print(f"    testing.json  : {len(test_out):,}   ({n_per_blur_test:,}/cell × "
          f"{len(speeds)} speeds × {N_BLUR} blur levels)")
    print(f"    all_data.json : {len(all_out):,}  ({n_per_blur_all:,}/cell × "
          f"{len(speeds)} speeds × {N_BLUR} blur levels  ← train+test)")
    print(f"    training + testing = {len(train_out)+len(test_out):,}  "
          f"(= all_data: {len(all_out):,} ✓)")
    print(f"\n  Original {RAW_FILE} was NOT modified.")

    return train_out, test_out, all_out


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — Verify
# ─────────────────────────────────────────────────────────────────────────────

def step3_verify(datasets):
    section("STEP 8 — VERIFICATION CHECKS")
    results = []

    def check(name, condition, detail=""):
        status = PASS_STR if condition else FAIL_STR
        msg    = f"  [{status}] {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        results.append((name, condition))

    for name, records in datasets.items():
        speed_counts = Counter(round(r['speed_scalar'], 1) for r in records)
        blur_counts  = Counter(r['blur_level'] for r in records)
        speed_vals   = [speed_counts.get(s, 0) for s in SPEEDS]
        blur_vals    = [blur_counts.get(b, 0) for b in BLUR_LEVELS]

        check(f"{name}: all speeds present",
              set(round(k, 1) for k in speed_counts) == set(SPEEDS),
              str(dict(sorted(speed_counts.items()))))
        check(f"{name}: all speeds equal count",
              len(set(speed_vals)) == 1,
              f"counts={speed_vals}")
        check(f"{name}: all blur levels present",
              set(blur_counts.keys()) == set(BLUR_LEVELS),
              str(dict(sorted(blur_counts.items()))))
        check(f"{name}: all blur levels equal count",
              len(set(blur_vals)) == 1,
              f"counts={blur_vals}")

        cell_counts = defaultdict(lambda: defaultdict(int))
        for r in records:
            cell_counts[round(r['speed_scalar'], 1)][r['blur_level']] += 1
        cell_vals = [cell_counts[s][b] for s in SPEEDS for b in BLUR_LEVELS]
        check(f"{name}: all speed×blur cells equal count",
              len(set(cell_vals)) == 1,
              f"cell_count={cell_vals[0]}, n_cells={len(cell_vals)}")

        r0 = records[0]
        check(f"{name}: required fields present",
              all(k in r0 for k in ['matrix', 'binary_commands', 'continuous_commands',
                                     'motor_velocities', 'speed_scalar', 'blur_level']))
        check(f"{name}: image is 4096 values", len(r0['matrix']) == 4096)

        sample_px = [v for r in records[:100] for v in r['matrix']]
        check(f"{name}: pixels in [0,255]",
              min(sample_px) >= 0 and max(sample_px) <= 255,
              f"min={min(sample_px)}, max={max(sample_px)}")

        n_uniform = sum(1 for r in records if min(r['matrix']) == max(r['matrix']))
        check(f"{name}: no uniform images", n_uniform == 0, f"{n_uniform} found")

        n_spikes = sum(1 for r in records if any(abs(v) > 150 for v in r['motor_velocities']))
        check(f"{name}: no motor spikes", n_spikes == 0, f"{n_spikes} found")
        print()

    n_train = len(datasets['training'])
    n_test  = len(datasets['testing'])
    n_all   = len(datasets['all_data'])
    check("training + testing == all_data count",
          n_train + n_test == n_all,
          f"{n_train}+{n_test}={n_train + n_test}  all_data={n_all}")

    blur0_vars = [np.var(r['matrix'])
                  for r in datasets['training'] if r['blur_level'] == 0][:50]
    blur4_vars = [np.var(r['matrix'])
                  for r in datasets['training'] if r['blur_level'] == 4][:50]
    check("training: blur-4 images have lower variance than blur-0",
          np.mean(blur4_vars) < np.mean(blur0_vars),
          f"mean_var blur0={np.mean(blur0_vars):.1f}  blur4={np.mean(blur4_vars):.1f}")

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n  CORRECTNESS: {passed}/{total} checks passed")
    if passed < total:
        print("  Failed checks:")
        for name, ok in results:
            if not ok:
                print(f"    ✗  {name}")

    print()
    for name, records in datasets.items():
        n            = len(records)
        speed_counts = Counter(round(r['speed_scalar'], 1) for r in records)
        blur_counts  = Counter(r['blur_level'] for r in records)
        cmd_fwd      = sum(1 for r in records if r['binary_commands'][0])
        cmd_yl       = sum(1 for r in records if r['binary_commands'][1])
        cmd_yr       = sum(1 for r in records if r['binary_commands'][2])
        all_motors   = [v for r in records for v in r['motor_velocities']]
        print(f"  {name} ({n:,} records)")
        print(f"    Speed: {dict(sorted(speed_counts.items()))}")
        print(f"    Blur:  {dict(sorted(blur_counts.items()))}")
        print(f"    Commands fwd/yl/yr: {cmd_fwd:,}/{cmd_yl:,}/{cmd_yr:,}")
        print(f"    Motor: min={min(all_motors):.1f}  max={max(all_motors):.1f}  "
              f"mean={np.mean(all_motors):.2f}  std={np.std(all_motors):.2f}")

    return passed, total


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4a — Split distribution visualization (7-panel figure)
# ─────────────────────────────────────────────────────────────────────────────

def step4a_visualize_split(train, test):
    section("STEP 9 — VISUALIZE SPLIT DISTRIBUTION")
    os.makedirs(VIZ_DIR, exist_ok=True)
    out_file = os.path.join(VIZ_DIR, 'split_distribution.png')

    tg    = speed_blur_grid(train)
    eg    = speed_blur_grid(test)
    total = len(train) + len(test)
    x     = np.arange(len(SPEEDS))

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Training Data — Split & Blur Distribution',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie([len(train), len(test)],
            labels=[f'Train\n{len(train):,}', f'Test\n{len(test):,}'],
            colors=[TRAIN_COLOR, TEST_COLOR],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Train / Test Split', fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    train_by_speed = [sum(tg[(s, b)] for b in BLUR_LEVELS) for s in SPEEDS]
    test_by_speed  = [sum(eg[(s, b)] for b in BLUR_LEVELS) for s in SPEEDS]
    bars_tr = ax2.bar(x, train_by_speed, color=TRAIN_COLOR, label='Train')
    ax2.bar(x, test_by_speed, bottom=train_by_speed, color=TEST_COLOR, label='Test')
    ax2.set_xticks(x); ax2.set_xticklabels(SPEED_LABELS)
    ax2.set_xlabel('Speed Scalar'); ax2.set_ylabel('Records')
    ax2.set_title('Records per Speed', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    for bar, val in zip(bars_tr, train_by_speed):
        ax2.text(bar.get_x() + bar.get_width() / 2, val / 2, f'{val:,}',
                 ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])
    train_by_blur = [sum(tg[(s, b)] for s in SPEEDS) for b in BLUR_LEVELS]
    test_by_blur  = [sum(eg[(s, b)] for s in SPEEDS) for b in BLUR_LEVELS]
    bars_tr2 = ax3.bar(x, train_by_blur, color=TRAIN_COLOR, label='Train')
    ax3.bar(x, test_by_blur, bottom=train_by_blur, color=TEST_COLOR, label='Test')
    ax3.set_xticks(x); ax3.set_xticklabels(BLUR_LABELS)
    ax3.set_xlabel('Blur Level (sigma)'); ax3.set_ylabel('Records')
    ax3.set_title('Records per Blur Level', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    for bar, val in zip(bars_tr2, train_by_blur):
        ax3.text(bar.get_x() + bar.get_width() / 2, val / 2, f'{val:,}',
                 ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 0:2])
    heat_train = np.array([[tg[(s, b)] for b in BLUR_LEVELS] for s in SPEEDS])
    im4 = ax4.imshow(heat_train, aspect='auto', cmap='Blues')
    ax4.set_xticks(range(len(BLUR_LEVELS))); ax4.set_xticklabels(BLUR_LABELS)
    ax4.set_yticks(range(len(SPEEDS)));      ax4.set_yticklabels(SPEED_LABELS)
    ax4.set_xlabel('Blur Level (sigma)'); ax4.set_ylabel('Speed Scalar')
    ax4.set_title('Training Set — Speed × Blur Counts', fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Records')
    for i, s in enumerate(SPEEDS):
        for j, b in enumerate(BLUR_LEVELS):
            ax4.text(j, i, f'{tg[(s, b)]:,}', ha='center', va='center',
                     fontsize=8,
                     color='white' if tg[(s, b)] > heat_train.max() * 0.5 else 'black')

    ax5 = fig.add_subplot(gs[1, 2])
    heat_test = np.array([[eg[(s, b)] for b in BLUR_LEVELS] for s in SPEEDS])
    im5 = ax5.imshow(heat_test, aspect='auto', cmap='Oranges')
    ax5.set_xticks(range(len(BLUR_LEVELS))); ax5.set_xticklabels(BLUR_LABELS)
    ax5.set_yticks(range(len(SPEEDS)));      ax5.set_yticklabels(SPEED_LABELS)
    ax5.set_xlabel('Blur Level (sigma)'); ax5.set_ylabel('Speed Scalar')
    ax5.set_title('Testing Set — Speed × Blur Counts', fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='Records')
    for i, s in enumerate(SPEEDS):
        for j, b in enumerate(BLUR_LEVELS):
            ax5.text(j, i, f'{eg[(s, b)]:,}', ha='center', va='center',
                     fontsize=8,
                     color='white' if eg[(s, b)] > heat_test.max() * 0.5 else 'black')

    ax6 = fig.add_subplot(gs[2, 0:2])
    bar_w = 0.15
    for bi, b in enumerate(BLUR_LEVELS):
        vals   = [tg[(s, b)] for s in SPEEDS]
        offset = (bi - 2) * bar_w
        ax6.bar(np.arange(len(SPEEDS)) + offset, vals, width=bar_w,
                label=f'Blur {b}', alpha=0.85)
    ax6.set_xticks(np.arange(len(SPEEDS))); ax6.set_xticklabels(SPEED_LABELS)
    ax6.set_xlabel('Speed Scalar'); ax6.set_ylabel('Records')
    ax6.set_title('Training — Blur Level Breakdown per Speed', fontweight='bold')
    ax6.legend(title='Blur level', fontsize=8, ncol=5)
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    lines = [
        f"Total records   : {total:,}",
        f"  Training      : {len(train):,}  ({100*len(train)/total:.0f}%)",
        f"  Testing       : {len(test):,}  ({100*len(test)/total:.0f}%)",
        "",
        f"Speeds          : {len(SPEEDS)}",
        f"Blur levels     : {len(BLUR_LEVELS)}  (σ = 0–4)",
        "",
        f"Train per speed : {len(train) // len(SPEEDS):,}",
        f"Test  per speed : {len(test) // len(SPEEDS):,}",
        "",
        f"Train per blur  : {len(train) // len(BLUR_LEVELS):,}",
        f"Test  per blur  : {len(test) // len(BLUR_LEVELS):,}",
        "",
        f"Per (speed×blur): {len(train) // (len(SPEEDS) * len(BLUR_LEVELS)):,} train",
        f"                  {len(test) // (len(SPEEDS) * len(BLUR_LEVELS)):,} test",
    ]
    ax7.text(0.05, 0.95, '\n'.join(lines), transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax7.set_title('Summary', fontweight='bold')

    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4b — Dataset visualizations (6 figures)
# ─────────────────────────────────────────────────────────────────────────────

def step4b_visualize_datasets(datasets):
    section("STEP 10 — GENERATE DATASET VISUALIZATIONS")
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Figure 1: Speed × Blur heatmaps (one per dataset)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Speed × Blur Level Cell Counts', fontsize=14, fontweight='bold')
    cmaps = ['Blues', 'Oranges', 'Greens']
    for ax, (name, records), cmap in zip(axes, datasets.items(), cmaps):
        matrix = np.zeros((len(SPEEDS), len(BLUR_LEVELS)), dtype=int)
        for r in records:
            si = SPEEDS.index(round(r['speed_scalar'], 1))
            matrix[si, r['blur_level']] += 1
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(BLUR_LEVELS)))
        ax.set_xticklabels([f'σ={s}' for s in BLUR_SIGMAS])
        ax.set_yticks(range(len(SPEEDS)))
        ax.set_yticklabels([f'{s:.1f}' for s in SPEEDS])
        ax.set_xlabel('Blur Level (sigma)'); ax.set_ylabel('Speed Scalar')
        ax.set_title(f'{name}\n({len(records):,} records)')
        for i in range(len(SPEEDS)):
            for j in range(len(BLUR_LEVELS)):
                ax.text(j, i, f'{matrix[i, j]:,}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    p = os.path.join(VIZ_DIR, 'dataset_heatmaps.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    # Figure 2: Command distribution bar charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Command Distribution per Dataset', fontsize=14, fontweight='bold')
    cmd_labels = ['Forward', 'Yaw Left', 'Yaw Right', 'Idle']
    colors     = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for ax, (name, records) in zip(axes, datasets.items()):
        n    = len(records)
        vals = [
            sum(1 for r in records if r['binary_commands'][0]
                and not r['binary_commands'][1] and not r['binary_commands'][2]),
            sum(1 for r in records if r['binary_commands'][1]),
            sum(1 for r in records if r['binary_commands'][2]),
            sum(1 for r in records if not any(r['binary_commands'][:3])),
        ]
        bars = ax.bar(cmd_labels, [v / n * 100 for v in vals], color=colors)
        ax.set_ylabel('% of records'); ax.set_title(f'{name}\n({n:,} records)')
        ax.set_ylim(0, 100)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{v:,}\n({v / n * 100:.1f}%)', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    p = os.path.join(VIZ_DIR, 'command_distribution.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    # Figure 3: Sample images at each blur level
    all_recs    = datasets['all_data']
    sample_recs = {}
    for r in all_recs:
        bl  = r['blur_level']
        spd = round(r['speed_scalar'], 1)
        if spd == 0.6 and bl not in sample_recs:
            sample_recs[bl] = r
        if len(sample_recs) == 5:
            break
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle('Sample Images at Each Blur Level (speed=0.6)', fontsize=14, fontweight='bold')
    blur_labels2 = ['Level 0\n(σ=0, no blur)', 'Level 1\n(σ=1, light)',
                    'Level 2\n(σ=2, moderate)', 'Level 3\n(σ=3, heavy)',
                    'Level 4\n(σ=4, maximum)']
    for i, (bl, label) in enumerate(zip(BLUR_LEVELS, blur_labels2)):
        ax = axes[i]
        if bl in sample_recs:
            img = np.array(sample_recs[bl]['matrix'], dtype=np.uint8).reshape(64, 64)
            ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            px_rng = f"px: {min(sample_recs[bl]['matrix'])}–{max(sample_recs[bl]['matrix'])}"
            ax.set_title(f'{label}\n{px_rng}', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'no sample', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(label, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    p = os.path.join(VIZ_DIR, 'blur_level_samples.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    # Figure 4: Records per speed per dataset (grouped bar)
    fig, ax = plt.subplots(figsize=(12, 5))
    x, width = np.arange(len(SPEEDS)), 0.25
    bcolors  = ['#4C72B0', '#DD8452', '#55A868']
    for i, (name, records) in enumerate(datasets.items()):
        sc   = Counter(round(r['speed_scalar'], 1) for r in records)
        vals = [sc.get(s, 0) for s in SPEEDS]
        bars = ax.bar(x + i * width, vals, width, label=name, color=bcolors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f'{v:,}', ha='center', va='bottom', fontsize=8, rotation=45)
    ax.set_xlabel('Speed Scalar'); ax.set_ylabel('Record Count')
    ax.set_title('Records per Speed per Dataset')
    ax.set_xticks(x + width); ax.set_xticklabels([str(s) for s in SPEEDS])
    ax.legend()
    ax.set_ylim(0, max(len(datasets['all_data']) // len(SPEEDS) * 1.2, 25000))
    plt.tight_layout()
    p = os.path.join(VIZ_DIR, 'speed_distribution.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    # Figure 5: Motor velocity distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Motor Velocity Distribution per Dataset', fontsize=14, fontweight='bold')
    for ax, (name, records) in zip(axes, datasets.items()):
        motors = [v for r in records for v in r['motor_velocities']]
        ax.hist(motors, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(np.mean(motors), color='red', linestyle='--', linewidth=1.5,
                   label=f'mean={np.mean(motors):.1f}')
        ax.set_xlabel('Motor Velocity (rad/s)'); ax.set_ylabel('Count')
        ax.set_title(f'{name}\n(std={np.std(motors):.1f})'); ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(VIZ_DIR, 'motor_velocity_dist.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    # Figure 6: Blur level × speed stacked bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Records per Blur Level × Speed (stacked bars)', fontsize=14, fontweight='bold')
    blur_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    for ax, (name, records) in zip(axes, datasets.items()):
        by_sb  = defaultdict(lambda: defaultdict(int))
        for r in records:
            by_sb[round(r['speed_scalar'], 1)][r['blur_level']] += 1
        bottom = np.zeros(len(SPEEDS))
        for bl in BLUR_LEVELS:
            vals = [by_sb[s][bl] for s in SPEEDS]
            ax.bar([str(s) for s in SPEEDS], vals, bottom=bottom,
                   label=f'Blur {bl} (σ={BLUR_SIGMAS[bl]})', color=blur_colors[bl])
            bottom += np.array(vals)
        ax.set_xlabel('Speed Scalar'); ax.set_ylabel('Record Count')
        ax.set_title(name); ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    p = os.path.join(VIZ_DIR, 'blur_speed_stacked.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RAW_FILE):
        print(f"ERROR: {RAW_FILE} not found.")
        sys.exit(1)

    print(f"Loading {RAW_FILE} ...")
    raw = load_jsonl(RAW_FILE)
    print(f"  {len(raw):,} records loaded")

    step1_analyze(raw)
    train_out, test_out, all_out = step2_clean_and_build(raw)

    datasets = {
        'training': train_out,
        'testing':  test_out,
        'all_data': all_out,
    }
    passed, total_checks = step3_verify(datasets)
    step4a_visualize_split(train_out, test_out)
    step4b_visualize_datasets(datasets)

    section("COMPLETE")
    print(f"  training.json : {len(train_out):,} records  → {TRAIN_FILE}")
    print(f"  testing.json  : {len(test_out):,} records  → {TEST_FILE}")
    print(f"  all_data.json : {len(all_out):,} records  → {ALL_FILE}")
    print(f"  Checks        : {passed}/{total_checks} passed")
    print(f"  Visualizations: {VIZ_DIR}/")
    print(f"    split_distribution.png")
    print(f"    dataset_heatmaps.png       command_distribution.png")
    print(f"    blur_level_samples.png     speed_distribution.png")
    print(f"    motor_velocity_dist.png    blur_speed_stacked.png")
    print()


if __name__ == '__main__':
    main()
