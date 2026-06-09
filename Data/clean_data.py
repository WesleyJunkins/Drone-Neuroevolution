#!/usr/bin/env python3
"""
clean_data.py — Produce the cleaned datasets from the raw session error logs.

This script lives in <repo root>/Data/. Put the raw results folder
ALL_FINAL_ERROR_RESULTS/ inside this same Data/ folder, then run the script.
It reads every {session}_error_log.csv from there and writes the cleaned data
into Data/ALL_FINAL_ERROR_RESULTS_CLEANED/ — nothing outside Data/ is touched.

(Ported verbatim from the former receivers/clean_results.py; only the input/
output paths were made relative to this Data/ folder, and the output-directory
creation was moved into main() so importing the module has no side effects.)

Both outputs share the same pre-processing:
  1. Restart artifact removal — drop rows before the last timestamp reset
     (timestamp drops > 10 s) within a test_index group.
  2. GPS-explosion filter — set pos_x/y/z/dist_from_center to NaN where
     pos_z is outside [0.3, 3.0] m.

Then two separate outputs:

  FOR_PLOTTING/  (rep 1 only — raw XY paths for path visualisation)
    - test_index 1–25 (repetition 1) extracted per session
    - All 25 configs equalized to the global minimum row count within the session
    - Columns: config_id, speed_scalar, blur_level, row_index,
               timestamp, error, pos_x, pos_y, pos_z,
               dist_from_center, out_0, out_1, out_2

  FOR_ANALYSIS/  (3-rep averaged 3D error — primary analysis metric)
    - All 75 test_indices processed; 3 reps per config averaged row-by-row
    - 3D error = sqrt((dist_2d - TRACK_RADIUS)² + (z - TARGET_Z)²)
      where dist_2d = sqrt((x - CX)² + (y - CY)²)
      GPS-invalid rows → error_3d = NaN; nanmean used so valid reps still contribute
    - All 25 configs equalized to the global minimum row count across all 3 reps
    - Columns: config_id, speed_scalar, blur_level, row_index,
               timestamp, error_px, error_3d, out_0, out_1, out_2

Usage (run from anywhere):
    python3 Data/clean_data.py
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_HERE, 'ALL_FINAL_ERROR_RESULTS')
_OUT_DIR  = os.path.join(_HERE, 'ALL_FINAL_ERROR_RESULTS_CLEANED')
_PLOT_DIR = os.path.join(_OUT_DIR, 'FOR_PLOTTING')
_ANAL_DIR = os.path.join(_OUT_DIR, 'FOR_ANALYSIS')

TEST_SPEEDS      = [0.2, 0.4, 0.6, 0.8, 1.0]
TEST_BLUR_LEVELS = [0, 1, 2, 3, 4]
ALT_MIN, ALT_MAX = 0.3, 3.0

TRACK_CX     = 0.000018
TRACK_CY     = -2.849774
TRACK_RADIUS = 2.845519
TARGET_Z     = 1.0

GPS_COLS = ['pos_x', 'pos_y', 'pos_z', 'dist_from_center']

PLOT_COLS = ['config_id', 'speed_scalar', 'blur_level', 'row_index',
             'timestamp', 'error', 'pos_x', 'pos_y', 'pos_z',
             'dist_from_center', 'out_0', 'out_1', 'out_2']

ANAL_COLS = ['config_id', 'speed_scalar', 'blur_level', 'row_index',
             'timestamp', 'error_px', 'error_3d', 'out_0', 'out_1', 'out_2']


def remove_restart_prefix(group_df):
    """Keep only rows from the last timestamp-reset boundary (timestamp drop > 10 s)."""
    ts = group_df['timestamp'].values
    last_reset = 0
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1] - 10.0:
            last_reset = i
    return group_df.iloc[last_reset:].reset_index(drop=True), last_reset


def error_3d(pos_x, pos_y, pos_z):
    """3D distance from the drone to the hovering circle surface."""
    dist_2d = np.sqrt((pos_x - TRACK_CX) ** 2 + (pos_y - TRACK_CY) ** 2)
    return np.sqrt((dist_2d - TRACK_RADIUS) ** 2 + (pos_z - TARGET_Z) ** 2)


def clean_session(session_name: str) -> bool:
    csv_in = os.path.join(_SRC_DIR, f'{session_name}_error_log.csv')
    df = pd.read_csv(csv_in)
    n_raw = len(df)

    # --- Shared step 1: restart artifact removal ----------------------------
    n_dropped_restart = 0
    parts = []
    for ti, grp in df.groupby('test_index', sort=True):
        clean_grp, n_drop = remove_restart_prefix(grp)
        n_dropped_restart += n_drop
        parts.append(clean_grp)
    df = pd.concat(parts, ignore_index=True)

    # --- Shared step 2: GPS-explosion filter (NaN, not drop) ----------------
    bad = (df['pos_z'] < ALT_MIN) | (df['pos_z'] > ALT_MAX)
    n_bad_gps = int(bad.sum())
    for col in GPS_COLS:
        df.loc[bad, col] = np.nan

    # =========================================================================
    # FOR PLOTTING — rep 1 (test_index 1–25) raw XY paths
    # =========================================================================
    rep1_configs = []
    for si, speed in enumerate(TEST_SPEEDS):
        for bi, blur in enumerate(TEST_BLUR_LEVELS):
            cid = si * 5 + bi + 1
            ti  = cid  # rep 1: test_index equals config_id
            grp = df[df['test_index'] == ti].reset_index(drop=True)
            rep1_configs.append((cid, speed, blur, grp))

    n_plot = min(len(g) for _, _, _, g in rep1_configs if len(g) > 0)

    plot_parts = []
    for cid, speed, blur, grp in rep1_configs:
        trimmed = grp.iloc[:n_plot].copy()
        trimmed['config_id']   = cid
        trimmed['speed_scalar'] = speed
        trimmed['blur_level']  = blur
        trimmed['row_index']   = np.arange(n_plot)
        for col in PLOT_COLS:
            if col not in trimmed.columns:
                trimmed[col] = np.nan
        plot_parts.append(trimmed[PLOT_COLS])

    pd.concat(plot_parts, ignore_index=True).to_csv(
        os.path.join(_PLOT_DIR, f'{session_name}_error_log.csv'),
        index=False, float_format='%.6f',
    )

    # =========================================================================
    # FOR ANALYSIS — 3 reps averaged, 3D hovering-circle error
    # =========================================================================
    # Pre-compute error_3d on every row (NaN where GPS is invalid)
    e3 = error_3d(df['pos_x'].values.astype(float),
                  df['pos_y'].values.astype(float),
                  df['pos_z'].values.astype(float))
    df['error_3d'] = e3  # NaN propagates automatically from NaN inputs

    anal_configs = []
    for si, speed in enumerate(TEST_SPEEDS):
        for bi, blur in enumerate(TEST_BLUR_LEVELS):
            cid = si * 5 + bi + 1
            reps = []
            for rep in range(1, 4):
                ti  = si * 5 + bi + 1 + (rep - 1) * 25
                grp = df[df['test_index'] == ti].reset_index(drop=True)
                reps.append(grp)
            anal_configs.append((cid, speed, blur, reps))

    n_anal = min(len(r) for _, _, _, reps in anal_configs for r in reps if len(r) > 0)

    anal_parts = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slice is expected
        for cid, speed, blur, reps in anal_configs:
            reps = [r.iloc[:n_anal] for r in reps]

            def avg(col):
                return np.nanmean(
                    np.stack([r[col].values for r in reps], axis=0).astype(float),
                    axis=0,
                )

            anal_df = pd.DataFrame({
                'config_id':    cid,
                'speed_scalar': speed,
                'blur_level':   blur,
                'row_index':    np.arange(n_anal),
                'timestamp':    avg('timestamp'),
                'error_px':     avg('error'),
                'error_3d':     avg('error_3d'),
                'out_0':        avg('out_0'),
                'out_1':        avg('out_1'),
                'out_2':        avg('out_2'),
            })
            anal_parts.append(anal_df)

    pd.concat(anal_parts, ignore_index=True)[ANAL_COLS].to_csv(
        os.path.join(_ANAL_DIR, f'{session_name}_error_log.csv'),
        index=False, float_format='%.6f',
    )

    print(f'  {session_name}')
    if n_dropped_restart:
        print(f'    restart rows removed : {n_dropped_restart:,}')
    if n_bad_gps:
        print(f'    GPS-invalid → NaN   : {n_bad_gps:,} ({n_bad_gps / n_raw * 100:.1f}%)')
    print(f'    FOR_PLOTTING : 25 configs × {n_plot} rows (rep 1)')
    print(f'    FOR_ANALYSIS : 25 configs × {n_anal} rows (3-rep avg, error_3d)')
    return True


def main():
    if not os.path.isdir(_SRC_DIR):
        print(f'ERROR: raw results not found at {_SRC_DIR}')
        print('Put the ALL_FINAL_ERROR_RESULTS/ folder inside Data/ and re-run.')
        return

    os.makedirs(_PLOT_DIR, exist_ok=True)
    os.makedirs(_ANAL_DIR, exist_ok=True)

    # Remove any stale flat CSVs left in the root of _OUT_DIR from a previous run
    for stale in glob.glob(os.path.join(_OUT_DIR, '*_error_log.csv')):
        os.remove(stale)

    sessions = sorted(
        f[:-len('_error_log.csv')]
        for f in os.listdir(_SRC_DIR)
        if f.endswith('_error_log.csv')
    )
    if not sessions:
        print(f'No CSVs found in {_SRC_DIR}')
        return

    print(f'Source : {_SRC_DIR}')
    print(f'Output : {_OUT_DIR}')
    print(f'  FOR_PLOTTING/  — rep 1 raw XY paths, GPS NaN-filtered, equalized')
    print(f'  FOR_ANALYSIS/  — 3-rep avg 3D error: sqrt((dist_2d−R)²+(z−1)²)')
    print(f'Sessions: {len(sessions)}\n')

    ok = sum(clean_session(s) for s in sessions)
    print(f'\nDone. {ok}/{len(sessions)} sessions written to both subfolders.')


if __name__ == '__main__':
    main()
