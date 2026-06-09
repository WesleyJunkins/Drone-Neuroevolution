#!/usr/bin/env python3
"""
analyze.py — Comprehensive thesis analysis script.

This script lives in <repo root>/Analysis/. It reads the datasets from
<repo root>/Data/ (the same folder Data/clean_data.py writes its cleaned output
to) and writes EVERY output into Analysis/Analysis_Results/ only. It never writes
to the thesis figures folder or anywhere else — the submitted thesis is frozen
and must not be modified.

Reads (relative to <repo root>):
  Data/ALL_FINAL_ERROR_RESULTS_CLEANED/FOR_ANALYSIS/   (primary: error_3d, error_px)
  Data/ALL_FINAL_ERROR_RESULTS_CLEANED/FOR_PLOTTING/  (spatial: pos_x/y/z, dist_from_center)
  Data/ALL_FINAL_ERROR_RESULTS/                        (raw: lap_number for lap analysis)
  Data/model_data/                                     (section 15: training-curve CSVs)

Writes (everything under Analysis/Analysis_Results/):
  Analysis_Results/ALL_FINAL_ANALYSIS/{01_overview ... 12_lap_analysis}/  (sections 01–12)
  Analysis_Results/ALL_FINAL_ANALYSIS/13_convergence/                     (section 13)
  Analysis_Results/charts/                                                (section 14)
  Analysis_Results/ALL_FINAL_ANALYSIS/training_convergence.png           (section 15)
  Analysis_Results/ALL_PATH_PLOTS/{session}_paths.png                    (section 16)

Sections 13–16 were merged in from the former standalone scripts
compute_convergence_time.py, plot_convergence_rate.py, make_overview_charts.py,
plot_training_convergence.py, and plot_all_paths.py, so a single full run
reproduces every analysis output. Sections 13–15 use their own matplotlib
rcParams via rc_context; sections 01–12 and 16 use the default rcParams.

Usage (run from anywhere):
    python3 Analysis/analyze.py
    python3 Analysis/analyze.py --sections 01 05 07
    python3 Analysis/analyze.py --sections 13 14 15 16
"""

import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon, f_oneway, kruskal

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
# Inputs: read from <repo root>/Data/ (same folder clean_data.py writes its output to)
_ANAL_SRC = os.path.join(_ROOT, 'Data', 'ALL_FINAL_ERROR_RESULTS_CLEANED', 'FOR_ANALYSIS')
_PLOT_SRC = os.path.join(_ROOT, 'Data', 'ALL_FINAL_ERROR_RESULTS_CLEANED', 'FOR_PLOTTING')
_RAW_SRC  = os.path.join(_ROOT, 'Data', 'ALL_FINAL_ERROR_RESULTS')
_MODEL_DATA_DIR = os.path.join(_ROOT, 'Data', 'model_data')      # section 15 training curves
# Outputs: EVERYTHING goes under Analysis/Analysis_Results/ — nothing else is written.
_RESULTS_BASE  = os.path.join(_HERE, 'Analysis_Results')
_OUT_DIR       = os.path.join(_RESULTS_BASE, 'ALL_FINAL_ANALYSIS')  # sections 01–13, 15
_CHARTS_DIR    = os.path.join(_RESULTS_BASE, 'charts')             # section 14 overview charts
_PATHS_OUT_DIR = os.path.join(_RESULTS_BASE, 'ALL_PATH_PLOTS')     # section 16 path plots

SUBDIRS = [
    '01_overview', '02_heatmaps', '03_speed_analysis', '04_blur_analysis',
    '05_rankings', '06_model_comparison', '07_baseline_comparison',
    '08_performance_score', '09_statistics', '10_temporal',
    '11_spatial', '12_lap_analysis', '13_convergence',
]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TRACK_RADIUS        = 2.845519
TARGET_Z            = 1.0
SPEEDS              = [0.2, 0.4, 0.6, 0.8, 1.0]
BLURS               = [0, 1, 2, 3, 4]
PRECISION_THRESHOLD = 1.0    # metres — error_3d above this counts as "off-track" for Precision score
PERF_SCORE_SCALE    = 2.0    # exponential decay scale for Accuracy / Stability
STABLE_SPEED_THR    = 0.5    # metres — "max stable speed" threshold
TRACK_LOSS_SENTINEL = 31.0   # error_px >= this means track not detected

# Session metadata: (name, family, output_type, trained)
SESSION_META = [
    ('baseline',                'baseline',           'ycommand',    False),
    ('train_ycommand',          'main_trained_NE',    'ycommand',    True),
    ('train_ycontinuous',       'main_trained_NE',    'ycontinuous', True),
    ('notrain_ycommand',        'main_untrained_NE',  'ycommand',    False),
    ('notrain_ycontinuous',     'main_untrained_NE',  'ycontinuous', False),
    ('mlp_256_128_ycommand',    'small_mlp',          'ycommand',    True),
    ('mlp_256_128_ycontinuous', 'small_mlp',          'ycontinuous', True),
    ('mlp_64_32_ycommand',      'small_mlp',          'ycommand',    True),
    ('mlp_64_32_ycontinuous',   'small_mlp',          'ycontinuous', True),
    ('mlp_16_8_ycommand',       'small_mlp',          'ycommand',    True),
    ('mlp_16_8_ycontinuous',    'small_mlp',          'ycontinuous', True),
    ('neat_ycommand',           'neat',               'ycommand',    False),
    ('neat_ycontinuous',        'neat',               'ycontinuous', False),
    ('features_ycommand',       'feature_mlp',        'ycommand',    True),
    ('features_ycontinuous',    'feature_mlp',        'ycontinuous', True),
    ('exp_e_ycommand',          'exp_e',              'ycommand',    True),
    ('exp_e_ycontinuous',       'exp_e',              'ycontinuous', True),
    ('exp_f_ycommand',          'exp_f',              'ycommand',    False),
    ('exp_f_ycontinuous',       'exp_f',              'ycontinuous', False),
]
SESSION_NAMES = [m[0] for m in SESSION_META]
META = {m[0]: {'family': m[1], 'output': m[2], 'trained': m[3]} for m in SESSION_META}

FAMILY_COLORS = {
    'baseline':           '#777777',
    'main_trained_NE':    '#1f77b4',   # deep blue
    'main_untrained_NE':  '#aec7e8',   # light blue
    'small_mlp':          '#2ca02c',
    'neat':               '#ff7f0e',
    'feature_mlp':        '#9467bd',
    'exp_e':              '#d62728',
    'exp_f':              '#8c564b',
}

# Display labels for figures. The two main-CNN sessions are split by training
# regime for plotting (so trained vs. untrained NE can be compared visually),
# but both share the same main_cnn architecture — the display names make that
# explicit. The family-level ANOVA pools them back into a single 'main_cnn'
# group (see run_09_statistics), so the figures and the statistics stay
# consistent: one architectural family, shown with its two training regimes.
FAMILY_DISPLAY = {
    'main_trained_NE':   'main_cnn (trained)',
    'main_untrained_NE': 'main_cnn (untrained)',
}
OUTPUT_COLORS  = {'ycommand': '#e377c2', 'ycontinuous': '#17becf'}
OUTPUT_MARKERS = {'ycommand': 'o', 'ycontinuous': 's'}
OUTPUT_LS      = {'ycommand': '--', 'ycontinuous': '-'}

_files_saved = 0


def _save(fig, path):
    global _files_saved
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {os.path.relpath(path, _ROOT)}')
    _files_saved += 1


def _save_csv(df, path, index=False):
    global _files_saved
    df.to_csv(path, index=index, float_format='%.6f')
    print(f'  saved: {os.path.relpath(path, _ROOT)}')
    _files_saved += 1


def _speed_idx(val):
    for i, s in enumerate(SPEEDS):
        if np.isclose(float(val), s, atol=1e-3):
            return i
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_analysis_data() -> dict:
    data = {}
    for name in SESSION_NAMES:
        path = os.path.join(_ANAL_SRC, f'{name}_error_log.csv')
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            print(f'  [WARN] missing FOR_ANALYSIS: {name}')
    return data


def load_plotting_data() -> dict:
    data = {}
    for name in SESSION_NAMES:
        path = os.path.join(_PLOT_SRC, f'{name}_error_log.csv')
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            print(f'  [WARN] missing FOR_PLOTTING: {name}')
    return data


def load_raw_lap_data() -> dict:
    data = {}
    cols = ['test_index', 'timestamp', 'lap_number', 'speed_scalar', 'blur_level']
    for name in SESSION_NAMES:
        path = os.path.join(_RAW_SRC, f'{name}_error_log.csv')
        if os.path.exists(path):
            data[name] = pd.read_csv(path, usecols=cols)
        else:
            print(f'  [WARN] missing raw: {name}')
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def per_config_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per config_id with key aggregate statistics."""
    records = []
    for cid, g in df.groupby('config_id'):
        e3d = g['error_3d'].values.astype(float)
        epx = g['error_px'].values.astype(float)
        n   = len(e3d)
        records.append({
            'config_id':      int(cid),
            'speed':          float(g['speed_scalar'].iloc[0]),
            'blur':           int(g['blur_level'].iloc[0]),
            'mean_e3d':       np.nanmean(e3d),
            'std_e3d':        np.nanstd(e3d),
            'median_e3d':     np.nanmedian(e3d),
            'mean_epx':       np.nanmean(epx),
            'std_epx':        np.nanstd(epx),
            'gps_valid_pct':  100.0 * np.sum(~np.isnan(e3d)) / n,
            'track_loss_pct': 100.0 * np.sum(epx >= TRACK_LOSS_SENTINEL) / n,
            'off_track_pct':  100.0 * np.nansum(e3d > PRECISION_THRESHOLD) / np.sum(~np.isnan(e3d)) if np.sum(~np.isnan(e3d)) > 0 else 100.0,
            'n_rows':         n,
        })
    return pd.DataFrame(records)


def per_speed_stats(cfg: pd.DataFrame) -> pd.DataFrame:
    return (cfg.groupby('speed')
               .agg(mean_e3d=('mean_e3d', lambda x: np.nanmean(x)),
                    std_e3d=('mean_e3d',  lambda x: np.nanstd(x)))
               .reset_index())


def per_blur_stats(cfg: pd.DataFrame) -> pd.DataFrame:
    return (cfg.groupby('blur')
               .agg(mean_e3d=('mean_e3d', lambda x: np.nanmean(x)),
                    std_e3d=('mean_e3d',  lambda x: np.nanstd(x)))
               .reset_index())


def compute_lap_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Mean laps completed and mean time-to-first-lap per speed (averaged across 3 reps)."""
    records = []
    for ti, grp in raw_df.groupby('test_index'):
        speed_mode = grp['speed_scalar'].mode()
        speed = float(speed_mode.iloc[0]) if len(speed_mode) > 0 else np.nan
        max_lap = int(grp['lap_number'].max())
        first_lap_ts = grp.loc[grp['lap_number'] >= 1, 'timestamp']
        t_first = float(first_lap_ts.min()) if len(first_lap_ts) > 0 else np.nan
        records.append({'test_index': int(ti), 'speed': speed,
                        'laps': max_lap, 't_first_lap': t_first})
    tmp = pd.DataFrame(records)
    out = (tmp.groupby('speed')
              .agg(mean_laps=('laps', 'mean'), std_laps=('laps', 'std'),
                   mean_t_first=('t_first_lap', 'mean'), std_t_first=('t_first_lap', 'std'))
              .reset_index())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Performance Score (adapted from thesis proposal)
# ─────────────────────────────────────────────────────────────────────────────

def performance_score_row(row) -> tuple:
    """Return (accuracy, stability, endurance, precision, composite)."""
    me3d = row['mean_e3d'] if not np.isnan(row['mean_e3d']) else 99.0
    se3d = row['std_e3d']  if not np.isnan(row['std_e3d'])  else 99.0
    accuracy  = float(np.exp(-me3d / PERF_SCORE_SCALE))
    stability = float(np.exp(-se3d / PERF_SCORE_SCALE))
    endurance = float(row['gps_valid_pct'] / 100.0)
    precision = float(1.0 - row['off_track_pct'] / 100.0)
    composite = 25.0 * (accuracy + stability + endurance + precision)
    return accuracy, stability, endurance, precision, composite


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pearson_spearman(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    xv, yv = x[mask], y[mask]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r,   r_p   = pearsonr(xv, yv)
        rho, rho_p = spearmanr(xv, yv)
    return float(r), float(r_p), float(rho), float(rho_p)


def _wilcoxon_vs_baseline(sess_vec, base_vec):
    """Paired Wilcoxon over 25 configs. Returns (W, p, median_diff, direction)."""
    s = np.asarray(sess_vec, dtype=float)
    b = np.asarray(base_vec, dtype=float)
    valid = ~(np.isnan(s) | np.isnan(b))
    if valid.sum() < 5:
        return np.nan, np.nan, np.nan, 'insufficient'
    sv, bv = s[valid], b[valid]
    diff = bv - sv   # positive = session is better than baseline
    median_diff = float(np.median(diff))
    try:
        W, p = wilcoxon(sv, bv, alternative='two-sided')
    except Exception:
        W, p = np.nan, np.nan
    direction = ('better' if median_diff > 0 else
                 'worse'  if median_diff < 0 else 'no diff')
    return float(W), float(p), median_diff, direction


def _anova_groups(groups: dict):
    """One-way ANOVA + Kruskal-Wallis. Returns (F, p_anova, eta_sq, H, p_kruskal)."""
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays]
    arrays = [a for a in arrays if len(a) >= 2]
    if len(arrays) < 2:
        return (np.nan,) * 5
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F, p_a = f_oneway(*arrays)
        H, p_k = kruskal(*arrays)
    grand_mean = np.concatenate(arrays).mean()
    ss_b = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_t = sum(((v - grand_mean) ** 2).sum() for v in arrays)
    eta_sq = ss_b / ss_t if ss_t > 0 else np.nan
    return float(F), float(p_a), float(eta_sq), float(H), float(p_k)


# ─────────────────────────────────────────────────────────────────────────────
# Section 01 — Overview
# ─────────────────────────────────────────────────────────────────────────────

def run_01_overview(anal_data: dict):
    out = os.path.join(_OUT_DIR, '01_overview')

    # session_summary.csv
    rows = []
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        sp = per_speed_stats(cs)
        stable = sp[sp['mean_e3d'] < STABLE_SPEED_THR]
        rows.append({
            'session':          name,
            'family':           META[name]['family'],
            'output':           META[name]['output'],
            'trained':          META[name]['trained'],
            'mean_error_3d':    float(np.nanmean(cs['mean_e3d'])),
            'std_error_3d':     float(np.nanstd(cs['mean_e3d'])),
            'mean_error_px':    float(np.nanmean(cs['mean_epx'])),
            'gps_valid_pct':    float(np.nanmean(cs['gps_valid_pct'])),
            'track_loss_pct':   float(np.nanmean(cs['track_loss_pct'])),
            'max_stable_speed': float(stable['speed'].max()) if len(stable) > 0 else 0.0,
        })
    summary = pd.DataFrame(rows).sort_values('mean_error_3d').reset_index(drop=True)
    summary.insert(0, 'rank', range(1, len(summary) + 1))
    _save_csv(summary, os.path.join(out, 'session_summary.csv'))

    # config_summary.csv
    cfg_rows = []
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        cs.insert(0, 'session', name)
        cfg_rows.append(cs)
    cfg_all = pd.concat(cfg_rows, ignore_index=True)
    _save_csv(cfg_all[['session', 'speed', 'blur', 'mean_e3d', 'std_e3d',
                        'gps_valid_pct', 'track_loss_pct', 'n_rows']],
              os.path.join(out, 'config_summary.csv'))

    # overall_heatmap_grid.png — small multiples
    names = list(anal_data.keys())
    n = len(names)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    config_grids = {}
    all_vals = []
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        grid = np.full((5, 5), np.nan)
        for _, row in cs.iterrows():
            si = _speed_idx(row['speed'])
            bi = BLURS.index(int(row['blur'])) if int(row['blur']) in BLURS else -1
            if si >= 0 and bi >= 0:
                grid[si, bi] = row['mean_e3d']
        config_grids[name] = grid
        all_vals.extend(grid[~np.isnan(grid)].tolist())

    vmax = min(float(np.nanpercentile(all_vals, 95)), 10.0) if all_vals else 5.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.4))
    axes = np.array(axes).flatten()
    im = None
    for i, name in enumerate(names):
        ax = axes[i]
        im = ax.imshow(config_grids[name], vmin=0, vmax=vmax,
                       cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['0','1','2','3','4'], fontsize=7)
        ax.set_yticks(range(5))
        ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7)
        ax.set_title(name.replace('_', '\n'), fontsize=7, fontweight='bold')
        if i % ncols == 0:
            ax.set_ylabel('Speed', fontsize=7)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='mean error_3d (m)')
    fig.suptitle('Mean 3D Error — All Sessions  (rows=speed, cols=blur)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    _save(fig, os.path.join(out, 'overall_heatmap_grid.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 02 — Per-Session Heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def _draw_heatmap(cs, col, title, filename, out_dir, cmap, vmax=None, annotate=True):
    grid = np.full((5, 5), np.nan)
    for _, row in cs.iterrows():
        si = _speed_idx(row['speed'])
        bi = BLURS.index(int(row['blur'])) if int(row['blur']) in BLURS else -1
        if si >= 0 and bi >= 0:
            grid[si, bi] = row[col]
    if vmax is None:
        vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, vmin=0, vmax=vmax, cmap=cmap, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'blur {b}' for b in BLURS], fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'{s}' for s in SPEEDS], fontsize=10)
    ax.set_xlabel('Blur Level', fontsize=11)
    ax.set_ylabel('Speed', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    if annotate:
        for si in range(5):
            for bi in range(5):
                v = grid[si, bi]
                if not np.isnan(v):
                    brightness = v / vmax if vmax > 0 else 0
                    txt_color = 'white' if brightness > 0.6 else 'black'
                    ax.text(bi, si, f'{v:.3f}', ha='center', va='center',
                            fontsize=8, color=txt_color)
    plt.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, filename))


def run_02_heatmaps(anal_data: dict):
    out = os.path.join(_OUT_DIR, '02_heatmaps')
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        vmax_e3d = min(float(np.nanmax(cs['mean_e3d'])) * 1.05, 10.0)
        vmax_epx = min(float(np.nanmax(cs['mean_epx'])) * 1.05, 32.0)
        _draw_heatmap(cs, 'mean_e3d', f'{name} — Mean 3D Error (m)',
                      f'{name}_error3d_heatmap.png', out, 'RdYlGn_r', vmax_e3d)
        _draw_heatmap(cs, 'mean_epx', f'{name} — Mean Pixel Error',
                      f'{name}_errorpx_heatmap.png', out, 'RdYlGn_r', vmax_epx)
        _draw_heatmap(cs, 'gps_valid_pct', f'{name} — GPS Valid % (100 = no explosion)',
                      f'{name}_gpsloss_heatmap.png', out, 'RdYlGn', 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Section 03 — Speed Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _speed_line_plot(anal_data, sessions, title, filename, out_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in sessions:
        if name not in anal_data:
            continue
        cs = per_config_stats(anal_data[name])
        sp = per_speed_stats(cs)
        color  = FAMILY_COLORS[META[name]['family']]
        marker = OUTPUT_MARKERS[META[name]['output']]
        ls     = OUTPUT_LS[META[name]['output']]
        ax.plot(sp['speed'], sp['mean_e3d'], marker=marker, ls=ls, color=color,
                lw=1.8, label=name, alpha=0.85)
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(SPEEDS)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, filename))


def run_03_speed_analysis(anal_data: dict):
    out = os.path.join(_OUT_DIR, '03_speed_analysis')

    _speed_line_plot(anal_data, SESSION_NAMES,
                     'Speed vs. Mean 3D Error — All Sessions',
                     'speed_vs_error3d_all.png', out)
    _speed_line_plot(anal_data, [n for n in SESSION_NAMES if META[n]['output'] == 'ycontinuous'],
                     'Speed vs. Mean 3D Error — ycontinuous Sessions',
                     'speed_vs_error3d_ycontinuous.png', out)
    _speed_line_plot(anal_data, [n for n in SESSION_NAMES if META[n]['output'] == 'ycommand'],
                     'Speed vs. Mean 3D Error — ycommand Sessions',
                     'speed_vs_error3d_ycommand.png', out)

    # By family
    fig, ax = plt.subplots(figsize=(11, 6))
    for fam in sorted(set(META[n]['family'] for n in SESSION_NAMES)):
        fnames = [n for n in SESSION_NAMES if META[n]['family'] == fam and n in anal_data]
        if not fnames:
            continue
        speed_e3d = {s: [] for s in SPEEDS}
        for name in fnames:
            cs = per_config_stats(anal_data[name])
            for _, row in per_speed_stats(cs).iterrows():
                if row['speed'] in speed_e3d:
                    speed_e3d[row['speed']].append(row['mean_e3d'])
        ys = [np.nanmean(speed_e3d[s]) if speed_e3d[s] else np.nan for s in SPEEDS]
        ax.plot(SPEEDS, ys, marker='o', color=FAMILY_COLORS[fam], lw=2.2, label=FAMILY_DISPLAY.get(fam, fam))
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('Speed vs. Mean 3D Error — By Model Family', fontsize=13, fontweight='bold')
    ax.set_xticks(SPEEDS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'speed_vs_error3d_by_family.png'))

    # Correlation table
    rows = []
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        sp = per_speed_stats(cs)
        r, rp, rho, rhop = _pearson_spearman(np.array(sp['speed']),
                                              np.array(sp['mean_e3d']))
        rows.append({'session': name, 'family': META[name]['family'],
                     'output': META[name]['output'],
                     'pearson_r': r, 'pearson_p': rp,
                     'spearman_rho': rho, 'spearman_p': rhop})
    _save_csv(pd.DataFrame(rows), os.path.join(out, 'speed_correlation_table.csv'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 04 — Blur Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _blur_line_plot(anal_data, sessions, title, filename, out_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in sessions:
        if name not in anal_data:
            continue
        cs = per_config_stats(anal_data[name])
        bp = per_blur_stats(cs)
        color  = FAMILY_COLORS[META[name]['family']]
        marker = OUTPUT_MARKERS[META[name]['output']]
        ls     = OUTPUT_LS[META[name]['output']]
        ax.plot(bp['blur'], bp['mean_e3d'], marker=marker, ls=ls, color=color,
                lw=1.8, label=name, alpha=0.85)
    ax.set_xlabel('Blur Level', fontsize=12)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(BLURS)
    ax.set_xticklabels(['0 (none)', '1 (light)', '2 (mod)', '3 (heavy)', '4 (max)'])
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, filename))


def run_04_blur_analysis(anal_data: dict):
    out = os.path.join(_OUT_DIR, '04_blur_analysis')

    _blur_line_plot(anal_data, SESSION_NAMES,
                    'Blur vs. Mean 3D Error — All Sessions',
                    'blur_vs_error3d_all.png', out)
    _blur_line_plot(anal_data, [n for n in SESSION_NAMES if META[n]['output'] == 'ycontinuous'],
                    'Blur vs. Mean 3D Error — ycontinuous Sessions',
                    'blur_vs_error3d_ycontinuous.png', out)
    _blur_line_plot(anal_data, [n for n in SESSION_NAMES if META[n]['output'] == 'ycommand'],
                    'Blur vs. Mean 3D Error — ycommand Sessions',
                    'blur_vs_error3d_ycommand.png', out)

    # By family
    fig, ax = plt.subplots(figsize=(11, 6))
    for fam in sorted(set(META[n]['family'] for n in SESSION_NAMES)):
        fnames = [n for n in SESSION_NAMES if META[n]['family'] == fam and n in anal_data]
        if not fnames:
            continue
        blur_e3d = {b: [] for b in BLURS}
        for name in fnames:
            cs = per_config_stats(anal_data[name])
            for _, row in per_blur_stats(cs).iterrows():
                if int(row['blur']) in blur_e3d:
                    blur_e3d[int(row['blur'])].append(row['mean_e3d'])
        ys = [np.nanmean(blur_e3d[b]) if blur_e3d[b] else np.nan for b in BLURS]
        ax.plot(BLURS, ys, marker='o', color=FAMILY_COLORS[fam], lw=2.2, label=FAMILY_DISPLAY.get(fam, fam))
    ax.set_xlabel('Blur Level', fontsize=12)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('Blur vs. Mean 3D Error — By Model Family', fontsize=13, fontweight='bold')
    ax.set_xticks(BLURS)
    ax.set_xticklabels(['0 (none)', '1 (light)', '2 (mod)', '3 (heavy)', '4 (max)'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'blur_vs_error3d_by_family.png'))

    rows = []
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        bp = per_blur_stats(cs)
        r, rp, rho, rhop = _pearson_spearman(np.array(bp['blur'], dtype=float),
                                              np.array(bp['mean_e3d']))
        rows.append({'session': name, 'family': META[name]['family'],
                     'output': META[name]['output'],
                     'pearson_r': r, 'pearson_p': rp,
                     'spearman_rho': rho, 'spearman_p': rhop})
    _save_csv(pd.DataFrame(rows), os.path.join(out, 'blur_correlation_table.csv'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 05 — Rankings
# ─────────────────────────────────────────────────────────────────────────────

def run_05_rankings(anal_data: dict):
    out = os.path.join(_OUT_DIR, '05_rankings')

    rows = []
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        row = {'session': name, 'family': META[name]['family'],
               'output': META[name]['output'],
               'overall_mean_e3d': float(np.nanmean(cs['mean_e3d']))}
        for speed in SPEEDS:
            sub = cs[np.isclose(cs['speed'], speed, atol=1e-3)]
            row[f'mean_e3d_s{speed}'] = float(np.nanmean(sub['mean_e3d'])) if len(sub) > 0 else np.nan
        for blur in BLURS:
            sub = cs[cs['blur'] == blur]
            row[f'mean_e3d_b{blur}'] = float(np.nanmean(sub['mean_e3d'])) if len(sub) > 0 else np.nan
        rows.append(row)

    rank_df = (pd.DataFrame(rows)
               .sort_values('overall_mean_e3d')
               .reset_index(drop=True))
    rank_df.insert(0, 'rank', range(1, len(rank_df) + 1))
    _save_csv(rank_df, os.path.join(out, 'rankings_table.csv'))

    # Overall bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [FAMILY_COLORS[META[n]['family']] for n in rank_df['session']]
    ax.barh(range(len(rank_df)), rank_df['overall_mean_e3d'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(rank_df)))
    ax.set_yticklabels(rank_df['session'], fontsize=9)
    ax.set_xlabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('Overall Rankings — All 19 Sessions by Mean 3D Error', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(rank_df['overall_mean_e3d']):
        if not np.isnan(v):
            ax.text(v + 0.05, i, f'{v:.4f}m', va='center', fontsize=8)
    legend_handles = [mpatches.Patch(facecolor=c, label=FAMILY_DISPLAY.get(f, f))
                      for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'overall_rankings.png'))

    # By speed (5-panel)
    fig, axes = plt.subplots(1, 5, figsize=(24, 8), sharey=False)
    for si, speed in enumerate(SPEEDS):
        ax = axes[si]
        srows = []
        for name, df in anal_data.items():
            cs = per_config_stats(df)
            sub = cs[np.isclose(cs['speed'], speed, atol=1e-3)]
            srows.append({'session': name, 'family': META[name]['family'],
                          'mean_e3d': float(np.nanmean(sub['mean_e3d'])) if len(sub) > 0 else np.nan})
        s_df = pd.DataFrame(srows).sort_values('mean_e3d')
        colors_s = [FAMILY_COLORS[r['family']] for _, r in s_df.iterrows()]
        ax.barh(range(len(s_df)), s_df['mean_e3d'], color=colors_s, edgecolor='white')
        ax.set_yticks(range(len(s_df)))
        ax.set_yticklabels(s_df['session'], fontsize=7)
        ax.set_title(f'Speed = {speed}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Mean error_3d (m)', fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle('Rankings by Speed Level', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'rankings_by_speed.png'))

    # By blur (5-panel)
    fig, axes = plt.subplots(1, 5, figsize=(24, 8), sharey=False)
    for bi, blur in enumerate(BLURS):
        ax = axes[bi]
        brows = []
        for name, df in anal_data.items():
            cs = per_config_stats(df)
            sub = cs[cs['blur'] == blur]
            brows.append({'session': name, 'family': META[name]['family'],
                          'mean_e3d': float(np.nanmean(sub['mean_e3d'])) if len(sub) > 0 else np.nan})
        b_df = pd.DataFrame(brows).sort_values('mean_e3d')
        colors_b = [FAMILY_COLORS[r['family']] for _, r in b_df.iterrows()]
        ax.barh(range(len(b_df)), b_df['mean_e3d'], color=colors_b, edgecolor='white')
        ax.set_yticks(range(len(b_df)))
        ax.set_yticklabels(b_df['session'], fontsize=7)
        ax.set_title(f'Blur = {blur}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Mean error_3d (m)', fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle('Rankings by Blur Level', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'rankings_by_blur.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 06 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_06_model_comparison(anal_data: dict):
    out = os.path.join(_OUT_DIR, '06_model_comparison')
    all_cs = {n: per_config_stats(df) for n, df in anal_data.items()}
    families = sorted(set(META[n]['family'] for n in SESSION_NAMES))

    # family_boxplot.png
    fig, ax = plt.subplots(figsize=(16, 6.5))
    positions, box_data, patch_colors, tick_labels = [], [], [], []
    pos = 0
    for fam in families:
        for otype in ['ycommand', 'ycontinuous']:
            snames = [n for n in SESSION_NAMES
                      if META[n]['family'] == fam and META[n]['output'] == otype and n in all_cs]
            if not snames:
                continue
            vals = np.concatenate([all_cs[n]['mean_e3d'].dropna().values for n in snames])
            # Log scale requires positive values; replace zeros with a small floor
            vals = np.maximum(vals, 1e-4)
            box_data.append(vals)
            positions.append(pos)
            patch_colors.append(OUTPUT_COLORS[otype])
            tick_labels.append(f'{FAMILY_DISPLAY.get(fam, fam)}\n{otype[:5]}')
            pos += 1
        pos += 0.4
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 2})
    for patch, color in zip(bp['boxes'], patch_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=15, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Mean error_3d per config (m, log scale)', fontsize=12)
    ax.set_title('Error Distribution by Family and Output Type', fontsize=13, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(facecolor=OUTPUT_COLORS['ycommand'], label='ycommand'),
                        mpatches.Patch(facecolor=OUTPUT_COLORS['ycontinuous'], label='ycontinuous')],
              fontsize=10, loc='upper right')
    ax.grid(axis='y', which='both', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'family_boxplot.png'))

    # family_bar_by_speed.png
    n_fam = len(families)
    width = 0.8 / n_fam
    fig, ax = plt.subplots(figsize=(14, 6))
    xs = np.arange(len(SPEEDS))
    for fi, fam in enumerate(families):
        fnames = [n for n in SESSION_NAMES if META[n]['family'] == fam and n in all_cs]
        ys = []
        for speed in SPEEDS:
            vals = np.concatenate([
                all_cs[n]['mean_e3d'][np.isclose(all_cs[n]['speed'], speed, atol=1e-3)].dropna().values
                for n in fnames
            ]) if fnames else np.array([np.nan])
            ys.append(float(np.nanmean(vals)))
        offset = fi * width - (n_fam - 1) * width / 2
        ax.bar(xs + offset, ys, width, label=FAMILY_DISPLAY.get(fam, fam), color=FAMILY_COLORS[fam], edgecolor='white')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in SPEEDS])
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('Mean 3D Error by Family and Speed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'family_bar_by_speed.png'))

    # family_bar_by_blur.png
    fig, ax = plt.subplots(figsize=(14, 6))
    xs = np.arange(len(BLURS))
    for fi, fam in enumerate(families):
        fnames = [n for n in SESSION_NAMES if META[n]['family'] == fam and n in all_cs]
        ys = []
        for blur in BLURS:
            vals = np.concatenate([
                all_cs[n]['mean_e3d'][all_cs[n]['blur'] == blur].dropna().values
                for n in fnames
            ]) if fnames else np.array([np.nan])
            ys.append(float(np.nanmean(vals)))
        offset = fi * width - (n_fam - 1) * width / 2
        ax.bar(xs + offset, ys, width, label=FAMILY_DISPLAY.get(fam, fam), color=FAMILY_COLORS[fam], edgecolor='white')
    ax.set_xticks(xs)
    ax.set_xticklabels(['0 (none)', '1 (light)', '2 (mod)', '3 (heavy)', '4 (max)'])
    ax.set_xlabel('Blur Level', fontsize=12)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('Mean 3D Error by Family and Blur Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'family_bar_by_blur.png'))

    # ycommand_vs_ycontinuous.png
    paired_fams = [fam for fam in families if fam != 'baseline' and
                   any(META[n]['output'] == 'ycommand'    for n in SESSION_NAMES if META[n]['family'] == fam) and
                   any(META[n]['output'] == 'ycontinuous' for n in SESSION_NAMES if META[n]['family'] == fam)]
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(len(paired_fams))
    yq_vals, yc_vals = [], []
    for fam in paired_fams:
        yq = [n for n in SESSION_NAMES if META[n]['family'] == fam and META[n]['output'] == 'ycommand'    and n in all_cs]
        yc = [n for n in SESSION_NAMES if META[n]['family'] == fam and META[n]['output'] == 'ycontinuous' and n in all_cs]
        yq_vals.append(float(np.nanmean(np.concatenate([all_cs[n]['mean_e3d'].dropna().values for n in yq]))) if yq else np.nan)
        yc_vals.append(float(np.nanmean(np.concatenate([all_cs[n]['mean_e3d'].dropna().values for n in yc]))) if yc else np.nan)
    ax.bar(xs - 0.2, yq_vals, 0.4, label='ycommand', color=OUTPUT_COLORS['ycommand'], edgecolor='white')
    ax.bar(xs + 0.2, yc_vals, 0.4, label='ycontinuous', color=OUTPUT_COLORS['ycontinuous'], edgecolor='white')
    ax.set_xticks(xs)
    ax.set_xticklabels(paired_fams, fontsize=10)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('ycommand vs. ycontinuous by Family', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'ycommand_vs_ycontinuous.png'))

    # trained_vs_untrained.png
    pairs = [('train_ycommand',    'notrain_ycommand'),
             ('train_ycontinuous', 'notrain_ycontinuous')]
    xlabels = ['train vs notrain\n(ycommand)', 'train vs notrain\n(ycontinuous)']
    fig, ax = plt.subplots(figsize=(13, 6))
    xs = np.arange(len(pairs))
    tv = [float(np.nanmean(all_cs[t]['mean_e3d'])) if t in all_cs else np.nan for t, _ in pairs]
    uv = [float(np.nanmean(all_cs[u]['mean_e3d'])) if u in all_cs else np.nan for _, u in pairs]
    ax.bar(xs - 0.2, tv, 0.4, label='Trained', color='#2196F3', edgecolor='white')
    ax.bar(xs + 0.2, uv, 0.4, label='Untrained', color='#FF9800', edgecolor='white')
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('Trained vs. Untrained — Matched Pairs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'trained_vs_untrained.png'))

    # output_type_summary.png
    baseline_vals = all_cs['baseline']['mean_e3d'].dropna().values if 'baseline' in all_cs else np.array([])
    yq_all = np.concatenate([all_cs[n]['mean_e3d'].dropna().values
                             for n in SESSION_NAMES if META[n]['output'] == 'ycommand' and n != 'baseline' and n in all_cs])
    yc_all = np.concatenate([all_cs[n]['mean_e3d'].dropna().values
                             for n in SESSION_NAMES if META[n]['output'] == 'ycontinuous' and n in all_cs])
    # Log scale requires positive values; floor at 1e-4 for visualization
    baseline_vals = np.maximum(baseline_vals, 1e-4)
    yq_all        = np.maximum(yq_all,        1e-4)
    yc_all        = np.maximum(yc_all,        1e-4)
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [baseline_vals, yq_all, yc_all]
    xlabels_ot = ['Baseline\n(rule-based)', 'ycommand\n(all NE)', 'ycontinuous\n(all NE)']
    grp_colors = ['#9E9E9E', OUTPUT_COLORS['ycommand'], OUTPUT_COLORS['ycontinuous']]
    bp = ax.boxplot(box_data, labels=xlabels_ot, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 2})
    for patch, color in zip(bp['boxes'], grp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_yscale('log')
    ax.set_ylabel('Mean error_3d per config (m, log scale)', fontsize=12)
    ax.set_title('Error Distribution by Output Type', fontsize=13, fontweight='bold')
    ax.grid(axis='y', which='both', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'output_type_summary.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 07 — Baseline Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_07_baseline_comparison(anal_data: dict):
    out = os.path.join(_OUT_DIR, '07_baseline_comparison')
    if 'baseline' not in anal_data:
        print('  [SKIP 07] baseline data not found')
        return

    base_cs = per_config_stats(anal_data['baseline'])
    base_vec = np.array([
        float(base_cs.loc[base_cs['config_id'] == cid, 'mean_e3d'].values[0])
        if cid in base_cs['config_id'].values else np.nan
        for cid in range(1, 26)
    ])
    base_mean = float(np.nanmean(base_vec))

    # wilcoxon_results.csv
    wrows = []
    for name, df in anal_data.items():
        if name == 'baseline':
            continue
        cs = per_config_stats(df)
        sess_vec = np.array([
            float(cs.loc[cs['config_id'] == cid, 'mean_e3d'].values[0])
            if cid in cs['config_id'].values else np.nan
            for cid in range(1, 26)
        ])
        W, p, med_diff, direction = _wilcoxon_vs_baseline(sess_vec, base_vec)
        wrows.append({
            'session': name, 'family': META[name]['family'], 'output': META[name]['output'],
            'mean_e3d_session': float(np.nanmean(sess_vec)),
            'mean_e3d_baseline': base_mean,
            'W_stat': W, 'p_value': p,
            'median_diff_vs_baseline': med_diff, 'direction': direction,
        })
    wdf = pd.DataFrame(wrows).sort_values('mean_e3d_session').reset_index(drop=True)
    _save_csv(wdf, os.path.join(out, 'wilcoxon_results.csv'))

    # all_vs_baseline_bar.png
    fig, ax = plt.subplots(figsize=(14, 7))
    diffs  = [base_mean - v for v in wdf['mean_e3d_session']]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in diffs]
    ax.barh(range(len(wdf)), diffs, color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.set_yticks(range(len(wdf)))
    ax.set_yticklabels(wdf['session'], fontsize=9)
    ax.set_xlabel('Improvement over Baseline (m) — positive = better', fontsize=11)
    ax.set_title(f'All Sessions vs. Baseline  (baseline mean = {base_mean:.4f} m)',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    for i, d in enumerate(diffs):
        ax.text(d + (0.02 if d >= 0 else -0.02), i, f'{d:+.4f}',
                va='center', fontsize=8, ha='left' if d >= 0 else 'right')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'all_vs_baseline_bar.png'))

    # best_ne_vs_baseline.png
    top5 = wdf.head(5)['session'].tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    sp_base = per_speed_stats(base_cs)
    ax.plot(sp_base['speed'], sp_base['mean_e3d'], 'k-o', lw=2.5, label='baseline', zorder=5)
    for name in top5:
        if name not in anal_data:
            continue
        sp = per_speed_stats(per_config_stats(anal_data[name]))
        ax.plot(sp['speed'], sp['mean_e3d'],
                marker=OUTPUT_MARKERS[META[name]['output']],
                color=FAMILY_COLORS[META[name]['family']],
                ls=OUTPUT_LS[META[name]['output']], lw=1.8, label=name)
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean error_3d (m)', fontsize=12)
    ax.set_title('Top 5 NE Models vs. Baseline — Speed Resolved', fontsize=13, fontweight='bold')
    ax.set_xticks(SPEEDS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'best_ne_vs_baseline.png'))

    # improvement_map.png  (all 19 × 25 configs)
    all_names = ['baseline'] + wdf['session'].tolist()
    imp_matrix = np.full((len(all_names), 25), np.nan)
    for si, name in enumerate(all_names):
        if name not in anal_data:
            continue
        cs = per_config_stats(anal_data[name])
        for _, row in cs.iterrows():
            cid  = int(row['config_id']) - 1
            bval = base_vec[cid]
            sval = row['mean_e3d']
            if not np.isnan(bval) and bval != 0 and not np.isnan(sval):
                imp_matrix[si, cid] = (bval - sval) / bval * 100

    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(imp_matrix, cmap='RdYlGn', vmin=-200, vmax=100, aspect='auto')
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xticks(range(25))
    ax.set_xticklabels([f'c{i+1}' for i in range(25)], fontsize=6, rotation=45)
    ax.set_xlabel('Config ID  (c1=s0.2/b0 … c25=s1.0/b4)', fontsize=11)
    ax.set_title('% Improvement over Baseline per Config  (green=better, red=worse)',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='% improvement')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'improvement_map.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 08 — Performance Score
# ─────────────────────────────────────────────────────────────────────────────

def run_08_performance_score(anal_data: dict):
    out = os.path.join(_OUT_DIR, '08_performance_score')

    score_rows = []
    for name, df in anal_data.items():
        cs = per_config_stats(df)
        for _, row in cs.iterrows():
            acc, stab, end, prec, comp = performance_score_row(row)
            score_rows.append({
                'session': name, 'family': META[name]['family'],
                'output': META[name]['output'],
                'speed': row['speed'], 'blur': int(row['blur']),
                'accuracy': acc, 'stability': stab,
                'endurance': end, 'precision': prec, 'composite': comp,
            })
    score_df = pd.DataFrame(score_rows)
    _save_csv(score_df, os.path.join(out, 'performance_scores.csv'))

    sess_comp = (score_df.groupby('session')['composite'].mean()
                         .reset_index()
                         .sort_values('composite', ascending=False)
                         .reset_index(drop=True))

    # performance_score_all.png
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [FAMILY_COLORS[META[n]['family']] for n in sess_comp['session']]
    ax.barh(range(len(sess_comp)), sess_comp['composite'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(sess_comp)))
    ax.set_yticklabels(sess_comp['session'], fontsize=9)
    ax.set_xlabel('Composite Performance Score (0–100)', fontsize=12)
    ax.set_title('Composite Performance Score — All Sessions', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(sess_comp['composite']):
        ax.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=8)
    legend_handles = [mpatches.Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'performance_score_all.png'))

    # score_components_grid.png (stacked bar)
    comps       = ['accuracy', 'stability', 'endurance', 'precision']
    comp_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    comp_labels = ['Accuracy', 'Stability', 'Endurance', 'Precision']
    sess_avg = score_df.groupby('session')[comps].mean()
    sess_avg = sess_avg.loc[sess_comp['session']]
    fig, ax = plt.subplots(figsize=(16, 7))
    bottom = np.zeros(len(sess_avg))
    for ci, (comp, color, label) in enumerate(zip(comps, comp_colors, comp_labels)):
        vals = sess_avg[comp].values * 25  # each component is 0-1, total 0-100
        ax.bar(range(len(sess_avg)), vals, bottom=bottom, color=color,
               label=label, edgecolor='white')
        bottom += vals
    ax.set_xticks(range(len(sess_avg)))
    ax.set_xticklabels(sess_avg.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score Component (0–100 total)', fontsize=12)
    ax.set_title('Performance Score Components — All Sessions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'score_components_grid.png'))

    # score_vs_speed_top5.png
    top5 = sess_comp.head(5)['session'].tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in top5:
        if name not in anal_data:
            continue
        spd = score_df[score_df['session'] == name].groupby('speed')['composite'].mean().reset_index()
        ax.plot(spd['speed'], spd['composite'],
                marker=OUTPUT_MARKERS[META[name]['output']],
                color=FAMILY_COLORS[META[name]['family']],
                lw=2.0, label=name)
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean Composite Score', fontsize=12)
    ax.set_title('Performance Score vs. Speed — Top 5 Sessions', fontsize=13, fontweight='bold')
    ax.set_xticks(SPEEDS)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'score_vs_speed_top5.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 09 — Statistics
# ─────────────────────────────────────────────────────────────────────────────

def run_09_statistics(anal_data: dict):
    out = os.path.join(_OUT_DIR, '09_statistics')
    all_cs = {n: per_config_stats(df) for n, df in anal_data.items()}

    # ANOVA by family. Main trained NE and Main untrained NE are treated as
    # separate families (8 groups total), matching the family taxonomy used in
    # the thesis tables: Architecture (main CNN) -> Family (main trained NE /
    # main untrained NE) -> Session. The figures use FAMILY_DISPLAY to show both
    # families under the shared main_cnn architecture name.
    family_groups = {}
    for name, cs in all_cs.items():
        fam = META[name]['family']
        family_groups.setdefault(fam, []).extend(cs['mean_e3d'].dropna().tolist())
    F, pa, eta, H, pk = _anova_groups(family_groups)
    _save_csv(pd.DataFrame([
        {'test': 'one_way_ANOVA',     'statistic': F, 'p_value': pa, 'eta_squared': eta, 'groups': 'family'},
        {'test': 'kruskal_wallis',    'statistic': H, 'p_value': pk, 'eta_squared': np.nan, 'groups': 'family'},
    ]), os.path.join(out, 'anova_by_family.csv'))

    # ANOVA by output type
    otype_groups = {}
    for name, cs in all_cs.items():
        ot = META[name]['output']
        otype_groups.setdefault(ot, []).extend(cs['mean_e3d'].dropna().tolist())
    F2, pa2, eta2, H2, pk2 = _anova_groups(otype_groups)
    _save_csv(pd.DataFrame([
        {'test': 'one_way_ANOVA',     'statistic': F2, 'p_value': pa2, 'eta_squared': eta2, 'groups': 'output_type'},
        {'test': 'kruskal_wallis',    'statistic': H2, 'p_value': pk2, 'eta_squared': np.nan, 'groups': 'output_type'},
    ]), os.path.join(out, 'anova_by_output_type.csv'))

    # Pairwise Wilcoxon matrix
    names = [n for n in SESSION_NAMES if n in all_cs]
    sess_vecs = {}
    for name in names:
        cs = all_cs[name]
        sess_vecs[name] = np.array([
            float(cs.loc[cs['config_id'] == cid, 'mean_e3d'].values[0])
            if cid in cs['config_id'].values else np.nan
            for cid in range(1, 26)
        ])

    n = len(names)
    p_mat = np.full((n, n), np.nan)
    for i, ni in enumerate(names):
        p_mat[i, i] = 1.0
        for j, nj in enumerate(names):
            if i == j:
                continue
            vi, vj = sess_vecs[ni], sess_vecs[nj]
            valid = ~(np.isnan(vi) | np.isnan(vj))
            if valid.sum() < 5:
                continue
            try:
                _, p = wilcoxon(vi[valid], vj[valid], alternative='two-sided')
                p_mat[i, j] = float(p)
            except Exception:
                pass

    pairwise_df = pd.DataFrame(p_mat, index=names, columns=names)
    _save_csv(pairwise_df, os.path.join(out, 'pairwise_wilcoxon_matrix.csv'), index=True)

    # Pairwise heatmap
    log_p = -np.log10(np.where(p_mat > 0, p_mat, 1e-10))
    np.fill_diagonal(log_p, 0)
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(log_p, cmap='Reds', vmin=0, vmax=8, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title('Pairwise Wilcoxon Significance  (−log₁₀ p)  —  darker = more significant',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='−log₁₀(p-value)')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'pairwise_wilcoxon_heatmap.png'))

    # Speed × blur interaction table
    irows = []
    for speed in SPEEDS:
        for blur in BLURS:
            row = {'speed': speed, 'blur': blur}
            for name, cs in all_cs.items():
                sub = cs[np.isclose(cs['speed'], speed, atol=1e-3) & (cs['blur'] == blur)]
                row[name] = float(sub['mean_e3d'].values[0]) if len(sub) > 0 else np.nan
            irows.append(row)
    _save_csv(pd.DataFrame(irows), os.path.join(out, 'speed_blur_interaction.csv'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Temporal Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_10_temporal(anal_data: dict):
    out = os.path.join(_OUT_DIR, '10_temporal')

    # Identify top 5
    sess_means = sorted((float(np.nanmean(per_config_stats(df)['mean_e3d'])), name)
                        for name, df in anal_data.items())
    top5 = [name for _, name in sess_means[:5]]

    # error_over_time_top5.png  (config 1 = speed=0.2, blur=0)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, name in enumerate(top5):
        ax = axes[i]
        df = anal_data[name]
        cfg = df[df['config_id'] == 1].copy()
        if len(cfg) == 0:
            cfg = df[np.isclose(df['speed_scalar'], 0.2, atol=1e-3) & (df['blur_level'] == 0)].copy()
        e3d = cfg['error_3d'].values.astype(float)
        window = max(1, len(e3d) // 200)
        smoothed = pd.Series(e3d).rolling(window, min_periods=1, center=True).mean().values
        ax.plot(np.arange(len(e3d)), smoothed, color=FAMILY_COLORS[META[name]['family']], lw=1.5)
        ax.set_title(name, fontsize=8, fontweight='bold')
        ax.set_xlabel('Row index', fontsize=8)
        if i == 0:
            ax.set_ylabel('error_3d (m)', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(bottom=0)
    fig.suptitle('Error 3D Over Time — Top 5 Sessions  (speed=0.2, blur=0)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'error_over_time_top5.png'))

    # error_over_time_baseline.png
    if 'baseline' in anal_data:
        df = anal_data['baseline']
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
        for si, speed in enumerate(SPEEDS):
            ax = axes[si]
            cfg = df[np.isclose(df['speed_scalar'], speed, atol=1e-3) & (df['blur_level'] == 0)].copy()
            if len(cfg) == 0:
                continue
            e3d = cfg['error_3d'].values.astype(float)
            window = max(1, len(e3d) // 200)
            smoothed = pd.Series(e3d).rolling(window, min_periods=1, center=True).mean().values
            ax.plot(np.arange(len(e3d)), smoothed, color='#555555', lw=1.5)
            ax.set_title(f'Speed = {speed}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Row index', fontsize=8)
            if si == 0:
                ax.set_ylabel('error_3d (m)', fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_ylim(bottom=0)
        fig.suptitle('Baseline Error 3D Over Time — blur=0, all speeds',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        _save(fig, os.path.join(out, 'error_over_time_baseline.png'))

    # within_test_stability.csv
    stab_rows = []
    for name, df in anal_data.items():
        for cid, grp in df.groupby('config_id'):
            e3d = grp['error_3d'].values.astype(float)
            half = len(e3d) // 2
            stab_rows.append({
                'session': name, 'config_id': int(cid),
                'speed': float(grp['speed_scalar'].iloc[0]),
                'blur': int(grp['blur_level'].iloc[0]),
                'first_half_mean': float(np.nanmean(e3d[:half])),
                'second_half_mean': float(np.nanmean(e3d[half:])),
                'delta': float(np.nanmean(e3d[half:]) - np.nanmean(e3d[:half])),
            })
    _save_csv(pd.DataFrame(stab_rows), os.path.join(out, 'within_test_stability.csv'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Spatial Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_11_spatial(plot_data: dict):
    out = os.path.join(_OUT_DIR, '11_spatial')

    # orbit_radius_heatmap.png  (small multiples)
    names = list(plot_data.keys())
    n = len(names)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    orbit_grids = {}
    all_orb = []
    for name, df in plot_data.items():
        grid = np.full((5, 5), np.nan)
        for cid, grp in df.groupby('config_id'):
            si = _speed_idx(float(grp['speed_scalar'].iloc[0]))
            bi_val = int(grp['blur_level'].iloc[0])
            bi = BLURS.index(bi_val) if bi_val in BLURS else -1
            if si >= 0 and bi >= 0:
                dfc = grp['dist_from_center'].values.astype(float)
                grid[si, bi] = float(np.nanmean(dfc))
        orbit_grids[name] = grid
        all_orb.extend(grid[~np.isnan(grid)].tolist())

    vmax_orb = min(float(np.nanpercentile(all_orb, 95)), 10.0) if all_orb else 5.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.4))
    axes = np.array(axes).flatten()
    im = None
    for i, name in enumerate(names):
        ax = axes[i]
        im = ax.imshow(orbit_grids[name], vmin=0, vmax=vmax_orb, cmap='RdYlBu', aspect='auto')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['0','1','2','3','4'], fontsize=7)
        ax.set_yticks(range(5))
        ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7)
        ax.set_title(name.replace('_', '\n'), fontsize=7, fontweight='bold')
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='mean dist_from_center (m)')
    fig.suptitle(f'Orbit Radius Heatmaps — All Sessions  (track radius = {TRACK_RADIUS:.3f} m)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    _save(fig, os.path.join(out, 'orbit_radius_heatmap.png'))

    # orbit_radius_vs_ideal.png
    orb_rows = []
    for name, df in plot_data.items():
        dfc = df['dist_from_center'].values.astype(float)
        mean_dfc = float(np.nanmean(dfc))
        orb_rows.append({'session': name, 'family': META[name]['family'],
                         'mean_dist_from_center': mean_dfc,
                         'delta_from_ideal': mean_dfc - TRACK_RADIUS})
    orb_df = pd.DataFrame(orb_rows).sort_values('delta_from_ideal')
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#4CAF50' if abs(d) < 0.3 else ('#FF9800' if abs(d) < 1.0 else '#F44336')
              for d in orb_df['delta_from_ideal']]
    ax.barh(range(len(orb_df)), orb_df['delta_from_ideal'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.set_yticks(range(len(orb_df)))
    ax.set_yticklabels(orb_df['session'], fontsize=9)
    ax.set_xlabel('Δ from ideal track radius (m)  [0 = perfectly on track]', fontsize=11)
    ax.set_title('Mean Orbit Radius Deviation from Track Centreline', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'orbit_radius_vs_ideal.png'))

    # altitude_deviation.png
    alt_rows = []
    for name, df in plot_data.items():
        pz = df['pos_z'].values.astype(float)
        valid = ~np.isnan(pz)
        alt_rows.append({'session': name, 'family': META[name]['family'],
                         'mean_alt_deviation': float(np.nanmean(np.abs(pz[valid] - TARGET_Z))) if valid.any() else np.nan})
    alt_df = pd.DataFrame(alt_rows).sort_values('mean_alt_deviation')
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [FAMILY_COLORS[META[n]['family']] for n in alt_df['session']]
    ax.barh(range(len(alt_df)), alt_df['mean_alt_deviation'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(alt_df)))
    ax.set_yticklabels(alt_df['session'], fontsize=9)
    ax.set_xlabel('Mean |pos_z − 1.0| (m)', fontsize=11)
    ax.set_title('Mean Altitude Deviation from Target (1.0 m)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'altitude_deviation.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — Lap Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_12_lap_analysis(raw_data: dict):
    out = os.path.join(_OUT_DIR, '12_lap_analysis')

    session_lap = {}
    all_lap_rows = []
    for name, df in raw_data.items():
        ls = compute_lap_stats(df)
        ls['session'] = name
        ls['family'] = META[name]['family']
        ls['output'] = META[name]['output']
        session_lap[name] = ls
        all_lap_rows.append(ls)

    lap_all = pd.concat(all_lap_rows, ignore_index=True)
    _save_csv(lap_all[['session', 'family', 'output', 'speed',
                        'mean_laps', 'std_laps', 'mean_t_first', 'std_t_first']],
              os.path.join(out, 'lap_summary.csv'))

    # laps_vs_speed_all.png
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, ls in session_lap.items():
        ls_s = ls.sort_values('speed')
        ax.plot(ls_s['speed'], ls_s['mean_laps'],
                marker=OUTPUT_MARKERS[META[name]['output']],
                color=FAMILY_COLORS[META[name]['family']],
                ls=OUTPUT_LS[META[name]['output']],
                lw=1.5, alpha=0.85, label=name)
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean Laps Completed', fontsize=12)
    ax.set_title('Laps Completed vs. Speed — All Sessions', fontsize=13, fontweight='bold')
    ax.set_xticks(SPEEDS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'laps_vs_speed_all.png'))

    # laps_vs_speed_by_family.png
    fig, ax = plt.subplots(figsize=(12, 7))
    for fam in sorted(set(META[n]['family'] for n in SESSION_NAMES)):
        fnames = [n for n in SESSION_NAMES if META[n]['family'] == fam and n in session_lap]
        if not fnames:
            continue
        speed_laps = {s: [] for s in SPEEDS}
        for name in fnames:
            ls = session_lap[name]
            for _, row in ls.iterrows():
                s = row['speed']
                if any(np.isclose(s, ss, atol=1e-3) for ss in SPEEDS):
                    speed_laps[min(SPEEDS, key=lambda ss: abs(ss - s))].append(row['mean_laps'])
        ys = [np.nanmean(speed_laps[s]) if speed_laps[s] else np.nan for s in SPEEDS]
        ax.plot(SPEEDS, ys, marker='o', color=FAMILY_COLORS[fam], lw=2.2, label=FAMILY_DISPLAY.get(fam, fam))
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean Laps Completed', fontsize=12)
    ax.set_title('Laps Completed vs. Speed — By Model Family', fontsize=13, fontweight='bold')
    ax.set_xticks(SPEEDS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(out, 'laps_vs_speed_by_family.png'))

    # time_to_first_lap.png
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, ls in session_lap.items():
        ls_s = ls.sort_values('speed').dropna(subset=['mean_t_first'])
        if len(ls_s) == 0:
            continue
        ax.plot(ls_s['speed'], ls_s['mean_t_first'],
                marker=OUTPUT_MARKERS[META[name]['output']],
                color=FAMILY_COLORS[META[name]['family']],
                ls=OUTPUT_LS[META[name]['output']],
                lw=1.5, alpha=0.85, label=name)
    ax.set_xlabel('Speed Scalar', fontsize=12)
    ax.set_ylabel('Mean Time to First Lap (s)', fontsize=12)
    ax.set_title('Time to First Lap vs. Speed — All Sessions', fontsize=13, fontweight='bold')
    ax.set_xticks(SPEEDS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    fig.tight_layout()
    _save(fig, os.path.join(out, 'time_to_first_lap.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Shared rcParams for the merged plotting sections (13–15).
# Applied via matplotlib.rc_context so sections 01–12 keep their own styling.
# ─────────────────────────────────────────────────────────────────────────────
_PLOT_RC = {
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 180,
}


def _with_plot_rc(rc=None):
    """Decorator: run a whole section under matplotlib.rc_context(_PLOT_RC) so
    the merged plotting sections (13–15) reproduce the standalone scripts'
    styling without disturbing the rcParams used by sections 01–12."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            with matplotlib.rc_context(rc if rc is not None else _PLOT_RC):
                return fn(*args, **kwargs)
        return wrapper
    return deco


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 — Convergence Time + Convergence Rate
# (merged from compute_convergence_time.py and plot_convergence_rate.py)
#
# Convergence time = elapsed seconds at which error_3d first drops below
# CONV_THRESHOLD and stays below it for CONV_SUSTAINED_SECONDS consecutively.
# ─────────────────────────────────────────────────────────────────────────────
CONV_THRESHOLD         = 0.10    # metres — 3D error must stay below this value
CONV_SUSTAINED_SECONDS = 30.0    # seconds of continuous below-threshold error required
CONV_TIMESTEP_S        = 0.032   # nominal simulation timestep (32 ms per row)
CONV_SUSTAINED_ROWS    = int(CONV_SUSTAINED_SECONDS / CONV_TIMESTEP_S)  # ~938 rows


def _conv_first_time(error_3d, timestamps):
    """Timestamp (s) at which error_3d first sustains CONV_SUSTAINED_ROWS
    consecutive rows below CONV_THRESHOLD; NaN if it never occurs."""
    n = len(error_3d)
    if n < CONV_SUSTAINED_ROWS:
        return np.nan
    below = (error_3d < CONV_THRESHOLD).astype(np.int32)
    cs = np.zeros(n + 1, dtype=np.int32)
    cs[1:] = np.cumsum(below)
    num_windows = n - CONV_SUSTAINED_ROWS + 1
    window_sums = cs[CONV_SUSTAINED_ROWS: CONV_SUSTAINED_ROWS + num_windows] - cs[:num_windows]
    hits = np.where(window_sums == CONV_SUSTAINED_ROWS)[0]
    if len(hits) == 0:
        return np.nan
    return float(timestamps[hits[0]])


def _conv_compute():
    """Compute convergence_time_raw.csv and convergence_time.csv."""
    global _files_saved
    out = os.path.join(_OUT_DIR, '13_convergence')
    os.makedirs(out, exist_ok=True)

    sessions = [
        'baseline',
        'train_ycommand',        'train_ycontinuous',
        'notrain_ycommand',      'notrain_ycontinuous',
        'mlp_256_128_ycommand',  'mlp_256_128_ycontinuous',
        'mlp_64_32_ycommand',    'mlp_64_32_ycontinuous',
        'mlp_16_8_ycommand',     'mlp_16_8_ycontinuous',
        'neat_ycommand',         'neat_ycontinuous',
        'features_ycommand',     'features_ycontinuous',
        'exp_e_ycommand',        'exp_e_ycontinuous',
        'exp_f_ycommand',        'exp_f_ycontinuous',
    ]

    raw_records = []
    for session in sessions:
        csv_path = os.path.join(_ANAL_SRC, f'{session}_error_log.csv')
        if not os.path.exists(csv_path):
            print(f'  [SKIP] {session} — file not found: {csv_path}')
            continue
        df = pd.read_csv(csv_path)
        for config_id in sorted(df['config_id'].unique()):
            sub = (df[df['config_id'] == config_id]
                   .sort_values('row_index')
                   .reset_index(drop=True))
            speed = float(sub['speed_scalar'].iloc[0])
            blur  = int(sub['blur_level'].iloc[0])
            sub_valid = sub.dropna(subset=['error_3d'])
            if len(sub_valid) < CONV_SUSTAINED_ROWS:
                conv_time = np.nan
            else:
                conv_time = _conv_first_time(
                    sub_valid['error_3d'].values,
                    sub_valid['timestamp'].values,
                )
            raw_records.append({
                'session':            session,
                'speed':              speed,
                'blur_level':         blur,
                'convergence_time_s': conv_time,
                'converged':          not np.isnan(conv_time),
                'n_rows':             len(sub_valid),
            })

    raw_df = pd.DataFrame(raw_records)
    raw_path = os.path.join(out, 'convergence_time_raw.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f'  saved: {os.path.relpath(raw_path, _ROOT)}')
    _files_saved += 1

    agg = (raw_df
           .groupby(['session', 'speed'])
           .agg(
               mean_conv_time_s=('convergence_time_s', 'mean'),
               std_conv_time_s=('convergence_time_s', lambda x: x.std(ddof=1)),
               pct_converged=('converged',
                              lambda x: round(100.0 * x.sum() / len(x), 1)),
               n_blur_levels=('blur_level', 'count'),
           )
           .reset_index())
    agg_path = os.path.join(out, 'convergence_time.csv')
    agg.to_csv(agg_path, index=False)
    print(f'  saved: {os.path.relpath(agg_path, _ROOT)}')
    _files_saved += 1


def _conv_plot():
    """convergence_rate_main.png + convergence_rate_heatmap.png."""
    global _files_saved
    out = os.path.join(_OUT_DIR, '13_convergence')
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(os.path.join(out, 'convergence_time.csv'))
    speeds = sorted(df['speed'].unique())

    STYLE = {
        'baseline':            ('#4dac26', 'o', '-',  'Baseline (rule-based)'),
        'train_ycontinuous':   ('#2166ac', 's', '-',  'Trained NE (continuous)'),
        'notrain_ycontinuous': ('#92c5de', '^', '--', 'Untrained NE (continuous)'),
        'train_ycommand':      ('#d6604d', 'D', '-',  'Trained NE (binary)'),
        'notrain_ycommand':    ('#f4a582', 'v', '--', 'Untrained NE (binary)'),
        'exp_e_ycontinuous':   ('#5aadff', 'P', ':',  'Backprop CNN (continuous)'),
        'features_ycontinuous':('#1a9641', 'X', ':',  'Feature MLP (continuous)'),
    }

    # FIGURE 1 — main sessions: convergence rate vs speed
    MAIN = ['baseline', 'train_ycontinuous', 'notrain_ycontinuous',
            'train_ycommand', 'notrain_ycommand']
    fig, ax = plt.subplots(figsize=(8, 5))
    for sess in MAIN:
        color, marker, ls, label = STYLE[sess]
        sub = df[df['session'] == sess].sort_values('speed')
        rate = []
        for s in speeds:
            row = sub[sub['speed'] == s]
            rate.append(float(row['pct_converged'].values[0]) if len(row) > 0 else 0.0)
        ax.plot(speeds, rate, marker=marker, linestyle=ls, color=color,
                label=label, linewidth=2, markersize=7)
    ax.set_xlabel('Speed Scalar')
    ax.set_ylabel('Convergence Rate (%)')
    ax.set_title('Convergence Rate vs. Speed — Main Sessions', fontweight='bold', pad=10)
    ax.set_xticks(speeds)
    ax.set_ylim(-5, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axhline(100, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.axhline(0,   color='grey', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.legend(framealpha=0.9, loc='upper left')
    ax.grid(linestyle=':', alpha=0.4)
    for path in [os.path.join(out, 'convergence_rate_main.png')]:
        plt.savefig(path)
        print(f'  saved: {os.path.relpath(path, _ROOT)}')
        _files_saved += 1
    plt.close()

    # FIGURE 2 — all 19 sessions: heatmap of convergence rate
    pivot = df.pivot_table(index='session', columns='speed',
                           values='pct_converged', aggfunc='first').fillna(0)
    pivot['_mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('_mean', ascending=False).drop(columns='_mean')
    LABELS = {
        'baseline':               'Baseline',
        'train_ycontinuous':      'Trained NE (continuous)',
        'exp_e_ycontinuous':      'Backprop CNN (continuous)',
        'features_ycontinuous':   'Feature MLP (continuous)',
        'notrain_ycontinuous':    'Untrained NE (continuous)',
        'mlp_16_8_ycontinuous':   'MLP-16-8 (continuous)',
        'mlp_64_32_ycontinuous':  'MLP-64-32 (continuous)',
        'mlp_256_128_ycontinuous':'MLP-256-128 (continuous)',
        'train_ycommand':         'Trained NE (binary)',
        'exp_e_ycommand':         'Backprop CNN (binary)',
        'mlp_16_8_ycommand':      'MLP-16-8 (binary)',
        'mlp_256_128_ycommand':   'MLP-256-128 (binary)',
        'notrain_ycommand':       'Untrained NE (binary)',
        'mlp_64_32_ycommand':     'MLP-64-32 (binary)',
        'neat_ycommand':          'NEAT (binary)',
        'neat_ycontinuous':       'NEAT (continuous)',
        'features_ycommand':      'Feature MLP (binary)',
        'exp_f_ycommand':         'Ext. NE (binary)',
        'exp_f_ycontinuous':      'Ext. NE (continuous)',
    }
    row_labels = [LABELS.get(s, s) for s in pivot.index]
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100, interpolation='nearest')
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([str(s) for s in pivot.columns])
    ax.set_xlabel('Speed Scalar')
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title('Convergence Rate (%) — All 19 Sessions', fontweight='bold', pad=10)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = int(pivot.values[i, j])
            txt_color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=8, color=txt_color, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Convergence Rate (%)', shrink=0.6)
    plt.tight_layout()
    for path in [os.path.join(out, 'convergence_rate_heatmap.png')]:
        plt.savefig(path)
        print(f'  saved: {os.path.relpath(path, _ROOT)}')
        _files_saved += 1
    plt.close()


def run_13_convergence():
    _conv_compute()
    with matplotlib.rc_context(_PLOT_RC):
        _conv_plot()


# ─────────────────────────────────────────────────────────────────────────────
# Section 14 — Overview Charts (merged from make_overview_charts.py)
# Reads section 01 + 08 outputs; writes 4 charts to charts/ at the repo root.
# ─────────────────────────────────────────────────────────────────────────────
@_with_plot_rc()
def run_14_overview_charts():
    global _files_saved
    os.makedirs(_CHARTS_DIR, exist_ok=True)
    session_df = pd.read_csv(os.path.join(_OUT_DIR, '01_overview', 'session_summary.csv'))
    config_df  = pd.read_csv(os.path.join(_OUT_DIR, '01_overview', 'config_summary.csv'))
    scores_df  = pd.read_csv(os.path.join(_OUT_DIR, '08_performance_score', 'performance_scores.csv'))

    C_CONT  = '#2166ac'   # ycontinuous  – blue
    C_BIN   = '#d6604d'   # ycommand     – red-orange
    C_BASE  = '#4dac26'   # baseline     – green

    def label_for(session):
        MAP = {
            'baseline':               'Baseline',
            'train_ycontinuous':      'Main Trained NE\n(continuous)',
            'train_ycommand':         'Main Trained NE\n(binary)',
            'notrain_ycontinuous':    'Main Untrained NE\n(continuous)',
            'notrain_ycommand':       'Main Untrained NE\n(binary)',
            'exp_e_ycontinuous':      'Backprop CNN\n(continuous)',
            'exp_e_ycommand':         'Backprop CNN\n(binary)',
            'exp_f_ycontinuous':      'Ext. NE CNN\n(continuous)',
            'exp_f_ycommand':         'Ext. NE CNN\n(binary)',
            'mlp_256_128_ycontinuous':'MLP-256-128\n(continuous)',
            'mlp_256_128_ycommand':   'MLP-256-128\n(binary)',
            'mlp_64_32_ycontinuous':  'MLP-64-32\n(continuous)',
            'mlp_64_32_ycommand':     'MLP-64-32\n(binary)',
            'mlp_16_8_ycontinuous':   'MLP-16-8\n(continuous)',
            'mlp_16_8_ycommand':      'MLP-16-8\n(binary)',
            'neat_ycontinuous':       'NEAT\n(continuous)',
            'neat_ycommand':          'NEAT\n(binary)',
            'features_ycontinuous':   'Feature MLP\n(continuous)',
            'features_ycommand':      'Feature MLP\n(binary)',
        }
        return MAP.get(session, session)

    def bar_color(row):
        if row['session'] == 'baseline':
            return C_BASE
        return C_CONT if row['output'] == 'ycontinuous' else C_BIN

    def _savefig(out_path):
        plt.savefig(out_path)
        plt.close()
        print(f'  saved: {os.path.relpath(out_path, _ROOT)}')
        global _files_saved
        _files_saved += 1

    # CHART 1 — all 19 sessions ranked by mean 3D error
    df = session_df.sort_values('mean_error_3d', ascending=True).reset_index(drop=True)
    labels = [label_for(s) for s in df['session']]
    values = df['mean_error_3d'].values
    errs   = df['std_error_3d'].values
    colors = [bar_color(r) for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))
    ax.barh(y, values, xerr=errs, color=colors, edgecolor='white',
            linewidth=0.5, capsize=3, error_kw={'elinewidth': 0.8, 'alpha': 0.7})
    ax.set_xscale('log')
    ax.set_xlabel('Mean 3D Error (m)  [log scale]')
    ax.set_title('All 19 Sessions Ranked by Mean 3D Tracking Error', pad=10, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    for i, (v, e) in enumerate(zip(values, errs)):
        ax.text(v * 1.05, i, f'{v:.4f} m', va='center', fontsize=8, color='#333333')
    bl = df[df['session'] == 'baseline']['mean_error_3d'].values[0]
    ax.axvline(bl, color=C_BASE, linestyle='--', linewidth=1.2, alpha=0.7, label=f'Baseline ({bl:.4f} m)')
    ax.set_xlim(left=0.003)
    ax.grid(axis='x', which='both', linestyle=':', alpha=0.4)
    legend_patches = [
        mpatches.Patch(color=C_BASE, label='Rule-based baseline'),
        mpatches.Patch(color=C_CONT, label='Continuous output (ycontinuous)'),
        mpatches.Patch(color=C_BIN,  label='Binary output (ycommand)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', framealpha=0.85)
    plt.tight_layout()
    _savefig(os.path.join(_CHARTS_DIR, 'chart1_all_sessions_ranked.png'))

    # CHART 2 — within-architecture output type comparison
    pairs = [
        ('Baseline\n(rule-based)',   'baseline',          None),
        ('Main Trained NE',          'train_ycontinuous',  'train_ycommand'),
        ('Main Untrained NE',        'notrain_ycontinuous','notrain_ycommand'),
        ('Backprop CNN\n(Exp E)',    'exp_e_ycontinuous',  'exp_e_ycommand'),
        ('Ext. NE CNN\n(Exp F)',     'exp_f_ycontinuous',  'exp_f_ycommand'),
        ('Feature MLP',              'features_ycontinuous','features_ycommand'),
        ('NEAT',                     'neat_ycontinuous',    'neat_ycommand'),
        ('MLP-16-8\n(Exp A)',        'mlp_16_8_ycontinuous','mlp_16_8_ycommand'),
    ]
    lookup = session_df.set_index('session')['mean_error_3d'].to_dict()
    std_lookup = session_df.set_index('session')['std_error_3d'].to_dict()
    n = len(pairs)
    x = np.arange(n)
    w = 0.35
    vals_cont, vals_bin, errs_cont, errs_bin = [], [], [], []
    for _, sc, sy in pairs:
        vals_cont.append(lookup.get(sc, np.nan))
        vals_bin.append(lookup.get(sy, np.nan) if sy else np.nan)
        errs_cont.append(std_lookup.get(sc, 0))
        errs_bin.append(std_lookup.get(sy, 0) if sy else 0)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar(x - w/2, vals_cont, w, color=C_CONT, label='Continuous output',
           edgecolor='white', capsize=4, yerr=errs_cont,
           error_kw={'elinewidth': 0.8, 'alpha': 0.7})
    ax.bar(x + w/2, vals_bin, w, color=C_BIN, label='Binary output',
           edgecolor='white', capsize=4, yerr=errs_bin,
           error_kw={'elinewidth': 0.8, 'alpha': 0.7})
    ax.set_yscale('log')
    ax.set_ylabel('Mean 3D Error (m)  [log scale]')
    ax.set_title('Output Type Comparison Within Each Model Architecture', pad=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], rotation=15, ha='right')
    ax.grid(axis='y', which='both', linestyle=':', alpha=0.4)
    bl = lookup['baseline']
    ax.axhline(bl, color=C_BASE, linestyle='--', linewidth=1.2, alpha=0.8,
               label=f'Baseline ({bl:.4f} m)')
    ax.legend(framealpha=0.85, loc='upper right')
    plt.tight_layout()
    _savefig(os.path.join(_CHARTS_DIR, 'chart2_within_architecture_comparison.png'))

    # CHART 3 — speed vs error for top performers vs baseline
    SESSIONS3 = [
        ('baseline',            'Baseline',             C_BASE,   '-o'),
        ('train_ycontinuous',   'Train CNN (cont.)',    C_CONT,   '-s'),
        ('exp_e_ycontinuous',   'Backprop CNN (cont.)', '#5aadff','-^'),
        ('features_ycontinuous','Feature MLP (cont.)',  '#1a9641','-D'),
    ]
    speeds3 = sorted(config_df['speed'].unique())
    speed_mean = (config_df.groupby(['session', 'speed'])['mean_e3d']
                           .mean().reset_index())
    fig, ax = plt.subplots(figsize=(8, 5))
    for sess, label, color, marker in SESSIONS3:
        sub = speed_mean[speed_mean['session'] == sess].sort_values('speed')
        ax.plot(sub['speed'], sub['mean_e3d'], marker[1], linestyle=marker[0],
                color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel('Speed Scalar')
    ax.set_ylabel('Mean 3D Error (m)')
    ax.set_title('Speed vs. Mean 3D Error — Top Performers vs. Baseline', pad=10, fontweight='bold')
    ax.set_xticks(speeds3)
    ax.set_xticklabels([str(s) for s in speeds3])
    ax.legend(framealpha=0.85)
    ax.grid(linestyle=':', alpha=0.4)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _savefig(os.path.join(_CHARTS_DIR, 'chart3_speed_vs_error_top4.png'))

    # CHART 4 — composite performance score, stacked components
    mean_comp = (scores_df.groupby('session')[['accuracy', 'stability', 'endurance', 'precision']]
                          .mean() * 25)
    mean_comp['composite'] = mean_comp.sum(axis=1)
    mean_comp = mean_comp.sort_values('composite', ascending=False).reset_index()
    labels = [label_for(s) for s in mean_comp['session']]
    x = np.arange(len(mean_comp))
    comp_colors = ['#2166ac', '#92c5de', '#4dac26', '#d6604d']
    components  = ['accuracy', 'stability', 'endurance', 'precision']
    comp_labels = ['Accuracy (25)', 'Stability (25)', 'Endurance (25)', 'Precision (25)']
    fig, ax = plt.subplots(figsize=(14, 5.5))
    bottom = np.zeros(len(mean_comp))
    for comp, color, clabel in zip(components, comp_colors, comp_labels):
        vals = mean_comp[comp].values
        ax.bar(x, vals, bottom=bottom, color=color, label=clabel, edgecolor='white', linewidth=0.4)
        bottom += vals
    for i, v in enumerate(mean_comp['composite']):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=7.5, color='#333333')
    ax.set_ylabel('Composite Performance Score (0–100)')
    ax.set_title('Composite Performance Score — All Sessions\n(Accuracy + Stability + Endurance + Precision)',
                 pad=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
    ax.set_ylim(0, 110)
    ax.axhline(99.11, color=C_BASE, linestyle='--', linewidth=1.2, alpha=0.8,
               label='Baseline score (99.11)')
    ax.legend(loc='lower left', framealpha=0.85, ncol=3)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    _savefig(os.path.join(_CHARTS_DIR, 'chart4_performance_score_stacked.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 — Training-time convergence (merged from plot_training_convergence.py)
# Reads per-model analytics/training_curve CSVs under data/model_data/.
# ─────────────────────────────────────────────────────────────────────────────
def run_15_training_convergence():
    global _files_saved
    SESSIONS = [
        ('train_ycontinuous',       'main_phase/train_ycontinuous/results/analytics.csv',
         'elapsed_seconds', 'best_fitness', 'neg'),
        ('train_ycommand',          'main_phase/train_ycommand/results/analytics.csv',
         'elapsed_seconds', 'best_fitness', 'neg'),
        ('notrain_ycontinuous',     'main_phase/notrain_ycontinuous/results/analytics.csv',
         'elapsed_seconds', 'best_fitness', 'neg'),
        ('notrain_ycommand',        'main_phase/notrain_ycommand/results/analytics.csv',
         'elapsed_seconds', 'best_fitness', 'neg'),
        ('mlp_256_128_ycontinuous', 'extra_observation/exp_a_small_mlp/mlp_256_128_ycontinuous/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('mlp_256_128_ycommand',    'extra_observation/exp_a_small_mlp/mlp_256_128_ycommand/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('mlp_64_32_ycontinuous',   'extra_observation/exp_a_small_mlp/mlp_64_32_ycontinuous/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('mlp_64_32_ycommand',      'extra_observation/exp_a_small_mlp/mlp_64_32_ycommand/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('mlp_16_8_ycontinuous',    'extra_observation/exp_a_small_mlp/mlp_16_8_ycontinuous/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('mlp_16_8_ycommand',       'extra_observation/exp_a_small_mlp/mlp_16_8_ycommand/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('neat_ycontinuous',        'extra_observation/exp_b_neat/neat_ycontinuous/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('neat_ycommand',           'extra_observation/exp_b_neat/neat_ycommand/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('features_ycontinuous',    'extra_observation/exp_c_features/features_ycontinuous/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('features_ycommand',       'extra_observation/exp_c_features/features_ycommand/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('exp_e_ycontinuous',       'extra_observation/exp_e_standard_backprop/ycontinuous/training_curve.csv',
         'elapsed_s', 'val_loss', 'val_loss'),
        ('exp_e_ycommand',          'extra_observation/exp_e_standard_backprop/ycommand/training_curve.csv',
         'elapsed_s', 'val_loss', 'val_loss'),
        ('exp_f_ycontinuous',       'extra_observation/exp_f_extended_notrain/ycontinuous/analytics.csv',
         'elapsed_s', 'best', 'neg'),
        ('exp_f_ycommand',          'extra_observation/exp_f_extended_notrain/ycommand/analytics.csv',
         'elapsed_s', 'best', 'neg'),
    ]
    FAMILY_STYLE = {
        'main_trained_NE':   ('#1f77b4', '-',  'Main Trained NE'),
        'main_untrained_NE': ('#aec7e8', '--', 'Main Untrained NE'),
        'small_mlp':         ('#2ca02c', ':',  'Small MLP (untrained NE)'),
        'neat':              ('#ff7f0e', '-.', 'NEAT'),
        'feature_mlp':       ('#9467bd', '-',  'Feature MLP (untrained NE)'),
        'exp_e':             ('#d62728', '-',  'Backprop CNN (Exp E)'),
        'exp_f':             ('#8c564b', '--', 'Extended NE (Exp F)'),
    }

    def family_key(label):
        if label.startswith('train_'):    return 'main_trained_NE'
        if label.startswith('notrain_'):  return 'main_untrained_NE'
        if label.startswith('mlp_'):      return 'small_mlp'
        if label.startswith('neat_'):     return 'neat'
        if label.startswith('features_'): return 'feature_mlp'
        if label.startswith('exp_e_'):    return 'exp_e'
        if label.startswith('exp_f_'):    return 'exp_f'
        return None

    def load_curve(rel_csv, time_col, val_col, transform):
        df = pd.read_csv(os.path.join(_MODEL_DATA_DIR, rel_csv))
        t = pd.to_numeric(df[time_col], errors='coerce').values
        v = pd.to_numeric(df[val_col],  errors='coerce').values
        if transform == 'neg':
            v = -v
        valid = (~np.isnan(t)) & (~np.isnan(v)) & (t > 0) & (v > 0)
        t = t[valid]
        v = v[valid]
        if len(t) > 1:
            decreases = np.where(np.diff(t) < 0)[0]
            if len(decreases) > 0:
                cut = decreases[0] + 1
                t = t[:cut]
                v = v[:cut]
        return t, v

    rc = dict(_PLOT_RC)
    rc['legend.fontsize'] = 8.5
    with matplotlib.rc_context(rc):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        used_top, used_bot = set(), set()
        for label, rel_csv, time_col, val_col, transform in SESSIONS:
            try:
                t, v = load_curve(rel_csv, time_col, val_col, transform)
            except FileNotFoundError:
                print(f'  [WARN] Missing: {rel_csv}')
                continue
            if len(t) == 0:
                continue
            fam = family_key(label)
            color, ls, fam_label = FAMILY_STYLE[fam]
            if label.endswith('ycontinuous'):
                ax, used = ax_top, used_top
            else:
                ax, used = ax_bot, used_bot
            legend_label = fam_label if fam not in used else None
            ax.plot(t, v, color=color, linestyle=ls, alpha=0.85, linewidth=1.7,
                    label=legend_label)
            ax.plot(t[0],  v[0],  marker='o', markersize=6,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=5)
            ax.plot(t[-1], v[-1], marker='o', markersize=6,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=5)
            used.add(fam)
        for ax, panel_title in [(ax_top, 'Continuous output (ycontinuous) sessions'),
                                (ax_bot, 'Binary output (ycommand) sessions')]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('MSE (log scale)', fontsize=11)
            ax.set_title(panel_title, fontweight='bold', pad=6, fontsize=12)
            ax.grid(which='both', linestyle=':', alpha=0.4)
            ax.legend(loc='upper right', framealpha=0.9, ncol=2,
                      title='Training family', fontsize=9)
        ax_bot.set_xlabel('Elapsed training time (seconds, log scale)', fontsize=11)
        fig.suptitle('Training-time convergence: fitness/loss vs. elapsed time',
                     fontweight='bold', y=0.995, fontsize=13)
        fig.text(0.98, 0.015,
                 'Main-phase (train_, notrain_) ran on Apple M2 Ultra MPS; '
                 'all supplementary experiments ran on Google Colab A100 GPU. '
                 'Cross-platform time comparisons are visual only.',
                 fontsize=8, color='#444444',
                 horizontalalignment='right', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='#f8f8f8', edgecolor='#888888', alpha=0.9))
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        os.makedirs(_OUT_DIR, exist_ok=True)
        for path in [os.path.join(_OUT_DIR, 'training_convergence.png')]:
            plt.savefig(path)
            print(f'  saved: {os.path.relpath(path, _ROOT)}')
            _files_saved += 1
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 16 — Per-session path plots (merged from plot_all_paths.py)
# Reads FOR_PLOTTING cleaned data; writes one {session}_paths.png per session.
# Uses default rcParams (no _PLOT_RC) so it matches the standalone script exactly.
# ─────────────────────────────────────────────────────────────────────────────
_PATH_TRACK_CX    = 0.000018
_PATH_TRACK_CY    = -2.849774   # circle centre Y (TRACK_RADIUS reused from module constants)
_PATH_PLOT_RADIUS = 10.0        # metres from track centre — clip paths beyond this
_PATH_PLOT_STEP   = 5           # downsample: 1 point per ~160 ms
_PATH_BLUR_STYLE = {
    0: ('#1a1aff', 'solid',  2.0, 'blur 0 (none)'),
    1: ('#00aaff', 'solid',  1.8, 'blur 1 (light)'),
    2: ('#33bb55', 'dashed', 1.8, 'blur 2 (moderate)'),
    3: ('#ff8833', 'dashed', 1.8, 'blur 3 (heavy)'),
    4: ('#cc2222', 'solid',  1.8, 'blur 4 (max)'),
}


def _path_draw_track_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(_PATH_TRACK_CX + TRACK_RADIUS * np.cos(theta),
            _PATH_TRACK_CY + TRACK_RADIUS * np.sin(theta),
            color='gray', lw=1.2, ls='-', alpha=0.35, zorder=1)
    ax.plot(_PATH_TRACK_CX, _PATH_TRACK_CY, 'k+', markersize=6, alpha=0.4, zorder=1)


def _path_plot_session(session_name, out_dir):
    csv_path  = os.path.join(_PLOT_SRC, f'{session_name}_error_log.csv')
    plot_path = os.path.join(out_dir, f'{session_name}_paths.png')
    if not os.path.exists(csv_path):
        print(f'  [SKIP] {session_name} — file not found')
        return False
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f'  [SKIP] {session_name} — empty CSV')
        return False

    fig, axes = plt.subplots(1, 5, figsize=(26, 6.5))
    fig.suptitle(session_name, fontsize=14, fontweight='bold', y=1.01)

    for col_idx, speed in enumerate(SPEEDS):
        ax = axes[col_idx]
        ax.set_aspect('equal')
        _path_draw_track_circle(ax)

        speed_df = df[np.isclose(df['speed_scalar'], speed)]
        legend_handles = [mpatches.Patch(color='gray', alpha=0.4,
                                          label='Track centreline')]

        for blur in BLURS:
            color, ls, lw, blur_label = _PATH_BLUR_STYLE[blur]
            cfg = speed_df[speed_df['blur_level'] == blur].sort_values('row_index')
            if cfg.empty:
                continue

            x_all = cfg['pos_x'].values[::_PATH_PLOT_STEP]
            y_all = cfg['pos_y'].values[::_PATH_PLOT_STEP]
            dist2d = np.sqrt((x_all - _PATH_TRACK_CX) ** 2 + (y_all - _PATH_TRACK_CY) ** 2)
            in_win = dist2d <= _PATH_PLOT_RADIUS   # NaN comparisons yield False → excluded

            x = x_all.copy().astype(float)
            y = y_all.copy().astype(float)
            x[~in_win] = np.nan
            y[~in_win] = np.nan

            n_valid = int(np.sum(~np.isnan(x)))
            pct_clip = (1 - in_win.mean()) * 100

            if n_valid > 1:
                valid_idx = np.where(~np.isnan(x))[0]
                ax.plot(x[valid_idx[0]],  y[valid_idx[0]],
                        's', color=color, markersize=6, zorder=5)
                ax.plot(x[valid_idx[-1]], y[valid_idx[-1]],
                        'D', color=color, markersize=6, zorder=5)
                ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=0.85, zorder=3)

                arrow_step = max(1, int(15 / 0.032 / _PATH_PLOT_STEP))
                for i in range(arrow_step, len(x), arrow_step):
                    if np.isnan(x[i]) or np.isnan(x[i - 1]):
                        continue
                    dx = x[i] - x[i - 1]
                    dy = y[i] - y[i - 1]
                    if abs(dx) + abs(dy) > 1e-4:
                        ax.annotate('', xy=(x[i], y[i]),
                                    xytext=(x[i - 1], y[i - 1]),
                                    arrowprops=dict(arrowstyle='->',
                                                    color=color, lw=1.0,
                                                    mutation_scale=10),
                                    zorder=4)

            px = cfg['pos_x'].values.astype(float)
            py = cfg['pos_y'].values.astype(float)
            pz = cfg['pos_z'].values.astype(float)
            valid = ~(np.isnan(px) | np.isnan(py) | np.isnan(pz))
            if valid.any():
                dist_2d = np.sqrt((px[valid] - _PATH_TRACK_CX)**2 + (py[valid] - _PATH_TRACK_CY)**2)
                err3d   = np.sqrt((dist_2d - TRACK_RADIUS)**2 + (pz[valid] - 1.0)**2)
                gps_note = f'  err3d={np.mean(err3d):.3f}m'
            else:
                gps_note = '  err3d=N/A'

            clip_note = f'  [{pct_clip:.0f}% clipped]' if pct_clip > 2.0 else ''
            legend_handles.append(mpatches.Patch(
                facecolor=color, linestyle=ls,
                label=f'{blur_label}{gps_note}  [{n_valid} pts]{clip_note}',
            ))

        ax.set_xlim(_PATH_TRACK_CX - _PATH_PLOT_RADIUS, _PATH_TRACK_CX + _PATH_PLOT_RADIUS)
        ax.set_ylim(_PATH_TRACK_CY - _PATH_PLOT_RADIUS, _PATH_TRACK_CY + _PATH_PLOT_RADIUS)
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(f'speed = {speed}', fontsize=10, fontweight='bold')
        ax.legend(handles=legend_handles, fontsize=7, loc='upper right',
                  framealpha=0.88, edgecolor='#aaaaaa')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {os.path.relpath(plot_path, _ROOT)}')
    return True


def run_16_path_plots():
    global _files_saved
    os.makedirs(_PATHS_OUT_DIR, exist_ok=True)
    if not os.path.isdir(_PLOT_SRC):
        print(f'  [WARN] FOR_PLOTTING data not found at {_PLOT_SRC}')
        return
    sessions = sorted(
        f[:-len('_error_log.csv')]
        for f in os.listdir(_PLOT_SRC)
        if f.endswith('_error_log.csv')
    )
    for s in sessions:
        if _path_plot_session(s, _PATHS_OUT_DIR):
            _files_saved += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Comprehensive thesis analysis script.')
    parser.add_argument('--sections', nargs='+', default=None,
                        help='Run only these section numbers, e.g. --sections 01 05 07')
    args = parser.parse_args()

    run_set = set(args.sections) if args.sections else None

    def should_run(num):
        return run_set is None or num in run_set

    # Create output directories
    os.makedirs(_OUT_DIR, exist_ok=True)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(_OUT_DIR, sub), exist_ok=True)

    t0 = time.time()
    print(f'Output : {_OUT_DIR}\n')

    # Determine which data to load
    needs_anal = any(should_run(n) for n in ['01','02','03','04','05','06','07','08','09','10'])
    needs_plot = should_run('11')
    needs_raw  = should_run('12')

    anal_data, plot_data, raw_data = {}, {}, {}

    if needs_anal:
        print('Loading FOR_ANALYSIS data...')
        anal_data = load_analysis_data()
        print(f'  {len(anal_data)} sessions loaded.\n')

    if needs_plot:
        print('Loading FOR_PLOTTING data...')
        plot_data = load_plotting_data()
        print(f'  {len(plot_data)} sessions loaded.\n')

    if needs_raw:
        print('Loading raw lap data...')
        raw_data = load_raw_lap_data()
        print(f'  {len(raw_data)} sessions loaded.\n')

    if should_run('01'):
        print('=== Section 01: Overview ===')
        run_01_overview(anal_data)
    if should_run('02'):
        print('=== Section 02: Per-Session Heatmaps ===')
        run_02_heatmaps(anal_data)
    if should_run('03'):
        print('=== Section 03: Speed Analysis ===')
        run_03_speed_analysis(anal_data)
    if should_run('04'):
        print('=== Section 04: Blur Analysis ===')
        run_04_blur_analysis(anal_data)
    if should_run('05'):
        print('=== Section 05: Rankings ===')
        run_05_rankings(anal_data)
    if should_run('06'):
        print('=== Section 06: Model Comparison ===')
        run_06_model_comparison(anal_data)
    if should_run('07'):
        print('=== Section 07: Baseline Comparison ===')
        run_07_baseline_comparison(anal_data)
    if should_run('08'):
        print('=== Section 08: Performance Score ===')
        run_08_performance_score(anal_data)
    if should_run('09'):
        print('=== Section 09: Statistics ===')
        run_09_statistics(anal_data)
    if should_run('10'):
        print('=== Section 10: Temporal Analysis ===')
        run_10_temporal(anal_data)
    if should_run('11'):
        print('=== Section 11: Spatial Analysis ===')
        run_11_spatial(plot_data)
    if should_run('12'):
        print('=== Section 12: Lap Analysis ===')
        run_12_lap_analysis(raw_data)
    if should_run('13'):
        print('=== Section 13: Convergence Time + Rate ===')
        run_13_convergence()
    if should_run('14'):
        print('=== Section 14: Overview Charts ===')
        run_14_overview_charts()
    if should_run('15'):
        print('=== Section 15: Training Convergence ===')
        run_15_training_convergence()
    if should_run('16'):
        print('=== Section 16: Path Plots ===')
        run_16_path_plots()

    elapsed = time.time() - t0
    print(f'\nDone. {_files_saved} files saved in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
