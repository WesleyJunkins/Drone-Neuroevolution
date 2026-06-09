#!/usr/bin/env python3
"""
Visualize the best NEAT genome for each of the three output types
(ycommand, ycontinuous, ymotors).

Produces one PNG per genome:
    NE_Framework/neat_vis_ycommand.png
    NE_Framework/neat_vis_ycontinuous.png
    NE_Framework/neat_vis_ymotors.png

Run from the project root:
    python3 NE_Framework/visualize_neat.py
"""

import os, pickle
import neat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_NEAT    = os.path.join(_ROOT, 'Data', 'model_data', 'extra_observation', 'exp_b_neat')

EXPERIMENTS = [
    {
        'name':        'neat_ycommand',
        'genome_path': os.path.join(_NEAT, 'neat_ycommand', 'best_genome.pkl'),
        'config_path': os.path.join(_NEAT, 'neat_ycommand', 'neat_config.ini'),
        'input_labels':  ['slope', 'line_at_middle', 'speed_scalar'],
        'output_labels': ['forward', 'yaw_increase', 'yaw_decrease'],
        'output_file': os.path.join(_HERE, 'neat_vis_ycommand.png'),
    },
    {
        'name':        'neat_ycontinuous',
        'genome_path': os.path.join(_NEAT, 'neat_ycontinuous', 'best_genome.pkl'),
        'config_path': os.path.join(_NEAT, 'neat_ycontinuous', 'neat_config.ini'),
        'input_labels':  ['slope', 'line_at_middle', 'speed_scalar'],
        'output_labels': ['forward_desired', 'yaw_desired'],
        'output_file': os.path.join(_HERE, 'neat_vis_ycontinuous.png'),
    },
    {
        'name':        'neat_ymotors',
        'genome_path': os.path.join(_NEAT, 'neat_ymotors', 'best_genome.pkl'),
        'config_path': os.path.join(_NEAT, 'neat_ymotors', 'neat_config.ini'),
        'input_labels':  ['slope', 'line_at_middle', 'speed_scalar'],
        'output_labels': ['m1', 'm2', 'm3', 'm4'],
        'output_file': os.path.join(_HERE, 'neat_vis_ymotors.png'),
    },
]

# ── Colour scale ──────────────────────────────────────────────────────────────
def weight_colour(w, alpha=1.0):
    """Blue for positive weights, red for negative."""
    if w >= 0:
        return (0.15, 0.45, 0.85, alpha)   # blue
    else:
        return (0.85, 0.20, 0.20, alpha)   # red

def weight_lw(w, max_lw=4.0, min_lw=0.8):
    return min_lw + (max_lw - min_lw) * min(abs(w) / 5.0, 1.0)

# ── Layout ────────────────────────────────────────────────────────────────────
def layout_nodes(genome, config, input_labels, output_labels):
    """
    Returns {node_key: (x, y)} positions and separate lists of
    input_keys, hidden_keys, output_keys.
    """
    input_keys  = config.genome_config.input_keys    # negative ints, e.g. [-1, -2, -3]
    output_keys = config.genome_config.output_keys   # e.g. [0, 1, 2]
    hidden_keys = [k for k in genome.nodes.keys() if k not in output_keys]

    n_in  = len(input_keys)
    n_out = len(output_keys)
    n_hid = len(hidden_keys)

    pos = {}

    # Inputs: x=0, spread vertically
    for i, k in enumerate(input_keys):
        y = (n_in - 1) / 2 - i
        pos[k] = (0.0, y)

    # Outputs: x=1 (or 2 if hidden), spread vertically
    x_out = 2.0 if n_hid > 0 else 1.0
    for i, k in enumerate(output_keys):
        y = (n_out - 1) / 2 - i
        pos[k] = (x_out, y)

    # Hidden: x=1, spread vertically
    for i, k in enumerate(hidden_keys):
        y = (n_hid - 1) / 2 - i
        pos[k] = (1.0, y)

    return pos, input_keys, hidden_keys, output_keys

# ── Draw one genome ───────────────────────────────────────────────────────────
def draw_genome(ax, genome, config, input_labels, output_labels, title):
    pos, input_keys, hidden_keys, output_keys = layout_nodes(
        genome, config, input_labels, output_labels)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)

    node_r = 0.18

    # Draw edges first (behind nodes)
    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        k_in, k_out = cg.key
        if k_in not in pos or k_out not in pos:
            continue
        x0, y0 = pos[k_in]
        x1, y1 = pos[k_out]
        col = weight_colour(cg.weight)
        lw  = weight_lw(cg.weight)
        ax.annotate('',
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->', color=col,
                lw=lw, connectionstyle='arc3,rad=0.08',
                mutation_scale=14,
            )
        )
        # Weight label near midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.07, f'{cg.weight:+.2f}',
                fontsize=6.5, ha='center', va='bottom',
                color=col[:3], alpha=0.85)

    # Draw disabled edges (dashed, grey)
    for cg in genome.connections.values():
        if cg.enabled:
            continue
        k_in, k_out = cg.key
        if k_in not in pos or k_out not in pos:
            continue
        x0, y0 = pos[k_in]
        x1, y1 = pos[k_out]
        ax.plot([x0, x1], [y0, y1], color='0.75', lw=0.8, ls='--', alpha=0.5)

    # Draw nodes
    for key, (x, y) in pos.items():
        if key in input_keys:
            face, edge, lbl_size = '#d0e8ff', '#2255aa', 8.5
            idx = list(input_keys).index(key)
            label = input_labels[idx] if idx < len(input_labels) else f'in{idx}'
            node_type = 'Input'
        elif key in output_keys:
            face, edge, lbl_size = '#d4f7d4', '#1a7a1a', 8.5
            idx = list(output_keys).index(key)
            label = output_labels[idx] if idx < len(output_labels) else f'out{idx}'
            node_type = 'Output'
        else:
            face, edge, lbl_size = '#fff3cc', '#aa7700', 8.0
            label = f'h{key}'
            node_type = 'Hidden'

        circle = plt.Circle((x, y), node_r, color=face, ec=edge, lw=1.6, zorder=3)
        ax.add_patch(circle)

        # Bias value for non-input nodes
        if key in genome.nodes:
            bias = genome.nodes[key].bias
            bias_txt = f'b={bias:+.2f}'
        else:
            bias_txt = ''

        ax.text(x, y + 0.04, label, fontsize=lbl_size, ha='center', va='center',
                fontweight='bold', zorder=4)
        if bias_txt:
            ax.text(x, y - 0.09, bias_txt, fontsize=6, ha='center', va='center',
                    color='#555555', zorder=4)

    # Column labels
    xs = sorted(set(v[0] for v in pos.values()))
    col_names = {0.0: 'Input', 1.0: 'Hidden', 2.0: 'Output'}
    if len(xs) == 2:
        col_names = {0.0: 'Input', 1.0: 'Output'}
    for x in xs:
        ys = [v[1] for k, v in pos.items() if v[0] == x]
        ax.text(x, max(ys) + node_r + 0.22,
                col_names.get(x, ''), fontsize=9, ha='center',
                va='bottom', color='#444444', fontstyle='italic')

    # Padding
    all_x = [v[0] for v in pos.values()]
    all_y = [v[1] for v in pos.values()]
    ax.set_xlim(min(all_x) - 0.55, max(all_x) + 0.55)
    ax.set_ylim(min(all_y) - 0.55, max(all_y) + 0.65)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='#d0e8ff', edgecolor='#2255aa', label='Input node'),
        mpatches.Patch(facecolor='#fff3cc', edgecolor='#aa7700', label='Hidden node'),
        mpatches.Patch(facecolor='#d4f7d4', edgecolor='#1a7a1a', label='Output node'),
        mpatches.Patch(facecolor='#4472d4', label='Positive weight'),
        mpatches.Patch(facecolor='#d43333', label='Negative weight'),
        mpatches.Patch(facecolor='0.80',    label='Disabled connection', linestyle='--'),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='lower center',
              ncol=3, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for exp in EXPERIMENTS:
        print(f"Processing {exp['name']} ...")

        with open(exp['genome_path'], 'rb') as f:
            genome = pickle.load(f)

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            exp['config_path'],
        )

        n_hid = len([k for k in genome.nodes
                     if k not in config.genome_config.output_keys])
        n_con_total   = len(genome.connections)
        n_con_enabled = sum(1 for c in genome.connections.values() if c.enabled)
        fitness = genome.fitness

        title = (f"{exp['name']}  |  "
                 f"hidden nodes: {n_hid}  |  "
                 f"connections: {n_con_enabled} enabled / {n_con_total} total  |  "
                 f"fitness: {fitness:.4f}")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_facecolor('#f8f8f8')
        ax.set_facecolor('#f8f8f8')

        draw_genome(ax, genome, config,
                    exp['input_labels'], exp['output_labels'], title)

        fig.tight_layout()
        fig.savefig(exp['output_file'], dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {exp['output_file']}")

    print("\nDone.")

if __name__ == '__main__':
    main()
