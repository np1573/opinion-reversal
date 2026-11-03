
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import model as M
from rebuild_phase import base_params

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)

C_NONE = '#808080'
C_FIX = '#D95F02'
C_ENDO = '#2166AC'

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=9)
    ax.grid(alpha=0.22, linewidth=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def _mc(res, key):
    arr = np.array(res[key])
    m = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
    return m, ci

def annotate(ax, t1_end, t2):
    ax.axvspan(1, t1_end, color='gray', alpha=0.16, lw=0, zorder=0)
    ax.axvline(t2, color='magenta', ls='--', lw=1.1, zorder=1)

def panel_label(ax, txt):
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=12,
            va='top', ha='left', fontweight='bold')

def draw_fig4(ax):
    groups = ['Slow consensus', 'Fast consensus']
    endo = [0.15, 0.72]
    fixd = [0.86, 0.77]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, endo, w, color=C_ENDO, label=r'Endogenous $S_i(t)$')
    b2 = ax.bar(x + w / 2, fixd, w, color=C_FIX, label=r'Fixed $S_i=1.0$')
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 0.02, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel('Reversal probability $P_{\\mathrm{rev}}$', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', direction='in', labelsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')

def main():
    params = base_params()
    t1_end = params['info1_duration']
    t2 = t1_end + params['post_info1_steps'] + 1

    t0 = time.time()
    print('内生易感性 (alpha=60)...', flush=True)
    res_endo = M.monte_carlo_experiment(params, num_runs=100)

    print('无易感性 (alpha=0)...', flush=True)
    res_none = M.monte_carlo_experiment(dict(params, alpha=0.0), num_runs=100)

    print('固定易感性 (S_i=1.0)...', flush=True)
    orig = M.update_skepticism

    def fixed_update(G, alpha, tau):
        for i in G.nodes():
            G.nodes[i]['S'] = 1.0
            G.nodes[i]['R'] = np.tanh(1.0)

    M.update_skepticism = fixed_update
    res_fixed = M.monte_carlo_experiment(params, num_runs=100)
    M.update_skepticism = orig

    cases = [
        ('No susceptibility\n($\\alpha=0$)', res_none, C_NONE),
        ('Fixed susceptibility\n($S_i=1.0$)', res_fixed, C_FIX),
        ('Endogenous susceptibility\n($\\alpha=60$)', res_endo, C_ENDO),
    ]
    P_rev = {name: float(np.mean(np.array(res['x_final']) > 0)) for name, res, _ in cases}
    print('耗时 %.0fs' % (time.time() - t0))
    for name, _, _ in cases:
        print(f'  {name.replace(chr(10), " ")}: P_rev = {P_rev[name]:.3f}')

    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          left=0.07, right=0.97, top=0.94, bottom=0.09)

    ax = fig.add_subplot(gs[0, 0])
    for name, res, c in cases:
        m, ci = _mc(res, 'avg_x')
        ax.plot(m, color=c, lw=1.6, label=name.split('\n')[0])
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.16, lw=0)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(-1, 1)
    annotate(ax, t1_end, t2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Mean opinion $\\bar{x}(t)$', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    panel_label(ax, '(a)')
    style_ax(ax)

    ax = fig.add_subplot(gs[0, 1])
    for name, res, c in cases:
        m, ci = _mc(res, 'avg_sigma')
        ax.plot(m, color=c, lw=1.6, label=name.split('\n')[0])
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.16, lw=0)
    ax.axhline(0.5, color='k', lw=0.8)
    ax.set_ylim(0, 1)
    annotate(ax, t1_end, t2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Support rate $P_+(t)$', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    panel_label(ax, '(b)')
    style_ax(ax)

    ax = fig.add_subplot(gs[1, 0])
    labels = ['No susceptibility', 'Fixed susceptibility', 'Endogenous susceptibility']
    colors = [C_NONE, C_FIX, C_ENDO]
    pvals = [P_rev['No susceptibility\n($\\alpha=0$)'],
             P_rev['Fixed susceptibility\n($S_i=1.0$)'],
             P_rev['Endogenous susceptibility\n($\\alpha=60$)']]
    x = np.arange(3)
    bars = ax.bar(x, pvals, 0.55, color=colors, edgecolor='k', linewidth=0.6)
    for xi, v in zip(x, pvals):
        ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Reversal probability $P_{\\mathrm{rev}}$', fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis='both', direction='in', labelsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, '(c)')

    ax = fig.add_subplot(gs[1, 1])
    draw_fig4(ax)
    panel_label(ax, '(d)')

    path = os.path.join(OUT, 'Fig3_three_mechanisms_preview.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('已保存:', path)

if __name__ == '__main__':
    main()
