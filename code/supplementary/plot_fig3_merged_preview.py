
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import monte_carlo_experiment
from rebuild_phase import base_params

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)

BLUE = '#2166AC'
ORANGE = '#D95F02'
GREEN = '#009E73'
RED = '#B2182B'
GRAY = '#4D4D4D'

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
    ax.axvspan(1, t1_end, color='gray', alpha=0.18, lw=0, zorder=0)
    ax.axvline(t2, color='magenta', ls='--', lw=1.2, zorder=1)

def panel_label(ax, txt):
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=12,
            va='top', ha='left', fontweight='bold')

def draw_fixedS(ax):
    groups = ['Slow consensus', 'Fast consensus']
    endo = [0.15, 0.72]
    fixd = [0.86, 0.77]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, endo, w, color=BLUE, label=r'Endogenous $S_i(t)$')
    b2 = ax.bar(x + w / 2, fixd, w, color=RED, label=r'Fixed $S_i=1.0$')
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
    print('重跑 full 模型 (100 MC)...', flush=True)
    res_full = monte_carlo_experiment(params, num_runs=100)
    print('  完成，P_rev(full) = %.3f' % (np.mean(np.array(res_full['x_final']) > 0)), flush=True)
    ab = dict(params, alpha=0.0)
    print('重跑 ablated 模型 (100 MC)...', flush=True)
    res_ab = monte_carlo_experiment(ab, num_runs=100)
    print('  完成，P_rev(ab) = %.3f，总耗时 %.0fs' %
          (np.mean(np.array(res_ab['x_final']) > 0), time.time() - t0), flush=True)

    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.09)

    ax = fig.add_subplot(gs[0, 0])
    m, ci = _mc(res_full, 'avg_x')
    ax.plot(m, color=BLUE, lw=1.6)
    ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=BLUE, alpha=0.2, lw=0)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(-1, 1)
    annotate(ax, t1_end, t2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Mean opinion $\\bar{x}(t)$', fontsize=9)
    panel_label(ax, '(a)')
    style_ax(ax)

    ax = fig.add_subplot(gs[0, 1])
    m, ci = _mc(res_ab, 'avg_x')
    ax.plot(m, color=ORANGE, lw=1.6)
    ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=ORANGE, alpha=0.2, lw=0)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(-1, 1)
    annotate(ax, t1_end, t2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Mean opinion $\\bar{x}(t)$', fontsize=9)
    panel_label(ax, '(b)')
    style_ax(ax)

    ax = fig.add_subplot(gs[0, 2])
    for res, c, lab in [(res_full, BLUE, 'Full'), (res_ab, ORANGE, 'Ablated')]:
        m, ci = _mc(res, 'avg_sigma')
        ax.plot(m, color=c, lw=1.6, label=lab)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.axhline(0.5, color='k', lw=0.8)
    ax.set_ylim(0, 1)
    annotate(ax, t1_end, t2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Support fraction $P_+(t)$', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    panel_label(ax, '(c)')
    style_ax(ax)

    ax = fig.add_subplot(gs[1, 0])
    for res, c, lab in [(res_full, BLUE, 'Full'), (res_ab, ORANGE, 'Ablated')]:
        m, ci = _mc(res, 'avg_R')
        ax.plot(m, color=c, lw=1.6, label=lab)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.set_ylim(0, 1)
    annotate(ax, t1_end, t2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Mean reversal prob. $\\bar{R}(t)$', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    panel_label(ax, '(d)')
    style_ax(ax)

    ax = fig.add_subplot(gs[1, 1])
    draw_fixedS(ax)
    panel_label(ax, '(e)')

    path = os.path.join(OUT, 'Fig3_merged_preview.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('已保存:', path)

if __name__ == '__main__':
    main()
