
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import monte_carlo_experiment
from rebuild_phase import base_params

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

BASE = '/Users/np/Desktop/小论文/重跑_正确语义版'

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def _mc(arr):
    m = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
    return m, ci

def annotate(ax, t1, t2):

    ax.axvspan(1, t1 + 1, color='gray', alpha=0.18, lw=0)
    ax.axvline(t2, color='m', linestyle='--', lw=1.4)

def main(post):
    os.environ['POST'] = str(post)
    params = base_params()
    t1 = params['info1_duration']
    t2 = t1 + params['post_info1_steps'] + 1
    figdir = os.path.join(BASE, f'post{post}/figs')
    subdir = os.path.join(BASE, f'post{post}/subfigs')
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(subdir, exist_ok=True)

    print(f'重跑 baseline POST={post} (100 MC)...', flush=True)
    res = monte_carlo_experiment(params, num_runs=100)
    avg_x = np.array(res['avg_x'])
    avg_sigma = np.array(res['avg_sigma'])
    avg_S = np.array(res['avg_S'])
    avg_R = np.array(res['avg_R'])

    density_path = os.path.join(BASE, f'post{post}/fig1_density.npz')
    d = np.load(density_path)

    def draw_panel(ax, kind):
        if kind == 'a':
            m, ci = _mc(avg_x)
            ax.plot(m, color='#2E86AB', lw=1.6)
            ax.fill_between(np.arange(len(m)), m - ci, m + ci, color='#2E86AB', alpha=0.2, lw=0)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_ylim(-1, 1)
            ax.set_ylabel('Opinion', fontsize=11)
        elif kind == 'b':
            m, ci = _mc(avg_sigma)
            ax.plot(m, color='#009E73', lw=1.6)
            ax.fill_between(np.arange(len(m)), m - ci, m + ci, color='#009E73', alpha=0.2, lw=0)
            ax.axhline(0.5, color='k', lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Support rate', fontsize=11)
        elif kind == 'c':
            mS, _ = _mc(avg_S)
            mR, _ = _mc(avg_R)
            ln1, = ax.plot(mS, color='#D55E00', lw=1.6, label='System susceptibility $\\chi(t)$')
            ax.set_ylabel('System susceptibility $\\chi(t)$', fontsize=11, color='#D55E00')
            ax.tick_params(axis='y', labelcolor='#D55E00')
            ax.set_ylim(0, 2.0)
            ax2 = ax.twinx()
            ln2, = ax2.plot(mR, color='#CC79A7', lw=1.6, label='Mean reversal probability $\\bar{R}(t)$')
            ax2.set_ylabel('Mean reversal probability $\\bar{R}(t)$', fontsize=11, color='#CC79A7')
            ax2.set_ylim(0, 1)
            ax2.spines['top'].set_visible(False)
            ax.legend(handles=[ln1, ln2], frameon=False, fontsize=9, loc='upper right')
        elif kind == 'd':
            H, yedges, vmin = d['H'], d['yedges'], float(d['vmin'])
            yc = (yedges[:-1] + yedges[1:]) / 2
            im = ax.pcolormesh(np.arange(H.shape[1]), yc, H, cmap='viridis',
                               norm=LogNorm(vmin=vmin, vmax=H.max()), shading='nearest')
            ax.set_ylabel('Opinion $x$', fontsize=11)
            cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
            cb.set_label('Density', fontsize=10)
            cb.ax.tick_params(labelsize=9)
        annotate(ax, t1, t2)
        ax.set_xlabel('Time step', fontsize=11)
        style_ax(ax)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, k in zip(axes.flat, ['a', 'b', 'c', 'd']):
        draw_panel(ax, k)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'Fig2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    for k in ['a', 'b', 'c', 'd']:
        f = plt.figure(figsize=(5.2, 3.9))
        ax = f.add_subplot(111)
        draw_panel(ax, k)
        f.tight_layout()
        f.savefig(os.path.join(subdir, f'Fig2_{k}.png'), dpi=300, bbox_inches='tight')
        plt.close(f)

    print(f'post{post} Fig2.png + subfigs/Fig2_a..d.png 已生成', flush=True)

if __name__ == '__main__':
    for p in (7, 5):
        main(p)
