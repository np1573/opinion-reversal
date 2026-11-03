
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

GROUPS = [1, 5, 8, 10]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

tr = np.load(os.path.join(OUT, 'speed_traces.npz'))
groups = []
for gi in GROUPS:
    p1 = float(tr[f'G{gi}_p']); d1 = float(tr[f'G{gi}_T'])
    groups.append(dict(p=p1, L=d1, v=1.0 / d1,
                       x=tr[f'G{gi}_avg_x'], S=tr[f'G{gi}_avg_S'],
                       t_pre=int(d1) + 7))

labels = [f"$p_{{0,1}}={g['p']:g},\\ L_1={int(g['L'])}$" for g in groups]

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def _mc(arr):
    m = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
    return m, ci

def draw_a(ax):
    for g, c in zip(groups, COLORS):
        m, ci = _mc(np.abs(g['x']))
        ax.plot(m, color=c, lw=1.6)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Consensus degree $|\\bar{x}(t)|$')
    style_ax(ax)

def draw_b(ax):
    for g, c in zip(groups, COLORS):
        m, _ = _mc(g['S'])
        ax.plot(m, color=c, lw=1.6)
    ax.set_xlabel('Time step')
    ax.set_ylabel('System susceptibility $\\chi(t)$')
    style_ax(ax)

def draw_c(ax):
    for g, c in zip(groups, COLORS):
        m, ci = _mc(g['x'])
        ax.plot(m, color=c, lw=1.6)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean opinion $\\bar{x}(t)$')
    style_ax(ax)

def draw_d(ax):
    for g, c in zip(groups, COLORS):
        chi_run = g['S'][:, g['t_pre']]
        xf_run = g['x'][:, -1]
        ax.scatter(chi_run, xf_run, s=14, color=c, alpha=0.35, edgecolors='none')
        ax.scatter(chi_run.mean(), xf_run.mean(), s=120, color=c,
                   edgecolors='k', linewidths=0.8, zorder=5, label=labels[groups.index(g)])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
    ax.set_ylabel('Final mean opinion $\\bar{x}_{\\mathrm{final}}$')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    style_ax(ax)

def panel_label(ax, txt):

    pass

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
draw_a(axes[0, 0]); panel_label(axes[0, 0], '(a)')
draw_b(axes[0, 1]); panel_label(axes[0, 1], '(b)')
draw_c(axes[1, 0]); panel_label(axes[1, 0], '(c)')
draw_d(axes[1, 1]); panel_label(axes[1, 1], '(d)')

handles = [plt.Line2D([0], [0], color=c, lw=2) for c in COLORS]
axes[0, 0].legend(handles, labels, frameon=False, fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_4groups.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('已保存 fig3_4groups.png')

fig, ax = plt.subplots(figsize=(4.8, 3.8))
for g, c in zip(groups, COLORS):
    chi_mean = g['S'][:, g['t_pre']].mean()
    xf_mean = g['x'][:, -1].mean()
    ax.scatter(chi_mean, xf_mean, s=120, color=c, edgecolors='k',
               linewidths=0.8, zorder=5, label=labels[groups.index(g)])
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
ax.set_ylabel('Final mean opinion $\\bar{x}_{\\mathrm{final}}$')
ax.legend(frameon=False, fontsize=8, loc='lower right')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_4groups.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('已保存 fig4_4groups.png')
