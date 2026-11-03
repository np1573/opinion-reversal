
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def panel_label(ax, txt):

    pass

def save(fig, name, use_tight=True):
    if use_tight:
        fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  已保存:', name)

dfA = pd.read_csv(os.path.join(OUT, 'speed_10_levels.csv'))
logA = json.load(open(os.path.join(OUT, 'speed_logistic.json')))
chi = dfA['chi_pre'].values
prev = dfA['P_rev'].values
xfin = dfA['x_final'].values
se = np.sqrt(np.clip(prev * (1 - prev), 1e-12, None) / 100.0)

def draw_A_panel_a(ax):
    ax.errorbar(chi, prev, yerr=1.96 * se, fmt='o', color='#B2182B', capsize=3,
                markersize=5, label='Simulation')
    g = np.linspace(chi.min() * 0.9, chi.max() * 1.05, 100)
    pf = 1 / (1 + np.exp(-(logA['a'] + logA['b'] * g)))
    ax.plot(g, pf, '-', color='k', lw=2, label='Logistic fit')
    ax.set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
    ax.set_ylabel('Reversal probability $P_{\\mathrm{rev}}$')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    style_ax(ax)

def draw_A_panel_b(ax):
    ax.plot(chi, xfin, 'o-', color='#2E86AB', lw=1.5, markersize=5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
    ax.set_ylabel('Final mean opinion $\\bar{x}_{\\mathrm{final}}$')
    style_ax(ax)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
draw_A_panel_a(axes[0]); panel_label(axes[0], '(a)')
draw_A_panel_b(axes[1]); panel_label(axes[1], '(b)')
save(fig, 'speed_combined.png')

fig, ax = plt.subplots(figsize=(4.6, 3.6)); draw_A_panel_a(ax); save(fig, 'speed_panel_a.png')
fig, ax = plt.subplots(figsize=(4.6, 3.6)); draw_A_panel_b(ax); save(fig, 'speed_panel_b.png')

N_vals = [250, 500, 1000, 2000, 4000]
recs = {N: pd.read_csv(os.path.join(OUT, f'fss_N{N}.csv')) for N in N_vals}
colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E']

def draw_B_panel_a(ax):
    for N, c in zip(N_vals, colors):
        ax.plot(recs[N]['I2'], recs[N]['P_rev'], 'o-', color=c, lw=1.6, ms=5, label=f'N={N}')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set_xlabel('Counter-information intensity $I_2$')
    ax.set_ylabel('Reversal probability $P_{\\mathrm{rev}}$')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    style_ax(ax)

def draw_B_panel_b(ax):
    for N, c in zip(N_vals, colors):
        ax.plot(recs[N]['I2'], recs[N]['M_mean'], 'o-', color=c, lw=1.6, ms=5, label=f'N={N}')
    ax.set_xlabel('Counter-information intensity $I_2$')
    ax.set_ylabel('Order parameter $M=\\langle|\\bar{x}_{\\mathrm{final}}|\\rangle$')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    style_ax(ax)

def draw_B_panel_c(ax):
    for N, c in zip(N_vals, colors):
        ax.plot(recs[N]['I2'], recs[N]['chi_rev'], 'o-', color=c, lw=1.6, ms=5, label=f'N={N}')
    ax.set_xlabel('Counter-information intensity $I_2$')
    ax.set_ylabel('Reversal fluctuation $F_{\\mathrm{rev}}=N(\\langle M^2\\rangle-\\langle M\\rangle^2)$')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    style_ax(ax)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
draw_B_panel_a(axes[0]); panel_label(axes[0], '(a)')
draw_B_panel_b(axes[1]); panel_label(axes[1], '(b)')
draw_B_panel_c(axes[2]); panel_label(axes[2], '(c)')
save(fig, 'fss_combined.png')

fig, ax = plt.subplots(figsize=(4.6, 3.6)); draw_B_panel_a(ax); save(fig, 'fss_panel_a.png')
fig, ax = plt.subplots(figsize=(4.6, 3.6)); draw_B_panel_b(ax); save(fig, 'fss_panel_b.png')
fig, ax = plt.subplots(figsize=(4.6, 3.6)); draw_B_panel_c(ax); save(fig, 'fss_panel_c.png')

dfIc = pd.read_csv(os.path.join(OUT, 'fss_Ic_summary.csv'))
Narr = dfIc['N'].values
ic = dfIc['Ic'].values
chi_peak = dfIc['chi_peak'].values

def draw_C_panel_a(ax):
    ax.plot(Narr, ic, 'o-', color='#B2182B', lw=1.6, ms=6)
    ax.set_xlabel('Network size $N$')
    ax.set_ylabel('Pseudo-critical point $I_c(N)$')
    style_ax(ax)

def draw_C_panel_b(ax):
    ax.plot(Narr, chi_peak, 's-', color='#2E86AB', lw=1.6, ms=6)
    ax.set_xlabel('Network size $N$')
    ax.set_ylabel('Peak of $F_{\\mathrm{rev}}$')
    style_ax(ax)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
draw_C_panel_a(axes[0]); panel_label(axes[0], '(a)')
draw_C_panel_b(axes[1]); panel_label(axes[1], '(b)')
save(fig, 'fss_ic_combined.png')

fig, ax = plt.subplots(figsize=(4.6, 3.6)); draw_C_panel_a(ax); save(fig, 'fss_ic_panel_a.png')
fig, ax = plt.subplots(figsize=(4.6, 3.6)); draw_C_panel_b(ax); save(fig, 'fss_ic_panel_b.png')

tr = np.load(os.path.join(OUT, 'speed_traces.npz'))
groups = []
for gi in range(1, 11):
    p1 = float(tr[f'G{gi}_p']); d1 = float(tr[f'G{gi}_T'])
    groups.append(dict(p=p1, T=d1, v=1.0 / d1,
                       x=tr[f'G{gi}_avg_x'], S=tr[f'G{gi}_avg_S'],
                       t_pre=int(d1) + 7))
cmapF3 = plt.cm.plasma
colsF3 = cmapF3(np.linspace(0.05, 0.82, len(groups)))
normF3 = plt.Normalize(vmin=min(g['v'] for g in groups), vmax=max(g['v'] for g in groups))

def _mc(arr):
    m = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
    return m, ci

def draw_F3_a(ax):
    for g, c in zip(groups, colsF3):
        m, ci = _mc(np.abs(g['x']))
        ax.plot(m, color=c, lw=1.5)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Consensus degree $|\\bar{x}(t)|$')
    style_ax(ax)

def draw_F3_b(ax):
    for g, c in zip(groups, colsF3):
        m, _ = _mc(g['S'])
        ax.plot(m, color=c, lw=1.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('System susceptibility $\\chi(t)$')
    style_ax(ax)

def draw_F3_c(ax):
    for g, c in zip(groups, colsF3):
        m, ci = _mc(g['x'])
        ax.plot(m, color=c, lw=1.5)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean opinion $\\bar{x}(t)$')
    style_ax(ax)

def draw_F3_d(ax):
    for g, c in zip(groups, colsF3):
        chi_run = g['S'][:, g['t_pre']]
        xf_run = g['x'][:, -1]
        ax.scatter(chi_run, xf_run, s=12, color=c, alpha=0.35, edgecolors='none')
        ax.scatter(chi_run.mean(), xf_run.mean(), s=110, color=c,
                   edgecolors='k', linewidths=0.8, zorder=5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
    ax.set_ylabel('Final mean opinion $\\bar{x}_{\\mathrm{final}}$')
    style_ax(ax)

def _add_colorbar(fig, ax_list):
    sm = plt.cm.ScalarMappable(cmap=cmapF3, norm=normF3)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_list, location='right', shrink=0.85, pad=0.02)
    cb.set_label('Consensus speed $v$', fontsize=10)
    cb.ax.tick_params(labelsize=9)

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), layout='constrained')
draw_F3_a(axes[0, 0]); panel_label(axes[0, 0], '(a)')
draw_F3_b(axes[0, 1]); panel_label(axes[0, 1], '(b)')
draw_F3_c(axes[1, 0]); panel_label(axes[1, 0], '(c)')
draw_F3_d(axes[1, 1]); panel_label(axes[1, 1], '(d)')
_add_colorbar(fig, list(axes.flat))
save(fig, 'fig3_combined.png', use_tight=False)

for name, fn in [('fig3_panel_a.png', draw_F3_a),
                 ('fig3_panel_b.png', draw_F3_b),
                 ('fig3_panel_c.png', draw_F3_c),
                 ('fig3_panel_d.png', draw_F3_d)]:
    fig, ax = plt.subplots(figsize=(5.0, 3.8), layout='constrained')
    fn(ax)
    _add_colorbar(fig, [ax])
    save(fig, name, use_tight=False)

print('全部完成')
