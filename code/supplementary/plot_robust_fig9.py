
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'output')
OUT = '/Users/np/Desktop/小论文/重跑_正确语义版/图片/attachments'
SUBF = os.path.join(OUT, 'subfigs')

PARAM_KEYS = ['alpha', 'mu', 'tau', 'N', 'k']
PARAM_LABEL = {'alpha': r'$\alpha$', 'mu': r'$\mu$', 'tau': r'$\tau$', 'N': r'$N$', 'k': r'$k$'}

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=11)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def panel_label(ax, txt):

    pass

net = pd.read_csv(os.path.join(DATA, 'robust_network.csv'))
params = {k: pd.read_csv(os.path.join(DATA, f'robust_{k}.csv')) for k in PARAM_KEYS}

def draw_a(ax, add_label=True):
    names = net['network'].tolist()

    disp = {'BA+SIT': 'BA-heterog.'}
    names = [disp.get(n, n) for n in names]
    xf = net['x_final'].tolist()
    ci = net['x_ci'].tolist() if 'x_ci' in net.columns else [0] * len(xf)
    colors = ['#2166AC', '#67A9CF', '#EF8A62', '#B2182B']
    ax.bar(names, xf, yerr=ci, color=colors, width=0.6, edgecolor='k',
           linewidth=0.6, capsize=3, error_kw=dict(ecolor='k', elinewidth=1))
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel(r'Final mean opinion $\bar{x}_{\mathrm{final}}$', fontsize=11)
    ax.tick_params(axis='both', direction='in', labelsize=11)
    ax.grid(axis='y', alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if add_label:
        panel_label(ax, '(a)')

def draw_param(ax, key, add_label=True):
    df = params[key]
    xs = df['value'].tolist()
    ys = df['x_final'].tolist()
    ci = df['x_ci'].tolist() if 'x_ci' in df.columns else [0] * len(ys)
    ax.errorbar(xs, ys, yerr=ci, fmt='o-', color='#D55E00', linewidth=1.6,
                markersize=4, capsize=3, ecolor='#D55E00', elinewidth=1)
    ax.axhline(0, color='k', linewidth=0.8)

    if key in ('mu', 'N', 'k'):
        ax.set_ylim(bottom=0)
    ax.set_xlabel(PARAM_LABEL[key], fontsize=11)
    ax.set_ylabel(r'Final mean opinion $\bar{x}_{\mathrm{final}}$', fontsize=11)
    style_ax(ax)
    if add_label:
        panel_label(ax, '(x)')

fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
draw_a(axes[0, 0], True)
for i, k in enumerate(PARAM_KEYS):
    r, c = divmod(i + 1, 3)
    draw_param(axes[r, c], k, True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig9.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('已保存 Fig9.png')

jobs = [('Fig9_a.png', 'a')] + [(f'Fig9_{chr(ord("b")+i)}.png', k) for i, k in enumerate(PARAM_KEYS)]
for name, key in jobs:
    f = plt.figure(figsize=(4.4, 3.4))
    ax = f.add_subplot(111)
    if key == 'a':
        draw_a(ax, add_label=False)
    else:
        draw_param(ax, key, add_label=False)
    f.tight_layout()
    f.savefig(os.path.join(SUBF, name), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(f)
print('已保存 subfigs/Fig9_a..f.png')
