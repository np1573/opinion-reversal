
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

BASE = '/Users/np/Desktop/小论文/重跑_正确语义版'
OUT = os.path.join(BASE, '图片/attachments')
SUBF = os.path.join(OUT, 'subfigs')
os.makedirs(SUBF, exist_ok=True)

I2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
TAGS = [('post5', 'a'), ('post7', 'b')]

def load(tag):
    d = os.path.join(BASE, tag)
    xs = np.load(os.path.join(d, 'phase_S_pre_axis.npy'))
    P = np.load(os.path.join(d, 'phase_P_rev.npy'))
    xb = np.empty(len(xs) + 1)
    xb[1:-1] = (xs[:-1] + xs[1:]) / 2
    xb[0] = xs[0] - (xs[1] - xs[0]) / 2
    xb[-1] = xs[-1] + (xs[-1] - xs[-2]) / 2
    return xs, xb, P

ys = I2
yb = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                     [ys[-1] + (ys[-1] - ys[-2]) / 2]])

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def draw_panel(ax, tag, with_cbar=False, fig=None):
    xs, xb, P = load(tag)
    im = ax.pcolormesh(xb, yb, P, cmap='RdYlBu_r', vmin=0, vmax=1, shading='flat')
    cs = ax.contour(xs, ys, P, levels=[0.5], colors='black', linewidths=2.0)
    ax.clabel(cs, fmt={0.5: r'$P_{\mathrm{rev}}=0.5$'}, fontsize=10,
              inline=True, inline_spacing=4)
    ax.set_xlabel(r'Pre-perturbation susceptibility $\chi_{\mathrm{pre}}$')
    ax.set_xlim(xb[0], xb[-1])
    ax.set_ylim(yb[0], yb[-1])
    style_ax(ax)
    if with_cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r'Reversal probability $P_{\mathrm{rev}}$', fontsize=10)
        cb.ax.tick_params(labelsize=9)
    return im

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
im = None
for ax, (tag, _) in zip(axes, TAGS):
    im = draw_panel(ax, tag, with_cbar=False)
axes[0].set_ylabel(r'Counter-information intensity $I_2$')
cb = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02)
cb.set_label(r'Reversal probability $P_{\mathrm{rev}}$', fontsize=10)
cb.ax.tick_params(labelsize=9)
fig.savefig(os.path.join(OUT, 'FigS1.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('已保存 FigS1.png')

for tag, letter in TAGS:
    f = plt.figure(figsize=(5.8, 4.4))
    ax = f.add_subplot(111)
    draw_panel(ax, tag, with_cbar=True, fig=f)
    f.tight_layout()
    f.savefig(os.path.join(SUBF, f'FigS1_{letter}.png'), dpi=300,
              bbox_inches='tight', facecolor='white')
    plt.close(f)
    print(f'已保存 subfigs/FigS1_{letter}.png')
