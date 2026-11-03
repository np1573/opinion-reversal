
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
chi = np.load(os.path.join(OUT, 'phase5_chi_axis.npy'))
P_rev = np.load(os.path.join(OUT, 'phase5_P_rev.npy'))
x_final = np.load(os.path.join(OUT, 'phase5_x_final.npy'))
I2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

xs = chi
xb = np.empty(len(xs) + 1)
xb[1:-1] = (xs[:-1] + xs[1:]) / 2
xb[0] = xs[0] - (xs[1] - xs[0]) / 2
xb[-1] = xs[-1] + (xs[-1] - xs[-2]) / 2
ys = I2
yb = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                     [ys[-1] + (ys[-1] - ys[-2]) / 2]])

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

im = axes[0].pcolormesh(xb, yb, P_rev, cmap='RdYlBu_r', vmin=0, vmax=1, shading='flat')
cs = axes[0].contour(xs, ys, P_rev, levels=[0.1, 0.5, 0.7],
                     colors='black', linewidths=[1.0, 1.8, 1.0])
axes[0].clabel(cs, fmt='%.1f', fontsize=10)
axes[0].set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
axes[0].set_ylabel('Counter-information intensity $I_2$')
style_ax(axes[0])
cb = fig.colorbar(im, ax=axes[0], pad=0.02)
cb.set_label('Reversal probability $P_{\\mathrm{rev}}$', fontsize=10)
cb.ax.tick_params(labelsize=9)

vmax = np.abs(x_final).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im2 = axes[1].pcolormesh(xb, yb, x_final, cmap='RdYlBu_r', norm=norm, shading='flat')
cs2 = axes[1].contour(xs, ys, x_final, levels=[0.0], colors='black', linewidths=1.8)
axes[1].clabel(cs2, fmt='$\\bar{x}_{\\mathrm{final}}=0$', fontsize=10)
axes[1].set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
axes[1].set_ylabel('Counter-information intensity $I_2$')
style_ax(axes[1])
cb2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
cb2.set_label('Final mean opinion $\\bar{x}_{\\mathrm{final}}$', fontsize=10)
cb2.ax.tick_params(labelsize=9)

fig.tight_layout()
path = os.path.join(OUT, 'Fig5_phase_fixed.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close(fig)
print('已保存:', path)
