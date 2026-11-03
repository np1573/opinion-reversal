
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase10_boot import (load_group, ic_from_prev, bootstrap_ic,
                                  I2, NG, OUT as BOOT_OUT)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

FIGDIR = '/Users/np/Desktop/小论文/Physica-A-new/图片/attachments'

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def load_all():
    chi = np.load(os.path.join(BOOT_OUT, 'chi_axis.npy'))
    X = [load_group(j) for j in range(NG)]
    P = np.array([[np.mean(X[j][i] > 0.0) for i in range(len(I2))] for j in range(NG)]).T
    order = np.argsort(chi)
    return chi[order], P[:, order], [X[j] for j in order]

def ic_ci_for_group(X):
    prev = np.array([np.mean(X[i] > 0.0) for i in range(len(I2))])
    ic, crossed = ic_from_prev(prev)
    ics, _ = bootstrap_ic(X)
    if not crossed:
        return None, None, None, True
    lo, hi = np.nanpercentile(ics, [2.5, 97.5])
    return ic, lo, hi, False

def panel_label(ax, txt):
    pass

def draw_panel_a(ax, fig, chi_s, P, ic_info, add_label=True):
    xs = chi_s
    xb = np.empty(len(xs) + 1)
    xb[1:-1] = (xs[:-1] + xs[1:]) / 2
    xb[0] = xs[0] - (xs[1] - xs[0]) / 2
    xb[-1] = xs[-1] + (xs[-1] - xs[-2]) / 2
    ys = I2
    yb = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                         [ys[-1] + (ys[-1] - ys[-2]) / 2]])

    im = ax.pcolormesh(xb, yb, P, cmap='RdYlBu_r', vmin=0, vmax=1, shading='flat')
    cs = ax.contour(xs, ys, P, levels=[0.5], colors='black', linewidths=2.0)
    ax.clabel(cs, fmt={0.5: r'$P_{\mathrm{rev}}=0.5$'}, fontsize=11,
              inline=True, inline_spacing=4)
    csd = ax.contour(xs, ys, P, levels=[0.1, 0.8], colors='black',
                     linewidths=[0.9, 0.9], linestyles='dashed')
    ax.clabel(csd, fmt={0.1: '0.1', 0.8: '0.8'}, fontsize=9, inline=True,
              inline_spacing=3)

    for x, (ic, lo, hi, cens) in zip(xs, ic_info):
        if cens:

            ax.plot([x], [ys[-1]], marker='^', markerfacecolor='none',
                    markeredgecolor='black', markersize=7, zorder=6)
            continue
        ax.plot([x, x], [lo, hi], color='black', lw=1.4, zorder=5)
        ax.scatter([x], [ic], marker='o', color='black', s=18, zorder=6)

    ax.set_xlabel(r'$\chi_{\mathrm{pre}}$ (consensus-matched, different speed)')
    ax.set_ylabel(r'Counter-information intensity $I_2$')
    ax.set_xlim(xb[0], xb[-1])
    ax.set_ylim(yb[0], yb[-1])
    style_ax(ax)
    if add_label:
        panel_label(ax, '(a)')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'Reversal probability $P_{\mathrm{rev}}$', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return ax

def draw_panel_b(ax, fig, chi_s, X, add_label=True):
    xs = chi_s
    xb = np.empty(len(xs) + 1)
    xb[1:-1] = (xs[:-1] + xs[1:]) / 2
    xb[0] = xs[0] - (xs[1] - xs[0]) / 2
    xb[-1] = xs[-1] + (xs[-1] - xs[-2]) / 2
    ys = I2
    yb = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                         [ys[-1] + (ys[-1] - ys[-2]) / 2]])
    x_mean = np.array([[np.mean(X[j][i]) for i in range(len(I2))] for j in range(len(X))]).T
    vmax = np.abs(x_mean).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = ax.pcolormesh(xb, yb, x_mean, cmap='RdYlBu_r', norm=norm, shading='flat')
    cs2 = ax.contour(xs, ys, x_mean, levels=[0.0], colors='black', linewidths=1.8)
    ax.clabel(cs2, fmt={0.0: r'$\bar{x}_{\mathrm{final}}=0$'}, fontsize=11,
              inline=True, inline_spacing=4)
    ax.set_xlabel(r'$\chi_{\mathrm{pre}}$')
    ax.set_ylabel(r'Counter-information intensity $I_2$')
    ax.set_xlim(xb[0], xb[-1])
    ax.set_ylim(yb[0], yb[-1])
    style_ax(ax)
    if add_label:
        panel_label(ax, '(b)')
    cb2 = fig.colorbar(im2, ax=ax, pad=0.02)
    cb2.set_label(r'Final mean opinion $\bar{x}_{\mathrm{final}}$', fontsize=10)
    cb2.ax.tick_params(labelsize=9)
    return ax

def main():
    chi_s, P, X = load_all()
    print('χ_pre（升序）:', np.round(chi_s, 3))
    ic_info = [ic_ci_for_group(X[j]) for j in range(NG)]
    for j, (x, (ic, lo, hi, cens)) in enumerate(zip(chi_s, ic_info)):
        if cens:
            print(f'  χ={x:.3f}: I_c > 1.0（右删失）')
        else:
            print(f'  χ={x:.3f}: I_c={ic:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]')

    os.makedirs(os.path.join(FIGDIR, 'subfigs'), exist_ok=True)

    FIG = plt.figure(figsize=(11.5, 4.8))
    axes = FIG.subplots(1, 2)
    draw_panel_a(axes[0], FIG, chi_s, P, ic_info, add_label=True)
    draw_panel_b(axes[1], FIG, chi_s, X, add_label=True)
    FIG.tight_layout()
    FIG.savefig(os.path.join(FIGDIR, 'Fig8.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(FIG)

    for name, draw in [('Fig8_a.png', draw_panel_a), ('Fig8_b.png', draw_panel_b)]:
        f = plt.figure(figsize=(5.8, 4.6))
        ax = f.add_subplot(111)
        if name.endswith('_a.png'):
            draw(ax, f, chi_s, P, ic_info, add_label=False)
        else:
            draw(ax, f, chi_s, X, add_label=False)
        f.tight_layout()
        f.savefig(os.path.join(FIGDIR, 'subfigs', name), dpi=300,
                  bbox_inches='tight', facecolor='white')
        plt.close(f)

    print('已生成 Fig8.png + subfigs/Fig8_a.png, Fig8_b.png（含 I_2=1 与 I_c 置信区间）')

if __name__ == '__main__':
    main()
