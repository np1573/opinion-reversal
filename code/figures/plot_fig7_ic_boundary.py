# -*- coding: utf-8 -*-
"""CSF Fig.7（旧 Fig.5）响应面：恢复 Physica-A 旧版样式（2026-09-24 用户要求改回）。

样式（与旧版 6N4XE8KP.png 一致）：
  左 panel：P_rev 热图 + 单条 P_rev=0.5 黑色等值线（clabel 内嵌标注），
            x 轴 "χ_pre (consensus-matched, different speed)"
  右 panel：x_final 热图 + x_final=0 黑色等值线（clabel 内嵌标注），x 轴 "χ_pre"
  不叠加 I_c 边界曲线，无 0.1/0.8 虚线（I_c 数值仅在正文中给出）。
数据：数据/补充实验/phase10_*.npy，不重跑仿真。

输出：合并图 Fig7.png（含 (a)(b) 图内标签，供 docx）+ 子图 subfigs/Fig7_a.png /
subfigs/Fig7_b.png（无图内标签，LaTeX subfigure 提供编号，供 PDF）。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

DATA = '/Users/np/Desktop/小论文/投稿-源码与图片/数据/补充实验'
OUT = '/Users/np/Desktop/小论文/投稿-源码与图片/图片/attachments'

chi = np.load(os.path.join(DATA, 'phase10_chi_axis.npy'))
P_raw = np.load(os.path.join(DATA, 'phase10_P_rev.npy'))
x_raw = np.load(os.path.join(DATA, 'phase10_x_final.npy'))
I2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

order = np.argsort(chi)
chi_s = chi[order]
P = P_raw[:, order]
X = x_raw[:, order]

# 非均匀网格边界（pcolormesh 用）
xs = chi_s
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


def panel_label(ax, txt):
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, fontsize=13,
            va='top', ha='left', fontweight='bold')


def draw_panel_a(ax, fig, add_label=True):
    """左 (a)：P_rev + P_rev=0.5 黑色等值线 + 0.1/0.8 虚线等值线。"""
    im = ax.pcolormesh(xb, yb, P, cmap='RdYlBu_r', vmin=0, vmax=1, shading='flat')
    cs = ax.contour(xs, ys, P, levels=[0.5], colors='black', linewidths=2.0)
    ax.clabel(cs, fmt={0.5: r'$P_{\mathrm{rev}}=0.5$'}, fontsize=11,
              inline=True, inline_spacing=4)
    csd = ax.contour(xs, ys, P, levels=[0.1, 0.8], colors='black',
                     linewidths=[0.9, 0.9], linestyles='dashed')
    ax.clabel(csd, fmt={0.1: '0.1', 0.8: '0.8'}, fontsize=9, inline=True,
              inline_spacing=3)
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


def draw_panel_b(ax, fig, add_label=True):
    """右 (b)：x_final + 0 等值线（旧版样式）。"""
    vmax = np.abs(X).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = ax.pcolormesh(xb, yb, X, cmap='RdYlBu_r', norm=norm, shading='flat')
    cs2 = ax.contour(xs, ys, X, levels=[0.0], colors='black', linewidths=1.8)
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


# ---------- 合并图（含 (a)(b) 标签，供 docx） ----------
FIG = plt.figure(figsize=(11.5, 4.8))
axes = FIG.subplots(1, 2)
draw_panel_a(axes[0], FIG, add_label=True)
draw_panel_b(axes[1], FIG, add_label=True)
FIG.tight_layout()
FIG.savefig(os.path.join(OUT, 'Fig7.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(FIG)

# ---------- 子图（无图内标签，LaTeX subfigure 提供编号，供 PDF） ----------
for name, draw in [('Fig7_a.png', draw_panel_a), ('Fig7_b.png', draw_panel_b)]:
    f = plt.figure(figsize=(5.8, 4.6))
    ax = f.add_subplot(111)
    draw(ax, f, add_label=False)
    f.tight_layout()
    f.savefig(os.path.join(OUT, 'subfigs', name), dpi=300,
              bbox_inches='tight', facecolor='white')
    plt.close(f)

print('Fig7.png + subfigs/Fig7_a/b.png 已生成（P_rev=0.5 实线 + 0.1/0.8 虚线等值线）')
