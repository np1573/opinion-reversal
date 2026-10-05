# -*- coding: utf-8 -*-
"""CSF Fig.3：消融对比（full vs ablated）正式 4 面板图，2026-09-24。

重跑 full（alpha=60）与 ablated（alpha=0）各 100 MC（N=500），绘制：
  (a) full-model mean opinion（单蓝曲线，无图例）
  (b) ablated-model mean opinion（单橙曲线，无图例）
  (c) positive-support fraction（full 蓝 vs ablated 橙，有图例）
  (d) system-level reversal probability（full vs ablated 柱状图）
a/b 与 Fig.2 一致为单曲线无图例；c 因双线对比保留图例；d 为柱状对比。
输出：合并图 attachments/Fig3.png（含 (a)(b)(c)(d) 图内标签，供 docx）+
子图 subfigs/Fig3_a/b/c/d.png（无图内标签，LaTeX subfigure 提供编号，供 PDF）。
"""
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

OUT = '/Users/np/Desktop/小论文/投稿-源码与图片/图片/attachments'

BLUE = '#2166AC'
ORANGE = '#D95F02'


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
    ax.axvspan(0, t1_end, color='gray', alpha=0.18, lw=0, zorder=0)
    ax.axvline(t2, color='magenta', ls='--', lw=1.2, zorder=1)


def panel_label(ax, txt):
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, fontsize=12,
            va='top', ha='left', fontweight='bold')


def draw_a(ax, res, add_label=True):
    """(a) full-model mean opinion（单蓝，无图例）。"""
    m, ci = _mc(res, 'avg_x')
    ax.plot(m, color=BLUE, lw=1.6)
    ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=BLUE, alpha=0.2, lw=0)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(-1, 1)
    annotate(ax, T1, T2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Mean opinion $\\bar{x}(t)$', fontsize=9)
    if add_label:
        panel_label(ax, '(a)')
    style_ax(ax)


def draw_b(ax, res, add_label=True):
    """(b) ablated-model mean opinion（单橙，无图例）。"""
    m, ci = _mc(res, 'avg_x')
    ax.plot(m, color=ORANGE, lw=1.6)
    ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=ORANGE, alpha=0.2, lw=0)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylim(-1, 1)
    annotate(ax, T1, T2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Mean opinion $\\bar{x}(t)$', fontsize=9)
    if add_label:
        panel_label(ax, '(b)')
    style_ax(ax)


def draw_c(ax, res_full, res_ab, add_label=True):
    """(c) positive-support fraction（双线，有图例）。"""
    for res, c, lab in [(res_full, BLUE, 'Full'), (res_ab, ORANGE, 'Ablated')]:
        m, ci = _mc(res, 'avg_sigma')
        ax.plot(m, color=c, lw=1.6, label=lab)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.axhline(0.5, color='k', lw=0.8)
    ax.set_ylim(0, 1)
    annotate(ax, T1, T2)
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Support fraction $P_+(t)$', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    if add_label:
        panel_label(ax, '(c)')
    style_ax(ax)


def draw_d(ax, res_full, res_ab, add_label=True):
    """(d) system-level reversal probability（柱状图，full vs ablated）。"""
    p_full = float(np.mean(np.array(res_full['x_final']) > 0))
    p_ab = float(np.mean(np.array(res_ab['x_final']) > 0))
    labels = ['Full', 'Ablated']
    vals = [p_full, p_ab]
    colors = [BLUE, ORANGE]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='k', linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Reversal probability $P_{\\mathrm{rev}}$', fontsize=9)
    ax.tick_params(axis='both', direction='in', labelsize=9)
    ax.grid(axis='y', alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if add_label:
        panel_label(ax, '(d)')


def main():
    global T1, T2
    params = base_params()
    T1 = params['info1_duration']                 # 5
    T2 = T1 + params['post_info1_steps'] + 1      # 21
    t0 = time.time()
    print('重跑 full 模型 (100 MC)...', flush=True)
    res_full = monte_carlo_experiment(params, num_runs=100)
    p_full = float(np.mean(np.array(res_full['x_final']) > 0))
    print(f'  full 完成，P_rev = {p_full:.3f}，耗时 {time.time()-t0:.0f}s', flush=True)
    ab = dict(params, alpha=0.0)
    print('重跑 ablated 模型 (alpha=0, 100 MC)...', flush=True)
    res_ab = monte_carlo_experiment(ab, num_runs=100)
    p_ab = float(np.mean(np.array(res_ab['x_final']) > 0))
    print(f'  ablated 完成，P_rev = {p_ab:.3f}，总耗时 {time.time()-t0:.0f}s', flush=True)

    # 合并图（2×2，含标签，供 docx）
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    draw_a(axes[0, 0], res_full, True)
    draw_b(axes[0, 1], res_ab, True)
    draw_c(axes[1, 0], res_full, res_ab, True)
    draw_d(axes[1, 1], res_full, res_ab, True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig3.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 子图（无标签，供 PDF）
    jobs = [('Fig3_a.png', draw_a, (4.8, 3.6), (res_full,)),
            ('Fig3_b.png', draw_b, (4.8, 3.6), (res_ab,)),
            ('Fig3_c.png', draw_c, (4.8, 3.6), (res_full, res_ab)),
            ('Fig3_d.png', draw_d, (4.8, 3.2), (res_full, res_ab))]
    for name, fn, fs, args in jobs:
        f = plt.figure(figsize=fs)
        ax = f.add_subplot(111)
        fn(ax, *args, add_label=False)
        f.tight_layout()
        f.savefig(os.path.join(OUT, 'subfigs', name), dpi=300,
                  bbox_inches='tight', facecolor='white')
        plt.close(f)

    print('Fig3.png + subfigs/Fig3_a/b/c/d.png 已生成')
    print(f'P_rev: full={p_full:.3f}, ablated={p_ab:.3f}')


if __name__ == '__main__':
    main()
