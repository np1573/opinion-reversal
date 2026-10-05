# -*- coding: utf-8 -*-
"""
第四章 改进版实验 + 绘图
=========================
按 ChatGPT 建议重组为机制导向的"核心图"：
  Figure 1  基准动力学（4-panel：平均意见/正向支持率/平均易感性/平均反转概率）
  Figure 2  怀疑机制消融（完整 vs 去除怀疑）
  Figure 3  内生易感性（共识形成速度 -> 易感性 -> 反转，含路径依赖散点）
  Figure 4  状态依赖的扰动响应相图（S_pre x I_2 -> P_rev，含反转边界）
  Figure 5  外部信息特征（强度/时机/范围/释放方式）
  Figure 6  稳健性汇总（网络结构 + 参数敏感性）

用法：
  python experiments.py <实验名> [num_runs]
  实验名: baseline | ablation | consensus | phase | info | robust | all
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (monte_carlo_experiment, run_single_simulation,
                   compute_statistics, reversal_probability)

# 中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)


def base_params():
    p = {
        'network_formation': ['BA', 4],
        'num_nodes': 500,
        'opinion_init_mode': 'random',
        'mu': 0.1, 'alpha': 60, 'tau': 10,
        'pre_info_steps': 0,
        'info1_value': -0.9, 'info1_reception_prob': 0.4, 'info1_duration': 5, 'info1_decay': True,
        'post_info1_steps': 15,
        'info2_value': 0.7, 'info2_reception_prob': 0.6, 'info2_duration': 1, 'info2_decay': True,
        'total_steps': 200,
    }
    p['post_info2_steps'] = (p['total_steps'] - p['pre_info_steps'] - p['info1_duration']
                             - p['post_info1_steps'] - p['info2_duration'])
    return p


def info_inject_times(params):
    """返回信息1、信息2的注入时刻区间（记录索引）"""
    t0 = params['pre_info_steps']
    d1 = params['info1_duration']
    p1 = params['post_info1_steps']
    d2 = params['info2_duration']
    info1 = range(t0 + 1, t0 + d1 + 1)
    info2 = range(t0 + d1 + p1 + 1, t0 + d1 + p1 + d2 + 1)
    return info1, info2


def plot_series_ci(ax, results, key, color, label, xlabel='Step'):
    """在 ax 上绘制时间序列均值 + 95% CI"""
    df = pd.DataFrame(results[key])
    summ = compute_statistics(df)
    steps = summ.index
    means = summ['mean']
    ci = summ['ci']
    ax.plot(steps, means, color=color, linewidth=2.0, label=label)
    ax.fill_between(steps, means - ci, means + ci, color=color, alpha=0.25, linewidth=0)


def annotate_injection(ax, params):
    """标注信息注入时刻"""
    info1, info2 = info_inject_times(params)
    if len(info1) > 1:
        ax.axvspan(info1[0], info1[-1], color='gray', alpha=0.18, zorder=0)
    else:
        ax.axvline(x=info1[0], color='black', linestyle='--', linewidth=1.0)
    if len(info2) > 1:
        ax.axvspan(info2[0], info2[-1], color='magenta', alpha=0.14, zorder=0)
    else:
        ax.axvline(x=info2[0], color='magenta', linestyle='--', linewidth=1.2)


# ===================== Figure 1: 基准动力学 =====================

def exp_baseline(num_runs=100):
    params = base_params()
    t0 = time.time()
    print("运行基准实验...")
    res = monte_carlo_experiment(params, num_runs=num_runs)
    print(f"  完成，耗时 {time.time()-t0:.1f}s，P_rev={reversal_probability(res):.3f}")

    info1, info2 = info_inject_times(params)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    plot_series_ci(axes[0, 0], res, 'avg_x', '#2E86AB', '平均意见 $\\overline{x}(t)$')
    axes[0, 0].axhline(0, color='k', linewidth=0.8)
    axes[0, 0].set_ylim(-1, 1)
    annotate_injection(axes[0, 0], params)
    axes[0, 0].set_title('(a) 平均意见 $\\overline{x}(t)$', fontsize=12)
    axes[0, 0].set_ylabel('意见值')

    plot_series_ci(axes[0, 1], res, 'avg_sigma', '#009E73', '正向支持率 $P_+(t)$')
    axes[0, 1].axhline(0.5, color='k', linewidth=0.8)
    axes[0, 1].set_ylim(0, 1)
    annotate_injection(axes[0, 1], params)
    axes[0, 1].set_title('(b) 正向支持率 $P_+(t)$', fontsize=12)
    axes[0, 1].set_ylabel('支持率')

    plot_series_ci(axes[1, 0], res, 'avg_S', '#D55E00', '平均易感性 $\\overline{S}(t)$')
    annotate_injection(axes[1, 0], params)
    axes[1, 0].set_title('(c) 平均易感性 $\\overline{S}(t)$', fontsize=12)
    axes[1, 0].set_ylabel('易感性')
    axes[1, 0].set_xlabel('时间步')

    plot_series_ci(axes[1, 1], res, 'avg_R', '#CC79A7', '平均反转概率 $\\overline{R}(t)$')
    axes[1, 1].set_ylim(0, 1)
    annotate_injection(axes[1, 1], params)
    axes[1, 1].set_title('(d) 平均反转概率 $\\overline{R}(t)$', fontsize=12)
    axes[1, 1].set_xlabel('时间步')

    for ax in axes.flat:
        ax.tick_params(axis='both', direction='in')
        ax.grid(alpha=0.25)

    fig.suptitle('Figure 1  基准动力学：共识形成 → 易感性上升 → 反向信息触发反转',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, 'Fig1_baseline.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  已保存:", path)


# ===================== Figure 2: 消融实验 =====================

def exp_ablation(num_runs=100):
    params = base_params()
    t0 = time.time()
    print("运行消融实验...")
    res_full = monte_carlo_experiment(params, num_runs=num_runs)
    print(f"  完整模型完成，P_rev={reversal_probability(res_full):.3f}")
    ab = dict(params, alpha=0.0)
    res_ab = monte_carlo_experiment(ab, num_runs=num_runs)
    print(f"  消融模型完成，P_rev={reversal_probability(res_ab):.3f}，耗时 {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, (res, title) in zip(axes, [(res_full, '完整模型'), (res_ab, '去除怀疑机制')]):
        plot_series_ci(ax, res, 'avg_x', '#2E86AB', '平均意见')
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_ylim(-1, 1)
        annotate_injection(ax, params)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('时间步')
        ax.set_ylabel('意见值')
        ax.tick_params(axis='both', direction='in')

    fig.suptitle('Figure 2  怀疑机制消融实验', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'Fig2_ablation.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  已保存:", path)


# ===================== Figure 3: 内生易感性（共识速度） =====================

def exp_consensus(num_runs=100):
    """同共识、不同形成速度 → 易感性 → 反转"""
    combos = [(0.1, 40), (0.2, 20), (0.4, 10), (0.8, 5)]
    t0 = time.time()
    print("运行共识形成速度实验...")
    all_res = {}
    for p1, d1 in combos:
        params = base_params()
        params.update({
            'info1_value': -0.3,
            'info1_reception_prob': p1,
            'info1_duration': d1,
            'info1_decay': False,   # 固定接收概率（无衰减）
            'post_info1_steps': 12,
            'info2_value': 0.7,
            'info2_reception_prob': 0.6,
            'info2_duration': 1,
        })
        params['post_info2_steps'] = (params['total_steps'] - params['pre_info_steps']
                                      - params['info1_duration'] - params['post_info1_steps']
                                      - params['info2_duration'])
        res = monte_carlo_experiment(params, num_runs=num_runs)
        all_res[(p1, d1)] = (params, res)
        print(f"  (p={p1}, duration={d1}): S_pre={np.mean(res['S_pre']):.3f}, "
              f"x_pre={np.mean(res['x_pre']):.3f}, x_final={np.mean(res['x_final']):.3f}, "
              f"P_rev={reversal_probability(res):.3f}")
    print(f"  完成，耗时 {time.time()-t0:.1f}s")

    speeds = [1.0 / d for _, d in combos]  # 共识形成速度代理
    colors = ['#2166AC', '#67A9CF', '#EF8A62', '#B2182B']

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for (p1, d1), c, s in zip(combos, colors, speeds):
        params, res = all_res[(p1, d1)]
        # (a) 共识度 f(t) = |avg_x(t)|
        df = pd.DataFrame(res['avg_x'])
        summ = compute_statistics(df)
        f_mean = np.abs(summ['mean'])
        f_ci = summ['ci']
        axes[0, 0].plot(summ.index, f_mean, color=c, linewidth=2.0,
                        label=f'p={p1}, T={d1} (v={s:.3f})')
        axes[0, 0].fill_between(summ.index, f_mean - f_ci, f_mean + f_ci,
                                color=c, alpha=0.2, linewidth=0)
        # (b) 易感性
        plot_series_ci(axes[0, 1], res, 'avg_S', c, f'p={p1}, T={d1}')
        # (c) 平均意见（反向信息作用）
        plot_series_ci(axes[1, 0], res, 'avg_x', c, f'p={p1}, T={d1}')

    axes[0, 0].set_title('(a) 共识度 $|\\overline{x}(t)|$', fontsize=12)
    axes[0, 0].set_ylabel('共识度')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title('(b) 平均易感性 $\\overline{S}(t)$', fontsize=12)
    axes[0, 1].set_ylabel('易感性')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_title('(c) 平均意见 $\\overline{x}(t)$', fontsize=12)
    axes[1, 0].axhline(0, color='k', linewidth=0.8)
    axes[1, 0].set_ylim(-1, 1)
    axes[1, 0].set_ylabel('意见值')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].legend(fontsize=8)

    # (d) 路径依赖散点：S_pre -> x_final（颜色 = 共识速度）
    axd = axes[1, 1]
    for (p1, d1), c in zip(combos, colors):
        params, res = all_res[(p1, d1)]
        S_pre = np.array(res['S_pre'])
        x_final = np.array(res['x_final'])
        axd.scatter(S_pre, x_final, s=22, alpha=0.5, color=c,
                    label=f'p={p1}, T={d1}')
        axd.scatter([np.mean(S_pre)], [np.mean(x_final)], s=120, color=c,
                    edgecolor='black', linewidth=1.2, zorder=3)
    axd.axhline(0, color='k', linewidth=0.8)
    axd.set_title('(d) 路径依赖：易感性 → 反转', fontsize=12)
    axd.set_xlabel('信息2投入前平均易感性 $\\overline{S}_{pre}$')
    axd.set_ylabel('最终平均意见 $\\overline{x}_{final}$')
    axd.legend(fontsize=8)

    for ax in axes.flat:
        ax.tick_params(axis='both', direction='in')
        ax.grid(alpha=0.25)

    fig.suptitle('Figure 3  共识形成速度 → 内生易感性 → 反转（最终共识相近、形成速度不同）',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, 'Fig3_consensus_speed.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  已保存:", path)

    # 汇总散点（单独一张，供正文引用核心结论）
    fig2, ax = plt.subplots(figsize=(5.2, 4))
    S_means, x_means, v_means = [], [], []
    for (p1, d1), c in zip(combos, colors):
        params, res = all_res[(p1, d1)]
        S_means.append(np.mean(res['S_pre']))
        x_means.append(np.mean(res['x_final']))
        v_means.append(1.0 / d1)
    ax.scatter(S_means, x_means, c=colors, s=140, edgecolor='black', linewidth=1.2)
    for i, (p1, d1) in enumerate(combos):
        ax.annotate(f'v={v_means[i]:.2f}', (S_means[i], x_means[i]),
                    textcoords='offset points', xytext=(8, -4), fontsize=9)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('$\overline{S}_{pre}$ (易感性)')
    ax.set_ylabel('$\overline{x}_{final}$ (最终平均意见)')
    ax.set_title('易感性越高 → 反转越彻底', fontsize=11)
    ax.grid(alpha=0.25)
    fig2.tight_layout()
    path2 = os.path.join(OUT, 'Fig3b_path_dependency.png')
    fig2.savefig(path2, dpi=160)
    plt.close(fig2)
    print("  已保存:", path2)


# ===================== Figure 4: 相图 S_pre x I_2 -> P_rev =====================

def exp_phase(num_runs=50):
    """核心创新图：状态依赖的扰动响应相图"""
    info1_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    info2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    t0 = time.time()
    print("运行相图实验 (S_pre x I_2)...")
    S_pre_axis = np.zeros(len(info1_probs))
    P_rev_matrix = np.zeros((len(info2_vals), len(info1_probs)))
    x_final_matrix = np.zeros_like(P_rev_matrix)

    for j, p1 in enumerate(info1_probs):
        params = base_params()
        params['info1_reception_prob'] = p1
        res = monte_carlo_experiment(params, num_runs=num_runs)
        S_pre_axis[j] = np.mean(res['S_pre'])
        for i, i2 in enumerate(info2_vals):
            params2 = dict(params, info2_value=i2)
            res2 = monte_carlo_experiment(params2, num_runs=num_runs)
            P_rev_matrix[i, j] = reversal_probability(res2)
            x_final_matrix[i, j] = np.mean(res2['x_final'])
        print(f"  info1_p={p1}: S_pre={S_pre_axis[j]:.3f}")

    print(f"  完成，耗时 {time.time()-t0:.1f}s")
    np.save(os.path.join(OUT, 'phase_S_pre_axis.npy'), S_pre_axis)
    np.save(os.path.join(OUT, 'phase_P_rev.npy'), P_rev_matrix)
    np.save(os.path.join(OUT, 'phase_x_final.npy'), x_final_matrix)

    # 反转概率相图
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    xs, ys = S_pre_axis, np.array(info2_vals)
    xb = np.empty(len(xs) + 1)
    xb[1:-1] = (xs[:-1] + xs[1:]) / 2
    xb[0] = xs[0] - (xs[1] - xs[0]) / 2
    xb[-1] = xs[-1] + (xs[-1] - xs[-2]) / 2
    yb = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                         [ys[-1] + (ys[-1] - ys[-2]) / 2]])
    im = ax.pcolormesh(xb, yb, P_rev_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('反转概率 $P_{rev}$')
    # 反转边界 P_rev = 0.5
    cs = ax.contour(xs, ys, P_rev_matrix, levels=[0.5], colors='black', linewidths=2.0)
    ax.clabel(cs, fmt='$P_{rev}=0.5$', fontsize=11)
    ax.set_xlabel('信息2投入前平均易感性 $\\overline{S}_{pre}$')
    ax.set_ylabel('反向信息强度 $I_2$')
    ax.set_title('Figure 4  状态依赖的扰动响应相图\n$P_{rev}=F(\\overline{S}_{pre}, I_2)$',
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', direction='in')
    fig.tight_layout()
    path = os.path.join(OUT, 'Fig4_phase_diagram.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  已保存:", path)

    # 最终平均意见热图（连续量，补充视角）
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.6))
    vmax = np.abs(x_final_matrix).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = ax2.pcolormesh(xb, yb, x_final_matrix, cmap='RdYlBu_r', norm=norm)
    cb2 = fig2.colorbar(im2, ax=ax2)
    cb2.set_label('最终平均意见 $\\overline{x}_{final}$')
    cs2 = ax2.contour(xs, ys, x_final_matrix, levels=[0.0], colors='black', linewidths=2.0)
    ax2.clabel(cs2, fmt='$\\overline{x}_{final}=0$', fontsize=11)
    ax2.set_xlabel('信息2投入前平均易感性 $\\overline{S}_{pre}$')
    ax2.set_ylabel('反向信息强度 $I_2$')
    ax2.set_title('最终平均意见（连续量视角）', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', direction='in')
    fig2.tight_layout()
    path2 = os.path.join(OUT, 'Fig4b_phase_xfinal.png')
    fig2.savefig(path2, dpi=160)
    plt.close(fig2)
    print("  已保存:", path2)


# ===================== Figure 5: 外部信息特征 =====================

def exp_info(num_runs=50):
    """强度 / 范围 / 时机 / 释放方式 的汇总"""
    t0 = time.time()

    # (a) 强度
    print("信息强度扫描...")
    I_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    strength_x, strength_p = [], []
    for i2 in I_vals:
        params = base_params(); params['info2_value'] = i2
        res = monte_carlo_experiment(params, num_runs=num_runs)
        strength_x.append(np.mean(res['x_final']))
        strength_p.append(reversal_probability(res))

    # (b) 范围
    print("信息范围扫描...")
    P_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    range_x, range_p = [], []
    for p2 in P_vals:
        params = base_params(); params['info2_reception_prob'] = p2
        res = monte_carlo_experiment(params, num_runs=num_runs)
        range_x.append(np.mean(res['x_final']))
        range_p.append(reversal_probability(res))

    # (c) 时机
    print("信息时机扫描...")
    T_vals = [0, 3, 6, 10, 15, 20, 30, 40, 55]
    timing_x, timing_p = [], []
    for dt in T_vals:
        params = base_params(); params['post_info1_steps'] = dt
        params['post_info2_steps'] = (params['total_steps'] - params['pre_info_steps']
                                      - params['info1_duration'] - dt - params['info2_duration'])
        res = monte_carlo_experiment(params, num_runs=num_runs)
        timing_x.append(np.mean(res['x_final']))
        timing_p.append(reversal_probability(res))

    # (d) 释放方式：脉冲 vs 持续
    print("释放方式对比...")
    pulse = base_params()
    res_pulse = monte_carlo_experiment(pulse, num_runs=num_runs)
    cont = base_params()
    cont.update({'info2_reception_prob': 0.2, 'info2_duration': 5})
    cont['post_info2_steps'] = (cont['total_steps'] - cont['pre_info_steps']
                                - cont['info1_duration'] - cont['post_info1_steps']
                                - cont['info2_duration'])
    res_cont = monte_carlo_experiment(cont, num_runs=num_runs)
    print(f"  完成，耗时 {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    axes[0, 0].plot(I_vals, strength_x, 'o-', color='#2E86AB', linewidth=2)
    axes[0, 0].axhline(0, color='k', linewidth=0.8)
    axes[0, 0].set_title('(a) 信息强度 $I_2$', fontsize=12)
    axes[0, 0].set_xlabel('$I_2$'); axes[0, 0].set_ylabel('$\overline{x}_{final}$')

    axes[0, 1].plot(P_vals, range_x, 's-', color='#009E73', linewidth=2)
    axes[0, 1].axhline(0, color='k', linewidth=0.8)
    axes[0, 1].set_title('(b) 传播范围 $P_2$', fontsize=12)
    axes[0, 1].set_xlabel('$P_2$'); axes[0, 1].set_ylabel('$\overline{x}_{final}$')

    axes[1, 0].plot(T_vals, timing_x, '^-', color='#D55E00', linewidth=2)
    axes[1, 0].axhline(0, color='k', linewidth=0.8)
    axes[1, 0].set_title('(c) 投入时机 $t_{inj}$', fontsize=12)
    axes[1, 0].set_xlabel('信息1后交互步数'); axes[1, 0].set_ylabel('$\overline{x}_{final}$')

    for ax, (res, c, lab) in zip([axes[1, 1]], [(res_pulse, '#CC79A7', '脉冲式'),
                                                 (res_cont, '#56B4E9', '持续式')]):
        pass
    plot_series_ci(axes[1, 1], res_pulse, 'avg_x', '#CC79A7', '脉冲式（单步）')
    plot_series_ci(axes[1, 1], res_cont, 'avg_x', '#56B4E9', '持续式（5步）')
    axes[1, 1].axhline(0, color='k', linewidth=0.8)
    axes[1, 1].set_ylim(-1, 1)
    axes[1, 1].set_title('(d) 释放方式', fontsize=12)
    axes[1, 1].set_xlabel('时间步'); axes[1, 1].set_ylabel('$\overline{x}(t)$')
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.25); ax.tick_params(axis='both', direction='in')
    fig.suptitle('Figure 5  外部信息特征的作用', fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, 'Fig5_info_features.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  已保存:", path)


# ===================== Figure 6: 稳健性 =====================

def exp_robust(num_runs=50):
    """网络结构 + 参数敏感性 -> 汇总"""
    t0 = time.time()

    # 网络结构
    nets = [('ER', ['ER', 0.016]), ('WS', ['WS', [8, 0.2]]),
            ('BA', ['BA', 4]), ('BA+SIT', ['BA', 4])]
    net_rows = []
    for name, cfg in nets:
        params = base_params()
        params['network_formation'] = cfg
        if name == 'BA+SIT':
            params['mu'] = 'Social_Impact'
        res = monte_carlo_experiment(params, num_runs=num_runs)
        net_rows.append((name, np.mean(res['x_final']), np.mean(res['S_pre']),
                         reversal_probability(res)))
        print(f"  网络 {name}: x_final={net_rows[-1][1]:.3f}, P_rev={net_rows[-1][3]:.3f}")

    # 参数敏感性
    param_sweeps = {
        'alpha': ([10, 20, 30, 40, 50, 60, 70, 80], 'alpha'),
        'mu': ([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8], 'mu'),
        'tau': ([1, 2, 3, 5, 8, 10, 15, 20], 'tau'),
        'N': ([300, 500, 700, 1000, 1500, 2000], 'num_nodes'),
        'k': ([2, 4, 6, 8, 10], 'k'),
    }
    param_rows = {}
    for pname, (vals, key) in param_sweeps.items():
        rows = []
        for v in vals:
            params = base_params()
            if key == 'k':
                params['network_formation'] = ['BA', v]
            else:
                params[key] = v
            res = monte_carlo_experiment(params, num_runs=num_runs)
            rows.append((v, np.mean(res['x_final']), reversal_probability(res)))
        param_rows[pname] = rows
        print(f"  参数 {pname}: 完成")

    print(f"  完成，耗时 {time.time()-t0:.1f}s")

    # 保存汇总 CSV
    net_df = pd.DataFrame(net_rows, columns=['network', 'x_final', 'S_pre', 'P_rev'])
    net_df.to_csv(os.path.join(OUT, 'robust_network.csv'), index=False)
    for pname, rows in param_rows.items():
        pd.DataFrame(rows, columns=['value', 'x_final', 'P_rev']).to_csv(
            os.path.join(OUT, f'robust_{pname}.csv'), index=False)

    # 绘图：网络结构柱状 + 参数曲线
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    names = [r[0] for r in net_rows]
    xf = [r[1] for r in net_rows]
    axes[0, 0].bar(names, xf, color=['#2166AC', '#67A9CF', '#EF8A62', '#B2182B'])
    axes[0, 0].axhline(0, color='k', linewidth=0.8)
    axes[0, 0].set_title('网络结构', fontsize=11)
    axes[0, 0].set_ylabel('$\overline{x}_{final}$')

    plot_specs = [('alpha', 0, 1), ('mu', 0, 2), ('tau', 1, 0), ('N', 1, 1), ('k', 1, 2)]
    for pname, ri, ci in plot_specs:
        rows = param_rows[pname]
        xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
        axes[ri, ci].plot(xs, ys, 'o-', color='#D55E00', linewidth=2)
        axes[ri, ci].axhline(0, color='k', linewidth=0.8)
        axes[ri, ci].set_title(f'参数 ${pname}$', fontsize=11)
        axes[ri, ci].set_xlabel(f'${pname}$'); axes[ri, ci].set_ylabel('$\overline{x}_{final}$')

    for ax in axes.flat:
        ax.grid(alpha=0.25); ax.tick_params(axis='both', direction='in')
    fig.suptitle('Figure 6  稳健性分析：机制在不同网络与参数下均保持', fontsize=13,
                 fontweight='bold', y=1.0)
    fig.tight_layout()
    path = os.path.join(OUT, 'Fig6_robustness.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  已保存:", path)


# ===================== Figure 4c: 纯净相图（同共识、扫 S_pre 与 I_2）=====================

def exp_phase_clean(num_runs=30):
    """
    纯净版相图：固定共识水平（x_pre ≈ -0.55），扫描 S_pre 与 I_2。
    用 4 组共识速度 (p, T) 作为 4 个 S_pre 水平，对每组扫描 I_2。
    """
    combos = [(0.1, 40), (0.2, 20), (0.4, 10), (0.8, 5)]
    I2_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

    t0 = time.time()
    print("运行纯净相图实验 (同共识、扫 S_pre 与 I_2)...")
    S_pre_axis = np.zeros(len(combos))
    P_rev_matrix = np.zeros((len(I2_vals), len(combos)))
    x_final_matrix = np.zeros_like(P_rev_matrix)
    x_pre_axis = np.zeros(len(combos))

    for j, (p1, d1) in enumerate(combos):
        params = base_params()
        params.update({
            'info1_value': -0.3,
            'info1_reception_prob': p1,
            'info1_duration': d1,
            'info1_decay': False,
            'post_info1_steps': 12,
            'info2_reception_prob': 0.6,
            'info2_duration': 1,
        })
        params['post_info2_steps'] = (params['total_steps'] - params['pre_info_steps']
                                      - params['info1_duration'] - params['post_info1_steps']
                                      - params['info2_duration'])
        S_pre_axis[j], x_pre_axis[j] = 0, 0
        for i, i2 in enumerate(I2_vals):
            p2 = dict(params, info2_value=i2)
            res = monte_carlo_experiment(p2, num_runs=num_runs)
            P_rev_matrix[i, j] = reversal_probability(res)
            x_final_matrix[i, j] = np.mean(res['x_final'])
            S_pre_axis[j] += np.mean(res['S_pre']) / len(I2_vals)
            x_pre_axis[j] += np.mean(res['x_pre']) / len(I2_vals)
        print(f"  (p={p1}, T={d1}): S_pre={S_pre_axis[j]:.3f}, x_pre={x_pre_axis[j]:.3f}")

    print(f"  完成，耗时 {time.time()-t0:.1f}s")
    np.save(os.path.join(OUT, 'phase_clean_S_pre.npy'), S_pre_axis)
    np.save(os.path.join(OUT, 'phase_clean_P_rev.npy'), P_rev_matrix)
    np.save(os.path.join(OUT, 'phase_clean_x_final.npy'), x_final_matrix)
    np.save(os.path.join(OUT, 'phase_clean_x_pre.npy'), x_pre_axis)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs, ys = S_pre_axis, np.array(I2_vals)
    xb = np.empty(len(xs) + 1)
    xb[1:-1] = (xs[:-1] + xs[1:]) / 2
    xb[0] = xs[0] - (xs[1] - xs[0]) / 2
    xb[-1] = xs[-1] + (xs[-1] - xs[-2]) / 2
    yb = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                         [ys[-1] + (ys[-1] - ys[-2]) / 2]])

    im = axes[0].pcolormesh(xb, yb, P_rev_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[0], label='反转概率 $P_{rev}$')
    cs = axes[0].contour(xs, ys, P_rev_matrix, levels=[0.5], colors='black', linewidths=2.0)
    axes[0].clabel(cs, fmt='$P_{rev}=0.5$', fontsize=11)
    axes[0].set_xlabel('$\\overline{S}_{pre}$ (共识相近、速度不同)')
    axes[0].set_ylabel('$I_2$')
    axes[0].set_title('(a) 纯净相图 $P_{rev}=F(\\overline{S}_{pre}, I_2)$', fontsize=12)

    vmax = np.abs(x_final_matrix).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = axes[1].pcolormesh(xb, yb, x_final_matrix, cmap='RdYlBu_r', norm=norm)
    fig.colorbar(im2, ax=axes[1], label='$\\overline{x}_{final}$')
    cs2 = axes[1].contour(xs, ys, x_final_matrix, levels=[0.0], colors='black', linewidths=2.0)
    axes[1].clabel(cs2, fmt='$\\overline{x}_{final}=0$', fontsize=11)
    axes[1].set_xlabel('$\\overline{S}_{pre}$')
    axes[1].set_ylabel('$I_2$')
    axes[1].set_title('(b) 最终平均意见', fontsize=12)

    for ax in axes:
        ax.tick_params(axis='both', direction='in')
    fig.suptitle(f'Figure 4c  纯净相图：共识水平相近 '
                 f'$\\overline{{x}}_{{pre}}\\in[{x_pre_axis.min():.2f}, {x_pre_axis.max():.2f}]$，'
                 f'扫描易感性与信息强度',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, 'Fig4c_phase_clean.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("  已保存:", path)


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'all'
    nr = int(sys.argv[2]) if len(sys.argv) > 2 else None
    jobs = {
        'baseline': lambda: exp_baseline(nr or 100),
        'ablation': lambda: exp_ablation(nr or 100),
        'consensus': lambda: exp_consensus(nr or 100),
        'phase': lambda: exp_phase(nr or 50),
        'phase_clean': lambda: exp_phase_clean(nr or 30),
        'info': lambda: exp_info(nr or 50),
        'robust': lambda: exp_robust(nr or 50),
    }
    if task == 'all':
        for fn in jobs.values():
            fn()
    else:
        jobs[task]()
