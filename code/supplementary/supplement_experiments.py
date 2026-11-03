
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import monte_carlo_experiment, reversal_probability

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

def speed_params(p1, d1):
    params = base_params()
    params.update({
        'info1_value': -0.3,
        'info1_reception_prob': p1,
        'info1_duration': d1,
        'info1_decay': False,
        'post_info1_steps': 7,
        'info2_value': 0.7,
        'info2_reception_prob': 0.6,
        'info2_duration': 1,
    })
    params['post_info2_steps'] = (params['total_steps'] - params['pre_info_steps']
                                  - params['info1_duration'] - params['post_info1_steps']
                                  - params['info2_duration'])
    return params

def exp_speed(num_runs=100):

    combos = [(0.10, 40), (0.20, 20), (4/15, 15), (1/3, 12), (0.40, 10),
              (0.50, 8), (4/7, 7), (2/3, 6), (0.80, 5), (1.00, 4)]
    t0 = time.time()
    rows = []
    traces = {}
    for gi, (p1, d1) in enumerate(combos, start=1):
        params = speed_params(p1, d1)
        res = monte_carlo_experiment(params, num_runs=num_runs)
        xf = np.array(res['x_final'])
        chi = float(np.mean(res['S_pre']))
        x_pre = float(np.mean(res['x_pre']))
        x_final = float(np.mean(xf))
        P_rev = float(reversal_probability(res))
        rev_cnt = int(np.sum(xf > 0))
        rows.append(dict(p=round(p1, 4), T=d1, v=round(1.0 / d1, 4), chi_pre=round(chi, 4),
                         x_pre=round(x_pre, 4), x_final=round(x_final, 4),
                         P_rev=round(P_rev, 4), rev_cnt=rev_cnt))
        traces[f'G{gi}_p'] = p1
        traces[f'G{gi}_T'] = d1
        traces[f'G{gi}_avg_x'] = np.array(res['avg_x'])
        traces[f'G{gi}_avg_S'] = np.array(res['avg_S'])
        print(f"  (p={p1:.3f}, T={d1:2d}, v={1/d1:.3f}): chi_pre={chi:.3f}, "
              f"x_pre={x_pre:.3f}, x_final={x_final:.3f}, P_rev={P_rev:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'speed_10_levels.csv'), index=False)
    np.savez_compressed(os.path.join(OUT, 'speed_traces.npz'), **traces)
    print(f"  轨迹已保存 speed_traces.npz ({len(traces)} arrays)")
    print(f"  实验 A 完成，耗时 {time.time()-t0:.1f}s")

    import statsmodels.api as sm
    X = sm.add_constant(df['chi_pre'].values)
    success = df['rev_cnt'].values.astype(float)
    failure = np.full(len(df), num_runs, dtype=float) - success
    y = np.column_stack([success, failure])
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    a, b = model.params
    b_se = model.bse[1]
    b_lo, b_hi = b - 1.96 * b_se, b + 1.96 * b_se
    pval_b = model.pvalues[1]
    print(f"  logistic 回归: logit(P_rev) = {a:.3f} + {b:.3f} * chi_pre")
    print(f"    b = {b:.3f} (SE={b_se:.3f}), 95% CI = [{b_lo:.3f}, {b_hi:.3f}], p = {pval_b:.4f}")
    json.dump(dict(a=float(a), b=float(b), b_se=float(b_se),
                   b_ci_lo=float(b_lo), b_ci_hi=float(b_hi), p_b=float(pval_b)),
              open(os.path.join(OUT, 'speed_logistic.json'), 'w'), indent=2)

    chi = df['chi_pre'].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    p_rev = df['P_rev'].values
    n = num_runs
    se = np.sqrt(np.clip(p_rev * (1 - p_rev), 1e-12, None) / n)
    ax.errorbar(chi, p_rev, yerr=1.96 * se, fmt='o', color='#B2182B', capsize=3, label='10 组（不同形成速度）')
    chi_grid = np.linspace(chi.min() * 0.9, chi.max() * 1.05, 100)
    p_fit = 1 / (1 + np.exp(-(a + b * chi_grid)))
    ax.plot(chi_grid, p_fit, '-', color='k', linewidth=2,
            label=f'logistic 拟合：logit(P)={a:.2f}+{b:.2f}·χ')
    ax.set_xlabel('信息2投入前系统易感性 χ_pre')
    ax.set_ylabel('系统级反转概率 P_rev')
    ax.set_title(f'(a) P_rev vs χ_pre（b={b:.2f}, p={pval_b:.4f}）')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax2 = axes[1]
    ax2.plot(chi, df['x_final'].values, 'o-', color='#2E86AB', linewidth=2)
    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_xlabel('χ_pre'); ax2.set_ylabel('最终平均意见 x̄_final')
    ax2.set_title('(b) 最终平均意见随易感性变化')
    ax2.grid(alpha=0.25)

    fig.suptitle('补充实验 A：同共识、不同速度（10 组）+ logistic 回归', fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT, 'S_A_speed_10_levels.png')
    fig.savefig(path, dpi=160); plt.close(fig)
    print("  已保存:", path)
    return df, dict(a=a, b=b, b_se=b_se, b_ci_lo=b_lo, b_ci_hi=b_hi, p_b=pval_b)

def exp_fss(num_runs=40):
    N_vals = [250, 500, 1000, 2000]
    I2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t0 = time.time()
    records = {N: {'I2': [], 'P_rev': [], 'M_mean': [], 'chi_rev': []} for N in N_vals}
    for N in N_vals:
        for i2 in I2_vals:
            params = speed_params(0.8, 5)
            params['num_nodes'] = N
            params['info2_value'] = i2
            res = monte_carlo_experiment(params, num_runs=num_runs)
            xf = np.array(res['x_final'])
            M = np.abs(xf)
            P_rev = reversal_probability(res)
            M_mean = float(np.mean(M))
            chi_rev = float(N * (np.mean(M ** 2) - M_mean ** 2))
            records[N]['I2'].append(i2)
            records[N]['P_rev'].append(P_rev)
            records[N]['M_mean'].append(M_mean)
            records[N]['chi_rev'].append(chi_rev)
        pd.DataFrame(records[N]).to_csv(os.path.join(OUT, f'fss_N{N}.csv'), index=False)
        print(f"  N={N} 完成，耗时 {time.time()-t0:.1f}s")

    Ic = {}
    for N in N_vals:
        i2 = np.array(records[N]['I2'])
        p = np.array(records[N]['P_rev'])
        idx = np.where(p >= 0.5)[0]
        if len(idx) == 0:
            Ic[N] = None
        else:
            j = idx[0]
            Ic[N] = i2[0] if j == 0 else i2[j - 1] + (0.5 - p[j - 1]) / (p[j] - p[j - 1]) * (i2[j] - i2[j - 1])
    print("  伪临界点 I_c(N):", {k: (round(v, 3) if v else None) for k, v in Ic.items()})

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(N_vals)))
    for N, c in zip(N_vals, colors):
        axes[0].plot(records[N]['I2'], records[N]['P_rev'], 'o-', color=c, linewidth=2, label=f'N={N}')
    axes[0].axhline(0.5, color='gray', ls='--', linewidth=1)
    axes[0].set_xlabel('反向信息强度 I_2'); axes[0].set_ylabel('P_rev')
    axes[0].set_title('(a) 反转概率曲线随 N 锐化'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    for N, c in zip(N_vals, colors):
        axes[1].plot(records[N]['I2'], records[N]['M_mean'], 'o-', color=c, linewidth=2, label=f'N={N}')
    axes[1].set_xlabel('I_2'); axes[1].set_ylabel('序参量 M = ⟨|x̄_final|⟩')
    axes[1].set_title('(b) 序参量'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    for N, c in zip(N_vals, colors):
        axes[2].plot(records[N]['I2'], records[N]['chi_rev'], 'o-', color=c, linewidth=2, label=f'N={N}')
    axes[2].set_xlabel('I_2'); axes[2].set_ylabel('χ_rev = N(⟨M²⟩−⟨M⟩²)')
    axes[2].set_title('(c) 反转涨落（临界点峰值）'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)

    fig.suptitle('补充实验 B：有限尺寸标度（finite-size scaling）', fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT, 'S_B_finite_size.png')
    fig.savefig(path, dpi=160); plt.close(fig)
    print("  已保存:", path)

    fig2, ax = plt.subplots(figsize=(6, 4.5))
    for N, c in zip(N_vals, colors):
        if Ic[N] is None:
            continue
        x = np.array(records[N]['I2']) - Ic[N]
        ax.plot(x, records[N]['P_rev'], 'o-', color=c, linewidth=2, label=f'N={N}')
    ax.axvline(0, color='gray', ls='--'); ax.axhline(0.5, color='gray', ls='--')
    ax.set_xlabel('I_2 − I_c(N)'); ax.set_ylabel('P_rev')
    ax.set_title('(d) 以各 N 的伪临界点对齐后的曲线')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig2.tight_layout()
    path2 = os.path.join(OUT, 'S_B_scaling_collapse.png')
    fig2.savefig(path2, dpi=160); plt.close(fig2)
    print("  已保存:", path2)

    ic_rows = []
    for N in N_vals:
        chi_arr = np.array(records[N]['chi_rev'])
        peak = float(chi_arr.max())
        peak_pos = records[N]['I2'][int(np.argmax(chi_arr))]
        ic_rows.append(dict(N=N, Ic=round(Ic[N], 4) if Ic[N] else None,
                            chi_peak=round(peak, 4), chi_peak_I2=peak_pos))
    pd.DataFrame(ic_rows).to_csv(os.path.join(OUT, 'fss_Ic_summary.csv'), index=False)
    print("  I_c 与峰值汇总:")
    for r in ic_rows:
        print("   ", r)
    return records, Ic

if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'all'
    nr = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if task in ('speed', 'all'):
        exp_speed(nr or 100)
    if task in ('fss', 'all'):
        exp_fss(nr or 40)
