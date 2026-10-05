# -*- coding: utf-8 -*-
"""任务 2：固定 χ_pre 的严格 finite-size scaling。

原 fss 用 speed_params(0.4,10) 固定参数，导致不同 N 的 χ_pre 漂移（0.83→0.94）。
本脚本对每个 N 先校准共识形成参数 T（保持 p1*T=4），使 χ_pre 对齐到目标值
（约 0.86），再扫描 I2 测 I_c，从而隔离出纯的有限尺寸效应。
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import monte_carlo_experiment, reversal_probability

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
TARGET_CHI = 0.86
I2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T_cands = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def measure_chi(N, T, num_runs=40):
    p1 = 4.0 / T
    params = speed_params(p1, T)
    params['num_nodes'] = N
    res = monte_carlo_experiment(params, num_runs=num_runs)
    return float(np.mean(res['S_pre']))


def calibrate(N):
    chis = []
    for T in T_cands:
        chi = measure_chi(N, T)
        chis.append(chi)
        print(f"    T={T}: chi={chi:.4f}", flush=True)
    chis = np.array(chis)
    best = int(np.argmin(np.abs(chis - TARGET_CHI)))
    T_star = T_cands[best]
    print(f"  N={N} 校准完成: T*={T_star}, chi={chis[best]:.4f}", flush=True)
    return T_star


def scan_ic(N, T_star, num_runs=40):
    p1 = 4.0 / T_star
    rows = []
    for i2 in I2_vals:
        params = speed_params(p1, T_star)
        params['num_nodes'] = N
        params['info2_value'] = i2
        res = monte_carlo_experiment(params, num_runs=num_runs)
        P_rev = reversal_probability(res)
        xf = np.array(res['x_final'])
        M = np.abs(xf)
        M_mean = float(np.mean(M))
        chi_rev = float(N * (np.mean(M ** 2) - M_mean ** 2))
        rows.append(dict(I2=i2, P_rev=P_rev, M_mean=M_mean, chi_rev=chi_rev))
        print(f"    I2={i2}: P_rev={P_rev:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f'fss_fixedchi_N{N}.csv'), index=False)
    i2 = df['I2'].values; p = df['P_rev'].values
    idx = np.where(p >= 0.5)[0]
    if len(idx) == 0:
        Ic = None
    elif idx[0] == 0:
        Ic = i2[0]
    else:
        j = idx[0]
        Ic = i2[j-1] + (0.5 - p[j-1]) / (p[j] - p[j-1]) * (i2[j] - i2[j-1])
    print(f"  N={N}: I_c={None if Ic is None else round(Ic,3)}", flush=True)
    return Ic


if __name__ == '__main__':
    Ns = [250, 500, 1000, 2000, 4000]
    res = {}
    t0 = time.time()
    for N in Ns:
        print(f"=== N={N} 校准 ===", flush=True)
        T_star = calibrate(N)
        print(f"=== N={N} 扫描 (T*={T_star}) ===", flush=True)
        Ic = scan_ic(N, T_star)
        res[N] = Ic
    print("=== 固定 χ_pre 后的 I_c(N) ===", flush=True)
    for N in Ns:
        print(f"  N={N}: I_c={res[N]}", flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)
