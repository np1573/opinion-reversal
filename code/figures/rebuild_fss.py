# -*- coding: utf-8 -*-
"""补跑有限尺寸标度：N=2000 和 N=4000（复用已有 N=250/500/1000 数据）。

与 exp_fss 完全一致的参数：固定"较快"共识形成 speed_params(0.4, 10)，
扫描 I2 ∈ {0.1,...,0.9}，num_runs=40。
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import monte_carlo_experiment, reversal_probability

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def run_N(N, num_runs=40):
    I2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t0 = time.time()
    records = {'I2': [], 'P_rev': [], 'M_mean': [], 'chi_rev': []}
    for i2 in I2_vals:
        params = speed_params(0.4, 10)
        params['num_nodes'] = N
        params['info2_value'] = i2
        res = monte_carlo_experiment(params, num_runs=num_runs)
        xf = np.array(res['x_final'])
        M = np.abs(xf)
        P_rev = reversal_probability(res)
        M_mean = float(np.mean(M))
        chi_rev = float(N * (np.mean(M ** 2) - M_mean ** 2))
        records['I2'].append(i2)
        records['P_rev'].append(P_rev)
        records['M_mean'].append(M_mean)
        records['chi_rev'].append(chi_rev)
        print(f"  N={N}, I2={i2}: P_rev={P_rev:.3f}", flush=True)
    pd.DataFrame(records).to_csv(os.path.join(OUT, f'fss_N{N}.csv'), index=False)
    print(f"N={N} 完成，耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    Ns = [int(x) for x in sys.argv[1:]] or [2000, 4000]
    for N in Ns:
        run_N(N)
