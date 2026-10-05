# -*- coding: utf-8 -*-
"""补跑 N=250/500/4000 的临界点 x 200 MC，得到准确的 I_c（N=1000/2000 已用 200 MC 验证）。"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import monte_carlo_experiment, reversal_probability


def run_points(N, I2_list, num_runs=200, n_workers=None):
    t0 = time.time()
    print(f"=== N={N}, num_runs={num_runs} ===", flush=True)
    for i2 in I2_list:
        params = speed_params(0.4, 10)
        params['num_nodes'] = N
        params['info2_value'] = i2
        kw = dict(num_runs=num_runs)
        if n_workers:
            kw['n_workers'] = n_workers
        res = monte_carlo_experiment(params, **kw)
        P_rev = reversal_probability(res)
        print(f"  I2={i2}: P_rev={P_rev:.3f}  (累计 {time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    run_points(250, [0.3, 0.4, 0.5, 0.6])
    run_points(500, [0.3, 0.4, 0.5, 0.6])
    # N=4000 慢且易 OOM，用 n_workers=2 + num_runs=100 折中
    run_points(4000, [0.3, 0.4, 0.5, 0.6], num_runs=100, n_workers=2)
    print("全部完成", flush=True)
