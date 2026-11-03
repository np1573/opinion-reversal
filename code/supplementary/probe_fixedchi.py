
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import monte_carlo_experiment

Ns = [250, 500, 1000, 2000, 4000]
T_cands = [4, 5, 6, 7, 8]

for N in Ns:
    line = []
    for T in T_cands:
        p1 = 4.0 / T
        params = speed_params(p1, T)
        params['num_nodes'] = N
        res = monte_carlo_experiment(params, num_runs=20)
        chi = float(np.mean(res['S_pre']))
        line.append(f"T={T}:{chi:.3f}")
    print(f"N={N}: " + "  ".join(line), flush=True)

print("=== 探测完成 ===", flush=True)
