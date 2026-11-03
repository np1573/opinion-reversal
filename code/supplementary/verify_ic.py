
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import monte_carlo_experiment, reversal_probability

def run_points(N, I2_list, num_runs=200):
    t0 = time.time()
    print(f"=== N={N}, num_runs={num_runs} ===", flush=True)
    for i2 in I2_list:
        params = speed_params(0.4, 10)
        params['num_nodes'] = N
        params['info2_value'] = i2
        res = monte_carlo_experiment(params, num_runs=num_runs)
        P_rev = reversal_probability(res)
        print(f"  I2={i2}: P_rev={P_rev:.3f}  (累计 {time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    run_points(1000, [0.5, 0.6, 0.7, 0.8])
    run_points(2000, [0.1, 0.2, 0.3, 0.4])
    print("全部完成", flush=True)
