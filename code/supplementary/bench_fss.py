
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import run_single_simulation

Ns = [250, 500, 1000, 2000, 4000]

if __name__ == '__main__':
    print("N\t单次耗时(s)\t预估 200x9 串行(小时)")
    for N in Ns:
        params = speed_params(0.8, 5)
        params['num_nodes'] = N

        ts = []
        for _ in range(3):
            t0 = time.time()
            run_single_simulation(params)
            ts.append(time.time() - t0)
        best = min(ts)
        total = best * 200 * 9 / 3600
        print(f"{N}\t{best:.3f}\t{total:.2f}", flush=True)
