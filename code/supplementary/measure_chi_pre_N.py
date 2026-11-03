
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import monte_carlo_experiment

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
Ns = [250, 500, 1000, 2000, 4000]

if __name__ == '__main__':
    chi = []
    t0 = time.time()
    for N in Ns:
        params = speed_params(0.8, 5)
        params['num_nodes'] = N
        res = monte_carlo_experiment(params, num_runs=100)
        c = float(np.mean(res['S_pre']))
        ci = 1.96 * float(np.std(res['S_pre']) / np.sqrt(len(res['S_pre'])))
        chi.append((N, c, ci))
        print(f'N={N}: chi_pre={c:.4f} +/- {ci:.4f}  (耗时 {time.time()-t0:.0f}s)', flush=True)
    np.save(os.path.join(OUT, 'chi_pre_vs_N.npy'),
            np.array([[x[0], x[1], x[2]] for x in chi]))
    print('已保存 chi_pre_vs_N.npy')
