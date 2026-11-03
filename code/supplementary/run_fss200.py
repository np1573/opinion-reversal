
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params
from model import monte_carlo_experiment

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'fss200')
os.makedirs(OUT, exist_ok=True)

Ns = [250, 500, 1000, 2000, 4000]
I2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_RUNS = 200

def npz_path(N, i2):
    return os.path.join(OUT, f'fss200_N{N}_I2{i2}.npz')

def run_cell(N, i2, n_workers):
    params = speed_params(0.8, 5)
    params['num_nodes'] = N
    params['info2_value'] = i2
    res = monte_carlo_experiment(params, num_runs=NUM_RUNS, n_workers=n_workers)
    x_final = np.array(res['x_final'])
    S_pre = np.array(res['S_pre'])
    np.savez_compressed(npz_path(N, i2), x_final=x_final, S_pre=S_pre)
    P_rev = float(np.mean(x_final > 0.0))
    return P_rev, float(np.mean(S_pre))

if __name__ == '__main__':
    t0 = time.time()
    for N in Ns:

        nw = 6 if N >= 4000 else 10
        for i2 in I2_vals:
            p = npz_path(N, i2)
            if os.path.exists(p):
                print(f'[skip] N={N} I2={i2} 已存在', flush=True)
                continue
            t1 = time.time()
            P_rev, chi = run_cell(N, i2, nw)
            print(f'[done] N={N} I2={i2}: P_rev={P_rev:.3f} chi_pre={chi:.4f} '
                  f'(+{time.time()-t1:.0f}s, 累计 {time.time()-t0:.0f}s)', flush=True)
    print(f'全部完成，总耗时 {time.time()-t0:.0f}s', flush=True)
