
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import monte_carlo_experiment, reversal_probability
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

COMBOS = [(0.1, 40), (0.2, 20), (0.27, 15), (0.33, 12), (0.4, 10),
          (0.5, 8), (0.57, 7), (0.67, 6), (0.8, 5), (1.0, 4)]
I2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def main(num_runs=50):
    t0 = time.time()
    chi_axis = []
    P_rev = np.zeros((len(I2_vals), len(COMBOS)))
    x_final = np.zeros_like(P_rev)

    for j, (p1, d1) in enumerate(COMBOS):
        params = speed_params(p1, d1)
        res0 = monte_carlo_experiment(params, num_runs=num_runs)
        chi_axis.append(float(np.mean(res0['S_pre'])))
        for i, i2 in enumerate(I2_vals):
            p2 = dict(params, info2_value=i2)
            r = monte_carlo_experiment(p2, num_runs=num_runs)
            P_rev[i, j] = reversal_probability(r)
            x_final[i, j] = float(np.mean(r['x_final']))
        print(f"  [{j+1}/10] p1={p1} L1={d1}: chi_pre={chi_axis[-1]:.3f}"
              f"  (累计 {time.time()-t0:.0f}s)", flush=True)

    chi_axis = np.array(chi_axis)
    np.save(os.path.join(OUT, 'phase10_chi_axis.npy'), chi_axis)
    np.save(os.path.join(OUT, 'phase10_P_rev.npy'), P_rev)
    np.save(os.path.join(OUT, 'phase10_x_final.npy'), x_final)
    print(f"完成，总耗时 {time.time()-t0:.0f}s", flush=True)
    print("chi_pre 轴:", np.round(chi_axis, 3), flush=True)
    print("P_rev:", np.round(P_rev, 2), flush=True)

if __name__ == '__main__':
    main()
