
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import monte_carlo_experiment
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'phase10boot')
os.makedirs(OUT, exist_ok=True)

COMBOS = [(0.10, 40), (0.20, 20), (4 / 15, 15), (1 / 3, 12), (0.40, 10),
          (0.50, 8), (4 / 7, 7), (2 / 3, 6), (0.80, 5), (1.00, 4)]
I2_VALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def main(num_runs=100):
    t0 = time.time()
    chi_axis = []

    for j, (p1, d1) in enumerate(COMBOS):
        params = speed_params(p1, d1)
        for i, i2 in enumerate(I2_VALS):
            p2 = dict(params, info2_value=i2)
            r = monte_carlo_experiment(p2, num_runs=num_runs, n_workers=10)
            xf = np.array(r['x_final'])
            np.savez(os.path.join(OUT, f'G{j}_I2{i2}.npz'), x_final=xf)
            if i == 0:
                chi_axis.append(float(np.mean(r['S_pre'])))
        print(f"[{j+1}/10] p1={p1:.4f} L1={d1}: chi_pre={chi_axis[-1]:.3f} "
              f"(累计 {time.time()-t0:.0f}s)", flush=True)

    np.save(os.path.join(OUT, 'chi_axis.npy'), np.array(chi_axis))
    print(f"完成，总耗时 {time.time()-t0:.0f}s", flush=True)
    print("chi_pre 轴:", np.round(chi_axis, 3), flush=True)

if __name__ == '__main__':
    main()
