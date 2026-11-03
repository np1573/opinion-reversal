
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import monte_carlo_experiment
from rebuild_phase import base_params

P1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TINJ = [5, 7]

def main(num_runs=100):
    for t in TINJ:
        chis, xpres = [], []
        for p1 in P1:
            p = base_params()
            p.update(dict(info1_reception_prob=p1, post_info1_steps=t))
            r = monte_carlo_experiment(p, num_runs=num_runs)
            chis.append(float(np.mean(r['S_pre'])))
            xpres.append(float(np.mean(r['x_pre'])))
            print(f"t_inj={t} p0,1={p1}: chi_pre={chis[-1]:.3f} "
                  f"x_pre={xpres[-1]:.3f} C_pre={abs(xpres[-1]):.3f}", flush=True)
        print(f"=== t_inj={t} 完成 ===\n"
              f"chi_pre = {np.round(chis,3)}\n"
              f"x_pre   = {np.round(xpres,3)}\n"
              f"C_pre   = {np.round(np.abs(xpres),3)}\n", flush=True)

if __name__ == '__main__':
    main()
