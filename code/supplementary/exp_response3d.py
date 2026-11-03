
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import base_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def run(num_runs=50):
    p1_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    I2_vals = [0.3, 0.5, 0.7, 0.9]
    chi_axis, x_pre_axis = [], []
    P_rev = np.zeros((len(I2_vals), len(p1_vals)))
    x_final = np.zeros_like(P_rev)

    for j, p1 in enumerate(p1_vals):
        params = base_params()
        params.update(dict(info1_reception_prob=p1, post_info1_steps=5))
        res0 = M.monte_carlo_experiment(params, num_runs=num_runs)
        chi_axis.append(float(np.mean(res0['S_pre'])))
        x_pre_axis.append(float(np.mean(res0['x_pre'])))
        for i, i2 in enumerate(I2_vals):
            p2 = dict(params, info2_value=i2)
            r = M.monte_carlo_experiment(p2, num_runs=num_runs)
            P_rev[i, j] = M.reversal_probability(r)
            x_final[i, j] = float(np.mean(r['x_final']))
        print(f"p0,1={p1}: chi_pre={chi_axis[-1]:.3f}, x_pre={x_pre_axis[-1]:.3f}, "
              f"C_pre={abs(x_pre_axis[-1]):.3f}", flush=True)

    out = dict(chi_axis=chi_axis, x_pre_axis=x_pre_axis,
               I2_axis=I2_vals, P_rev=P_rev.tolist(), x_final=x_final.tolist())
    json.dump(out, open(os.path.join(OUT, 'response3d.json'), 'w'), indent=2)
    print('已保存 response3d.json')

if __name__ == '__main__':
    run()
