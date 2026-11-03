
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def run(num_runs=100):
    combos = {'slow': (0.10, 40), 'fast': (1.00, 4)}
    out = {}
    for name, (p1, d1) in combos.items():
        params = speed_params(p1, d1)

        res_endo = M.monte_carlo_experiment(params, num_runs=num_runs)
        out[name] = dict(
            P_rev=float(M.reversal_probability(res_endo)),
            x_pre=float(np.mean(res_endo['x_pre'])),
            S_pre=float(np.mean(res_endo['S_pre'])))

        orig = M.update_skepticism

        def fixed_update(G, alpha, tau):
            for i in G.nodes():
                G.nodes[i]['S'] = 1.0
                G.nodes[i]['R'] = np.tanh(1.0)

        M.update_skepticism = fixed_update
        res_fixed = M.monte_carlo_experiment(params, num_runs=num_runs)
        M.update_skepticism = orig
        out[f'fixed_{name}'] = dict(
            P_rev=float(M.reversal_probability(res_fixed)),
            x_pre=float(np.mean(res_fixed['x_pre'])),
            S_pre=1.0)
        print(f"[{name}] endo P_rev={out[name]['P_rev']:.3f} "
              f"(S_pre={out[name]['S_pre']:.3f}, x_pre={out[name]['x_pre']:.3f}) | "
              f"fixed P_rev={out[f'fixed_{name}']['P_rev']:.3f} "
              f"(x_pre={out[f'fixed_{name}']['x_pre']:.3f})", flush=True)
    json.dump(out, open(os.path.join(OUT, 'E2_fixedS_rerun.json'), 'w'), indent=2)
    print('已保存 E2_fixedS_rerun.json')

if __name__ == '__main__':
    run()
