
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

        res_full = M.monte_carlo_experiment(params, num_runs=num_runs)
        out[f'{name}_full'] = dict(
            P_rev=float(M.reversal_probability(res_full)),
            x_final=float(np.mean(res_full['x_final'])))

        res_internalize = M.monte_carlo_experiment(dict(params, alpha=0.0), num_runs=num_runs)
        out[f'{name}_internalize'] = dict(
            P_rev=float(M.reversal_probability(res_internalize)),
            x_final=float(np.mean(res_internalize['x_final'])))

        orig_inject = M.inject_information

        def inject_no_internalize(G, info_value, reception_prob, update_actions_func):
            info_sign = np.sign(info_value)
            for i in G.nodes():
                if random.random() < reception_prob:
                    x_i = G.nodes[i]['x']
                    if info_value * x_i < 0 and random.random() < G.nodes[i]['R']:
                        G.nodes[i]['x'] = -x_i
                        G.nodes[i]['reversal'] = 1
            update_actions_func(G)

        import random
        M.inject_information = inject_no_internalize
        res_fliponly = M.monte_carlo_experiment(params, num_runs=num_runs)
        M.inject_information = orig_inject
        out[f'{name}_fliponly'] = dict(
            P_rev=float(M.reversal_probability(res_fliponly)),
            x_final=float(np.mean(res_fliponly['x_final'])))

        print(f"[{name}] full={out[f'{name}_full']['P_rev']:.3f} | "
              f"internalize-only={out[f'{name}_internalize']['P_rev']:.3f} | "
              f"flip-only={out[f'{name}_fliponly']['P_rev']:.3f}", flush=True)

    json.dump(out, open(os.path.join(OUT, 'mechanism_decomp.json'), 'w'), indent=2)
    print('已保存 mechanism_decomp.json')

if __name__ == '__main__':
    run()
