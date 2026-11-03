
import os
import sys
import json
import numpy as np
import multiprocessing

multiprocessing.set_start_method('fork', force=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
NUM_RUNS = 100
N_WORKERS = 8

def interaction_step_unweighted(G, mu, update_actions_func):
    new_opinions = {}
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if not neighbors:
            new_opinions[i] = G.nodes[i]['x']
            continue
        avg = float(np.mean([G.nodes[j]['x'] for j in neighbors]))
        new_opinions[i] = np.clip(G.nodes[i]['x'] + (avg - G.nodes[i]['x']) * G.nodes[i]['mu'], -1, 1)
    for i, val in new_opinions.items():
        G.nodes[i]['x'] = val
    update_actions_func(G)

def main():

    M.interaction_step = interaction_step_unweighted

    groups = [('Slow', 0.10, 40), ('Medium', 0.40, 10), ('Faster', 0.667, 6), ('Fast', 1.00, 4)]
    out = {}
    print('unweighted 邻居均值对照（100 MC each）:', flush=True)
    for name, p1, d1 in groups:

        if name == 'Faster':
            p1 = 2.0 / 3.0
        params = speed_params(p1, d1)
        res = M.monte_carlo_experiment(params, num_runs=NUM_RUNS, n_workers=N_WORKERS)
        chi_pre = float(np.mean(res['S_pre']))
        x_pre = float(np.mean(res['x_pre']))
        P_rev = float(M.reversal_probability(res))
        out[name] = dict(p1=p1, L1=d1, chi_pre=chi_pre, x_pre=x_pre, P_rev=P_rev)
        print(f"  {name}: p1={p1:.3f} L1={d1} -> chi_pre={chi_pre:.4f} "
              f"x_pre={x_pre:.4f} P_rev={P_rev:.3f}", flush=True)

    out['degree_weighted_reference'] = {
        'Slow': dict(chi_pre=0.101, P_rev=0.020),
        'Medium': dict(chi_pre=0.402, P_rev=0.160),
        'Faster': dict(chi_pre=0.745, P_rev=0.620),
        'Fast': dict(chi_pre=1.104, P_rev=0.910),
    }

    json.dump(out, open(os.path.join(OUT, 'unweighted_control.json'), 'w'), indent=2)
    print('\n已保存 unweighted_control.json')

if __name__ == '__main__':
    main()
