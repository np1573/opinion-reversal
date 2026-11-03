
import os
import sys
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def run_single_with_details(params):
    seed = params.get('seed', 0)
    np.random.seed(seed)
    random.seed(seed)

    G = M.initialize_network(params['network_formation'], params['num_nodes'],
                             params['tau'], params.get('opinion_init_mode', 'random'))
    M.set_mu(G, params['mu'])
    update_actions = M.update_actions_sgn
    update_actions(G)

    for _ in range(params['pre_info_steps']):
        M.interaction_step(G, params['mu'], update_actions)
        M.update_skepticism(G, params['alpha'], params['tau'])

    info1_decay = params.get('info1_decay', True)
    for step in range(params['info1_duration']):
        if info1_decay:
            prob = params['info1_reception_prob'] * (params['info1_duration'] - step) / params['info1_duration']
        else:
            prob = params['info1_reception_prob']
        M.inject_information(G, params['info1_value'], prob, update_actions)
        M.interaction_step(G, params['mu'], update_actions)
        M.update_skepticism(G, params['alpha'], params['tau'])

    for node in G.nodes():
        G.nodes[node]['reversal'] = 0

    for _ in range(params['post_info1_steps']):
        M.interaction_step(G, params['mu'], update_actions)
        M.update_skepticism(G, params['alpha'], params['tau'])

    xs = np.array([G.nodes[i]['x'] for i in G.nodes()])
    Ss = np.array([G.nodes[i]['S'] for i in G.nodes()])
    degs = np.array([len(list(G.neighbors(i))) for i in G.nodes()])

    x_pre = float(xs.mean())
    x_var = float(xs.var())
    p_plus = float((xs > 0).mean())
    deg_weighted_x = float((degs * xs).sum() / degs.sum())
    S_mean = float(Ss.mean())
    S_var = float(Ss.var())
    S_max = float(Ss.max())

    info2_decay = params.get('info2_decay', True)
    for step in range(params['info2_duration']):
        if info2_decay:
            prob = params['info2_reception_prob'] * (params['info2_duration'] - step) / params['info2_duration']
        else:
            prob = params['info2_reception_prob']
        M.inject_information(G, params['info2_value'], prob, update_actions)
        M.interaction_step(G, params['mu'], update_actions)
        M.update_skepticism(G, params['alpha'], params['tau'])

    for _ in range(params['post_info2_steps']):
        M.interaction_step(G, params['mu'], update_actions)
        M.update_skepticism(G, params['alpha'], params['tau'])

    x_final = float(np.mean([G.nodes[i]['x'] for i in G.nodes()]))

    return dict(x_pre=x_pre, x_var=x_var, p_plus=p_plus, deg_weighted_x=deg_weighted_x,
                S_mean=S_mean, S_var=S_var, S_max=S_max, x_final=x_final)

def run(num_runs=200):
    combos = {'slow': (0.10, 40), 'fast': (1.00, 4)}
    data = {}
    for name, (p1, d1) in combos.items():
        params = speed_params(p1, d1)
        recs = []
        for r in range(num_runs):
            params['seed'] = r
            recs.append(run_single_with_details(params))
        data[name] = recs

    stats = {}
    for name in combos:
        R = data[name]
        keys = ['x_pre', 'x_var', 'p_plus', 'deg_weighted_x', 'S_mean', 'S_var', 'S_max', 'x_final']
        stats[name] = {k: float(np.mean([r[k] for r in R])) for k in keys}
        print(f"[{name}] " + " | ".join(
            f"{k}={stats[name][k]:.4f}" for k in keys), flush=True)

    lo, hi = -0.505, -0.495
    paired = {}
    for name in combos:
        R = data[name]
        sub = [r for r in R if lo <= r['x_pre'] <= hi]
        P_rev = float(np.mean([r['x_final'] > 0 for r in sub]))
        paired[name] = dict(n=len(sub), x_pre=float(np.mean([r['x_pre'] for r in sub])),
                            S_mean=float(np.mean([r['S_mean'] for r in sub])),
                            P_rev=P_rev)
        print(f"[配对 {name}] n={len(sub)}, x_pre={paired[name]['x_pre']:.4f}, "
              f"S_mean={paired[name]['S_mean']:.4f}, P_rev={P_rev:.3f}", flush=True)

    json.dump(dict(stats=stats, paired=paired),
              open(os.path.join(OUT, 'state_matching.json'), 'w'), indent=2)
    print('已保存 state_matching.json')

if __name__ == '__main__':
    run()
