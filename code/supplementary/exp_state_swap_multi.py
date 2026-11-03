
import os
import sys
import json
import copy
import random
import multiprocessing
from multiprocessing import Pool
import numpy as np
import networkx as nx

multiprocessing.set_start_method('fork', force=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
N_NODES = 500
TAU = 10
MU = 0.1
NUM_RUNS = 100
N_WORKERS = 8
NS = 10

def build_network(seed):
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(N_NODES, 4)
    nx.set_node_attributes(G, {i: MU for i in G.nodes()}, 'mu')
    return G

def init_opinions(G, seed):
    random.seed(seed)
    opinions = {i: random.uniform(-1, 1) for i in G.nodes()}
    nx.set_node_attributes(G, opinions, 'x')
    nx.set_node_attributes(G, 0.0, 'S')
    nx.set_node_attributes(G, 0.0, 'R')
    nx.set_node_attributes(G, {i: [0] * TAU for i in G.nodes()}, 'consensus_history')
    nx.set_node_attributes(G, 0, 'info')
    nx.set_node_attributes(G, 0, 'reversal')
    nx.set_node_attributes(G, 0, 'sigma')
    nx.set_node_attributes(G, {i: MU for i in G.nodes()}, 'mu')
    return G

def form_consensus(G_base, params):
    G = copy.deepcopy(G_base)
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

    snap = {
        'x': {i: G.nodes[i]['x'] for i in G.nodes()},
        'hist': {i: list(G.nodes[i]['consensus_history']) for i in G.nodes()},
        'sigma': {i: G.nodes[i]['sigma'] for i in G.nodes()},
    }
    S = np.array([G.nodes[i]['S'] for i in range(N_NODES)])
    return snap, S

def run_info2(G_base, snap, S_vec, params, seed):
    G = copy.deepcopy(G_base)
    np.random.seed(seed)
    random.seed(seed)
    for i in G.nodes():
        G.nodes[i]['x'] = snap['x'][i]
        G.nodes[i]['consensus_history'] = list(snap['hist'][i])
        G.nodes[i]['sigma'] = snap['sigma'][i]
        G.nodes[i]['reversal'] = 0
        G.nodes[i]['S'] = float(S_vec[i])
        G.nodes[i]['R'] = float(np.tanh(S_vec[i]))

    update_actions = M.update_actions_sgn

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

    return float(np.mean([G.nodes[i]['x'] for i in G.nodes()]))

def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

_G = None
_SNAP = None
_SVEC = None
_PARAMS = None

def _worker(seed):
    return run_info2(_G, _SNAP, _SVEC, _PARAMS, seed)

def process_one_state(args):
    global _G, _SNAP, _SVEC, _PARAMS
    net_seed, op_seed = args

    G_base = build_network(net_seed)
    G_slow = init_opinions(copy.deepcopy(G_base), op_seed)
    G_fast = init_opinions(copy.deepcopy(G_base), op_seed)

    params_slow = speed_params(0.10, 40)
    params_fast = speed_params(1.00, 4)

    snap_slow, S_slow = form_consensus(G_slow, params_slow)
    snap_fast, S_fast = form_consensus(G_fast, params_fast)

    x_slow = np.array([snap_slow['x'][i] for i in range(N_NODES)])
    x_fast = np.array([snap_fast['x'][i] for i in range(N_NODES)])

    combos = {
        '(x_slow, S_slow)': (snap_slow, S_slow, params_slow),
        '(x_slow, S_fast)': (snap_slow, S_fast, params_slow),
        '(x_fast, S_slow)': (snap_fast, S_slow, params_fast),
        '(x_fast, S_fast)': (snap_fast, S_fast, params_fast),
    }

    out = {}
    for name, (snap, S_vec, params) in combos.items():
        _G, _SNAP, _SVEC, _PARAMS = G_base, snap, S_vec, params
        with Pool(processes=N_WORKERS) as pool:
            finals = pool.map(_worker, list(range(NUM_RUNS)))
        k = int(sum(1 for xf in finals if xf > 0))
        P_rev = k / NUM_RUNS
        lo, hi = clopper_pearson(k, NUM_RUNS)
        out[name] = dict(P_rev=P_rev, ci_lo=lo, ci_hi=hi, x_final=float(np.mean(finals)))

    return dict(
        net_seed=net_seed, op_seed=op_seed,
        x_slow_mean=float(x_slow.mean()), x_slow_var=float(x_slow.var()),
        x_fast_mean=float(x_fast.mean()), x_fast_var=float(x_fast.var()),
        chi_slow=float(S_slow.mean()), chi_fast=float(S_fast.mean()),
        combos=out)

def main():
    seeds = [(s, 1000 + s) for s in range(NS)]
    results = []
    for args in seeds:
        r = process_one_state(args)
        results.append(r)
        print(f"[state net={args[0]},op={args[1]}] "
              f"chi_slow={r['chi_slow']:.3f} chi_fast={r['chi_fast']:.3f} "
              f"x_slow={r['x_slow_mean']:.3f} x_fast={r['x_fast_mean']:.3f} "
              f"P(x_slow,S_fast)={r['combos']['(x_slow, S_fast)']['P_rev']:.3f} "
              f"P(x_fast,S_slow)={r['combos']['(x_fast, S_slow)']['P_rev']:.3f}", flush=True)

    summary = {}
    for combo in ['(x_slow, S_slow)', '(x_slow, S_fast)', '(x_fast, S_slow)', '(x_fast, S_fast)']:
        ps = np.array([r['combos'][combo]['P_rev'] for r in results])
        summary[combo] = dict(
            mean=float(ps.mean()), min=float(ps.min()), max=float(ps.max()),
            std=float(ps.std(ddof=1)), values=[round(float(p), 3) for p in ps])

    chi_slow_all = [r['chi_slow'] for r in results]
    chi_fast_all = [r['chi_fast'] for r in results]
    summary['chi_slow'] = dict(mean=float(np.mean(chi_slow_all)),
                               min=float(min(chi_slow_all)), max=float(max(chi_slow_all)))
    summary['chi_fast'] = dict(mean=float(np.mean(chi_fast_all)),
                               min=float(min(chi_fast_all)), max=float(max(chi_fast_all)))

    json.dump(dict(NS=NS, results=results, summary=summary),
              open(os.path.join(OUT, 'state_swap_multi.json'), 'w'), indent=2)

    print('\n=== 跨状态汇总 ===')
    for combo, s in summary.items():
        if isinstance(s, dict) and 'mean' in s:
            if 'std' in s:
                print(f"{combo}: mean={s['mean']:.3f} min={s['min']:.3f} max={s['max']:.3f} std={s['std']:.3f}")
            else:
                print(f"{combo}: mean={s['mean']:.3f} min={s['min']:.3f} max={s['max']:.3f}")
    print('已保存 state_swap_multi.json')

if __name__ == '__main__':
    main()
