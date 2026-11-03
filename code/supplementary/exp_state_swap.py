
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
NETWORK_SEED = 0
OPINION_SEED = 42

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

    x = np.array([G.nodes[i]['x'] for i in range(N_NODES)])
    S = np.array([G.nodes[i]['S'] for i in range(N_NODES)])
    snap = {
        'x': {i: G.nodes[i]['x'] for i in G.nodes()},
        'hist': {i: list(G.nodes[i]['consensus_history']) for i in G.nodes()},
        'sigma': {i: G.nodes[i]['sigma'] for i in G.nodes()},
    }
    return snap, x, S

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

_GBASE = None
_SNAP = None
_SVEC = None
_PARAMS = None

def _worker(seed):
    return run_info2(_GBASE, _SNAP, _SVEC, _PARAMS, seed)

def run():
    global _GBASE, _SNAP, _SVEC, _PARAMS
    G_base = build_network(NETWORK_SEED)
    G_slow = init_opinions(copy.deepcopy(G_base), OPINION_SEED)
    G_fast = init_opinions(copy.deepcopy(G_base), OPINION_SEED)

    params_slow = speed_params(0.10, 40)
    params_fast = speed_params(1.00, 4)

    snap_slow, x_slow, S_slow = form_consensus(G_slow, params_slow)
    snap_fast, x_fast, S_fast = form_consensus(G_fast, params_fast)

    print(f"x_slow mean={x_slow.mean():.4f}, var={x_slow.var():.5f}, "
          f"chi_slow={S_slow.mean():.4f}")
    print(f"x_fast mean={x_fast.mean():.4f}, var={x_fast.var():.5f}, "
          f"chi_fast={S_fast.mean():.4f}")

    combos = {
        '(x_slow, S_slow)': (snap_slow, S_slow, params_slow),
        '(x_slow, S_fast)': (snap_slow, S_fast, params_slow),
        '(x_fast, S_slow)': (snap_fast, S_slow, params_fast),
        '(x_fast, S_fast)': (snap_fast, S_fast, params_fast),
    }

    out = {}
    for name, (snap, S_vec, params) in combos.items():
        _GBASE, _SNAP, _SVEC, _PARAMS = G_base, snap, S_vec, params
        with Pool(processes=N_WORKERS) as pool:
            finals = pool.map(_worker, list(range(NUM_RUNS)))
        k = int(sum(1 for xf in finals if xf > 0))
        P_rev = k / NUM_RUNS
        lo, hi = clopper_pearson(k, NUM_RUNS)
        out[name] = dict(
            P_rev=P_rev, ci_lo=lo, ci_hi=hi,
            x_final=float(np.mean(finals)),
            k=k, n=NUM_RUNS)
        print(f"[{name}] P_rev={P_rev:.3f} [{lo:.3f},{hi:.3f}] "
              f"x_final={out[name]['x_final']:.4f} ({k}/{NUM_RUNS})", flush=True)

    json.dump(dict(
        x_slow_mean=float(x_slow.mean()), x_slow_var=float(x_slow.var()),
        chi_slow=float(S_slow.mean()),
        x_fast_mean=float(x_fast.mean()), x_fast_var=float(x_fast.var()),
        chi_fast=float(S_fast.mean()),
        combos=out), open(os.path.join(OUT, 'state_swap.json'), 'w'), indent=2)
    print('已保存 state_swap.json')

if __name__ == '__main__':
    run()
