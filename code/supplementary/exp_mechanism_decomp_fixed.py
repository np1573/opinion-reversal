
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

def form_consensus_full(G_base, params):
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
        'S': {i: G.nodes[i]['S'] for i in G.nodes()},
        'R': {i: G.nodes[i]['R'] for i in G.nodes()},
        'hist': {i: list(G.nodes[i]['consensus_history']) for i in G.nodes()},
        'sigma': {i: G.nodes[i]['sigma'] for i in G.nodes()},
    }
    chi_pre = float(np.mean([G.nodes[i]['S'] for i in G.nodes()]))
    x_pre = float(np.mean([G.nodes[i]['x'] for i in G.nodes()]))
    return snap, chi_pre, x_pre

def inject_internalize_only(G, info_value, reception_prob, update_actions_func):
    info_sign = np.sign(info_value)
    for i in G.nodes():
        if random.random() < reception_prob:
            x_i = G.nodes[i]['x']
            update = abs(info_value) * (info_sign - x_i) / 2
            G.nodes[i]['x'] = np.clip(x_i + update, -1, 1)
    update_actions_func(G)

def inject_flip_only(G, info_value, reception_prob, update_actions_func):
    info_sign = np.sign(info_value)
    for i in G.nodes():
        if random.random() < reception_prob:
            x_i = G.nodes[i]['x']
            if info_value * x_i < 0 and random.random() < G.nodes[i]['R']:
                G.nodes[i]['x'] = -x_i
                G.nodes[i]['reversal'] = 1
    update_actions_func(G)

def run_info2_variant(G_base, snap, params, inject_fn, seed):
    G = copy.deepcopy(G_base)
    np.random.seed(seed)
    random.seed(seed)
    for i in G.nodes():
        G.nodes[i]['x'] = snap['x'][i]
        G.nodes[i]['S'] = snap['S'][i]
        G.nodes[i]['R'] = snap['R'][i]
        G.nodes[i]['consensus_history'] = list(snap['hist'][i])
        G.nodes[i]['sigma'] = snap['sigma'][i]
        G.nodes[i]['reversal'] = 0

    update_actions = M.update_actions_sgn

    info2_decay = params.get('info2_decay', True)
    for step in range(params['info2_duration']):
        if info2_decay:
            prob = params['info2_reception_prob'] * (params['info2_duration'] - step) / params['info2_duration']
        else:
            prob = params['info2_reception_prob']
        inject_fn(G, params['info2_value'], prob, update_actions)
        M.interaction_step(G, params['mu'], update_actions)
        M.update_skepticism(G, params['alpha'], params['tau'])

    for _ in range(params['post_info2_steps']):
        M.interaction_step(G, params['mu'], update_actions)
        M.update_skepticism(G, params['alpha'], params['tau'])

    return float(np.mean([G.nodes[i]['x'] for i in G.nodes()]))

_GBASE = None
_SNAP = None
_PARAMS = None
_INJECT = None

def _worker(seed):
    return run_info2_variant(_GBASE, _SNAP, _PARAMS, _INJECT, seed)

def run():
    global _GBASE, _SNAP, _PARAMS, _INJECT
    G_base = build_network(NETWORK_SEED)

    combos = {'slow': speed_params(0.10, 40), 'fast': speed_params(1.00, 4)}
    variants = {
        'full': M.inject_information,
        'internalize_only': inject_internalize_only,
        'flip_only': inject_flip_only,
    }

    out = {}
    for name, params in combos.items():
        G_init = init_opinions(copy.deepcopy(G_base), OPINION_SEED)
        snap, chi_pre, x_pre = form_consensus_full(G_init, params)
        out[name] = dict(chi_pre=chi_pre, x_pre=x_pre, variants={})
        print(f"[{name}] chi_pre={chi_pre:.4f}, x_pre={x_pre:.4f}", flush=True)
        for vname, inject_fn in variants.items():
            _GBASE, _SNAP, _PARAMS, _INJECT = G_base, snap, params, inject_fn
            with Pool(processes=N_WORKERS) as pool:
                finals = pool.map(_worker, list(range(NUM_RUNS)))
            k = int(sum(1 for xf in finals if xf > 0))
            P_rev = k / NUM_RUNS
            out[name]['variants'][vname] = dict(
                P_rev=P_rev, x_final=float(np.mean(finals)), k=k, n=NUM_RUNS)
            print(f"  [{name} / {vname}] P_rev={P_rev:.3f} "
                  f"x_final={out[name]['variants'][vname]['x_final']:.4f} "
                  f"({k}/{NUM_RUNS})", flush=True)

    json.dump(out, open(os.path.join(OUT, 'mechanism_decomp_fixed.json'), 'w'), indent=2)
    print('已保存 mechanism_decomp_fixed.json')

if __name__ == '__main__':
    run()
