
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import networkx as nx
import random
from model import initialize_network, set_mu, update_actions_sgn, interaction_step, inject_information

def make_update_skepticism(beta):
    def update_skepticism(G, alpha, tau):
        for i in G.nodes():
            neighbors = list(G.neighbors(i))
            if not neighbors:
                G.nodes[i]['S'], G.nodes[i]['R'] = 0, 0.5
                continue
            consistent_sum = sum(abs(G.nodes[j]['x']) for j in neighbors
                                 if G.nodes[j]['sigma'] == G.nodes[i]['sigma'])
            consistent_cnt = sum(1 for j in neighbors
                                 if G.nodes[j]['sigma'] == G.nodes[i]['sigma'])
            f_i = consistent_sum / len(neighbors) if consistent_cnt > 0 else 0
            G.nodes[i]['consensus_history'].append(f_i)
            if len(G.nodes[i]['consensus_history']) > tau + 1:
                G.nodes[i]['consensus_history'].pop(0)
            if len(G.nodes[i]['consensus_history']) > 1:
                speeds = np.abs(np.diff(G.nodes[i]['consensus_history']))
                v_i = np.mean(speeds[-tau:]) if len(speeds) > 0 else 0.0
            else:
                v_i = 0.0
            raw = alpha * np.log(1 + v_i * f_i + 1e-8)
            prev = G.nodes[i]['S']
            G.nodes[i]['S'] = beta * prev + (1 - beta) * raw
            G.nodes[i]['R'] = np.tanh(G.nodes[i]['S'])
    return update_skepticism

def run(params, update_skep):
    np.random.seed(params['seed'])
    random.seed(params['seed'])
    G = initialize_network(params['network_formation'], params['num_nodes'], params['tau'],
                            params.get('opinion_init_mode', 'random'))
    set_mu(G, params['mu'])
    update_actions = update_actions_sgn
    update_actions(G)
    avg_x, avg_S = [], []
    def record():
        avg_x.append(np.mean([G.nodes[i]['x'] for i in G.nodes()]))
        avg_S.append(np.mean([G.nodes[i]['S'] for i in G.nodes()]))
    record()
    for _ in range(params['pre_info_steps']):
        interaction_step(G, params['mu'], update_actions); update_skep(G, params['alpha'], params['tau']); record()
    for step in range(params['info1_duration']):
        if params.get('info1_decay', True):
            prob = params['info1_reception_prob'] * (params['info1_duration'] - step) / params['info1_duration']
        else:
            prob = params['info1_reception_prob']
        inject_information(G, params['info1_value'], prob, update_actions)
        interaction_step(G, params['mu'], update_actions); update_skep(G, params['alpha'], params['tau']); record()
    for node in G.nodes(): G.nodes[node]['reversal'] = 0
    for _ in range(params['post_info1_steps']):
        interaction_step(G, params['mu'], update_actions); update_skep(G, params['alpha'], params['tau']); record()
    S_pre = avg_S[-1]; x_pre = avg_x[-1]
    for step in range(params['info2_duration']):
        if params.get('info2_decay', True):
            prob = params['info2_reception_prob'] * (params['info2_duration'] - step) / params['info2_duration']
        else:
            prob = params['info2_reception_prob']
        inject_information(G, params['info2_value'], prob, update_actions)
        interaction_step(G, params['mu'], update_actions); update_skep(G, params['alpha'], params['tau']); record()
    for _ in range(params['post_info2_steps']):
        interaction_step(G, params['mu'], update_actions); update_skep(G, params['alpha'], params['tau']); record()
    return {'avg_x': avg_x, 'avg_S': avg_S, 'S_pre': S_pre, 'x_pre': x_pre, 'x_final': avg_x[-1]}

def base(t_inj, tau, beta, seed):
    return dict(network_formation=['BA', 4], num_nodes=500, tau=tau, mu=0.1, alpha=60,
                pre_info_steps=0, info1_value=-0.9, info1_reception_prob=0.4, info1_duration=5,
                info1_decay=True, post_info1_steps=t_inj,
                info2_value=0.7, info2_reception_prob=0.6, info2_duration=1, info2_decay=True,
                post_info2_steps=40, seed=seed)

NMC = 20
t_inj_list = [0, 2, 4, 6, 8, 10, 12]
configs = [
    ('tau=10 beta=0 (基准)', 10, 0.0),
    ('tau=30 beta=0',        30, 0.0),
    ('tau=10 beta=0.5',      10, 0.5),
    ('tau=10 beta=0.8',      10, 0.8),
]

for name, tau, beta in configs:
    up = make_update_skepticism(beta)
    print(f"\n===== {name} =====")
    print(f"{'t_inj':>5} {'chi_pre':>9} {'x_pre':>8} {'x_final':>8} {'P_rev':>7}")
    for t_inj in t_inj_list:
        chi_l, xpre_l, xfin_l = [], [], []
        for s in range(NMC):
            r = run(base(t_inj, tau, beta, s), up)
            chi_l.append(r['S_pre']); xpre_l.append(r['x_pre']); xfin_l.append(r['x_final'])
        chi = np.mean(chi_l); xpre = np.mean(xpre_l)
        xfin = np.mean(xfin_l); prev = np.mean(np.array(xfin_l) > 0)
        print(f"{t_inj:>5} {chi:>9.3f} {xpre:>8.3f} {xfin:>8.3f} {prev:>7.2f}")
