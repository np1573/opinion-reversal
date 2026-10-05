import time, networkx as nx, numpy as np, random, warnings
warnings.filterwarnings("ignore")

def initialize_network(network_formation, num_nodes, tau):
    if network_formation[0] == 'BA':
        G = nx.barabasi_albert_graph(num_nodes, network_formation[1])
    elif network_formation[0] == 'ER':
        G = nx.erdos_renyi_graph(num_nodes, network_formation[1])
    elif network_formation[0] == 'WS':
        G = nx.watts_strogatz_graph(num_nodes, network_formation[1][0], network_formation[1][1])
    opinions = {i: random.uniform(-1, 1) for i in G.nodes()}
    nx.set_node_attributes(G, opinions, 'x')
    nx.set_node_attributes(G, 0.0, 'S'); nx.set_node_attributes(G, 0.0, 'R')
    nx.set_node_attributes(G, [0]*tau, 'consensus_history')
    nx.set_node_attributes(G, 0, 'info'); nx.set_node_attributes(G, 0, 'reversal'); nx.set_node_attributes(G, 0, 'sigma')
    return G

def update_actions_sgn(G):
    for i in G.nodes():
        G.nodes[i]['sigma'] = 1 if G.nodes[i]['x'] >= 0 else -1

def set_mu(G, mu):
    if mu == 'Social_Impact':
        deg = [len(list(G.neighbors(i))) for i in G.nodes()]
        k_min, k_max = min(deg), max(deg)
        mu_dict = {i: (0.5 if k_max==k_min else 0.5*(1+(-deg[i]+k_min)/(k_max-k_min))) for i in G.nodes()}
    else:
        mu_dict = {i: mu for i in G.nodes()}
    nx.set_node_attributes(G, mu_dict, 'mu')

def update_skepticism(G, alpha, tau):
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if not neighbors:
            G.nodes[i]['S'], G.nodes[i]['R'] = 0, 0.5; continue
        consistent_sum = sum(abs(G.nodes[j]['x']) for j in neighbors if G.nodes[j]['sigma']==G.nodes[i]['sigma'])
        consistent_cnt = sum(1 for j in neighbors if G.nodes[j]['sigma']==G.nodes[i]['sigma'])
        f_i = consistent_sum / len(neighbors) if consistent_cnt > 0 else 0
        G.nodes[i]['consensus_history'].append(f_i)
        if len(G.nodes[i]['consensus_history']) > tau+1:
            G.nodes[i]['consensus_history'].pop(0)
        speeds = np.abs(np.diff(G.nodes[i]['consensus_history'])) if len(G.nodes[i]['consensus_history'])>1 else np.array([])
        v_i = np.mean(speeds[-tau:]) if len(speeds)>0 else 0.0
        s_i = alpha*np.log(1 + v_i*f_i + 1e-8)
        G.nodes[i]['S'], G.nodes[i]['R'] = s_i, np.tanh(s_i)

def interaction_step(G, mu, update_actions_func):
    new_opinions = {}
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if not neighbors:
            new_opinions[i] = G.nodes[i]['x']; continue
        weighted_sum = sum(G.nodes[j]['x']*len(list(G.neighbors(j))) for j in neighbors)
        total_weight = sum(len(list(G.neighbors(j))) for j in neighbors)
        avg = weighted_sum/total_weight if total_weight>0 else 0
        new_opinions[i] = np.clip(G.nodes[i]['x'] + (avg-G.nodes[i]['x'])*G.nodes[i]['mu'], -1, 1)
    for i,val in new_opinions.items(): G.nodes[i]['x']=val
    update_actions_func(G)

def inject_information(G, info_value, reception_prob, update_actions_func):
    info_sign = np.sign(info_value)
    for i in G.nodes():
        if random.random() < reception_prob:
            x_i = G.nodes[i]['x']
            if info_value*x_i < 0 and random.random() < G.nodes[i]['R']:
                G.nodes[i]['x'] = -x_i; G.nodes[i]['reversal']=1; x_i = G.nodes[i]['x']
            update = abs(info_value)*(info_sign - x_i)/2
            G.nodes[i]['x'] = np.clip(x_i+update, -1, 1)
    update_actions_func(G)

def run_single(params):
    np.random.seed(params['seed'])
    G = initialize_network(params['network_formation'], params['num_nodes'], params['tau'])
    set_mu(G, params['mu'])
    update_actions = update_actions_sgn
    update_actions(G)
    for _ in range(params['pre_info_steps']):
        interaction_step(G, params['mu'], update_actions); update_skepticism(G, params['alpha'], params['tau'])
    for step in range(params['info1_duration']):
        prob = params['info1_reception_prob']*(params['info1_duration']-step)/params['info1_duration']
        inject_information(G, params['info1_value'], prob, update_actions)
        interaction_step(G, params['mu'], update_actions); update_skepticism(G, params['alpha'], params['tau'])
    for node in G.nodes(): G.nodes[node]['reversal']=0
    for _ in range(params['post_info1_steps']):
        interaction_step(G, params['mu'], update_actions); update_skepticism(G, params['alpha'], params['tau'])
    for step in range(params['info2_duration']):
        prob = params['info2_reception_prob']*(params['info2_duration']-step)/params['info2_duration']
        inject_information(G, params['info2_value'], prob, update_actions)
        interaction_step(G, params['mu'], update_actions); update_skepticism(G, params['alpha'], params['tau'])
    for _ in range(params['post_info2_steps']):
        interaction_step(G, params['mu'], update_actions); update_skepticism(G, params['alpha'], params['tau'])
    return np.mean([G.nodes[i]['x'] for i in G.nodes()])

BASE = dict(network_formation=['BA',4], num_nodes=500, tau=10, mu=0.1, alpha=60,
    pre_info_steps=0, info1_value=-0.9, info1_reception_prob=0.4, info1_duration=5,
    post_info1_steps=15, info2_value=0.7, info2_reception_prob=0.6, info2_duration=1, total_steps=200)
BASE['post_info2_steps'] = BASE['total_steps']-BASE['pre_info_steps']-BASE['info1_duration']-BASE['post_info1_steps']-BASE['info2_duration']

p = BASE.copy(); p['seed']=0
t0=time.time(); run_single(p); t1=time.time()
print(f"单次仿真耗时: {t1-t0:.3f} 秒")
# 再测3次取平均
t0=time.time()
for s in range(3): 
    pp=p.copy(); pp['seed']=s; run_single(pp)
t1=time.time()
print(f"3次平均: {(t1-t0)/3:.3f} 秒/次")
