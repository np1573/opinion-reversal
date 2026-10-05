# -*- coding: utf-8 -*-
"""
第四章 改进版仿真模型
=====================
在原始模型基础上做以下改进（对应 ChatGPT 建议）：
1. 新增系统级指标：平均易感性 S_bar(t)、系统级反转概率 P_rev；
2. 单次仿真返回"时间序列 + 关键标量(S_pre, x_pre, x_final, sigma_final)"，
   用于绘制 S_pre--I_2 反转概率相图与"同共识、不同速度"路径依赖散点图；
3. 支持"无衰减(固定接收概率)"传播模式，用于共识形成速度实验；
4. 修正共识度 f_i 的分母（邻居数而非一致邻居数）。

模型机制（与原文一致）：
   意见 x_i in [-1,1]，行动 sigma_i = sign(x_i)；
   局部共识度 f_i = sum(|x_j| : 邻居 j 与 i 同向) / 邻居数；
   共识变化速度 v_i = mean(|diff(history)|, 最近 tau 步)；
   怀疑度 S_i = alpha * ln(1 + v_i * f_i)；
   反转概率 R_i = tanh(S_i)；
   外部信息注入：若与自身意见异号且 rand < R_i，则意见反转 x -> -x，
   再按 update = |I|*(sign(I)-x)/2 内化。
"""

import os
import networkx as nx
import numpy as np
import pandas as pd
import random
import warnings
import multiprocessing
from multiprocessing import Pool, cpu_count

multiprocessing.set_start_method('fork', force=True)
warnings.filterwarnings("ignore")


# ===================== 1. 模型核心函数 =====================

def initialize_network(network_formation, num_nodes, tau, opinion_init_mode='random'):
    """初始化网络和主体属性"""
    if network_formation[0] == 'BA':
        G = nx.barabasi_albert_graph(num_nodes, network_formation[1])
    elif network_formation[0] == 'ER':
        G = nx.erdos_renyi_graph(num_nodes, network_formation[1])
    elif network_formation[0] == 'WS':
        G = nx.watts_strogatz_graph(num_nodes, network_formation[1][0], network_formation[1][1])

    if opinion_init_mode == 'random':
        opinions = {i: random.uniform(-1, 1) for i in G.nodes()}
    else:
        opinions = {i: 0.0 for i in G.nodes()}
    nx.set_node_attributes(G, opinions, 'x')

    nx.set_node_attributes(G, 0.0, 'S')
    nx.set_node_attributes(G, 0.0, 'R')
    nx.set_node_attributes(G, [0] * tau, 'consensus_history')
    nx.set_node_attributes(G, 0, 'info')
    nx.set_node_attributes(G, 0, 'reversal')
    nx.set_node_attributes(G, 0, 'sigma')
    return G


def update_actions_sgn(G):
    """基于意见符号更新行动"""
    for i in G.nodes():
        G.nodes[i]['sigma'] = 1 if G.nodes[i]['x'] >= 0 else -1


def set_mu(G, mu):
    """设置更新步长 mu（支持社会影响力异质性 mu）"""
    if mu == 'Social_Impact':
        deg = [len(list(G.neighbors(i))) for i in G.nodes()]
        k_min, k_max = min(deg), max(deg)
        if k_max == k_min:
            mu_dict = {i: 0.5 for i in G.nodes()}
        else:
            mu_dict = {i: 0.5 * (1 + (-deg[i] + k_min) / (k_max - k_min)) for i in G.nodes()}
    else:
        mu_dict = {i: mu for i in G.nodes()}
    nx.set_node_attributes(G, mu_dict, 'mu')


def update_skepticism(G, alpha, tau):
    """计算共识度、怀疑度、反转概率"""
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if not neighbors:
            G.nodes[i]['S'], G.nodes[i]['R'] = 0, 0.5
            continue

        consistent_sum = sum(abs(G.nodes[j]['x']) for j in neighbors
                             if G.nodes[j]['sigma'] == G.nodes[i]['sigma'])
        consistent_cnt = sum(1 for j in neighbors
                             if G.nodes[j]['sigma'] == G.nodes[i]['sigma'])
        # 分母为邻居数（修正）
        f_i = consistent_sum / len(neighbors) if consistent_cnt > 0 else 0

        G.nodes[i]['consensus_history'].append(f_i)
        if len(G.nodes[i]['consensus_history']) > tau + 1:
            G.nodes[i]['consensus_history'].pop(0)

        if len(G.nodes[i]['consensus_history']) > 1:
            speeds = np.abs(np.diff(G.nodes[i]['consensus_history']))
            v_i = np.mean(speeds[-tau:]) if len(speeds) > 0 else 0.0
        else:
            v_i = 0.0

        s_i = alpha * np.log(1 + v_i * f_i + 1e-8)
        G.nodes[i]['S'], G.nodes[i]['R'] = s_i, np.tanh(s_i)


def interaction_step(G, mu, update_actions_func):
    """社会影响力交互（意见向邻居加权平均移动）"""
    new_opinions = {}
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if not neighbors:
            new_opinions[i] = G.nodes[i]['x']
            continue
        weighted_sum = sum(G.nodes[j]['x'] * len(list(G.neighbors(j))) for j in neighbors)
        total_weight = sum(len(list(G.neighbors(j))) for j in neighbors)
        avg = weighted_sum / total_weight if total_weight > 0 else 0
        new_opinions[i] = np.clip(G.nodes[i]['x'] + (avg - G.nodes[i]['x']) * G.nodes[i]['mu'], -1, 1)

    for i, val in new_opinions.items():
        G.nodes[i]['x'] = val
    update_actions_func(G)


def inject_information(G, info_value, reception_prob, update_actions_func):
    """信息注入：怀疑触发反转 + 信息内化"""
    info_sign = np.sign(info_value)
    for i in G.nodes():
        if random.random() < reception_prob:
            x_i = G.nodes[i]['x']
            # 怀疑触发的离散反转
            if info_value * x_i < 0 and random.random() < G.nodes[i]['R']:
                G.nodes[i]['x'] = -x_i
                G.nodes[i]['reversal'] = 1
                x_i = G.nodes[i]['x']
            # 信息内化
            update = abs(info_value) * (info_sign - x_i) / 2
            G.nodes[i]['x'] = np.clip(x_i + update, -1, 1)
    update_actions_func(G)


# ===================== 2. 单次仿真 =====================

def run_single_simulation(params):
    """
    运行单次仿真，返回：
      - 时间序列: avg_x, avg_sigma, avg_S, avg_R
      - 关键标量: S_pre, R_pre, x_pre, sigma_pre, x_final, sigma_final
    """
    seed = params.get('seed', 0)
    np.random.seed(seed)
    random.seed(seed)

    G = initialize_network(params['network_formation'], params['num_nodes'],
                           params['tau'], params.get('opinion_init_mode', 'random'))
    set_mu(G, params['mu'])
    update_actions = update_actions_sgn
    update_actions(G)

    avg_x, avg_sigma, avg_S, avg_R = [], [], [], []

    def record():
        xs = [G.nodes[i]['x'] for i in G.nodes()]
        ss = [G.nodes[i]['sigma'] for i in G.nodes()]
        Ss = [G.nodes[i]['S'] for i in G.nodes()]
        Rs = [G.nodes[i]['R'] for i in G.nodes()]
        avg_x.append(np.mean(xs))
        avg_sigma.append((1 + np.mean(ss)) / 2)  # 正向支持率 P_+
        avg_S.append(np.mean(Ss))
        avg_R.append(np.mean(Rs))

    record()  # t = 0

    # 初始交互
    for _ in range(params['pre_info_steps']):
        interaction_step(G, params['mu'], update_actions)
        update_skepticism(G, params['alpha'], params['tau'])
        record()

    # 信息 1（负向）
    info1_decay = params.get('info1_decay', True)
    for step in range(params['info1_duration']):
        if info1_decay:
            prob = params['info1_reception_prob'] * (params['info1_duration'] - step) / params['info1_duration']
        else:
            prob = params['info1_reception_prob']
        inject_information(G, params['info1_value'], prob, update_actions)
        interaction_step(G, params['mu'], update_actions)
        update_skepticism(G, params['alpha'], params['tau'])
        record()

    # 重置反转标记
    for node in G.nodes():
        G.nodes[node]['reversal'] = 0

    # 信息 1 后交互
    for _ in range(params['post_info1_steps']):
        interaction_step(G, params['mu'], update_actions)
        update_skepticism(G, params['alpha'], params['tau'])
        record()

    # 信息 2（反向）投入前一刻的标量
    t_pre = len(avg_x) - 1  # 当前最后一条记录即投入前一刻
    S_pre = avg_S[t_pre]
    R_pre = avg_R[t_pre]
    x_pre = avg_x[t_pre]
    sigma_pre = avg_sigma[t_pre]

    # 信息 2（反向）
    info2_decay = params.get('info2_decay', True)
    for step in range(params['info2_duration']):
        if info2_decay:
            prob = params['info2_reception_prob'] * (params['info2_duration'] - step) / params['info2_duration']
        else:
            prob = params['info2_reception_prob']
        inject_information(G, params['info2_value'], prob, update_actions)
        interaction_step(G, params['mu'], update_actions)
        update_skepticism(G, params['alpha'], params['tau'])
        record()

    # 信息 2 后交互
    for _ in range(params['post_info2_steps']):
        interaction_step(G, params['mu'], update_actions)
        update_skepticism(G, params['alpha'], params['tau'])
        record()

    x_final = avg_x[-1]
    sigma_final = avg_sigma[-1]

    return {
        'avg_x': avg_x, 'avg_sigma': avg_sigma, 'avg_S': avg_S, 'avg_R': avg_R,
        'S_pre': S_pre, 'R_pre': R_pre, 'x_pre': x_pre, 'sigma_pre': sigma_pre,
        'x_final': x_final, 'sigma_final': sigma_final,
    }


# ===================== 3. Monte Carlo =====================

def _worker(args):
    run_id, params = args
    p = params.copy()
    p['seed'] = run_id
    return run_id, run_single_simulation(p)


def monte_carlo_experiment(params, num_runs=100, n_workers=None):
    """多进程 Monte Carlo 实验"""
    if n_workers is None:
        n_workers = min(cpu_count(), 10)
    args_list = [(run, params) for run in range(num_runs)]
    with Pool(processes=n_workers) as pool:
        sim_results = pool.map(_worker, args_list)
    sim_results.sort(key=lambda x: x[0])

    results = {'avg_x': [], 'avg_sigma': [], 'avg_S': [], 'avg_R': [],
               'S_pre': [], 'R_pre': [], 'x_pre': [], 'sigma_pre': [],
               'x_final': [], 'sigma_final': []}
    for run, sim in sim_results:
        for k in results:
            results[k].append(sim[k])
    return results


def compute_statistics(df_key):
    """计算均值与 95% 置信区间"""
    col_means = df_key.mean()
    col_std = df_key.std()
    col_ci = 1.96 * col_std / np.sqrt(len(df_key))
    return pd.DataFrame({'mean': col_means, 'std': col_std, 'ci': col_ci})


def reversal_probability(results, threshold=0.0):
    """系统级反转概率 P_rev = P(x_final > threshold)，基于多次 MC 的结果"""
    x_final = np.array(results['x_final'])
    return float(np.mean(x_final > threshold))
