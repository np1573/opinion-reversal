
import os
import sys
import json
import random
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from exp_alt_susceptibility import make_update_skepticism
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
TARGET = 0.10
NUM_RUNS = 100
N_WORKERS = 8
TOL = 0.003
MAX_ITER = 8

INITIAL_ALPHA = {
    'S1_f': 0.22898976803473733,
    'S2_v': 32.74265469505255,
    'S3_vf_abs': 59.62859035373707,
    'S3_vf_pos': 70.73132634850987,
    'S3_vf_net': 85.73436783520366,
}

def inject_information_counting(G, info_value, reception_prob, update_actions_func):
    info_sign = np.sign(info_value)
    flips = 0
    for i in G.nodes():
        if random.random() < reception_prob:
            x_i = G.nodes[i]['x']
            if info_value * x_i < 0 and random.random() < G.nodes[i]['R']:
                G.nodes[i]['x'] = -x_i
                G.nodes[i]['reversal'] = 1
                flips += 1
                x_i = G.nodes[i]['x']
            update = abs(info_value) * (info_sign - x_i) / 2
            G.nodes[i]['x'] = np.clip(x_i + update, -1, 1)
    update_actions_func(G)
    return flips

def _flip_worker(args):
    run_id, params = args
    p = params.copy()
    np.random.seed(run_id)
    random.seed(run_id)
    G = M.initialize_network(p['network_formation'], p['num_nodes'],
                             p['tau'], p.get('opinion_init_mode', 'random'))
    M.set_mu(G, p['mu'])
    M.update_actions_sgn(G)
    for _ in range(p['pre_info_steps']):
        M.interaction_step(G, p['mu'], M.update_actions_sgn)
        M.update_skepticism(G, p['alpha'], p['tau'])
    info1_decay = p.get('info1_decay', True)
    flips = []
    for step in range(p['info1_duration']):
        if info1_decay:
            prob = p['info1_reception_prob'] * (p['info1_duration'] - step) / p['info1_duration']
        else:
            prob = p['info1_reception_prob']
        flips.append(inject_information_counting(G, p['info1_value'], prob, M.update_actions_sgn))
        M.interaction_step(G, p['mu'], M.update_actions_sgn)
        M.update_skepticism(G, p['alpha'], p['tau'])
    return flips

def count_info1_flips(params, num_runs=NUM_RUNS, n_workers=N_WORKERS):
    with Pool(n_workers) as pool:
        results = pool.map(_flip_worker, [(r, params) for r in range(num_runs)])
    return np.array(results)

def calibrate_iterative(s_form, v_form, vname):
    M.update_skepticism = make_update_skepticism(s_form, v_form)
    alpha = INITIAL_ALPHA.get(vname, 60.0)
    history = []
    for it in range(MAX_ITER):
        params = speed_params(0.10, 40)
        params['alpha'] = alpha
        res = M.monte_carlo_experiment(params, num_runs=NUM_RUNS, n_workers=N_WORKERS)
        chi_pre = float(np.mean(res['S_pre']))
        history.append(round(chi_pre, 4))
        print(f"  [{vname}] iter={it}: alpha={alpha:.4f} -> chi_pre={chi_pre:.4f}", flush=True)
        if abs(chi_pre - TARGET) < TOL:
            return alpha, chi_pre, history
        alpha = alpha * TARGET / chi_pre
    return alpha, chi_pre, history

def run():
    variants = [
        ('S1_f', 'f', 'abs'),
        ('S2_v', 'v', 'abs'),
        ('S3_vf_abs', 'vf', 'abs'),
        ('S3_vf_pos', 'vf', 'pos'),
        ('S3_vf_net', 'vf', 'net'),
    ]
    combos = {'slow': (0.10, 40), 'fast': (1.00, 4)}
    out = {}

    for vname, s_form, v_form in variants:
        alpha_cal, chi_slow, hist = calibrate_iterative(s_form, v_form, vname)
        print(f"[{vname}] 校准完成: alpha_cal={alpha_cal:.4f}, slow chi_pre={chi_slow:.4f}, 迭代轨迹={hist}", flush=True)

        M.update_skepticism = make_update_skepticism(s_form, v_form)
        entry = {'alpha_cal': alpha_cal, 'calib_history': hist}
        for name, (p1, d1) in combos.items():
            params = speed_params(p1, d1)
            params['alpha'] = alpha_cal
            res = M.monte_carlo_experiment(params, num_runs=NUM_RUNS, n_workers=N_WORKERS)
            entry[name] = dict(
                P_rev=float(M.reversal_probability(res)),
                chi_pre=float(np.mean(res['S_pre'])),
                x_pre=float(np.mean(res['x_pre'])))
            print(f"  [{vname} / {name}] chi_pre={entry[name]['chi_pre']:.4f} "
                  f"P_rev={entry[name]['P_rev']:.3f} x_pre={entry[name]['x_pre']:.3f}", flush=True)

        slow_params = speed_params(0.10, 40)
        slow_params['alpha'] = alpha_cal
        flips = count_info1_flips(slow_params)
        total_flips_per_run = flips.sum(axis=1)
        entry['info1_flip_stats'] = dict(
            mean_flips_per_run=float(np.mean(total_flips_per_run)),
            mean_flips_per_step=float(np.mean(flips)),
            first_step_mean=float(np.mean(flips[:, 0])),
            max_per_run=float(np.max(total_flips_per_run)),
        )
        print(f"  [{vname}] 信息1阶段翻转统计: 每run平均翻转 {entry['info1_flip_stats']['mean_flips_per_run']:.2f} 次, "
              f"每步平均 {entry['info1_flip_stats']['mean_flips_per_step']:.2f} 次, "
              f"第一步平均 {entry['info1_flip_stats']['first_step_mean']:.2f} 次", flush=True)
        out[vname] = entry

    json.dump(out, open(os.path.join(OUT, 'alt_susceptibility_iter.json'), 'w'), indent=2)
    print('已保存 alt_susceptibility_iter.json')

if __name__ == '__main__':
    run()
