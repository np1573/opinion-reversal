
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
TARGET = 0.10
NUM_RUNS = 100
N_WORKERS = 8

def make_update_skepticism(s_form='vf', v_form='abs'):
    def update(G, alpha, tau):
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
            hist = G.nodes[i]['consensus_history']
            if len(hist) > 1:
                diff = np.diff(hist)
                if v_form == 'abs':
                    v_i = float(np.mean(np.abs(diff[-tau:]))) if len(diff) > 0 else 0.0
                elif v_form == 'pos':
                    v_i = float(np.mean(np.maximum(diff[-tau:], 0))) if len(diff) > 0 else 0.0
                elif v_form == 'net':
                    v_i = float(max(0.0, f_i - hist[0]) / tau)
                else:
                    v_i = 0.0
            else:
                v_i = 0.0

            if s_form == 'vf':
                stim = v_i * f_i
            elif s_form == 'f':
                stim = f_i
            elif s_form == 'v':
                stim = v_i
            else:
                stim = v_i * f_i
            s_i = alpha * np.log(1 + stim + 1e-8)
            G.nodes[i]['S'], G.nodes[i]['R'] = s_i, np.tanh(s_i)
    return update

def calibrate(s_form, v_form):
    M.update_skepticism = make_update_skepticism(s_form, v_form)
    params = speed_params(0.10, 40)
    res = M.monte_carlo_experiment(params, num_runs=NUM_RUNS, n_workers=N_WORKERS)
    scale = float(np.mean(res['S_pre'])) / params['alpha']
    alpha_cal = TARGET / scale
    return alpha_cal, scale

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
    orig = M.update_skepticism

    for vname, s_form, v_form in variants:
        alpha_cal, scale = calibrate(s_form, v_form)
        print(f"[{vname}] scale={scale:.6f} alpha_cal={alpha_cal:.3f}", flush=True)
        M.update_skepticism = make_update_skepticism(s_form, v_form)
        entry = {'alpha_cal': alpha_cal, 'scale': scale}
        for name, (p1, d1) in combos.items():
            params = speed_params(p1, d1)
            params['alpha'] = alpha_cal
            res = M.monte_carlo_experiment(params, num_runs=NUM_RUNS, n_workers=N_WORKERS)
            entry[name] = dict(
                P_rev=float(M.reversal_probability(res)),
                chi_pre=float(np.mean(res['S_pre'])),
                x_pre=float(np.mean(res['x_pre'])))
            print(f"  [{vname} / {name}] chi_pre={entry[name]['chi_pre']:.3f} "
                  f"P_rev={entry[name]['P_rev']:.3f}", flush=True)
        out[vname] = entry

    M.update_skepticism = orig
    json.dump(out, open(os.path.join(OUT, 'alt_susceptibility_cal.json'), 'w'), indent=2)
    print('已保存 alt_susceptibility_cal.json')

if __name__ == '__main__':
    run()
