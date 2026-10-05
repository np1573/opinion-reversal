# -*- coding: utf-8 -*-
"""远程跑 N=4000 的 200 次 MC（I2=0.3/0.4/0.5），内联 speed_params 避免 import 整个 supplement_experiments。"""
import json
import time
import numpy as np
from model import monte_carlo_experiment, reversal_probability


def base_params():
    p = {
        'network_formation': ['BA', 4],
        'num_nodes': 500,
        'opinion_init_mode': 'random',
        'mu': 0.1, 'alpha': 60, 'tau': 10,
        'pre_info_steps': 0,
        'info1_value': -0.9, 'info1_reception_prob': 0.4, 'info1_duration': 5, 'info1_decay': True,
        'post_info1_steps': 15,
        'info2_value': 0.7, 'info2_reception_prob': 0.6, 'info2_duration': 1, 'info2_decay': True,
        'total_steps': 200,
    }
    p['post_info2_steps'] = (p['total_steps'] - p['pre_info_steps'] - p['info1_duration']
                             - p['post_info1_steps'] - p['info2_duration'])
    return p


def speed_params(p1, d1):
    params = base_params()
    params.update({
        'info1_value': -0.3,
        'info1_reception_prob': p1,
        'info1_duration': d1,
        'info1_decay': False,
        'post_info1_steps': 12,
        'info2_value': 0.7,
        'info2_reception_prob': 0.6,
        'info2_duration': 1,
    })
    params['post_info2_steps'] = (params['total_steps'] - params['pre_info_steps']
                                  - params['info1_duration'] - params['post_info1_steps']
                                  - params['info2_duration'])
    return params


if __name__ == '__main__':
    N = 4000
    I2_list = [0.3, 0.4, 0.5]
    t0 = time.time()
    results = []
    for i2 in I2_list:
        params = speed_params(0.4, 10)
        params['num_nodes'] = N
        params['info2_value'] = i2
        res = monte_carlo_experiment(params, num_runs=200, n_workers=10)
        P = reversal_probability(res)
        results.append((i2, P))
        print(f'I2={i2}: P_rev={P:.4f}  (累计 {time.time()-t0:.0f}s)', flush=True)
    print('=== SUMMARY ===', flush=True)
    for i2, P in results:
        print(f'{i2} {P:.4f}', flush=True)
    with open('N4000_result.txt', 'w') as f:
        f.write('\n'.join(f'{i2} {P:.4f}' for i2, P in results))
    print('结果已写入 N4000_result.txt', flush=True)
