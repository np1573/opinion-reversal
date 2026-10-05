# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')
from model import *
import numpy as np

BASE_PARAMS = {
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
BASE_PARAMS['post_info2_steps'] = (BASE_PARAMS['total_steps'] - BASE_PARAMS['pre_info_steps']
    - BASE_PARAMS['info1_duration'] - BASE_PARAMS['post_info1_steps'] - BASE_PARAMS['info2_duration'])

# 单次
r = run_single_simulation(dict(BASE_PARAMS, seed=0))
print("单次: 时间点数 =", len(r['avg_x']),
      "| x_pre = %.3f" % r['x_pre'],
      "| S_pre = %.3f" % r['S_pre'],
      "| x_final = %.3f" % r['x_final'])

# 多次 MC（小规模验证）
res = monte_carlo_experiment(BASE_PARAMS, num_runs=20, n_workers=8)
print("MC 20次: S_pre均值=%.3f, x_pre均值=%.3f, x_final均值=%.3f, P_rev=%.2f" % (
    np.mean(res['S_pre']), np.mean(res['x_pre']), np.mean(res['x_final']),
    reversal_probability(res)))

# 消融（alpha=0）
ab = dict(BASE_PARAMS, alpha=0.0)
res_ab = monte_carlo_experiment(ab, num_runs=20, n_workers=8)
print("消融 alpha=0: x_pre=%.3f, x_final=%.3f, P_rev=%.2f" % (
    np.mean(res_ab['x_pre']), np.mean(res_ab['x_final']), reversal_probability(res_ab)))
