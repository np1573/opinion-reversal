
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supplement_experiments import speed_params, base_params
from model import monte_carlo_experiment

params = speed_params(0.8, 5)
print('speed_params(0.8,5) num_nodes =', params['num_nodes'])
print('speed_params(0.8,5) info2_value =', params['info2_value'])
res = monte_carlo_experiment(params, num_runs=100, n_workers=10)
chi = float(np.mean(res['S_pre']))
std = float(np.std(res['S_pre']))
print(f'G9 (info2=0.7, N=500, 100MC): chi_pre={chi:.4f} std={std:.4f}')

params2 = speed_params(0.8, 5)
params2['info2_value'] = 0.1
res2 = monte_carlo_experiment(params2, num_runs=100, n_workers=10)
chi2 = float(np.mean(res2['S_pre']))
print(f'G9 (info2=0.1, N=500, 100MC): chi_pre={chi2:.4f}')

print('前 5 个 S_pre (info2=0.7):', [round(float(x),6) for x in res['S_pre'][:5]])
print('前 5 个 S_pre (info2=0.1):', [round(float(x),6) for x in res2['S_pre'][:5]])
