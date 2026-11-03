
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import speed_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

NETS = [
    ('BA', ['BA', 4]),
    ('ER', ['ER', 0.016]),
    ('WS', ['WS', [8, 0.2]]),
]
SPEEDS = [('slow', 0.10, 40), ('fast', 1.00, 4)]
NUM_RUNS = 200
N_WORKERS = 10
B = 4000
SEED = 12345

def run_all():
    store = {}
    meta = {}
    for net_name, cfg in NETS:
        for speed_name, p1, d1 in SPEEDS:
            key = f'{net_name}_{speed_name}'
            params = speed_params(p1, d1)
            params['network_formation'] = cfg
            res = M.monte_carlo_experiment(params, num_runs=NUM_RUNS,
                                           n_workers=N_WORKERS)
            xf = np.array(res['x_final'])
            store[key] = xf
            k = int(np.sum(xf > 0.0))
            meta[key] = dict(
                chi_pre=float(np.mean(res['S_pre'])),
                x_pre=float(np.mean(res['x_pre'])),
                P_rev=k / NUM_RUNS, k=k, n=NUM_RUNS)
            print(f'[{key}] chi_pre={meta[key]["chi_pre"]:.4f} '
                  f'x_pre={meta[key]["x_pre"]:.4f} '
                  f'P_rev={meta[key]["P_rev"]:.3f} ({k}/{NUM_RUNS})', flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for net_name, cfg in NETS:
        xf_slow = store[f'{net_name}_slow'] > 0.0
        xf_fast = store[f'{net_name}_fast'] > 0.0
        obs = xf_fast.mean() - xf_slow.mean()
        draws = []
        for _ in range(B):
            bs = rng.integers(0, NUM_RUNS, NUM_RUNS)
            bf = rng.integers(0, NUM_RUNS, NUM_RUNS)
            draws.append(xf_fast[bf].mean() - xf_slow[bs].mean())
        draws = np.array(draws)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(dict(
            network=net_name,
            P_slow=meta[f'{net_name}_slow']['P_rev'],
            P_fast=meta[f'{net_name}_fast']['P_rev'],
            dP_rev=obs, dP_lo=lo, dP_hi=hi,
            chi_slow=meta[f'{net_name}_slow']['chi_pre'],
            chi_fast=meta[f'{net_name}_fast']['chi_pre'],
            xpre_slow=meta[f'{net_name}_slow']['x_pre'],
            xpre_fast=meta[f'{net_name}_fast']['x_pre']))
        print(f'[{net_name}] ΔP_rev={obs:+.3f} 95% CI=[{lo:+.3f}, {hi:+.3f}]',
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'dprev_summary.csv'), index=False)
    np.savez_compressed(
        os.path.join(OUT, 'dprev_per_run.npz'),
        **{f'{net}_{sp}': store[f'{net}_{sp}']
           for net, _ in NETS for sp, _, _ in SPEEDS})
    json.dump(meta, open(os.path.join(OUT, 'dprev_meta.json'), 'w'), indent=2)
    print('已保存 dprev_summary.csv / dprev_per_run.npz / dprev_meta.json')

if __name__ == '__main__':
    run_all()
