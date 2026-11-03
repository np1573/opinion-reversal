
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import monte_carlo_experiment, reversal_probability
from rebuild_phase import base_params

def extract(post):
    os.environ['POST'] = str(post)
    params = base_params()
    t1 = params['info1_duration']
    t2 = t1 + params['post_info1_steps']

    res = monte_carlo_experiment(params, num_runs=100)
    avg_x = np.array(res['avg_x'])
    avg_sigma = np.array(res['avg_sigma'])
    avg_S = np.array(res['avg_S'])
    avg_R = np.array(res['avg_R'])

    mx = avg_x.mean(axis=0)
    ms = avg_sigma.mean(axis=0)
    mS = avg_S.mean(axis=0)
    mR = avg_R.mean(axis=0)

    chi_peak = float(mS.max())
    chi_peak_t = int(mS.argmax())
    R_peak = float(mR.max())
    R_peak_t = int(mR.argmax())

    ab = dict(params, alpha=0.0)
    res_ab = monte_carlo_experiment(ab, num_runs=100)
    abx = np.array(res_ab['avg_x']).mean(axis=0)

    out = {
        'POST': post,
        't1': t1, 't2': t2,
        'x_stage1_end': float(mx[t1]),
        'p_stage1_end': float(ms[t1]),
        'x_min_stage1': float(mx[:t2].min()),
        'x_t_pre': float(mx[t2]),
        'p_t_pre': float(ms[t2]),
        'chi_peak': chi_peak, 'chi_peak_t': chi_peak_t,
        'R_peak': R_peak, 'R_peak_t': R_peak_t,
        'x_after_info2': float(mx[t2 + 1]),
        'p_after_info2': float(ms[t2 + 1]),
        'x_after_info2_peak': float(mx[t2 + 1: t2 + 4].max()),
        'x_final_full': float(np.mean(res['x_final'])),
        'P_rev_full': float(reversal_probability(res)),
        'x_final_ablated': float(np.mean(res_ab['x_final'])),
        'P_rev_ablated': float(reversal_probability(res_ab)),
        'x_final_ablated_min': float(abx.min()),
        'x_final_ablated_end': float(abx[-1]),
    }
    return out

if __name__ == '__main__':
    import time
    t0 = time.time()
    results = {}
    for post in (7, 5):
        print(f'=== POST={post} ===', flush=True)
        r = extract(post)
        results[str(post)] = r
        for k, v in r.items():
            if isinstance(v, float):
                print(f'  {k} = {v:.4f}', flush=True)
            else:
                print(f'  {k} = {v}', flush=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'output', 'baseline_extract.json')
    json.dump(results, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f'已保存 {out_path}，总耗时 {time.time()-t0:.0f}s', flush=True)
