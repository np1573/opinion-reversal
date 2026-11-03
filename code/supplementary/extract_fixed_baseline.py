
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from rebuild_phase import base_params

def run(post):
    os.environ['POST'] = str(post)
    params = base_params()
    r = M.monte_carlo_experiment(params, num_runs=100)
    endogenous = dict(P_rev=M.reversal_probability(r),
                      x_pre=float(np.mean(r['x_pre'])),
                      S_pre=float(np.mean(r['S_pre'])))

    S0 = 1.0
    orig = M.update_skepticism

    def fixed_update(G, alpha, tau):
        for i in G.nodes():
            G.nodes[i]['S'] = S0
            G.nodes[i]['R'] = np.tanh(S0)

    M.update_skepticism = fixed_update
    r2 = M.monte_carlo_experiment(base_params(), num_runs=100)
    M.update_skepticism = orig
    fixed = dict(P_rev=M.reversal_probability(r2),
                 x_pre=float(np.mean(r2['x_pre'])))
    return dict(POST=post, endogenous=endogenous, fixed=fixed)

if __name__ == '__main__':
    out = {}
    for post in (7, 5):
        r = run(post)
        out[str(post)] = r
        print(f"POST={post}: endogenous P_rev={r['endogenous']['P_rev']:.3f} "
              f"(S_pre={r['endogenous']['S_pre']:.3f}), "
              f"fixed P_rev={r['fixed']['P_rev']:.3f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'output', 'baseline_fixed.json'), 'w'), indent=2)
    print('已保存 baseline_fixed.json')
