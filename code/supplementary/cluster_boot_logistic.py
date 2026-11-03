
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
B = 4000
N_RUNS = 100
rng = np.random.default_rng(20261101)

def load_y():
    d = np.load(os.path.join(OUT, 'speed_traces.npz'))
    Y = np.zeros((10, N_RUNS), dtype=int)
    chi = []
    for j in range(1, 11):
        avg_x = d[f'G{j}_avg_x']
        x_final = avg_x[:, -1]
        Y[j - 1] = (x_final > 0.0).astype(int)

    import csv
    with open(os.path.join(OUT, 'speed_10_levels.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    chi = np.array([float(r['chi_pre']) for r in rows])
    return Y, chi

def fit_b(Y, chi):
    import statsmodels.api as sm
    success = Y.sum(axis=1).astype(float)
    failure = np.full(10, N_RUNS, dtype=float) - success
    X = sm.add_constant(chi)
    model = sm.GLM(np.column_stack([success, failure]), X,
                   family=sm.families.Binomial()).fit()
    return float(model.params[0]), float(model.params[1])

def main():
    Y, chi = load_y()

    print('各组反转计数（应= 2,4,5,13,16,35,43,62,70,91）:', Y.sum(axis=1).tolist())
    print('chi_pre:', np.round(chi, 4).tolist())

    a0, b0 = fit_b(Y, chi)
    print(f'原始点估计: a={a0:.4f}, b={b0:.4f}')

    bs = np.zeros(B)
    for b in range(B):
        idx = rng.choice(N_RUNS, size=N_RUNS, replace=True)
        Yb = Y[:, idx]
        _, bb = fit_b(Yb, chi)
        bs[b] = bb

    b_se_boot = float(np.std(bs, ddof=1))
    b_lo, b_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    from scipy import stats as scipy_stats
    z = b0 / b_se_boot
    p_boot = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))

    print(f'\n聚类 bootstrap（B={B}，按 run index 联合重采样）:')
    print(f'  b = {b0:.4f}, 聚类稳健 SE = {b_se_boot:.4f}')
    print(f'  聚类稳健 95% CI = [{b_lo:.4f}, {b_hi:.4f}]')
    print(f'  聚类稳健 p = {p_boot:.3e}')
    print(f'  对照（原独立假设）: b={b0:.4f}, SE=0.349, CI=[4.963, 6.333], p=9.2e-59')

    out = dict(
        method='run-index clustered bootstrap (joint resampling across ten groups)',
        B=B, n_runs=N_RUNS, n_groups=10,
        b_point=round(b0, 4), a_point=round(a0, 4),
        b_se_clustered=round(b_se_boot, 4),
        b_ci_clustered=[round(b_lo, 4), round(b_hi, 4)],
        p_clustered=round(p_boot, 6),

        b_se_naive=0.3495, b_ci_naive=[4.9633, 6.3331], p_naive=9.2e-59,
    )
    with open(os.path.join(OUT, 'speed_logistic_clustered.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n已保存 speed_logistic_clustered.json')

if __name__ == '__main__':
    main()
