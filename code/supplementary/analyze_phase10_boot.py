
import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'phase10boot')
I2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
NG = 10
B = 4000
rng = np.random.default_rng(20261101)

def load_group(j):
    X = np.zeros((len(I2), 100))
    for i, i2 in enumerate(I2):
        f = os.path.join(OUT, f'G{j}_I2{i2}.npz')
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        X[i] = np.load(f)['x_final']
    return X

def rev(p):
    return p > 0.0

def ic_from_prev(prev):
    if prev[0] >= 0.5:
        return float(I2[0]), True
    for i in range(1, len(I2)):
        if prev[i] >= 0.5:
            ic = I2[i - 1] + (0.5 - prev[i - 1]) / (prev[i] - prev[i - 1]) * (I2[i] - I2[i - 1])
            return float(ic), True
    return None, False

def bootstrap_ic(X):
    n = X.shape[1]
    ics = np.full(B, np.nan)
    crossed_cnt = 0
    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        prev = np.array([np.mean(X[i, idx] > 0.0) for i in range(len(I2))])
        ic, crossed = ic_from_prev(prev)
        if crossed:
            ics[b] = ic
            crossed_cnt += 1
    return ics, crossed_cnt

def main():
    chi_path = os.path.join(OUT, 'chi_axis.npy')
    chi = None
    if os.path.exists(chi_path):
        chi = np.load(chi_path)
        print('chi_axis.npy 已存在：', np.round(chi, 3))

    results = []
    print('\n=== 主响应边界 I_c（点估计 + bootstrap 95% CI）===')
    print(f"{'组':>2} {'χ_pre':>7} | {'P_rev (0.1~1.0)':>24} | {'I_c':>7} | {'95% CI':>16} | 穿越比例")
    for j in range(NG):
        try:
            X = load_group(j)
        except FileNotFoundError as e:
            print(f'  [组 {j}] 数据不完整，跳过：{os.path.basename(str(e))}')
            continue

        prev = np.array([np.mean(X[i] > 0.0) for i in range(len(I2))])
        ic, crossed = ic_from_prev(prev)
        ics, crossed_cnt = bootstrap_ic(X)
        frac = crossed_cnt / B

        if crossed:
            lo, hi = np.nanpercentile(ics, [2.5, 97.5])
            ci_str = f'[{lo:.3f}, {hi:.3f}]'
            ic_str = f'{ic:.3f}'
        else:
            ci_str = '>1.0 (右删失)'
            ic_str = '>1.0'
            lo = hi = np.nan

        prev_str = ' '.join(f'{p:.2f}' for p in prev)
        chi_str = f'{chi[j]:.3f}' if chi is not None else '?'
        print(f'{j:>2} {chi_str:>7} | {prev_str:>24} | {ic_str:>7} | {ci_str:>16} | {frac:.2f}')

        results.append({
            'group': j,
            'chi_pre': float(chi[j]) if chi is not None else None,
            'P_rev': [round(float(p), 4) for p in prev],
            'Ic_point': round(float(ic), 4) if crossed else None,
            'Ic_ci_lo': round(float(lo), 4) if crossed else None,
            'Ic_ci_hi': round(float(hi), 4) if crossed else None,
            'crossing_fraction': round(float(frac), 4),
            'censored': not crossed,
        })

    if results:
        with open(os.path.join(OUT, 'ic_bootstrap.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print('\n已保存 ic_bootstrap.json')

if __name__ == '__main__':
    main()
