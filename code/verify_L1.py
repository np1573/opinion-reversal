
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from model import run_single_simulation

def base(L1, post_info1_steps=30, seed=0):
    p = dict(
        network_formation=['BA', 4], num_nodes=500, tau=10, mu=0.1, alpha=60,
        pre_info_steps=0,
        info1_value=-0.9, info1_reception_prob=0.4, info1_duration=L1,
        info1_decay=True,
        post_info1_steps=post_info1_steps,
        info2_value=0.7, info2_reception_prob=0.6, info2_duration=1,
        info2_decay=True,
        post_info2_steps=30,
        seed=seed,
    )
    return p

L1_list = [5, 8, 10, 15, 20]
NMC = 40

print(f"{'L1':>3} {'chi_peak':>9} {'chi_pre':>9} {'x_pre':>8} {'x_final':>8} {'窗口(半衰)':>10}")
for L1 in L1_list:
    chi_peak_list, chi_pre_list, x_pre_list, x_final_list, half_list = [], [], [], [], []
    for s in range(NMC):
        p = base(L1, seed=s)
        r = run_single_simulation(p)
        S = np.array(r['avg_S'])
        X = np.array(r['avg_x'])

        end_info1 = L1
        chi_peak = S[:end_info1+1].max()
        chi_pre = r['S_pre']
        x_pre = r['x_pre']
        x_final = r['x_final']

        peak = chi_peak
        decayed = S[end_info1:]
        half = None
        for k, val in enumerate(decayed):
            if val <= peak * 0.5:
                half = k
                break
        chi_peak_list.append(chi_peak)
        chi_pre_list.append(chi_pre)
        x_pre_list.append(x_pre)
        x_final_list.append(x_final)
        half_list.append(half if half is not None else np.nan)

    def fmt(a):
        a = np.array(a)
        return f"{np.nanmean(a):.3f}"

    half_mean = np.nanmean(half_list)
    print(f"{L1:>3} {fmt(chi_peak_list):>9} {fmt(chi_pre_list):>9} {fmt(x_pre_list):>8} {fmt(x_final_list):>8} {half_mean:>9.1f}")

print("\n注：chi_peak=信息1期间易感性峰值；chi_pre=信息2注入前一刻易感性；")
print("    x_pre=信息2前一刻平均意见（共识强度，越负=惯性越强）；")
print("    窗口(半衰)=信息1结束后 chi 衰减到峰值一半所需步数（越长=易感窗口越长）。")
