
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from model import run_single_simulation

def base(L1, post_info1_steps, seed=0):
    return dict(
        network_formation=['BA', 4], num_nodes=500, tau=10, mu=0.1, alpha=60,
        pre_info_steps=0,
        info1_value=-0.9, info1_reception_prob=0.4, info1_duration=L1,
        info1_decay=True,
        post_info1_steps=post_info1_steps,
        info2_value=0.7, info2_reception_prob=0.6, info2_duration=1,
        info2_decay=True,
        post_info2_steps=40,
        seed=seed,
    )

NMC = 30
t_inj_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20]

for L1 in [5, 10]:
    print(f"\n===== L1 = {L1} =====")
    print(f"{'t_inj':>5} {'chi_pre':>9} {'x_pre':>8} {'x_final':>8} {'P_rev':>7}")
    for t_inj in t_inj_list:
        chi_pre_l, x_pre_l, x_final_l = [], [], []
        for s in range(NMC):
            r = run_single_simulation(base(L1, t_inj, seed=s))
            chi_pre_l.append(r['S_pre'])
            x_pre_l.append(r['x_pre'])
            x_final_l.append(r['x_final'])
        chi_pre = np.mean(chi_pre_l)
        x_pre = np.mean(x_pre_l)
        x_final = np.mean(x_final_l)
        p_rev = np.mean(np.array(x_final_l) > 0)
        print(f"{t_inj:>5} {chi_pre:>9.3f} {x_pre:>8.3f} {x_final:>8.3f} {p_rev:>7.2f}")
