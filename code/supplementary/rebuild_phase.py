
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import monte_carlo_experiment, reversal_probability

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def base_params():
    p = {
        'network_formation': ['BA', 4],
        'num_nodes': 500,
        'opinion_init_mode': 'random',
        'mu': 0.1, 'alpha': 60, 'tau': 10,
        'pre_info_steps': 0,
        'info1_value': -0.9, 'info1_reception_prob': 0.4, 'info1_duration': 5, 'info1_decay': True,
        'post_info1_steps': int(os.environ.get('POST', '5')),
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
        'post_info1_steps': 7,
        'info2_value': 0.7,
        'info2_reception_prob': 0.6,
        'info2_duration': 1,
    })
    params['post_info2_steps'] = (params['total_steps'] - params['pre_info_steps']
                                  - params['info1_duration'] - params['post_info1_steps']
                                  - params['info2_duration'])
    return params

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def main(num_runs=50):
    combos = [(0.1, 40), (0.2, 20), (0.4, 10), (0.8, 5)]
    I2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    t0 = time.time()
    chi_axis = []
    P_rev = np.zeros((len(I2_vals), len(combos)))
    x_final = np.zeros_like(P_rev)

    for j, (p1, d1) in enumerate(combos):
        params = speed_params(p1, d1)
        res0 = monte_carlo_experiment(params, num_runs=num_runs)
        chi_axis.append(float(np.mean(res0['S_pre'])))
        for i, i2 in enumerate(I2_vals):
            p2 = dict(params, info2_value=i2)
            r = monte_carlo_experiment(p2, num_runs=num_runs)
            P_rev[i, j] = reversal_probability(r)
            x_final[i, j] = float(np.mean(r['x_final']))
        print(f"  χ_pre={chi_axis[-1]:.3f} 完成，耗时 {time.time()-t0:.1f}s")

    chi_axis = np.array(chi_axis)
    I2_arr = np.array(I2_vals)
    print(f"相图完成，总耗时 {time.time()-t0:.1f}s")
    print("χ_pre 轴:", np.round(chi_axis, 3))
    np.save(os.path.join(OUT, 'phase5_chi_axis.npy'), chi_axis)
    np.save(os.path.join(OUT, 'phase5_P_rev.npy'), P_rev)
    np.save(os.path.join(OUT, 'phase5_x_final.npy'), x_final)

    xs = chi_axis
    xb = np.empty(len(xs) + 1)
    xb[1:-1] = (xs[:-1] + xs[1:]) / 2
    xb[0] = xs[0] - (xs[1] - xs[0]) / 2
    xb[-1] = xs[-1] + (xs[-1] - xs[-2]) / 2
    ys = I2_arr
    yb = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                         [ys[-1] + (ys[-1] - ys[-2]) / 2]])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    im = axes[0].pcolormesh(xb, yb, P_rev, cmap='RdYlBu_r', vmin=0, vmax=1, shading='flat')
    cs = axes[0].contour(xs, ys, P_rev, levels=[0.5], colors='black', linewidths=1.8)
    axes[0].clabel(cs, fmt='$P_{\\mathrm{rev}}=0.5$', fontsize=10)
    axes[0].set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
    axes[0].set_ylabel('Counter-information intensity $I_2$')
    style_ax(axes[0])
    cb = fig.colorbar(im, ax=axes[0], pad=0.02)
    cb.set_label('Reversal probability $P_{\\mathrm{rev}}$', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    vmax = np.abs(x_final).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = axes[1].pcolormesh(xb, yb, x_final, cmap='RdYlBu_r', norm=norm, shading='flat')
    cs2 = axes[1].contour(xs, ys, x_final, levels=[0.0], colors='black', linewidths=1.8)
    axes[1].clabel(cs2, fmt='$\\bar{x}_{\\mathrm{final}}=0$', fontsize=10)
    axes[1].set_xlabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
    axes[1].set_ylabel('Counter-information intensity $I_2$')
    style_ax(axes[1])
    cb2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
    cb2.set_label('Final mean opinion $\\bar{x}_{\\mathrm{final}}$', fontsize=10)
    cb2.ax.tick_params(labelsize=9)

    fig.tight_layout()
    path = os.path.join(OUT, 'Fig5_phase_fixed.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('已保存:', path)

if __name__ == '__main__':
    main()
