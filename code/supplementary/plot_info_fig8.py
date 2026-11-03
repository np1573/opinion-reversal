
import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import monte_carlo_experiment, reversal_probability

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = '/Users/np/Desktop/小论文/重跑_正确语义版/图片/attachments'
SUBF = os.path.join(OUT, 'subfigs')

NUM_RUNS = 100

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

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=11)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def _cp(k, n, alpha=0.05):
    from scipy.stats import beta
    if k <= 0:
        lo = 0.0
    else:
        lo = beta.ppf(alpha / 2, k, n - k + 1)
    if k >= n:
        hi = 1.0
    else:
        hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

def _summarize(res):
    xf = np.array(res['x_final'])
    n = len(xf)
    m = float(xf.mean())
    ci = float(1.96 * xf.std() / np.sqrt(n))
    k = int(np.sum(xf > 0))
    prev = k / n
    lo, hi = _cp(k, n)
    return m, ci, prev, (lo, hi)

def run(num_runs=NUM_RUNS):

    I_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    strength_x, strength_ci, strength_p, strength_p_ci = [], [], [], []
    for i2 in I_vals:
        params = base_params(); params['info2_value'] = i2
        res = monte_carlo_experiment(params, num_runs=num_runs)
        m, ci, prev, pci = _summarize(res)
        strength_x.append(m); strength_ci.append(ci)
        strength_p.append(prev); strength_p_ci.append(pci)
    print('  (a) 强度完成', flush=True)

    P_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    range_x, range_ci, range_p, range_p_ci = [], [], [], []
    for p2 in P_vals:
        params = base_params(); params['info2_reception_prob'] = p2
        res = monte_carlo_experiment(params, num_runs=num_runs)
        m, ci, prev, pci = _summarize(res)
        range_x.append(m); range_ci.append(ci)
        range_p.append(prev); range_p_ci.append(pci)
    print('  (b) 范围完成', flush=True)

    T_vals = [0, 3, 6, 10, 15, 20, 30, 40, 55]
    timing_x, timing_ci, timing_p, timing_p_ci = [], [], [], []
    for dt in T_vals:
        params = base_params(); params['post_info1_steps'] = dt
        params['post_info2_steps'] = (params['total_steps'] - params['pre_info_steps']
                                      - params['info1_duration'] - dt - params['info2_duration'])
        res = monte_carlo_experiment(params, num_runs=num_runs)
        m, ci, prev, pci = _summarize(res)
        timing_x.append(m); timing_ci.append(ci)
        timing_p.append(prev); timing_p_ci.append(pci)
    print('  (c) 时机完成', flush=True)

    pulse = base_params()
    res_pulse = monte_carlo_experiment(pulse, num_runs=num_runs)
    cont = base_params()
    cont.update({'info2_reception_prob': 0.2, 'info2_duration': 5})
    cont['post_info2_steps'] = (cont['total_steps'] - cont['pre_info_steps']
                                - cont['info1_duration'] - cont['post_info1_steps']
                                - cont['info2_duration'])
    res_cont = monte_carlo_experiment(cont, num_runs=num_runs)
    print('  (d) 释放方式完成', flush=True)

    post = os.environ.get('POST', '5')
    data = dict(
        POST=post,
        num_runs=num_runs,
        intensity=dict(I2=I_vals, x_final=strength_x, x_ci=strength_ci,
                       P_rev=strength_p, P_rev_ci=strength_p_ci),
        coverage=dict(p02=P_vals, x_final=range_x, x_ci=range_ci,
                      P_rev=range_p, P_rev_ci=range_p_ci),
        timing=dict(t_inj=T_vals, x_final=timing_x, x_ci=timing_ci,
                    P_rev=timing_p, P_rev_ci=timing_p_ci),
        release=dict(pulse_P_rev=float(reversal_probability(res_pulse)),
                     sustained_P_rev=float(reversal_probability(res_cont))),
    )
    jpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'output', f'info_post{post}.json')
    json.dump(data, open(jpath, 'w'), indent=2)
    print(f'  已保存 {jpath}', flush=True)

    return dict(I_vals=I_vals, strength_x=strength_x, strength_ci=strength_ci,
                P_vals=P_vals, range_x=range_x, range_ci=range_ci,
                T_vals=T_vals, timing_x=timing_x, timing_ci=timing_ci,
                res_pulse=res_pulse, res_cont=res_cont)

def draw_a(ax, d, add_label=True):
    ax.errorbar(d['I_vals'], d['strength_x'], yerr=d['strength_ci'], fmt='o-',
                color='#2E86AB', linewidth=1.6, markersize=4, capsize=3, ecolor='#2E86AB', elinewidth=1)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel(r'Information intensity $I_2$', fontsize=11)
    ax.set_ylabel(r'Final mean opinion $\bar{x}_{\mathrm{final}}$', fontsize=11)
    style_ax(ax)
    if add_label:
        pass

def draw_b(ax, d, add_label=True):
    ax.errorbar(d['P_vals'], d['range_x'], yerr=d['range_ci'], fmt='s-',
                color='#009E73', linewidth=1.6, markersize=4, capsize=3, ecolor='#009E73', elinewidth=1)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel(r'Propagation coverage $p_{0,2}$', fontsize=11)
    ax.set_ylabel(r'Final mean opinion $\bar{x}_{\mathrm{final}}$', fontsize=11)
    style_ax(ax)
    if add_label:
        pass

def draw_c(ax, d, add_label=True):
    ax.errorbar(d['T_vals'], d['timing_x'], yerr=d['timing_ci'], fmt='^-',
                color='#D55E00', linewidth=1.6, markersize=4, capsize=3, ecolor='#D55E00', elinewidth=1)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel(r'Injection timing $t_{\mathrm{inj}}$', fontsize=11)
    ax.set_ylabel(r'Final mean opinion $\bar{x}_{\mathrm{final}}$', fontsize=11)
    style_ax(ax)
    if add_label:
        pass

def draw_d(ax, d, add_label=True):
    for res, c, lab in [(d['res_pulse'], '#CC79A7', 'Impulsive'),
                        (d['res_cont'], '#56B4E9', 'Sustained')]:
        arr = np.array(res['avg_x'])
        m = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
        ax.plot(m, color=c, lw=1.6, label=lab)
        ax.fill_between(np.arange(len(m)), m - ci, m + ci, color=c, alpha=0.15, lw=0)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Time step', fontsize=11)
    ax.set_ylabel(r'Mean opinion $\bar{x}(t)$', fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    style_ax(ax)
    if add_label:
        pass

if __name__ == '__main__':
    t0 = time.time()
    d = run()
    print(f'  数据完成，耗时 {time.time()-t0:.0f}s', flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    draw_a(axes[0, 0], d, True)
    draw_b(axes[0, 1], d, True)
    draw_c(axes[1, 0], d, True)
    draw_d(axes[1, 1], d, True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig8.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('已保存 Fig8.png')

    for name, fn in [('Fig8_a.png', draw_a), ('Fig8_b.png', draw_b),
                     ('Fig8_c.png', draw_c), ('Fig8_d.png', draw_d)]:
        f = plt.figure(figsize=(4.8, 3.6))
        ax = f.add_subplot(111)
        fn(ax, d, add_label=False)
        f.tight_layout()
        f.savefig(os.path.join(SUBF, name), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(f)
    print('已保存 subfigs/Fig8_a..d.png')
