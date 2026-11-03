
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multiprocessing
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (initialize_network, set_mu, update_actions_sgn,
                   interaction_step, update_skepticism, inject_information)

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
        'post_info1_steps': int(os.environ.get('POST', '7')),
        'info2_value': 0.7, 'info2_reception_prob': 0.6, 'info2_duration': 1, 'info2_decay': True,
        'total_steps': 200,
    }
    p['post_info2_steps'] = (p['total_steps'] - p['pre_info_steps'] - p['info1_duration']
                             - p['post_info1_steps'] - p['info2_duration'])
    return p

def info_inject_times(params):
    t0 = params['pre_info_steps']
    d1 = params['info1_duration']
    p1 = params['post_info1_steps']
    d2 = params['info2_duration']
    info1 = range(t0 + 1, t0 + d1 + 1)
    info2 = range(t0 + d1 + p1 + 1, t0 + d1 + p1 + d2 + 1)
    return info1, info2

def run_with_records(args):
    run_id, params = args
    p = params.copy()
    p['seed'] = run_id
    np.random.seed(p['seed'])
    random.seed(p['seed'])

    beta = p.get('beta', 0.5)

    G = initialize_network(p['network_formation'], p['num_nodes'],
                           p['tau'], p.get('opinion_init_mode', 'random'))
    set_mu(G, p['mu'])
    update_actions = update_actions_sgn
    update_actions(G)

    nodes = list(G.nodes())
    xs = [np.array([G.nodes[i]['x'] for i in nodes], dtype=np.float32)]

    def step_record():
        xs.append(np.array([G.nodes[i]['x'] for i in nodes], dtype=np.float32))

    for _ in range(p['pre_info_steps']):
        interaction_step(G, p['mu'], update_actions)
        update_skepticism(G, p['alpha'], p['tau'], beta)
        step_record()

    info1_decay = p.get('info1_decay', True)
    for step in range(p['info1_duration']):
        prob = (p['info1_reception_prob'] * (p['info1_duration'] - step) / p['info1_duration']
                if info1_decay else p['info1_reception_prob'])
        inject_information(G, p['info1_value'], prob, update_actions)
        interaction_step(G, p['mu'], update_actions)
        update_skepticism(G, p['alpha'], p['tau'], beta)
        step_record()

    for _ in range(p['post_info1_steps']):
        interaction_step(G, p['mu'], update_actions)
        update_skepticism(G, p['alpha'], p['tau'], beta)
        step_record()

    info2_decay = p.get('info2_decay', True)
    for step in range(p['info2_duration']):
        prob = (p['info2_reception_prob'] * (p['info2_duration'] - step) / p['info2_duration']
                if info2_decay else p['info2_reception_prob'])
        inject_information(G, p['info2_value'], prob, update_actions)
        interaction_step(G, p['mu'], update_actions)
        update_skepticism(G, p['alpha'], p['tau'], beta)
        step_record()

    for _ in range(p['post_info2_steps']):
        interaction_step(G, p['mu'], update_actions)
        update_skepticism(G, p['alpha'], p['tau'], beta)
        step_record()

    return run_id, np.array(xs)

def main(num_runs=100):
    params = base_params()
    import time
    t0 = time.time()
    args = [(r, params) for r in range(num_runs)]
    with Pool(processes=min(multiprocessing.cpu_count(), 10)) as pool:
        results = pool.map(run_with_records, args)
    results.sort(key=lambda x: x[0])
    trajs = np.concatenate([r[1] for r in results], axis=1)
    print(f"基准实验完成: {trajs.shape}, 耗时 {time.time()-t0:.1f}s")

    ny = 200
    yedges = np.linspace(-1.0, 1.0, ny + 1)
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    bw = np.diff(yedges)
    nt = trajs.shape[0]
    H = np.zeros((ny, nt))
    for t in range(nt):
        counts, _ = np.histogram(trajs[t], bins=yedges)
        H[:, t] = counts / (counts.sum() * bw)

    vmin = 1e-2
    Hm = np.ma.masked_less(H, vmin)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    norm = LogNorm(vmin=vmin, vmax=H.max())
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('white')
    im = ax.pcolormesh(np.arange(nt + 1), yedges, Hm, cmap=cmap, norm=norm,
                       shading='flat')

    info1, info2 = info_inject_times(params)
    ax.axvspan(info1[0], info1[-1], color='gray', alpha=0.18, zorder=0)
    ax.axvline(x=info2[0], color='magenta', linestyle='--', linewidth=1.2)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Opinion $x$')
    ax.set_xlim(0, nt - 1)
    ax.set_ylim(-1, 1)
    ax.tick_params(axis='both', direction='in', labelsize=10)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(False)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Density', fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    path = os.path.join(OUT, 'Fig1_d_fixed.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('已保存:', path)

    np.savez_compressed(os.path.join(OUT, 'fig1_density.npz'),
                        H=H, yedges=yedges, vmin=vmin)
    print('密度矩阵已保存 fig1_density.npz')

if __name__ == '__main__':
    main()
