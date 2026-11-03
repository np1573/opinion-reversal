
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/np/Desktop/小论文')
import model as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

def speed_params(p1, d1, post_info1=7):
    p = {
        'network_formation': ['BA', 4], 'num_nodes': 500,
        'opinion_init_mode': 'random', 'mu': 0.1, 'alpha': 60, 'tau': 10,
        'pre_info_steps': 0,
        'info1_value': -0.3, 'info1_reception_prob': p1, 'info1_duration': d1, 'info1_decay': False,
        'post_info1_steps': post_info1,
        'info2_value': 0.7, 'info2_reception_prob': 0.6, 'info2_duration': 1, 'info2_decay': True,
        'total_steps': 200,
    }
    p['post_info2_steps'] = (p['total_steps'] - p['pre_info_steps'] - p['info1_duration']
                             - p['post_info1_steps'] - p['info2_duration'])
    return p

def run_E1(num_runs=50):
    print('=== E1: t_inj x I2 ===', flush=True)
    t_inj = [2, 8, 15, 22, 30]
    I2 = [0.3, 0.5, 0.7, 0.9]
    P = np.zeros((len(t_inj), len(I2)))
    for i, tj in enumerate(t_inj):
        for j, i2 in enumerate(I2):
            p = speed_params(0.8, 5, post_info1=tj)
            p['info2_value'] = i2
            r = M.monte_carlo_experiment(p, num_runs=num_runs)
            P[i, j] = M.reversal_probability(r)
            print(f'  t_inj={tj} I2={i2}: P_rev={P[i,j]:.2f}', flush=True)
    np.save(os.path.join(OUT, 'E1_P_rev.npy'), P)
    np.save(os.path.join(OUT, 'E1_t_inj.npy'), np.array(t_inj))
    np.save(os.path.join(OUT, 'E1_I2.npy'), np.array(I2))
    print('E1 done', flush=True)

def run_E2(num_runs=100):
    print('=== E2: fixed-S baseline ===', flush=True)
    S0 = 1.0
    orig = M.update_skepticism

    def fixed_update(G, alpha, tau, beta=0.5):
        for i in G.nodes():
            G.nodes[i]['S'] = S0
            G.nodes[i]['R'] = np.tanh(S0)

    res = {}

    for name, (p1, d1) in [('slow', (0.1, 40)), ('fast', (0.8, 5))]:
        p = speed_params(p1, d1)
        r = M.monte_carlo_experiment(p, num_runs=num_runs)
        res[name] = dict(P_rev=M.reversal_probability(r),
                         x_pre=float(np.mean(r['x_pre'])), S_pre=float(np.mean(r['S_pre'])))
        print(f'  endogenous {name}: P_rev={res[name]["P_rev"]:.2f} '
              f'x_pre={res[name]["x_pre"]:.3f} S_pre={res[name]["S_pre"]:.3f}', flush=True)

    M.update_skepticism = fixed_update
    for name, (p1, d1) in [('slow', (0.1, 40)), ('fast', (0.8, 5))]:
        p = speed_params(p1, d1)
        r = M.monte_carlo_experiment(p, num_runs=num_runs)
        res['fixed_' + name] = dict(P_rev=M.reversal_probability(r),
                                    x_pre=float(np.mean(r['x_pre'])), S_pre=float(np.mean(r['S_pre'])))
        print(f'  fixed-S {name}: P_rev={res["fixed_"+name]["P_rev"]:.2f} '
              f'x_pre={res["fixed_"+name]["x_pre"]:.3f}', flush=True)
    M.update_skepticism = orig
    import json
    with open(os.path.join(OUT, 'E2_fixedS.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('E2 done', flush=True)

def run_E3():
    print('=== E3: P(S_i) distribution ===', flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    dists = {}
    for name, (p1, d1) in [('slow', (0.1, 40)), ('fast', (0.8, 5))]:
        p = speed_params(p1, d1)

        np.random.seed(0)
        import random
        random.seed(0)
        G = M.initialize_network(p['network_formation'], p['num_nodes'], p['tau'], p.get('opinion_init_mode', 'random'))
        M.set_mu(G, p['mu'])
        ua = M.update_actions_sgn
        ua(G)
        for _ in range(p['pre_info_steps']):
            M.interaction_step(G, p['mu'], ua); M.update_skepticism(G, p['alpha'], p['tau'], p.get('beta', 0.5))
        for step in range(p['info1_duration']):
            prob = p['info1_reception_prob']
            M.inject_information(G, p['info1_value'], prob, ua)
            M.interaction_step(G, p['mu'], ua); M.update_skepticism(G, p['alpha'], p['tau'], p.get('beta', 0.5))
        for node in G.nodes():
            G.nodes[node]['reversal'] = 0
        for _ in range(p['post_info1_steps']):
            M.interaction_step(G, p['mu'], ua); M.update_skepticism(G, p['alpha'], p['tau'], p.get('beta', 0.5))
        S_vals = np.array([G.nodes[i]['S'] for i in G.nodes()])
        dists[name] = S_vals
        print(f'  {name}: mean S_i={S_vals.mean():.3f}, std={S_vals.std():.3f}, '
              f'max={S_vals.max():.3f}', flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, color in [('slow', '#2166AC'), ('fast', '#B2182B')]:
        ax.hist(dists[name], bins=40, density=True, alpha=0.55, color=color,
                label=('Slow consensus' if name == 'slow' else 'Fast consensus'))
    ax.set_xlabel('Individual susceptibility $S_i$ before counter-information')
    ax.set_ylabel('Density')
    ax.tick_params(axis='both', direction='in', labelsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'E3_Sdist.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('E3 done', flush=True)

if __name__ == '__main__':
    run_E1(50)
    run_E2(100)
    run_E3()
    print('ALL DONE', flush=True)
