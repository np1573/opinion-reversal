
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def clopper_pearson(k, n, alpha=0.05):
    k = int(round(k)); n = int(round(n))
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

_e2 = json.load(open(os.path.join(OUT, 'E2_fixedS_rerun.json')))
slow_end = _e2['slow']['P_rev']
fast_end = _e2['fast']['P_rev']
slow_fix = _e2['fixed_slow']['P_rev']
fast_fix = _e2['fixed_fast']['P_rev']

N_RUNS = 100
groups = ['Slow consensus', 'Fast consensus']
endo = [slow_end, fast_end]
fixd = [slow_fix, fast_fix]

def ci(v):
    lo, hi = clopper_pearson(v * N_RUNS, N_RUNS)
    return np.array([[v - lo], [hi - v]])

endo_err = [ci(v) for v in endo]
fixd_err = [ci(v) for v in fixd]

x = np.arange(2)
w = 0.34

fig, ax = plt.subplots(figsize=(5.8, 4.4))

b1 = ax.bar(x - w/2, endo, w, yerr=np.hstack(endo_err), capsize=4,
            color='#2166AC', label=r'Endogenous $S_i(t)$', error_kw=dict(lw=1.0, color='k'))
b2 = ax.bar(x + w/2, fixd, w, yerr=np.hstack(fixd_err), capsize=4,
            color='#B2182B', label=r'Fixed $S_i=1.0$', error_kw=dict(lw=1.0, color='k'))

for i, (r_end, r_fix) in enumerate(zip(b1, b2)):
    hi_end = endo[i] + endo_err[i][1][0]
    ax.text(r_end.get_x() + r_end.get_width()/2, hi_end + 0.012,
            f'{endo[i]:.2f}', ha='center', va='bottom', fontsize=10)
    hi_fix = fixd[i] + fixd_err[i][1][0]
    ax.text(r_fix.get_x() + r_fix.get_width()/2, hi_fix + 0.012,
            f'{fixd[i]:.2f}', ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel('Reversal probability $P_{\\mathrm{rev}}$', fontsize=11)
ax.set_ylim(0, 1.08)
ax.tick_params(axis='both', direction='in', labelsize=10)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.grid(axis='y', alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9.5, loc='upper left')

fig.tight_layout()
path = os.path.join(OUT, 'E2_fixedS_bar.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close(fig)
print('已保存:', path)
