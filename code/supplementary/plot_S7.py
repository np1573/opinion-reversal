
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
data = np.load(os.path.join(OUT, 'chi_pre_vs_N.npy'))
N = data[:, 0]
chi = data[:, 1]
ci = data[:, 2]

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.errorbar(N, chi, yerr=ci, fmt='o-', color='#B2182B', lw=1.6, ms=6, capsize=3)
ax.set_xlabel('Network size $N$')
ax.set_ylabel('Pre-perturbation susceptibility $\\chi_{\\mathrm{pre}}$')
ax.set_xscale('log')
ax.set_xticks([250, 500, 1000, 2000, 4000])
ax.set_xticklabels(['250', '500', '1000', '2000', '4000'])
ax.set_ylim(0, 1.2)
ax.tick_params(axis='both', direction='in', labelsize=10)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
path = os.path.join(OUT, 'fig_S7_chi_pre_N.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close(fig)
print('已保存:', path)
