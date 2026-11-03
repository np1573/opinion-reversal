
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
_df = pd.read_csv(os.path.join(OUT, 'fss_Ic_summary.csv')).sort_values('N')
N = _df['N'].tolist()
Ic = _df['Ic'].tolist()
chi_peak = _df['chi_peak'].tolist()

def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))

axes[0].plot(N, Ic, 'o-', color='#B2182B', lw=1.8, ms=7)
axes[0].set_xlabel('Network size $N$')
axes[0].set_ylabel('Response boundary $I_c(N)$')
axes[0].set_xscale('log')
axes[0].set_xticks(N)
axes[0].set_xticklabels([str(x) for x in N])
style_ax(axes[0]); axes[0].grid(alpha=0.25, linewidth=0.5); axes[0].set_axisbelow(False)

axes[1].plot(N, chi_peak, 's-', color='#2E86AB', lw=1.8, ms=7)
axes[1].set_xlabel('Network size $N$')
axes[1].set_ylabel('Peak reversal fluctuation $F_{\\mathrm{rev}}$')
axes[1].set_xscale('log')
axes[1].set_xticks(N)
axes[1].set_xticklabels([str(x) for x in N])
style_ax(axes[1]); axes[1].grid(alpha=0.25, linewidth=0.5); axes[1].set_axisbelow(False)

fig.tight_layout()
path = '/Users/np/Desktop/小论文/重跑_正确语义版/代码/补充实验/output/fss_ic_combined.png'
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close(fig)
print('已保存:', path)
