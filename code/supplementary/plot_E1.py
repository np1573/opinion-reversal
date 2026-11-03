
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
P = np.load(os.path.join(OUT, 'E1_P_rev.npy'))
t_inj = np.load(os.path.join(OUT, 'E1_t_inj.npy'))
I2 = np.load(os.path.join(OUT, 'E1_I2.npy'))

fig, ax = plt.subplots(figsize=(6.4, 4.6))
cmap = plt.cm.RdYlBu_r
im = ax.pcolormesh(t_inj, I2, P.T, cmap=cmap, vmin=0, vmax=1, shading='nearest')
ax.set_xlabel('Injection timing $t_{\\mathrm{inj}}$ (steps after Information 1)')
ax.set_ylabel('Counter-information intensity $I_2$')
ax.set_xticks(t_inj)
ax.tick_params(axis='both', direction='in', labelsize=10)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label('Reversal probability $P_{\\mathrm{rev}}$', fontsize=10)

for i, tj in enumerate(t_inj):
    for j, i2 in enumerate(I2):
        v = P[i, j]
        r, g, b, _ = cmap(v)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        color = 'white' if luminance < 0.5 else 'black'
        ax.text(tj, i2, f'{v:.2f}', ha='center', va='center', fontsize=9,
                color=color)
fig.tight_layout()
path = os.path.join(OUT, 'E1_tinj_I2.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
plt.close(fig)
print('已保存:', path)
