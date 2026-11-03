
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
N_vals = [250, 500, 1000, 2000, 4000]

recs = {}
for N in N_vals:
    df = pd.read_csv(os.path.join(OUT, f'fss_N{N}.csv'))
    recs[N] = df

print("=== finite-size scaling 汇总 ===")
Ic = {}
for N in N_vals:
    df = recs[N]
    i2 = df['I2'].values
    p = df['P_rev'].values
    idx = np.where(p >= 0.5)[0]
    if len(idx) == 0:
        Ic[N] = None
    else:
        j = idx[0]
        Ic[N] = i2[0] if j == 0 else i2[j-1] + (0.5 - p[j-1]) / (p[j] - p[j-1]) * (i2[j] - i2[j-1])
    chi = df['chi_rev'].values
    peak = chi.max()
    peak_pos = i2[int(np.argmax(chi))]

    seg = df[(df['P_rev'] >= 0.2) & (df['P_rev'] <= 0.8)]
    slope = (seg['P_rev'].max() - seg['P_rev'].min()) / (seg['I2'].max() - seg['I2'].min()) if len(seg) >= 2 else np.nan
    print(f"  N={N}: I_c={Ic[N]:.3f}, chi_peak={peak:.3f} @ I2={peak_pos}, slope={slope:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
colors = plt.cm.viridis(np.linspace(0, 0.8, len(N_vals)))
for N, c in zip(N_vals, colors):
    df = recs[N]
    axes[0].plot(df['I2'], df['P_rev'], 'o-', color=c, linewidth=2, label=f'N={N}')
axes[0].axhline(0.5, color='gray', ls='--', linewidth=1)
axes[0].set_xlabel('反向信息强度 I_2'); axes[0].set_ylabel('P_rev')
axes[0].set_title('(a) 反转概率曲线'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

for N, c in zip(N_vals, colors):
    df = recs[N]
    axes[1].plot(df['I2'], df['M_mean'], 'o-', color=c, linewidth=2, label=f'N={N}')
axes[1].set_xlabel('I_2'); axes[1].set_ylabel('序参量 M = ⟨|x̄_final|⟩')
axes[1].set_title('(b) 序参量'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

for N, c in zip(N_vals, colors):
    df = recs[N]
    axes[2].plot(df['I2'], df['chi_rev'], 'o-', color=c, linewidth=2, label=f'N={N}')
axes[2].set_xlabel('I_2'); axes[2].set_ylabel('χ_rev = N(⟨M²⟩−⟨M⟩²)')
axes[2].set_title('(c) 反转涨落'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)

fig.suptitle('补充实验 B：有限尺寸标度（N=250/500/1000）', fontweight='bold')
fig.tight_layout()
path = os.path.join(OUT, 'S_B_finite_size.png')
fig.savefig(path, dpi=160); plt.close(fig)
print("  已保存:", path)

fig2, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(N_vals, [Ic[N] for N in N_vals], 'o-', color='#B2182B', linewidth=2)
ax[0].set_xlabel('N'); ax[0].set_ylabel('伪临界点 I_c(N)')
ax[0].set_title('(d) I_c(N) 随 N 右移'); ax[0].grid(alpha=0.25)
peaks = [recs[N]['chi_rev'].max() for N in N_vals]
ax[1].plot(N_vals, peaks, 's-', color='#2E86AB', linewidth=2)
ax[1].set_xlabel('N'); ax[1].set_ylabel('χ_rev 峰值')
ax[1].set_title('(e) 反转涨落峰值随 N 增长'); ax[1].grid(alpha=0.25)
fig2.tight_layout()
path2 = os.path.join(OUT, 'S_B_Ic_chi_peak.png')
fig2.savefig(path2, dpi=160); plt.close(fig2)
print("  已保存:", path2)

rows = []
for N in N_vals:
    chi = recs[N]['chi_rev'].values
    i2 = recs[N]['I2'].values
    rows.append(dict(N=N, Ic=round(Ic[N], 4) if Ic[N] else None,
                     chi_peak=round(float(chi.max()), 4),
                     chi_peak_I2=float(i2[int(np.argmax(chi))])))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'fss_Ic_summary.csv'), index=False)
print("  汇总:", rows)
