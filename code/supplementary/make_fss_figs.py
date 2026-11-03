
import os, json
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'output', 'fss200')
OUT = os.path.join(BASE, 'output')
ATT = '/Users/np/Desktop/小论文/Physica-A-new/图片/attachments'
SUBF = os.path.join(ATT, 'subfigs')
os.makedirs(SUBF, exist_ok=True)

Ns = [250, 500, 1000, 2000, 4000]
I2_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
B_BOOT = 4000
SEED = 20261030
RED = '#B2182B'
CMAP = plt.cm.viridis(np.linspace(0.0, 0.82, len(Ns)))

def clopper_pearson(k, n, alpha=0.05):
    k = int(round(k)); n = int(round(n))
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha/2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha/2, k + 1, n - k)
    return lo, hi

def interpolate_ic(i2_arr, p_arr):
    idx = np.where(p_arr >= 0.5)[0]
    if len(idx) == 0:
        return None
    if idx[0] == 0:
        return float(i2_arr[0])
    j = idx[0]
    return float(i2_arr[j-1] + (0.5 - p_arr[j-1]) / (p_arr[j] - p_arr[j-1]) * (i2_arr[j] - i2_arr[j-1]))

def style_ax(ax, xlog=False):
    ax.tick_params(axis='both', direction='in', labelsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    if xlog:
        ax.set_xscale('log')

def load_all():
    xf = {N: {} for N in Ns}
    chi = {N: [] for N in Ns}
    for N in Ns:
        for i2 in I2_vals:
            d = np.load(os.path.join(DATA, f'fss200_N{N}_I2{i2}.npz'))
            xf[N][i2] = d['x_final']
            if i2 == I2_vals[0]:

                chi[N] = d['S_pre'].tolist()
    return xf, chi

def compute_summary(xf, chi):
    i2_arr = np.array(I2_vals)
    rng = np.random.default_rng(SEED)
    rows = []
    p_all, p_lo_all, p_hi_all = {}, {}, {}
    for N in Ns:
        p = {i2: float(np.mean(xf[N][i2] > 0.0)) for i2 in I2_vals}
        p_lo, p_hi = {}, {}
        for i2 in I2_vals:
            x = xf[N][i2]
            k = int(np.sum(x > 0.0)); n = len(x)
            lo, hi = clopper_pearson(k, n)
            p_lo[i2], p_hi[i2] = lo, hi
        p_all[N] = p
        p_lo_all[N] = p_lo
        p_hi_all[N] = p_hi
        ic_pt = interpolate_ic(i2_arr, np.array([p[i2] for i2 in I2_vals]))

        n_runs = len(xf[N][I2_vals[0]])
        ic_boot = []
        for _ in range(B_BOOT):
            idx = rng.choice(n_runs, size=n_runs, replace=True)
            pb = [np.mean(xf[N][i2][idx] > 0.0) for i2 in I2_vals]
            icb = interpolate_ic(i2_arr, np.array(pb))
            if icb is not None:
                ic_boot.append(icb)
        ic_boot = np.array(ic_boot)
        ic_lo = np.percentile(ic_boot, 2.5) if len(ic_boot) else np.nan
        ic_hi = np.percentile(ic_boot, 97.5) if len(ic_boot) else np.nan
        c = np.array(chi[N])
        rows.append(dict(N=N, chi_pre=float(c.mean()),
                         chi_pre_ci=1.96*float(c.std()/np.sqrt(len(c))),
                         Ic=ic_pt, Ic_boot_lo=ic_lo, Ic_boot_hi=ic_hi,
                         boot_used=int(len(ic_boot)), boot_censored=int(B_BOOT-len(ic_boot))))
        print(f'N={N}: chi_pre={c.mean():.4f}±{1.96*c.std()/np.sqrt(len(c)):.4f} | '
              f'Ic={ic_pt} 95%=[{ic_lo:.3f},{ic_hi:.3f}]', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'fss200_summary.csv'), index=False)
    with open(os.path.join(OUT, 'fss200_summary.json'), 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)
    return df, p_all, p_lo_all, p_hi_all, xf

def plot_S4_panels(xf, p, p_lo, p_hi):

    panels = {}

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for ci, N in enumerate(Ns):
        xs = np.array(I2_vals)
        ys = np.array([p[N][i2] for i2 in I2_vals])
        lo = np.array([p_lo[N][i2] for i2 in I2_vals])
        hi = np.array([p_hi[N][i2] for i2 in I2_vals])
        ax.errorbar(xs, ys, yerr=np.vstack([ys-lo, hi-ys]), fmt='o-', color=CMAP[ci],
                    lw=1.3, ms=4, capsize=2, elinewidth=0.8, label=f'$N={N}$')
    ax.axhline(0.5, color='0.45', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('$I_2$')
    ax.set_ylabel('$P_{\\mathrm{rev}}$')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, loc='lower right', ncol=1)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(SUBF, 'FigS4_a.png'), dpi=300, bbox_inches='tight')
    panels['a'] = fig
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for ci, N in enumerate(Ns):
        xs = np.array(I2_vals)
        m = np.array([np.mean(np.abs(xf[N][i2])) for i2 in I2_vals])
        ax.plot(xs, m, 'o-', color=CMAP[ci], lw=1.3, ms=4, label=f'$N={N}$')
    ax.set_xlabel('$I_2$')
    ax.set_ylabel('$\\langle m\\rangle$')
    ax.legend(frameon=False, fontsize=8, loc='best')
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(SUBF, 'FigS4_b.png'), dpi=300, bbox_inches='tight')
    panels['b'] = fig
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for ci, N in enumerate(Ns):
        xs = np.array(I2_vals)
        F = []
        for i2 in I2_vals:
            m = np.abs(xf[N][i2])
            F.append(N * (np.mean(m**2) - np.mean(m)**2))
        ax.plot(xs, F, 'o-', color=CMAP[ci], lw=1.3, ms=4, label=f'$N={N}$')
    ax.set_xlabel('$I_2$')
    ax.set_ylabel('$F_{\\mathrm{rev}}$')
    ax.legend(frameon=False, fontsize=8, loc='best')
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(SUBF, 'FigS4_c.png'), dpi=300, bbox_inches='tight')
    panels['c'] = fig
    plt.close(fig)

    from PIL import Image
    ims = [Image.open(os.path.join(SUBF, f'FigS4_{l}.png')) for l in 'abc']
    h = max(im.height for im in ims)
    pad = 18
    W = sum(im.width for im in ims) + pad * 2
    canvas = Image.new('RGB', (W, h), 'white')
    x0 = 0
    for im in ims:
        canvas.paste(im, (x0, (h - im.height)//2))
        x0 += im.width + pad
    canvas.save(os.path.join(ATT, 'FigS4.png'), dpi=(300, 300))
    print('合并图 FigS4.png ->', canvas.size)

def plot_S5_panels(df, xf):

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    N_arr = df['N'].values
    ic = df['Ic'].values
    lo = df['Ic_boot_lo'].values
    hi = df['Ic_boot_hi'].values
    ax.errorbar(N_arr, ic, yerr=np.vstack([ic-lo, hi-ic]), fmt='o-', color=RED,
                lw=1.5, ms=6, capsize=3, elinewidth=1.0)
    ax.set_xlabel('$N$')
    ax.set_ylabel('$I_c(N)$')
    style_ax(ax, xlog=True)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(int(x)) for x in Ns])
    fig.tight_layout()
    fig.savefig(os.path.join(SUBF, 'FigS5_a.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    peaks = []
    for N in Ns:
        F = []
        for i2 in I2_vals:
            m = np.abs(xf[N][i2])
            F.append(N * (np.mean(m**2) - np.mean(m)**2))
        peaks.append(max(F))
    ax.plot(Ns, peaks, 's-', color='#2E86AB', lw=1.5, ms=6)
    ax.set_xlabel('$N$')
    ax.set_ylabel('$F_{\\mathrm{rev}}$ peak')
    style_ax(ax, xlog=True)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(int(x)) for x in Ns])
    fig.tight_layout()
    fig.savefig(os.path.join(SUBF, 'FigS5_b.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    from PIL import Image
    ims = [Image.open(os.path.join(SUBF, f'FigS5_{l}.png')) for l in 'ab']
    h = max(im.height for im in ims)
    pad = 18
    W = sum(im.width for im in ims) + pad
    canvas = Image.new('RGB', (W, h), 'white')
    x0 = 0
    for im in ims:
        canvas.paste(im, (x0, (h - im.height)//2))
        x0 += im.width + pad
    canvas.save(os.path.join(ATT, 'FigS5.png'), dpi=(300, 300))
    print('合并图 FigS5.png ->', canvas.size)

def plot_S6(df):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.errorbar(df['N'].values, df['chi_pre'].values, yerr=df['chi_pre_ci'].values,
                fmt='o-', color=RED, lw=1.6, ms=6, capsize=3)
    ax.set_xlabel('Network size $N$')
    ax.set_ylabel('$\\chi_{\\mathrm{pre}}$')
    style_ax(ax, xlog=True)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(int(x)) for x in Ns])
    fig.tight_layout()
    fig.savefig(os.path.join(SUBF, 'FigS6.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(ATT, 'FigS6.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('图 FigS6.png ->', os.path.join(ATT, 'FigS6.png'))

if __name__ == '__main__':
    xf, chi = load_all()
    df, p, p_lo, p_hi, _ = compute_summary(xf, chi)
    plot_S4_panels(xf, p, p_lo, p_hi)
    plot_S5_panels(df, xf)
    plot_S6(df)
    print('全部完成')
