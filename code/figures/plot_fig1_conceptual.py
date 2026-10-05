# -*- coding: utf-8 -*-
"""CSF 新增 Fig.1：概念机制图（意见三十一）。

Consensus formation (intensity f_i + speed v_i)
  -> Endogenous susceptibility chi_pre (path-dependent latent state)
  -> Response to counter-information (high-chi reversal vs low-chi no reversal)
底部强调：same consensus != same susceptibility。
风格与全文一致：Times New Roman、无标题。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2166AC'
RED = '#B2182B'
GRAY = '#4D4D4D'
LIGHT_BLUE = '#D6E4F0'
LIGHT_RED = '#F4D7D7'
LIGHT_GRAY = '#EFEFEF'

fig, ax = plt.subplots(figsize=(11.2, 4.4))
ax.set_xlim(0, 112)
ax.set_ylim(0, 44)
ax.axis('off')


def box(x, y, w, h, fc, ec, lw=1.4, r=0.02):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.6,rounding_size={r*100}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(b)


def arrow(x0, y0, x1, y1, color=GRAY, lw=2.2):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                        mutation_scale=18, color=color, lw=lw, zorder=5)
    ax.add_patch(a)


# ---------- 框 1：Consensus formation ----------
box(2, 10, 28, 26, LIGHT_BLUE, BLUE)
ax.text(16, 32.6, 'Consensus formation', ha='center', va='center',
        fontsize=13, fontweight='bold', color=BLUE)
# 两个子框（间距 2.0，与粉框内子框间距一致）
box(3.5, 13.5, 11.5, 13, 'white', BLUE, lw=1.0)
ax.text(9.25, 22.4, 'Local consensus\nintensity', ha='center', va='center', fontsize=10, color='#1a1a1a')
ax.text(9.25, 15.9, r'$f_i(t)$', ha='center', va='center', fontsize=12, color='#1a1a1a')
box(17.0, 13.5, 11.5, 13, 'white', BLUE, lw=1.0)
ax.text(22.75, 22.4, 'Formation\nspeed', ha='center', va='center', fontsize=10, color='#1a1a1a')
ax.text(22.75, 15.9, r'$v_i(t)$', ha='center', va='center', fontsize=12, color='#1a1a1a')

# ---------- 箭头 1 ----------
arrow(30.5, 23, 38.5, 23)

# ---------- 框 2：Endogenous susceptibility ----------
box(39, 10, 30, 26, '#FDF3E7', '#C77F3B')
ax.text(54, 32.6, 'Endogenous susceptibility', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#A85F1E')
ax.text(54, 25.5, r'$\chi_{\mathrm{pre}}$', ha='center', va='center', fontsize=16, color='#1a1a1a')
ax.text(54, 19.5, 'path-dependent\nlatent state', ha='center', va='center',
        fontsize=9.5, color='#5a5a5a', style='italic')
ax.text(54, 13.2, r'$S_i=\alpha\ln(1+v_i f_i)\ \to\ R_i=\tanh(S_i)$',
        ha='center', va='center', fontsize=9.5, color='#5a5a5a')

# ---------- 箭头 2 ----------
arrow(69.5, 23, 77.5, 23)

# ---------- 框 3：Response ----------
box(78, 10, 32, 26, LIGHT_RED, RED)
ax.text(94, 32.6, 'Response to counter-\ninformation', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=RED)
box(80, 13, 13.5, 13, 'white', RED, lw=1.0)
ax.text(86.75, 22.6, 'High $\\chi_{\\mathrm{pre}}$', ha='center', va='center', fontsize=10, color='#1a1a1a')
ax.text(86.75, 17.2, 'reversal\n(even limited $I_2$)', ha='center', va='center', fontsize=9.5, color=RED)
box(95.5, 13, 13.5, 13, 'white', RED, lw=1.0)
ax.text(102.25, 22.6, 'Low $\\chi_{\\mathrm{pre}}$', ha='center', va='center', fontsize=10, color='#1a1a1a')
ax.text(102.25, 17.2, 'no reversal\n(even strong $I_2$)', ha='center', va='center',
        fontsize=9.5, color='#7a3030')

# ---------- 底部强调条（纯文字，与正文表述一致）----------
box(4, 1.5, 104, 6, LIGHT_GRAY, GRAY, lw=1.2)
ax.text(56, 4.5, 'Same consensus level does not imply the same susceptibility — '
        'the formation trajectory, not the final state, sets the response.',
        ha='center', va='center', fontsize=10.5, color='#1a1a1a')

fig.savefig('/Users/np/Desktop/小论文/投稿-源码与图片/图片/attachments/Fig1.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Fig1.png 已生成')
