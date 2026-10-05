# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

IMG = '/Users/np/Desktop/小论文/投稿-源码与图片/图片/attachments'

LINE = '#2E86AB'      # 态度曲线（蓝）
SHADE = '#F5DCC8'     # 信息1期阴影（浅橙）
C_INFO = '#4A90C4'    # Info 1 标注（蓝）
C_RAPID = '#D95F02'   # Rapid consensus 标注（橙红）
C_COUNTER = '#D62728' # Counter-info 标注 + 虚线（红）
C_REV = '#009E73'     # Opinion reversal（绿）

ARROW_KW = dict(arrowstyle='-|>', mutation_scale=14, lw=1.5, shrinkA=2, shrinkB=1)


def style_ax(ax):
    ax.tick_params(axis='both', direction='in', labelsize=11)
    ax.grid(alpha=0.22, linewidth=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def yfmt(v, _):
    return {1: '+1', 0.5: '+0.5', 0: '0', -0.5: '−0.5', -1: '−1'}.get(v, str(v))


def broken_axis(ax, xc, size=0.030):
    """单轴断裂标记：在数据横坐标 xc 处把 bottom spine 真正断开，画两条紧凑斜杠（'//'）。

    - bottom spine 手动拆成左右两段，中间（断裂区）留白，不再有连续的轴线；
    - 两条斜杠紧邻断裂中心、向右上倾斜，间距紧凑；
    - 态度曲线本身保持连续穿过此区。
    """
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    y0 = ylim[0]
    span_y = ylim[1] - ylim[0]
    dy = span_y * size
    slash_w = dy * 0.55          # 每条斜杠的水平半宽
    xl = xc - slash_w * 1.6      # 左斜杠中心
    xr = xc + slash_w * 1.6      # 右斜杠中心

    # 隐藏默认 spine，手动重画左右两段（在斜杠外侧断开）
    ax.spines['bottom'].set_visible(False)
    ax.plot([xlim[0], xl - slash_w], [y0, y0], color='k', lw=0.8,
            clip_on=False, zorder=6)
    ax.plot([xr + slash_w, xlim[1]], [y0, y0], color='k', lw=0.8,
            clip_on=False, zorder=6)

    # 两条紧凑斜杠（'//'）
    for x in (xl, xr):
        ax.plot([x - slash_w, x + slash_w], [y0 - dy, y0 + dy],
                color='k', lw=1.2, clip_on=False, zorder=7)


def phase_marker(ax, x0, x1, y, text, color, fontsize=10, fontweight='normal',
                 text_dy=-0.05, va='top'):
    """用双向箭头标记一个阶段区间 [x0, x1]，文字放在箭头中点附近。

    箭头画在 y（水平双向箭头，表示一段过程而非单个数据点）；
    text_dy 为文字相对箭头的垂直偏移（负值=箭头下方，正值=上方）。
    """
    ax.annotate('', xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.6,
                                shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y + text_dy, text, ha='center', va=va,
            fontsize=fontsize, color=color, fontweight=fontweight)


def liu_case(ax):
    """(a) Liu Qiangdong 案例：单轴断裂轴，曲线连续。

    事件：Sep 2（信息1）→ Sep 3（快速共识）→ Dec 21（反向信息，检方不改判）
          → Dec 24（舆论反转）。
    x 位置在 Sep 3 与 Dec 21 之间拉开（时间跳跃），但态度曲线用一条连续折线连接，
    不在断裂处断开；断裂仅体现在时间轴刻度 + 底边斜线标记上。
    """
    xs = [0, 0.5, 1.0, 3.4, 3.7, 4.0]
    ys = [0, -0.55, -0.85, -0.72, 0.0, 0.30]
    ax.set_xlim(-0.5, 4.55)
    ax.set_ylim(-1.3, 1.35)
    ax.axhline(0, color='k', lw=0.8, zorder=1)
    ax.axvspan(0, 1, color=SHADE, alpha=0.55, lw=0, zorder=0)

    # 连续曲线（一条折线，跨过断裂区不断）
    ax.plot(xs, ys, color=LINE, lw=2.4, zorder=4, solid_joinstyle='round',
            solid_capstyle='round')
    ax.scatter(xs, ys, s=32, color=LINE, zorder=5, edgecolors='white', linewidths=0.5)

    # counter-info 虚线（Dec 21，画到 y≈0.9，不到顶部文字区）
    ax.axvline(3.4, color=C_COUNTER, ls='--', lw=1.8, zorder=3, ymax=0.83)

    ax.set_xticks([0, 1.0, 3.4, 4.0])
    ax.set_xticklabels(['Sep 2', 'Sep 3', 'Dec 21', 'Dec 24'])
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
    ax.set_ylabel('Opinion attitude\n(negative → positive)', fontsize=11)

    # 断裂标记：Sep 3(1.0) 与 Dec 21(3.4) 中点 x=2.2 处（spine 断开 + 紧凑斜杠）
    broken_axis(ax, 2.2)

    # Info 1 → 起点
    ax.annotate('Info 1: arrest exposed (Sep 2)\n→ over 6,000 reports by Sep 3',
                xy=(0, 0.02), xytext=(-0.42, 0.62), ha='left', va='center',
                fontsize=10, color=C_INFO,
                arrowprops=dict(ARROW_KW, color=C_INFO))
    # Rapid consensus（阶段：Sep 2 → Sep 3 的下降段，双向箭头）
    phase_marker(ax, 0.0, 1.0, -1.05, 'Rapid consensus\nformation',
                 C_RAPID, text_dy=-0.05, va='top')
    # Counter-info → 虚线（文字在虚线左上方，斜箭头指向虚线）
    ax.annotate('Counter-info: prosecutors\ndecline to change (Dec 21)',
                xy=(3.34, 0.62), xytext=(2.62, 0.95), ha='center', va='center',
                fontsize=10, color=C_COUNTER,
                arrowprops=dict(ARROW_KW, color=C_COUNTER))
    # Opinion reversal（阶段：Dec 21 → Dec 24 的反转段，双向箭头）
    phase_marker(ax, 3.4, 4.0, 0.55, 'Opinion reversal',
                 C_REV, fontsize=10.5, fontweight='bold', text_dy=0.05, va='bottom')
    style_ax(ax)


def covington_case(ax):
    """(b) Covington Catholic incident：单轴连续时间轴。"""
    xs = [0, 0.5, 1.0, 2.0, 2.5, 3.0]
    ys = [0, -0.68, -0.85, -0.70, 0.15, 0.50]
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-1.3, 1.35)
    ax.axhline(0, color='k', lw=0.8, zorder=1)
    ax.axvspan(0, 1.0, color=SHADE, alpha=0.55, lw=0, zorder=0)
    ax.plot(xs, ys, color=LINE, lw=2.4, zorder=4, solid_joinstyle='round',
            solid_capstyle='round')
    ax.scatter(xs, ys, s=32, color=LINE, zorder=5, edgecolors='white', linewidths=0.5)
    ax.axvline(2.0, color=C_COUNTER, ls='--', lw=1.8, zorder=3, ymax=0.83)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Jan 18', 'Jan 19', 'Jan 20', 'Jan 21'])
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
    ax.set_ylabel('Opinion attitude\n(negative → positive)', fontsize=11)

    # Info 1 → 起点
    ax.annotate('Info 1: short video spread (Jan 18)',
                xy=(-0.02, 0.02), xytext=(-0.42, 0.60), ha='left', va='center',
                fontsize=10, color=C_INFO,
                arrowprops=dict(ARROW_KW, color=C_INFO))
    # Rapid negative judgment（阶段：Jan 18 → Jan 19 的下降段，双向箭头）
    phase_marker(ax, 0.0, 1.0, -1.05, 'Rapid negative judgment\n(fragmented information)',
                 C_RAPID, text_dy=-0.05, va='top')
    # Counter-info → 虚线（文字在虚线左上方，斜箭头指向虚线）
    ax.annotate('Counter-info: full video and\nother angles (Jan 20)',
                xy=(1.95, 0.62), xytext=(1.55, 0.95), ha='center', va='center',
                fontsize=10, color=C_COUNTER,
                arrowprops=dict(ARROW_KW, color=C_COUNTER))
    # Opinion reversal（阶段：Jan 20 → Jan 21 的反转段，双向箭头）
    phase_marker(ax, 2.0, 3.0, 0.75, 'Opinion reversal\n(critics apologized)',
                 C_REV, fontsize=10.5, fontweight='bold', text_dy=0.05, va='bottom')
    style_ax(ax)


def main():
    # 合并图：上=(a) 单轴断裂轴，下=(b) 单轴连续
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.4), gridspec_kw=dict(hspace=0.30))
    liu_case(axes[0])
    covington_case(axes[1])
    fig.savefig(f'{IMG}/Fig10.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 子图（PDF 版式用，并排 a/b；两图统一 figsize，保证并排时大小一致）
    fa, axa = plt.subplots(figsize=(8.2, 4.4))
    liu_case(axa)
    fa.tight_layout()
    fa.savefig(f'{IMG}/subfigs/Fig10_a.png', dpi=300, bbox_inches='tight',
               facecolor='white')
    plt.close(fa)
    fb, axb = plt.subplots(figsize=(8.2, 4.4))
    covington_case(axb)
    fb.tight_layout()
    fb.savefig(f'{IMG}/subfigs/Fig10_b.png', dpi=300, bbox_inches='tight',
               facecolor='white')
    plt.close(fb)
    print('Fig10.png + Fig10_a/b.png 已生成（(a) 单轴断裂轴 + 曲线连续，整体拉长）')


if __name__ == '__main__':
    main()
