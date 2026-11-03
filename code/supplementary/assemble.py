
import os, shutil
from PIL import Image

BASE = '/Users/np/Desktop/小论文/重跑_正确语义版'
OUT = os.path.join(BASE, '代码/补充实验/output')
ATT = os.path.join(BASE, '图片/attachments')
SUBF = os.path.join(ATT, 'subfigs')
os.makedirs(SUBF, exist_ok=True)

def cp(src, dst):
    s = os.path.join(OUT, src)
    if not os.path.exists(s):
        print(f'  [跳过] {src} 不存在')
        return
    shutil.copy2(s, os.path.join(ATT, dst))
    print(f'{src} -> attachments/{dst}')

def cp_sub(src, dst):
    s = os.path.join(OUT, src)
    if not os.path.exists(s):
        print(f'  [跳过] {src} 不存在')
        return
    shutil.copy2(s, os.path.join(SUBF, dst))
    print(f'{src} -> subfigs/{dst}')

def split2x2(src, fig):
    s = os.path.join(OUT, src)
    if not os.path.exists(s):
        print(f'  [跳过] {src} 不存在')
        return
    im = Image.open(s).convert('RGB')
    W, H = im.size
    for r in range(2):
        for c in range(2):
            letter = chr(97 + r * 2 + c)
            sub = im.crop((c * W // 2, r * H // 2, (c + 1) * W // 2, (r + 1) * H // 2))
            sub.save(os.path.join(SUBF, f'{fig}_{letter}.png'), dpi=(300, 300))
            print(f'{src} -> subfigs/{fig}_{letter}.png')

print('=== 合并图 ===')
cp('E2_fixedS_bar.png', 'Fig4.png')
cp('fig3_4groups.png', 'Fig5.png')
cp('fig4_4groups.png', 'Fig6.png')
cp('fig3_combined.png', 'FigS3.png')
cp('speed_combined.png', 'FigS4.png')
cp('fss_combined.png', 'FigS5.png')
cp('fss_ic_combined.png', 'FigS6.png')
cp('fig_S7_chi_pre_N.png', 'FigS7.png')
cp('E3_Sdist.png', 'FigS8.png')
cp('E1_tinj_I2.png', 'FigS9.png')

print('=== 子图 ===')
split2x2('fig3_4groups.png', 'Fig5')
cp_sub('fig3_combined.png', 'FigS3.png')
cp_sub('speed_panel_a.png', 'FigS4_a.png')
cp_sub('speed_panel_b.png', 'FigS4_b.png')
cp_sub('fss_panel_a.png', 'FigS5_a.png')
cp_sub('fss_panel_b.png', 'FigS5_b.png')
cp_sub('fss_panel_c.png', 'FigS5_c.png')
cp_sub('fss_ic_panel_a.png', 'FigS6_a.png')
cp_sub('fss_ic_panel_b.png', 'FigS6_b.png')
cp_sub('fig_S7_chi_pre_N.png', 'FigS7.png')
cp_sub('E3_Sdist.png', 'FigS8.png')
cp_sub('E1_tinj_I2.png', 'FigS9.png')

print('\n组装完成')
