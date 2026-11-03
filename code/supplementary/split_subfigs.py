
import os
from PIL import Image
import numpy as np

ATT = '/Users/np/Desktop/小论文/重跑_正确语义版/源文件/attachments'
OUT = os.path.join(ATT, 'subfigs')
os.makedirs(OUT, exist_ok=True)

JOBS = [
    ('N8JFPVTQ.png', 1, 2, 2, None),
    ('97MD7IXQ.png', 2, 2, 2, (0, 0, 70, 58)),
    ('JKSRBC82.png', 3, 2, 2, (170, 45, 275, 140)),
    ('6N4XE8KP.png', 5, 1, 2, None),
    ('DQ7UWZ35.png', 6, 2, 2, None),
    ('FYD66FB7.png', 7, 2, 3, None),
    ('W3ASF3LE.png', 8, 2, 1, None),
    ('7N5R75VT.png', 9, 1, 3, None),
]

PAD = 14

def trim(im):
    a = np.array(im.convert('L'))
    mask = a < 250
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return im
    y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD, a.shape[0])
    x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD, a.shape[1])
    return im.crop((x0, y0, x1, y1))

def erase_label(sub_im, box):
    a = np.array(sub_im.convert('RGB')).astype(int)
    x0, y0, x1, y1 = box
    region = a[y0:y1, x0:x1]
    mx = region.max(axis=2)
    mn = region.min(axis=2)
    grayish = (mx < 225) & ((mx - mn) < 45)
    region[grayish] = [255, 255, 255]
    return Image.fromarray(a.astype(np.uint8))

total = 0
for fname, fig, rows, cols, label_box in JOBS:
    im = Image.open(os.path.join(ATT, fname)).convert('RGB')
    W, H = im.size
    cw, rh = W // cols, H // rows
    idx = 0
    for r in range(rows):
        for c in range(cols):
            idx += 1
            letter = chr(96 + idx)
            x0, y0 = c * cw, r * rh
            sub = im.crop((x0, y0, x0 + cw, y0 + rh))
            if label_box is not None:
                sub = erase_label(sub, label_box)
            sub = trim(sub)
            out = os.path.join(OUT, f'Fig{fig}_{letter}.png')
            sub.save(out, dpi=(200, 200))
            print(f'Fig{fig}_{letter}.png  {sub.size[0]}x{sub.size[1]}')
            total += 1
print(f'\n共生成 {total} 个子图 -> {OUT}')

def split_si():
    n = 0

    def save(im, name):
        im.save(os.path.join(OUT, name), dpi=(200, 200))
        print(f'{name}  {im.size[0]}x{im.size[1]}')
        nonlocal n
        n += 1

    for src, name in [('N9ZV6D8H.png', 'FigS1.png'), ('L3YE4WXW.png', 'FigS2.png'),
                      ('SI_fig7_ic_compare.png', 'FigS7.png'),
                      ('SI_fig8_fixedS.png', 'FigS8.png'),
                      ('SI_fig9_Sdist.png', 'FigS9.png'),
                      ('SI_fig10_tinj_I2.png', 'FigS10.png')]:
        save(Image.open(os.path.join(ATT, src)).convert('RGB'), name)

    im = Image.open(os.path.join(ATT, 'SI_fig6_fss_ic.png')).convert('RGB')
    W, H = im.size
    for i in range(2):
        sub = trim(im.crop((i * W // 2, 0, (i + 1) * W // 2, H)))
        save(sub, f'FigS6_{chr(97+i)}.png')

    im = Image.open(os.path.join(ATT, 'SI_fig4_logistic.png')).convert('RGB')
    W, H = im.size
    for i in range(2):
        sub = im.crop((i * W // 2, 0, (i + 1) * W // 2, H))
        sub = erase_label(sub, (182, 50, 272, 135))
        save(trim(sub), f'FigS4_{chr(97+i)}.png')

    im = Image.open(os.path.join(ATT, 'SI_fig5_fss.png')).convert('RGB')
    W, H = im.size
    for i in range(3):
        sub = im.crop((i * W // 3, 0, (i + 1) * W // 3, H))
        sub = erase_label(sub, (170, 55, 258, 130))
        save(trim(sub), f'FigS5_{chr(97+i)}.png')

    im = Image.open(os.path.join(ATT, 'SI_fig3_10groups.png')).convert('RGB')
    W, H = im.size
    MW = 2890
    cw, rh = MW // 2, H // 2

    boxes = {(0, 0): (180, 45, 268, 130), (0, 1): (205, 45, 285, 130),
             (1, 0): (180, 45, 268, 130), (1, 1): (205, 45, 285, 130)}
    subs = {}
    for r in range(2):
        for c in range(2):
            sub = im.crop((c * cw, r * rh, (c + 1) * cw, (r + 1) * rh))
            subs[(r, c)] = erase_label(sub, boxes[(r, c)])

    cb = trim(im.crop((2895, 0, 3140, H)))
    save(cb, 'FigS3_cb.png')
    for r in range(2):
        for c in range(2):
            letter = chr(97 + r * 2 + c)
            save(trim(subs[(r, c)]), f'FigS3_{letter}.png')

    print(f'\nSI 共生成 {n} 个文件 -> {OUT}')

split_si()
