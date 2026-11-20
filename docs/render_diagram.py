"""Render the training diagram as editable SVG and PNG. Requires Pillow only."""
from pathlib import Path
from html import escape
import math
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
W,H=1600,2070
im=Image.new('RGB',(W,H),'#f5f7fb');draw=ImageDraw.Draw(im)
fontroot=Path('C:/Windows/Fonts')
def font(size,bold=False):
    if fontroot.exists(): return ImageFont.truetype(str(fontroot/('arialbd.ttf' if bold else 'arial.ttf')),size)
    return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',size)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     '<rect width="100%" height="100%" fill="#f5f7fb"/>']
def rect(x,y,w,h,fill,stroke='#d2dbe8',r=15):
    draw.rounded_rectangle((x,y,x+w,y+h),r,fill=fill,outline=stroke,width=2)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def text(x,y,value,size=23,bold=False,color='#1e293b',center=True):
    f=font(size,bold);anchor='mm' if center else 'lm'
    draw.text((x,y),value,font=f,anchor=anchor,fill=color)
    svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial,DejaVu Sans,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{"middle" if center else "start"}" dominant-baseline="central">{escape(value)}</text>')
def box(x,y,w,h,lines,fill='#ffffff',stroke='#d2dbe8',size=22):
    rect(x,y,w,h,fill,stroke)
    spacing=size+10
    for i,line in enumerate(lines): text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*spacing,line,size,i==0)
def arrow(points,color='#64748b'):
    draw.line(points,fill=color,width=3)
    end=points[-1];pre=points[-2];a=math.atan2(end[1]-pre[1],end[0]-pre[0])
    tri=[end,(end[0]-12*math.cos(a-.45),end[1]-12*math.sin(a-.45)),(end[0]-12*math.cos(a+.45),end[1]-12*math.sin(a+.45))]
    draw.polygon(tri,fill=color)
    coords=' '.join(f'{x},{y}' for x,y in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
    svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in tri)}" fill="{color}"/>')

text(65,65,'ALFA CONNECTOME · ОБУЧЕНИЕ И ИНФЕРЕНС',36,True,center=False)
text(65,109,'Выбранное решение: 1 GRU + AE/MLP · Public WP 0.65490 · 646 / 5800',24,center=False,color='#52627b')
box(365,155,870,100,['train.parquet: 10 607 × 20 000 строк','Исходный fold: 8 485 train + 2 122 holdout'],fill='#e7eefb')
arrow([(800,255),(800,285)])
box(365,285,870,90,['Одна зафиксированная нормализация','mean/std по 256 последовательностям fold-train; clip ±10'],size=22)
rect(45,420,720,880,'#edf4ff','#b7cae5')
rect(805,420,750,880,'#f3effb','#cfc0e5')
text(405,450,'ВЕТВЬ GRU · все 10 607',27,True,color='#1d4e89')
text(1180,450,'ВЕТВЬ AE20 · 8 485 + holdout 2 122',26,True,color='#6b3a91')
arrow([(540,375),(540,395),(405,395),(405,420)])
arrow([(1060,375),(1060,395),(1180,395),(1180,420)])
box(85,485,640,95,['Обучение с нуля · seed 20260928','112 → GRU(128) → GRU(128) → Linear(2)'],size=22)
arrow([(405,580),(405,615)])
box(85,615,640,120,['Эпохи 1–4 · LR 0.0003','Weighted MSE · AdamW · EMA весов 0.999','16 последовательностей / TBPTT 512'],size=23)
arrow([(405,735),(405,775)])
box(150,775,510,80,['EMA-чекпойнт эпохи 4'],fill='#dce9ff',stroke='#6e99cc',size=25)
arrow([(405,855),(405,895)])
box(85,895,640,145,['Эпоха 5 · LR 0.0001','Старт с EMA4, а не с raw-весов','Новые AdamW / scaler / EMA / RNG','Те же 10 607 последовательностей'],size=22)
arrow([(405,1040),(405,1080)])
box(150,1080,510,80,['Финальные EMA-веса эпохи 5'],fill='#dce9ff',stroke='#6e99cc',size=24)
text(405,1210,'Фиксированные эпохи; без выбора по holdout.',21)
text(405,1244,'Рецепт ранее выбирался на исходном fold.',21)

box(840,485,680,95,['Причинная история: 112 × 5 = 560','z; Δ1; Δ20; z−EMA20; z−EMA100'],size=23)
arrow([(1180,580),(1180,615)])
box(840,615,680,100,['Encoder: 560 → 128 → 64 · ReLU','Шум std 0.03 только на входе Encoder'],fill='#ebe1f7',stroke='#b09acb',size=23)
arrow([(1010,715),(1010,760)])
arrow([(1360,715),(1360,760)])
box(840,760,330,110,['Основная MLP','[чистые 560 + latent 64]','624 → 128 → 64 → 2'],size=19)
arrow([(840,530),(820,530),(820,815),(840,815)])
box(1200,760,320,110,['Decoder','64 → 128 → 560','Восстановление входа'],size=21)
arrow([(1360,870),(1360,900)])
box(1200,900,320,80,['Auxiliary head','560 → 2'],size=21)
arrow([(1005,870),(1005,1010)])
arrow([(1360,980),(1360,1010)])
box(840,1010,680,100,['Общее обучение трёх задач','WMSE(main) + 0.25 WMSE(aux) + 0.1 MSE(rec)'],size=21)
arrow([(1200,815),(1185,815),(1185,1010)])
arrow([(1180,1110),(1180,1140)])
box(840,1140,680,130,['Выбрана AE эпохи 20 по holdout','AdamW + ReduceLROnPlateau; без EMA весов','В исходном запуске: 17 × 0.0003 + 3 × 0.00015'],fill='#ebe1f7',stroke='#b09acb',size=21)

rect(45,1350,1510,440,'#eaf5f0','#b7d8c8')
text(800,1385,'ИНФЕРЕНС · обучение моделей завершено',28,True,color='#25634b')
arrow([(660,1120),(745,1120),(745,1320),(405,1320),(405,1430)],'#356b99')
arrow([(1180,1270),(1180,1430)],'#78539b')
box(130,1430,550,85,['Прогноз GRU: 2 × tanh','Hidden обновляется на каждом шаге'],size=23)
box(890,1430,580,85,['Encoder + основная MLP: 2 × tanh','Без Decoder, auxiliary, шума и dropout'],size=22)
arrow([(405,1515),(405,1540),(620,1540),(620,1570)])
arrow([(1180,1515),(1180,1540),(980,1540),(980,1570)])
box(360,1570,880,90,['t0 = 0.70 × GRU + 0.30 × AE','t1 = 0.65 × GRU + 0.35 × AE'],fill='#d5edde',stroke='#7ab398',size=26)
arrow([(800,1660),(800,1695)])
box(125,1695,1350,65,['ONNX float32 · 21 узел · C-история · 1 CPU · состояние сбрасывается между seq'],size=22)

box(125,1830,1350,95,['Отдельная official validation: 1 873 последовательности','Маска need_prediction & is_scored; Global WP по всем выбранным строкам'],size=23)
text(800,1975,'Warm-up: шаги 0–98 обновляют состояние, но прогноз не возвращается.',23)
text(800,2014,'EMA весов GRU и EMA признаков истории — разные механизмы.',23,True)
svg.append('</svg>')
(OUT/'training_flow.svg').write_text('\n'.join(svg),encoding='utf-8')
im.save(OUT/'training_flow.png')
print(OUT/'training_flow.svg')
print(OUT/'training_flow.png')
