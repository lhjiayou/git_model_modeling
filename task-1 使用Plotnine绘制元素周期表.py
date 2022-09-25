# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:57:47 2022

@author: 18721
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from plotnine import *
# 读入元素周期表数据
element = pd.read_csv('./data/element.csv')   
element  #[118 rows x 21 columns]


'''特征处理1，将缺失值填充为-1'''
#group特征就是该元素所在的族，发现group有的值中用 '-' 标记，说明它不属于任何族，它们应该是镧系或者锕系元素，'-' 符号最好用﹣1表示。
element.loc[element['group'] == '-', 'group'] = -1
element['group'] = element['group'].astype('int')
element['group'].unique()
element.info()
 0   atomic number             118 non-null    int64  
 1   symbol                    118 non-null    object 
 2   name                      118 non-null    object 
 3   atomic mass               118 non-null    object 
 4   CPK                       118 non-null    object 
 5   electronic configuration  118 non-null    object 
 6   electronegativity         97 non-null     float64
 7   atomic radius             71 non-null     float64
 8   ion radius                92 non-null     object 
 9   van der Waals radius      38 non-null     float64
 10  IE-1                      102 non-null    float64
 11  EA                        85 non-null     float64
 12  standard state            99 non-null     object 
 13  bonding type              98 non-null     object      文本类型
 14  melting point             101 non-null    float64
 15  boiling point             94 non-null     float64
 16  density                   96 non-null     float64
 17  metal                     118 non-null    object    文本类型
 18  year discovered           118 non-null    object 
 19  group                     118 non-null    int32  
 20  period                    118 non-null    int64  
 
 '''特征工程2：将文本转换成类别'''
element['bonding type'] = element['bonding type'].astype('category')
element['metal'] = element['metal'].astype('category')

# 将原子数atomic number变成字符串
element['atomic_number'] = element['atomic number'].astype(str)


# 一般来说，元素周期表有两个部分，top和bottom。top部分主要绘制的是能找到group 不等于 -1的元素，而bottom部分主要绘制的是group等于-1的元素
## 分别用top和bottom变量引用上下部分元素集合
top = element.loc[element['group'] != -1, :]   #90*22
bottom = element.loc[element['group'] == -1, :]  #28*22


# 元素周期表中横向表示的是族（group），纵向表示的是周期（period）
top['x'] = top['group'].copy()
top['y'] = top['period'].copy()


# 除了top的部分之外，bottom的锕系和镧系元素也是类似的，但是横坐标不能用 group ，因为bottom的group为﹣1。
# hs 和 vs 分别表示横、纵间距，这样可以为锕系和镧系元素设置横纵坐标值。
nrows = 2
hs = 3.5
vs = 3
bottom['x'] = np.tile(np.arange(len(bottom) // nrows), nrows) + hs  #需要人工设置
bottom['y'] = bottom.period + vs


#最终使用plotpine绘图
# 开始绘制元素周期表：
p1 = (ggplot(aes('x', 'y'))+ 
    geom_tile(top, aes(width=0.95, height=0.95))+
    geom_tile(bottom, aes(width=0.95, height=0.95))+
    scale_y_reverse()+
    aes(fill='metal')+  # 将tile颜色填充
    geom_text(top, aes(label='atomic_number'),nudge_x=-0.40, nudge_y=-.40,ha='left', va='top', fontweight='normal', size=6)+   # 添加top的atomic_number（原子序数）注释
    geom_text(top, aes(label='symbol'),nudge_y=.1, size=9)+      #  添加top的symbol（元素符号）注释
    geom_text(top, aes(label='name'),nudge_y=-0.125, fontweight='normal', size=4.5)+   #  添加top的name（元素名称）注释
    geom_text(top, aes(label='atomic mass'),nudge_y=-.3, fontweight='normal', size=4.5)+   # 添加top的atomic mass（原子数）注释
    geom_text(bottom, aes(label='atomic_number'),nudge_x=-0.40, nudge_y=-.40,ha='left', va='top', fontweight='normal', size=6)+
    geom_text(bottom, aes(label='symbol'),nudge_y=.1, size=9)+
    geom_text(bottom, aes(label='name'),nudge_y=-0.125, fontweight='normal', size=4.5)+
    geom_text(bottom, aes(label='atomic mass'),nudge_y=-.3, fontweight='normal', size=4.5)+
    coord_equal(expand=False)+ # 设置坐标轴， expand=False, 代表坐标系的大小由数据决定
    theme(figure_size=(16,8))    
)
print(p1)
