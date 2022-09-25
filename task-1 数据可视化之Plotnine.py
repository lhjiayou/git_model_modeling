# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 22:44:53 2022

@author: 18721
"""

# 假设检验、方差分析、回归分析与分类分析  是不是应该看一下
# python三大数据可视化工具库的简介：Matplotlib、Seaborn和Plotnine   用的最多的还是matplotlib，其实seaborn也还不错



'''1.什么是数据可视化'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from plotnine import *
# %matplotlib inline   对于spyder来说其实并不需要
plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']  #设置字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import seaborn as sns 
df = pd.DataFrame({
    'variable': ['gender', 'gender', 'age', 'age', 'age', 'income', 'income', 'income', 'income'],
    'category': ['Female', 'Male', '1-24', '25-54', '55+', 'Lo', 'Lo-Med', 'Med', 'High'],
    'value': [60, 40, 50, 30, 20, 10, 25, 25, 40],
})  #根据字典创建dataframe
df.info()  #查看目前的数据类型
 # 0   variable  9 non-null      object   目前是本文数据
 # 1   category  9 non-null      object
 # 2   value     9 non-null      int64 

#转换成分类数据
df['variable'] = pd.Categorical(df['variable'], categories=['gender', 'age', 'income'])
df['category'] = pd.Categorical(df['category'], categories=df['category'])
df.info()
# 0   variable  9 non-null      category   此时已经转换成分类数据了
# 1   category  9 non-null      category
# 2   value     9 non-null      int64  

#df
#   variable category  value
# 0   gender   Female     60
# 1   gender     Male     40
# 2      age     1-24     50
# 3      age    25-54     30
# 4      age      55+     20
# 5   income       Lo     10
# 6   income   Lo-Med     25
# 7   income      Med     25
# 8   income     High     40

'''1.堆叠柱状图'''
from plotnine import *
(
    ggplot(df, aes(x='variable', y='value', fill='category'))+
    geom_col()
)
'''2.柱状图'''
from plotnine import *
(
    ggplot(df, aes(x='variable', y='value', fill='category'))+
    geom_col(stat='identity', position='dodge')
)
#堆叠柱状图能够表现出相对大小的情况






'''2.Python三大数据可视化工具库的简介：Matplotlib、Seaborn和Plotnine'''
#目前我最常见的还是使用matplotlib工具包
# （1）Matplotlib：  其实存在两种风格
'''eg1'''
# 创建数据
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,6))
plt.scatter(x=x, y=y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = sin(x)')
plt.show()
'''eg2：其实每一次plt.figure()都会生成一个新的对象，然后在新的对象上画图'''
# 准备数据
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
# 绘制第一个图：
fig1 = plt.figure(figsize=(6,4), num='first')
fig1.suptitle('y = sin(x)')
plt.scatter(x=x, y=y1)
plt.xlabel('x')
plt.ylabel('y')
# 绘制第二个图：
fig2 = plt.figure(figsize=(6,4), num='second')
fig2.suptitle('y = cos(x)')
plt.scatter(x=x, y=y2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

'''Seaborn是在Matplotlib的基础上的再次封装，是对Matplotlib绘制统计图表的简化。下面，我们一起看看Seaborn的基本绘图逻辑。'''
# （2）Seaborn：
# Seaborn主要用于统计分析绘图的，它是基于Matplotlib进行了更高级的API封装。Seaborn比matplotlib更加易用，尤其在统计图表的绘制上
# ，因为它避免了matplotlib中多种参数的设置。Seaborn与matplotlib关系，可以把Seaborn视为matplotlib的补充。
'''下面的例子是两个库的对比'''
'''matplotlib的方式'''
# 准备数据
x = np.linspace(-10, 10, 100)
y = 2 * x + 1 + np.random.randn(100)
df = pd.DataFrame({'x':x, 'y':y})
# 使用matplotlib绘制带有拟合直线效果的散点图
func = np.polyfit(x,y,1)  # 拟合直线
# func  输出的是array([1.98193639, 1.01024378])，其实就是拟合函数的两个参数值
poly = np.poly1d(func)  # 设置拟合函数，这个好像不是一个变量，而是一个函数了，这些其实在sklearn中可以调用线性模型实现
y_pred = poly(x) # 预测
plt.scatter(x, y)  # 绘制点图，原始数据
plt.plot(x, y_pred)  # 绘制拟合直线图，拟合数据
plt.xlabel('x')
plt.ylabel('y')
plt.show()

'''sns的方式，就是一行调用即可'''
sns.lmplot(x="x",y="y",data=df)   #默认的order是1

#如果我们设置的是二次函数，sns会自动分析出order是2吗？
x = np.linspace(-10, 10, 100)
yyy = 2 * x**2+ 2 * x + 1 + np.random.randn(100)
df = pd.DataFrame({'x':x, 'yyy':yyy})
sns.lmplot("x","yyy",data=df)   #看来还是需要手动设置order
sns.lmplot("x","yyy",data=df,order=2) 

# （3）Plotnine：
'''为什么这个好用？
ggplot2奠定了R语言数据可视化在R语言数据科学的统治地位，R语言的数据可视化是大一统的，提到R语言数据可视化首先想到的就是ggplot2。
数据可视化一直是Python的短板，即使有Matplotlib、Seaborn等数据可视化包，也无法与R语言的ggplot2相媲美，原因在于当绘制复杂图表时，
Matplotlib和Seaborn由于“每一句代码都是往纸上添加一个绘图特征”的特性而需要大量代码语句。Plotnine可以说是ggplot2在Python上的移植版，使
用的基本语法与R语言ggplot2语法“一模一样”，使得Python的数据可视化能力大幅度提升，为什么ggplot2和Plotnine会更适合数据可视化呢？
原因可以类似于PhotoShop绘图和PPT绘图的区别，与PPT一笔一画的绘图方式不同的是，PhotoShop绘图采用了“图层”的概念，
每一句代码都是相当于往图像中添加一个图层，一个图层就是一类绘图动作，这无疑给数据可视化工作大大减负，同时更符合绘图者的认知。'''
from plotnine import *     # 将Plotnine所有模块引入
from plotnine.data import mpg  # 引入PLotnine自带数据集 234*11的数据集
# mpg数据集记录了美国1999年和2008年部分汽车的制造厂商，型号，类别，驱动程序和耗油量。
mpg.columns  #['manufacturer', 'model', 'displ', 'year', 'cyl', 'trans', 'drv', 'cty','hwy', 'fl', 'class']

# 绘制汽车在不同驱动系统下，发动机排量与耗油量的关系
p1 = (
    ggplot(mpg, aes(x='displ', y='hwy', color='drv'))     
    # 设置数据映射图层，数据集使用mpg，x数据使用mpg['displ']，y数据使用mpg['hwy']，颜色映射使用mog['drv']
    + geom_point()       
    # 绘制散点图图层
    + geom_smooth(method='lm')        
    # 绘制平滑线图层
    + labs(x='displacement', y='horsepower')     
    # 绘制x、y标签图层
)
print(p1)   # 展示p1图像
'''Plotnine的绘图逻辑是：一句话一个图层！！！！！！！
因此，在Plotnine中少量的代码就能画图非常漂亮的图表，而且可以画出很多很复杂的图表'''

#注意：一旦设置ggplot风格，在接下来的所有运行的notebook代码绘图的风格都是ggplot，除非我们重启环境，才会回复默认风格。
x = np.linspace(-10, 10, 100)
y = 2 * x + 1 + np.random.randn(100)
df = pd.DataFrame({'x':x, 'y':y})

# 使用matplotlib绘制带有拟合直线效果的散点图
plt.style.use("ggplot")   #风格使用ggplot
func = np.polyfit(x,y,1)  # 拟合直线
poly = np.poly1d(func)  # 设置拟合函数
y_pred = poly(x) # 预测
plt.scatter(x, y)  # 绘制点图
plt.plot(x, y_pred)  # 绘制拟合直线图
plt.xlabel('x')
plt.ylabel('y')
plt.show()





'''3.Matplotlib绘图基础'''
# Matplotlib 图表的组成元素包括: 图形 (figure)、坐标图形 (axes)、图名 (title)、图例 (legend)、 主要刻度 (major tick)、
# 次要刻度 (minor tick)、主要刻度标签(major tick label)、次要刻度标签 (minor tick label)、 Y轴名 (Y axis label)、
# X轴名 ( X axis label)、边框图 (line)、数据标记 (markers)、网格 (grid) 线等。具体如图所示。
'''一般来说，Mayplotlib绘图元素包括：基本绘图类型与容器绘图类型！！！
基本绘图类型（graphic primitives）：点 (marker)、线 (line)、文本 (text)、图例 (legend)、网格线 (grid)、 标题 (title)、图片 (image) 等
基本绘图类型其实就是matlab风格的每一个笔画

（2）容器绘图类型（containers）：
Figure：最重要的元素，代表整个图像，所有的其他元素都是绘制在其上（如果有多个子图，子图也绘制在figure上）。Figure 对象包含一些特殊的 artist 对象，如图名 (title)、图例 (legend)。
Axes：第二重要的元素，代表 subplot（子图），数据都是显示在这个区域。一个Figure至少含有一个Axes对象，当绘制多个子图时
Axis：代表坐标轴对象，本质是一种带装饰的 spines，一般分为 xaxis 和 yaxis，Axis对象主要用于控制数据轴上的刻度位置和显示数值。
可见axes更像是子图的概念，而axis就是非常具体的坐标轴
Spines：表示数据显示区域的边界，可以显示或不显示。
Artist：表示任何显示在 Figure 上的元素，Artist 是很通用的概念，几乎任何需要绘制的元素都可以当成是 Artist，但是一个 Artist 只能存在于一个 Axes 之上。
'''

'''一般来说，要使用Matplotlib画出一副图表，需要设置一个容器绘图类型，
再在容器内添加基本绘图对象，如数据元素：点 (marker)、线 (line)、文本 (text)、图例 (legend)、网格线 (grid)、 标题 (title)、图片 (image) 等。
除了数据元素以外，还需要设置图表元素，包括图表尺寸、坐标轴的轴名及其标签、刻度、图例、网格线等。

'''
#eg：ticks代表在哪个位置修改，labels代表修改的具体值是多少还是很有必要的，如果不写这个，那么坐标上就是默认的小数
# 准备数据
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
# 开始绘图
fig = plt.figure(figsize=(16,6))
ax1 = fig.add_axes([0.1,0.1,0.4,0.8])  # [left, bottom, width, height], 它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度
plt.scatter(x=x, y=y1, label=r"$y=sin(x)$")   # 在axes1中绘制y=sin(x)散点图, label是为了legend能够知道是哪个绘图的图例
plt.title("y = sin(x)")   # 在axes1设置标题
plt.xlabel("x")    # 在axes1中设置x标签
plt.ylabel("y")    # 在axes1中设置y标签
plt.axis(xmin=-2*np.pi, xmax=2*np.pi, ymin=-1, ymax=1)  # 在axes1中设置x轴和y轴显示范围
# plt.xticks(
#     ticks=[-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 3/2*np.pi, 2*np.pi], 
#     labels=[r'$-2 \pi$', r'$-\frac{3 \pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3 \pi}{2}$', r'$2\pi$']
#     )   # 在axes1中设置x轴刻度的值，ticks代表在哪个位置修改，labels代表修改的具体值是多少
plt.grid(b=True, which='both')  # 在axes1中设置设置网格线
plt.legend(loc=1)   # 在axes1中设置图例，

ax2 = fig.add_axes([0.6,0.1,0.4,0.8])  # [left, bottom, width, height], 它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度
plt.scatter(x=x, y=y2, label=r"$y=cos(x)$")   # 在axes2中绘制y=cos(x)散点图，label是为了legend能够知道是哪个绘图的图例
plt.title("y = cos(x)")  # 在axes2设置标题
plt.xlabel("x")     # 在axes2中设置x轴标签
plt.ylabel("y")    # 在axes2中设置y标签
plt.axis(xmin=-2*np.pi, xmax=2*np.pi, ymin=-1, ymax=1)   # 在axes2中设置x轴和y轴显示范围
# plt.xticks(
#     ticks=[-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 3/2*np.pi, 2*np.pi], 
#     labels=[r'$-2 \pi$', r'$-\frac{3 \pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3 \pi}{2}$', r'$2\pi$']
#     )  # 在axes2中设置x轴刻度的值，ticks代表在哪个位置修改，labels代表修改的具体值是多少
plt.grid(b=True, which='both')   # 在axes2中设置设置网格线
plt.legend(loc=1)   # 在axes2中设置图例，loc表示的是图例的位置

plt.show() #除了plt.show之外，其它的语句其实每个子图都应该写一次



'''4.Plotnine绘图基础'''
# Plotnine绘图使用图层的概念，其中Plotnine中的图层可以分为：必备图层和可选图层。
'''
（1）必备图层：ggplot()图层与geom_xxx()/stat_xxx()图层：
1-1ggplot()图层：底层绘图函数，ggplot()函数可以将绘图和数据分离，在ggplot内可以设置数据以及数据的映射，
如：ggplot(data, aes(x='col_x', y='y_value', fill='col_class'))。
因此，在ggplot()中，除了设置数据外，还可以设置变量的映射aes()，用来表示x和y，还可以在aes()内控制颜色color、大小size和形状shape等等。

1-2geom_xxx()图层：几何对象，即我们在图中实际看到的图形元素，比如：
点图geom_point()、柱状图geom_bar()、折线图geom_line()、直方图geom_histogram()等等。
同样的，我们也可以仅仅通过改变几何对象来生成不同的几何图形。
通常来说，通常只使用geom_xxx()就可以绘制绝大多数的统计图表，但是如果涉及复杂的统计变换，那么则需要使用stat_xxx()图层。

1-3stat_xxx()图层：统计变换图层，比如求均值，求方差等，当我们需要展示出某个变量的某种统计特征的时候，则需要用到统计变换。'''
#eg1：
# 准备数据
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)
y_label = np.random.choice(['1','2','3'], 100)   #给他们随机设置标签
data = pd.DataFrame({'x':x, 'y':y, 'label':y_label})
# 绘制散点图geom_point()
p1 = (
    ggplot(data, aes(x='x', y='y', fill='y_label', size='y', shape='y_label'))+  #颜色和shape都是和标签相关的，size是与y正相关的
    geom_point()  #散点图
)
print(p1)


'''（2）可选图层：必备图层其实已经囊括了绝大对数绘图函数，基本图形已经可以绘制。
包括：scale_xxx()、facet_xxx()、guides_xxx()、coord_xxx()以及theme()。'''
# 2-1scale_xxx()图层：标度（scale）是用于调整数据映射的图形属性，scale_xxx()获取数据并对其进行调整以适应视觉的不同方面，即长度、颜色、大小和形状等。一般来说，scale_xxx()的基本格式为：scale_映射类型_数据类型（）。其中，映射类型包括：xy轴，size, color(颜色), fill(填充颜色), shape, alpha(透明度), linetype；数据类型包括：连续型continuous，离散型discrete，自定义manual，同一型identity。
p1 = (
    ggplot(mpg, aes(x='displ', y='hwy', color='drv', shape='drv'))     
    # 设置数据映射图层，数据集使用mpg，x数据使用mpg['displ']，y数据使用mpg['hwy']，颜色映射使用mog['drv']
    + geom_point()       
    # 绘制散点图图层
    + labs(x='displacement', y='horsepower')     
    # x、y标签
    # + scale_shape_manual(values=('o', 's', '*'))  
    # 添加shape映射美化
)
print(p1)   # 展示p1图像

#eg:
# 绘制汽车在不同的车辆类型下，在不同的驱动系统，发动机排量与耗油量的关系
p1 = (
    ggplot(mpg, aes(x='displ', y='hwy', color='drv'))     # 设置数据映射图层，数据集使用mpg，x数据使用mpg['displ']，y数据使用mpg['hwy']，颜色映射使用mog['drv']
    + geom_point()       # 绘制散点图图层
    + labs(x='displacement', y='horsepower')     # 绘制x、y标签图层
    # + facet_grid('.~ class', labeller='label_value') # 按照车辆类型分面
)
print(p1)   # 展示p1图像


'''5.基本图表的Quick Start'''
# 5.1 类别型图表：
# （1）柱状图:  柱状图一般分为单系列柱状图、多系列柱状图、堆叠柱状图和百分比柱状图：
'''单柱状图'''
##方式1，用matplotlib绘图
 # Matplotlib绘制单系列柱状图：不同城市的房价对比
data = pd.DataFrame({'city':['深圳', '上海', '北京', '广州', '成都'], 'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]})

fig = plt.figure(figsize=(10,6))  #图形大小
ax1 = fig.add_axes([0.15,0.15,0.7,0.7])  # [left, bottom, width, height], 它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度
#axes代表的是子图
plt.bar(data['city'], data['house_price(w)'],    #数据
        width=0.6,   #柱子的宽度
        align='center',    #柱子的对其方式，中心还是比较合适的
        orientation='vertical',   #垂直的，不是水平柱状图
        label='城市')
"""
x 表示x坐标，数据类型为int或float类型，也可以为str
height 表示柱状图的高度，也就是y坐标值，数据类型为int或float类型
width 表示柱状图的宽度，取值在0~1之间，默认为0.8
bottom 柱状图的起始位置，也就是y轴的起始坐标
align 柱状图的中心位置，"center","lege"边缘
color 柱状图颜色
edgecolor 边框颜色
linewidth 边框宽度
tick_label 下标标签
log 柱状图y周使用科学计算方法，bool类型
orientation 柱状图是竖直还是水平，竖直："vertical"，水平条："horizontal"
"""
plt.title("不同城市的房价对比图")   # 在axes1设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.grid(b=True, which='both')  # 在axes1中设置设置网格线，蓝色的网格
for i in range(len(data)):
    plt.text(i-0.05,    #x坐标
             data.iloc[i,]['house_price(w)']+0.01,   #y坐标
             data.iloc[i,]['house_price(w)'],fontsize=13)   #文本内容
    # 添加数据注释
plt.legend()
plt.show()

#方式2，用plotpine绘图
data = pd.DataFrame({'city':['深圳', '上海', '北京', '广州', '成都'], 'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]})
p_single_bar = (
    ggplot(data, aes(x='city', y='house_price(w)', fill='city', label='house_price(w)'))+
    geom_bar(stat='identity')+
    labs(x="城市", y="房价(w)", title="不同城市的房价对比图")+
    geom_text(nudge_y=0.08)+
    theme(text = element_text(family = "Songti SC"))
)
print(p_single_bar)

'''多柱状图'''
## Matplotlib绘制多系列柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_axes([0.15,0.15,0.7,0.7])  # [left, bottom, width, height], 它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度
plt.bar(
    np.arange(len(np.unique(data['城市'])))-0.15, 
    data.loc[data['年份']==2021,'房价(w)'], 
    width=0.3, 
    align='center', 
    orientation='vertical', 
    label='年份：2021'    #左边的标签
    )
plt.bar(
    np.arange(len(np.unique(data['城市'])))+0.15, 
    data.loc[data['年份']==2022,'房价(w)'], 
    width=0.3, 
    align='center', 
    orientation='vertical', 
    label='年份：2022'   #右边的标签
    )
plt.title("不同城市的房价对比图")   # 在axes1设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.xticks(np.arange(len(np.unique(data['城市']))), np.array(['深圳', '上海', '北京', '广州', '成都']))
plt.grid(b=True, which='both')  # 在axes1中设置设置网格线

data_2021 = data.loc[data['年份']==2021,:]
for i in range(len(data_2021)):
    plt.text(i-0.15-0.05, data_2021.iloc[i,2]+0.05, data_2021.iloc[i,2],fontsize=13)   # 添加数据注释
    #plt.text(x,y,s)

data_2022 = data.loc[data['年份']==2022,:]
for i in range(len(data_2022)):
    plt.text(i+0.15-0.05, data_2022.iloc[i,2]+0.05, data_2022.iloc[i,2],fontsize=13)   # 添加数据注释
plt.legend()
plt.show()
## Plotnine绘制多系列柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})

data['年份'] = pd.Categorical(data['年份'], ordered=True, categories=data['年份'].unique())
p_mult_bar = (
    ggplot(data, aes(x='城市', y='房价(w)', fill='年份'))+
    geom_bar(stat='identity',width=0.6, position='dodge')+
    scale_fill_manual(values = ["#f6e8c3", "#5ab4ac"])+
    labs(x="城市", y="房价(w)", title="不同城市的房价对比图")+
    geom_text(aes(label='房价(w)'), position = position_dodge2(width = 0.6, preserve = 'single'))+
    theme(text = element_text(family = "Songti SC"))
)
print(p_mult_bar)

'''堆叠柱状图'''
## Matplotlib绘制堆叠柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
tmp=data.set_index(['城市','年份'])['房价(w)'].unstack()
data=tmp.rename_axis(columns=None).reset_index()
data.columns = ['城市','2021房价','2022房价']
print(data)

plt.figure(figsize=(10,6))
plt.bar(
    data['城市'], 
    data['2021房价'], 
    width=0.6, 
    align='center', 
    orientation='vertical', 
    label='年份：2021'
    )
plt.bar(
    data['城市'], 
    data['2022房价'], 
    width=0.6, 
    align='center', 
    orientation='vertical', 
    bottom=data['2021房价'],
    label='年份：2022'
    )
plt.title("不同城市2121-2022年房价对比图")   # 在axes1设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.legend()
plt.show()
## Plotnine绘制堆叠柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})

data['年份'] = pd.Categorical(data['年份'], ordered=True, categories=data['年份'].unique())
p_mult_bar = (
    ggplot(data, aes(x='城市', y='房价(w)', fill='年份'))+
    geom_bar(stat='identity',width=0.6, position='stack')+   # 只需要改变position='stack'
    scale_fill_manual(values = ["#f6e8c3", "#5ab4ac"])+
    labs(x="城市", y="房价(w)", title="不同城市2121-2022年房价对比图")+
    theme(text = element_text(family = "Songti SC"))
)
print(p_mult_bar)

'''百分比柱状图'''
## Matplotlib绘制百分比柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
tmp=data.set_index(['城市','年份'])['房价(w)'].unstack()
data=tmp.rename_axis(columns=None).reset_index()
data.columns = ['城市','2021房价','2022房价']
print(data)

plt.figure(figsize=(10,6))
plt.bar(
    data['城市'], 
    data['2021房价']/(data['2021房价']+data['2022房价']), 
    width=0.4, 
    align='center', 
    orientation='vertical', 
    label='年份：2021'
    )
plt.bar(
    data['城市'], 
    data['2022房价']/(data['2021房价']+data['2022房价']), 
    width=0.4, 
    align='center', 
    orientation='vertical', 
    bottom=data['2021房价']/(data['2021房价']+data['2022房价']),
    label='年份：2022'
    )
plt.title("不同城市2121-2022年房价对比图")   # 设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.legend()
plt.show()

## Plotnine绘制百分比柱状图：不同城市在不同年份的房价对比
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})

data['年份'] = pd.Categorical(data['年份'], ordered=True, categories=data['年份'].unique())
p_mult_bar = (
    ggplot(data, aes(x='城市', y='房价(w)', fill='年份'))+   
    geom_bar(stat='identity',width=0.6, position='fill')+     # 只需要改变position='fill'
    scale_fill_manual(values = ["#f6e8c3", "#5ab4ac"])+
    labs(x="城市", y="房价(w)", title="不同城市的房价对比图")+
    theme(text = element_text(family = "Songti SC"))
)
print(p_mult_bar)


# （2）火柴图：（棒棒糖图）
# 由于柱状图在表达数据的数值大小时使用的是不等高的长方形柱子，柱子会占据大量的绘图面积，因此当类别较多时，会出现画不下的情况。再者，柱子的宽度并没有表达什么信息，因此可以省略柱子或者将柱子替换为直线就可以节省大量的绘图空间，这样的图就是火柴图。
## 使用Matplotlib绘制火柴图（棒棒糖图）
data = pd.DataFrame({'city':['深圳', '上海', '北京', '广州', '成都'], 'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]})

plt.figure(figsize=(10,6))
markerline, stemlines, baseline = plt.stem(data['city'],data['house_price(w)'],bottom=0, label='城市')
plt.setp(markerline, color='red', marker='o',ms=8)  # marker点：火柴头 ms=markersize
plt.setp(stemlines, color='#FF9900', lw=3, ls=':' )    # 火柴杆  lw=linewidth
plt.setp(baseline, color='white', linewidth=2, ls='-')   # 基准线 ls=linestyle
plt.title("不同城市房价对比图")   # 设置标题
plt.xlabel("城市")    # 在axes1中设置x标签
plt.ylabel("房价/w")    # 在axes1中设置y标签
plt.legend()
plt.show()

## 使用Plotnine绘制火柴图（棒棒糖图）
data = pd.DataFrame({'city':['深圳', '上海', '北京', '广州', '成都'], 'house_price(w)':[3.5, 4.0, 4.2, 2.1, 1.5]})
data['city'] = pd.Categorical(data['city'], ordered=True, categories=data['city'].unique())

p1 = (
    ggplot(data, aes(x='city', y='house_price(w)'))+
    geom_segment(aes(x='city', y=0, xend='city', yend='house_price(w)'), linetype="dotted")+
    geom_point(shape='o', size=3, color='black', fill='#FD4E07')+
    theme(text = element_text(family = "Songti SC"))
)
print(p1)



# （3）哑铃图：

# 由于火柴杆图只能展示一个纬度的数值对比情况，如果想像多系列柱状图一样对比两个或多个因素的数值变化情况，又不想像柱状图一样浪费许多绘图空间，那么哑铃图是个不错的选择。
## Matplotlib绘制哑铃图：对比不同城市2021-2022年的房价情况
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
tmp=data.set_index(['城市','年份'])['房价(w)'].unstack()
data=tmp.rename_axis(columns=None).reset_index()
data.columns = ['城市','2021房价','2022房价']
print(data)

plt.figure(figsize=(10,6))
plt.vlines(
    data.loc[data['2022房价'] - data['2021房价'] >= 0, '城市'],  
    ymin=data.loc[data['2022房价'] - data['2021房价'] >= 0, '2021房价'],
    ymax=data.loc[data['2022房价'] - data['2021房价'] >= 0, '2022房价'],
    color='red',
    label='房价上涨',
    zorder=1, 
    lw=3,
    )                       # 绘制端点之间的连线
plt.vlines(
    data.loc[data['2022房价'] - data['2021房价'] <= 0, '城市'], 
    ymin=data.loc[data['2022房价'] - data['2021房价'] <= 0, '2021房价'],
    ymax=data.loc[data['2022房价'] - data['2021房价'] <= 0, '2022房价'],
    color='yellow',
    label='房价下跌',
    zorder=1, 
    lw=3,
    )                       # 绘制端点之间的连线
plt.scatter(x=data['城市'], y=data['2021房价'], color='#00589F', s=100,  label='2021房价')  # 绘制哑铃图的端点
plt.scatter(x=data['城市'], y=data['2022房价'], color='#F68F00', s=100,  label='2022房价')  # 绘制哑铃图的另一个端点
plt.title("不同城市2021-2022年房价对比图")   # 设置标题
plt.xlabel("城市")    # 设置x标签
plt.ylabel("房价/w")    # 设置y标签
plt.legend()
plt.show()

## Plotnine绘制哑铃图：对比不同城市2021-2022年的房价情况
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
data['年份'] = pd.Categorical(data['年份'], ordered=True, categories=data['年份'].unique())

p1 = (
    ggplot(data, aes(x='城市', y='房价(w)', fill='年份'))+
    geom_line(aes(group='城市'))+
    geom_point(shape='o', size=5, color='black')+
    scale_fill_manual(values=('#00AFBB', '#FC4E07', '#36BED9'))+
    theme(text = element_text(family = "Songti SC"))
)
print(p1)

# （4）坡度图：

# 以上的柱状图、多系列柱状图以及火柴杆和哑铃图等等都是机遇柱状图的推广，坡度图可以看多折线图的推广。坡度图可以很好的比较各个类别在两个不同时间点或者两种不同状态下的数值数据的变化，表现的内容与哑铃图大同小异。
## Matplotlib绘制坡度图：对比不同城市2021-2022年的房价情况
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
tmp=data.set_index(['城市','年份'])['房价(w)'].unstack()
data=tmp.rename_axis(columns=None).reset_index()
data.columns = ['城市','2021房价','2022房价']
print(data)

plt.figure(figsize=(10,6))
plt.vlines(x=[2021, 2022], ymin=0, ymax=len(data)+1, lw=3, color='gray', zorder=1)
for i in range(len(data)):
    plt.plot([2021, 2022], data.iloc[i, 1:3].values, color='black', zorder=2)
    plt.text(x=2021-0.4, y=data.iloc[i, 1], s=data.iloc[i, 0]+"房价: "+str(data.iloc[i, 1]))
    plt.text(x=2022+0.1, y=data.iloc[i, 2], s=data.iloc[i, 0]+"房价: "+str(data.iloc[i, 2]))
plt.scatter(x=[2021]*len(data), y=data['2021房价'], s=100, color='#00AFBB', zorder=3)
plt.scatter(x=[2022]*len(data), y=data['2022房价'], s=100, color='#FC4E07', zorder=3)
plt.xlim(2020.5,2022.5)
plt.xticks([2021,2022], ['2021', '2022'])
plt.yticks([]) # 不显示y轴
plt.xlabel("年份")
plt.show()


## Plotnine绘制坡度图：对比不同城市2021-2022年的房价情况
data = pd.DataFrame({
    '城市':['深圳', '上海', '北京', '广州', '成都', '深圳', '上海', '北京', '广州', '成都'],
    '年份':[2021,2021,2021,2021,2021,2022,2022,2022,2022,2022],
    '房价(w)':[3.5, 4.0, 4.2, 2.1, 1.5, 4.0, 4.2, 4.3, 1.6, 1.9]
})
tmp=data.set_index(['城市','年份'])['房价(w)'].unstack()
data=tmp.rename_axis(columns=None).reset_index()
data.columns = ['城市','2021房价','2022房价']
print(data)

left_label = data.apply(lambda x: x['城市']+": "+str(x['2021房价']), axis=1)
right_label = data.apply(lambda x: x['城市']+": "+str(x['2022房价']), axis=1)

p1 = (
    ggplot(data)+
    geom_segment(aes(x=1, xend=2, y='2021房价', yend='2022房价'), size=0.75, color='black', show_legend=False)+
    geom_vline(xintercept=1, linetype='solid', size=0.2)+
    geom_vline(xintercept=2, linetype='solid', size=0.2)+
    geom_point(aes(x=1, y='2021房价'), size=3, shape='o', fill='#00AFBB', color='black')+
    geom_point(aes(x=2, y='2022房价'), size=3, shape='o', fill='#FC4E07', color='black')+
    xlim(0.75, 2.25)+
    ylim(0.95*np.min(np.min(data[['2021房价', '2022房价']])), 1.05*np.max(np.max(data[['2021房价', '2022房价']])))+
    scale_x_discrete(limits = ("2021", "2022"))+
    xlab("年份")+
    ylab("房价")+
    geom_text(label=left_label, x=0.95, y=data['2021房价'], size=10, ha='right')+
    geom_text(label=right_label, x=2.05, y=data['2022房价'], size=10, ha='left')+
    geom_text(label='2021', x=1, y=1.05*np.max(np.max(data[['2021房价', '2022房价']])), size=12)+
    geom_text(label='2022', x=2, y=1.05*np.max(np.max(data[['2021房价', '2022房价']])), size=12)+
    theme(text = element_text(family = "Songti SC"))
)
print(p1)

# （5）雷达图：

# 雷达图可以展示多个变量在不同属性上的数值对比，如：十个员工在职场中五种能力的分数对比、英雄联盟每个英雄在每个属性的数值分数对比等等。由于Plotnine没有实现极坐标，因此只能使用Matplotlib绘制雷达图。
# 使用Matplotlib绘制雷达图：英雄联盟几位英雄的能力对比
data = pd.DataFrame({
    '属性': ['血量', '攻击力', '攻速', '物抗', '魔抗'],
    '艾希':[3, 7, 8, 2, 2],
    '诺手':[8, 6, 3, 6, 6]
})

plt.figure(figsize=(8,8))
theta = np.linspace(0, 2*np.pi, len(data), endpoint=False)   # 每个坐标点的位置
theta = np.append(theta, theta[0])  # 让数据封闭
aixi = np.append(data['艾希'].values,data['艾希'][0])  #让数据封闭
nuoshou = np.append(data['诺手'].values,data['诺手'][0])  # 让数据封闭
shuxing = np.append(data['属性'].values,data['属性'][0])  # 让数据封闭

plt.polar(theta, aixi, 'ro-', lw=2, label='艾希') # 画出雷达图的点和线
plt.fill(theta, aixi, facecolor='r', alpha=0.5) # 填充
plt.polar(theta, nuoshou, 'bo-', lw=2, label='诺手')  # 画出雷达图的点和线
plt.fill(theta, nuoshou, facecolor='b', alpha=0.5) # 填充
plt.thetagrids(theta/(2*np.pi)*360, shuxing)  # 为每个轴添加标签
plt.ylim(0,10)
plt.legend()
plt.show()

#散点图
# 使用Matplotlib和四个图说明相关关系：
x = np.random.randn(100)*10
y1 = np.random.randn(100)*10
y2 = 2 * x + 1 + np.random.randn(100)
y3 = -2 * x + 1 + np.random.randn(100)
y4 = x**2 + 1 + np.random.randn(100)


plt.figure(figsize=(12, 12))

plt.subplot(2,2,1)  #创建两行两列的子图，并绘制第一个子图
plt.scatter(x, y1, c='dodgerblue', marker=".", s=50)
plt.xlabel("x")
plt.ylabel("y1")
plt.title("y1与x不存在关联关系")

plt.subplot(2,2,2)  #创建两行两列的子图，并绘制第二个子图
plt.scatter(x, y2, c='tomato', marker="o", s=10)
plt.xlabel("x")
plt.ylabel("y2")
plt.title("y2与x存在正相关")

plt.subplot(2,2,3)  #创建两行两列的子图，并绘制第三个子图
plt.scatter(x, y3, c='magenta', marker="o", s=10)
plt.xlabel("x")
plt.ylabel("y3")
plt.title("y3与x存在负相关")

plt.subplot(2,2,4)  #创建两行两列的子图，并绘制第四个子图
plt.scatter(x, y4, c='deeppink', marker="s", s=10)
plt.xlabel("x")
plt.ylabel("y3")
plt.title("y4与x存在二次关系")

plt.suptitle('各种关系')
plt.show()


# 使用Plotnine和四个图说明相关关系：
x = np.random.randn(100)*10
y1 = np.random.randn(100)*10
y2 = 10 * x + 1 + np.random.randn(100)
y3 = -10 * x + 1 + np.random.randn(100)
y4 = x**2 + 1 + np.random.randn(100)

df = pd.DataFrame({
    'x': np.concatenate([x,x,x,x]),
    'y': np.concatenate([y1, y2, y3, y4]),
    'class': ['y1']*100 + ['y2']*100 + ['y3']*100 + ['y4']*100
})

p1 = (
    ggplot(df)+
    geom_point(aes(x='x', y='y', fill='class', shape='class'), color='black', size=2)+
    scale_fill_manual(values=('#00AFBB', '#FC4E07', '#00589F', '#F68F00'))+
    theme(text = element_text(family = "Songti SC"))
)
print(p1)


# （2）带趋势线的散点图：

# 一般来说，靠肉眼观察到的趋势性规律往往不具备说服力，在统计学中能够解释趋势关系的方法是：回归。因此，如果在散点图的基础上添加回归的趋势线，将会使得我们的关系更加具备说服力。回归方法在统计学中常有：参数型回归和非参数型回归，参数回归的代表为线性回归、多项式回归、指数回归、对数回归等等，而非参数回归的代表有LOESS数据平滑方法、GAM模型、样条数据平滑方法。
# 使用Matplotlib绘制具备趋势线的散点图
from sklearn.linear_model import LinearRegression  #线性回归等参数回归
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures  # 构造多项式
x = np.linspace(-10, 10, 100)
y = np.square(x) + np.random.randn(100)*100
x_poly2 = PolynomialFeatures(degree=2).fit_transform(x.reshape(-1, 1))


y_linear_pred = LinearRegression().fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
y_poly_pred = LinearRegression().fit(x_poly2, y).predict(x_poly2)
y_exp_pred = LinearRegression().fit(np.exp(x).reshape(-1, 1), y).predict(np.exp(x).reshape(-1, 1))
y_loess_pred = sm.nonparametric.lowess(x, y, frac=2/3)[:, 1]

plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_linear_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带线性趋势线的散点图")

plt.subplot(2,2,2)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_poly_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带二次趋势线的散点图")

plt.subplot(2,2,3)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_exp_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带指数趋势线的散点图")

plt.subplot(2,2,4)
plt.scatter(x, y, c='tomato', marker="o", s=10)
plt.plot(x, y_loess_pred, c='dodgerblue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("带 loess平滑线的散点图")
plt.suptitle('带拟合的散点图')
plt.show()


# 使用Matplotlib绘制QQ图与PP图
from scipy import stats
sample_norm = np.random.randn(100)   #构造正态分布的样本
sample_exp = np.random.exponential(1, 100) #构造指数分布的样本
sort_sample_norm = np.sort(sample_norm)  #假设为正态分布样本的分位数
sort_sample_exp = np.sort(sample_exp)  #假设为指数分布样本的分位数
theory_norm = stats.norm.ppf(np.arange(100) / 100) #正态分布理论分位数
sample_norm_cdf = stats.norm.cdf(sort_sample_norm) # 正态分布理论累积分布
sample_exp_cdf = stats.norm.cdf(sort_sample_exp) # 正态分布理论累积分布

plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.scatter(theory_norm, sort_sample_norm, c='tomato', marker="o", s=20)
plt.plot(theory_norm, theory_norm, c='dodgerblue')
plt.xlabel("理论分位数点")
plt.ylabel("样本分位数点")
plt.title("Q-Q图：样本基本服从标准正态分布")

plt.subplot(2,2,2)
plt.scatter(theory_norm, sort_sample_exp, c='tomato', marker="o", s=20)
plt.plot(theory_norm, theory_norm, c='dodgerblue')
plt.xlabel("理论分位数点")
plt.ylabel("样本分位数点")
plt.title("Q-Q图：样本不服从标准正态分布")

plt.subplot(2,2,3)
stats.probplot(sample_norm, plot=plt)
plt.xlabel("理论累积比例")
plt.ylabel("样本累积比例")
plt.title("P-P图：样本服从标准正态分布")

plt.subplot(2,2,4)
stats.probplot(sample_exp, plot=plt)
plt.xlabel("理论累积比例")
plt.ylabel("样本累积比例")
plt.title("P-P图：样本不服从标准正态分布")

plt.show()


## 使用Plotnine绘制QQ图
sample_norm = np.random.randn(100)   #构造正态分布的样本
sample_exp = np.random.exponential(1, 100) #构造指数分布的样本
df = pd.DataFrame({
    'sample_norm': sample_norm,
    'sample_exp': sample_exp
})

p1 = (
    ggplot(data=df, )+
    geom_qq(aes(sample='sample_norm'),shape='o', fill='#FC4E07', color='black', size=2)+
    geom_qq_line(aes(sample='sample_norm'), color='blue', size=1)+
    labs(x="理论分位数点", y="样本分位数点", title="Q-Q图：样本服从标准正态分布")+
    theme(text = element_text(family = "Songti SC"))
)

p2 = (
    ggplot(data=df, )+
    geom_qq(aes(sample='sample_exp'),shape='o', fill='#FC4E07', color='black', size=2)+
    geom_qq_line(aes(sample='sample_exp'), color='blue', size=1)+
    labs(x="理论分位数点", y="样本分位数点", title="Q-Q图：样本不服从标准正态分布")+
    theme(text = element_text(family = "Songti SC"))
)
print([p1, p2])


# 使用Matplotlib绘制聚类散点图
from sklearn.datasets import load_iris  #家在鸢尾花数据集
iris = load_iris()
X = iris.data
label = iris.target
feature = iris.feature_names
df = pd.DataFrame(X, columns=feature)
df['label'] = label

label_unique = np.unique(df['label']).tolist()
plt.figure(figsize=(10, 6))
for i in label_unique:
    df_label = df.loc[df['label'] == i, :]
    plt.scatter(x=df_label['sepal length (cm)'], y=df_label['sepal width (cm)'], s=20, label=i)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('sepal width (cm)~sepal length (cm)')
plt.legend()
plt.show()


# 使用Plotnine绘制聚类散点图
from sklearn.datasets import load_iris  #家在鸢尾花数据集
iris = load_iris()
X = iris.data
label = iris.target
feature = iris.feature_names
df = pd.DataFrame(X, columns=feature)
df['label'] = label

p1 = (
    ggplot(df, aes(x='sepal length (cm)', y='sepal width (cm)', fill='factor(label)'))+
    geom_point(alpha=0.2, color='black', size=2)+
    labs(title="sepal width (cm)~sepal length (cm)")+
    theme(text = element_text(family = "Songti SC"))
)
print(p1)


# 使用Matplotlib/Seaborn绘制相关系数矩阵图
uniform_data = np.random.rand(10, 12)
sns.heatmap(uniform_data)

# 使用plotnine绘制相关系数矩阵图：
from plotnine.data import mtcars
corr_mat = np.round(mtcars.corr(), 1).reset_index() #计算相关系数矩阵
df = pd.melt(corr_mat, id_vars='index', var_name='variable', value_name='corr_xy') #将矩阵宽数据变成长数据
df['abs_corr'] = np.abs(df['corr_xy'])
p1 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', size='abs_corr'))+
    geom_point(shape='o', color='black')+
    scale_size_area(max_size=11, guide=False)+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="Variable", y="Variable")+
    theme(dpi=100, figure_size=(4.5,4.55))
)
p2 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', size='abs_corr'))+
    geom_point(shape='s', color='black')+
    scale_size_area(max_size=10, guide=False)+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="Variable", y="Variable")+
    theme(dpi=100, figure_size=(4.5,4.55))
)
p3 = (
    ggplot(df, aes(x='index', y='variable', fill='corr_xy', label='corr_xy'))+
    geom_tile(color='black')+
    geom_text(size=8, color='white')+
    scale_fill_cmap(name='RdYIBu_r')+
    coord_equal()+
    labs(x="Variable", y="Variable")+
    theme(dpi=100, figure_size=(4.5,4.55))
)
print([p1, p2, p3])


# 使用matplotlib绘制直方图：
plt.figure(figsize=(8, 6))
plt.hist(mtcars['mpg'], bins=20, alpha=0.85)
plt.xlabel("mpg")
plt.ylabel("count")
plt.show()


# 使用plotnine绘制核密度图：
from plotnine.data import mtcars

p1 = (
    ggplot(mtcars, aes(x='mpg', fill='factor(carb)'))+
    geom_density(bw=1, alpha=0.6, color='black', size=0.25)+
    scale_fill_hue(s=0.90, l=0.65, h=0.0417, color_space='husl')
)
print(p1)


# 使用plotnine绘制的箱线图：
from plotnine.data import mtcars
p1 = (
    ggplot(mtcars, aes(x='carb', y='mpg', fill='factor(carb)'))+
    geom_boxplot(show_legend=False)+
    geom_jitter(fill='black', shape='.', width=0.2, size=3, stroke=0.1, show_legend=False)
)
print(p1)

# 使用matplotlib绘制箱线图
import seaborn as sns 
from plotnine.data import mtcars

data = mtcars
data['carb'] = data['carb'].astype('category')
plt.figure(figsize=(8, 6))
sns.boxenplot(x='carb', y='mpg', data=mtcars, linewidth=0.2, palette=sns.husl_palette(6, s=0.9, l=0.65, h=0.0417))
plt.show()


# 使用Plotnine绘制提琴图：
from plotnine.data import mtcars

p1 = (
    ggplot(mtcars, aes(x='factor(carb)', y='mpg', fill='factor(carb)'))+
    geom_violin(show_legend=False)+
    geom_boxplot(fill='white', width=0.1, show_legend=False)
   )
print(p1)


# 使用Matplotlib绘制饼状图：实心的
from matplotlib import cm, colors
df = pd.DataFrame({
    '己方': ['寒冰', '布隆', '发条', '盲僧', '青钢影'],
    '敌方': ['女警', '拉克丝', '辛德拉', '赵信', '剑姬'],
    '己方输出': [26000, 5000, 23000, 4396, 21000],
    '敌方输出': [25000, 12000, 21000, 10000, 18000]
})

df_our = df[['己方', '己方输出']].sort_values(by='己方输出', ascending=False).reset_index()
df_other = df[['敌方', '敌方输出']].sort_values(by='敌方输出', ascending=False).reset_index()
color_list = [cm.Set3(i) for i in range(len(df))]
plt.figure(figsize=(16, 10))
plt.subplot(1,2,1)
plt.pie(df_our['己方输出'].values, startangle=90, shadow=True, colors=color_list, labels=df_our['己方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')


plt.subplot(1,2,2)
plt.pie(df_other['敌方输出'].values, startangle=90, shadow=True, colors=color_list, labels=df_other['敌方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')

plt.show()



# 使用Matplotlib绘制环状图：
from matplotlib import cm, colors
df = pd.DataFrame({
    '己方': ['寒冰', '布隆', '发条', '盲僧', '青钢影'],
    '敌方': ['女警', '拉克丝', '辛德拉', '赵信', '剑姬'],
    '己方输出': [26000, 5000, 23000, 4396, 21000],
    '敌方输出': [25000, 12000, 21000, 10000, 18000]
})

df_our = df[['己方', '己方输出']].sort_values(by='己方输出', ascending=False).reset_index()
df_other = df[['敌方', '敌方输出']].sort_values(by='敌方输出', ascending=False).reset_index()
color_list = [cm.Set3(i) for i in range(len(df))]
wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
plt.figure(figsize=(16, 10))
plt.subplot(1,2,1)
plt.pie(df_our['己方输出'].values, startangle=90, shadow=True, colors=color_list, wedgeprops=wedgeprops, labels=df_our['己方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')
plt.text(0, 0, '己方' , ha='center', va='center', fontsize=30)

plt.subplot(1,2,2)
plt.pie(df_other['敌方输出'].values, startangle=90, shadow=True, colors=color_list, wedgeprops=wedgeprops, labels=df_other['敌方'].tolist(), explode=(0,0,0,0,0.3), autopct='%.2f%%')
plt.text(0, 0, '敌方' , ha='center', va='center', fontsize=30)
plt.show()


path='D:\！datawhale学习\GitModel数据分析和统计建模\Modeling-Universe-main\Data Analysis and Statistical Modeling\task_01 数据可视化\data/AirPassengers.csv'
df = pd.read_csv(
    './data/AirPassengers.csv'
    )  # 航空数据1949-1960
df['date'] = pd.to_datetime(df['date'])
# 使用Matplotlib绘制时间序列折线图
plt.figure(figsize=(8,6))
plt.plot(df['date'], df['value'], color='red')
plt.xlabel("date")
plt.ylabel("value")
plt.show()

# Matplotlib 绘制多系列折线图
date_list = pd.date_range('2022-01-01', '2022-03-31').astype('str').tolist()
value_list1 = np.random.randn(len(date_list))
value_list2 = np.random.randn(len(date_list))
data = pd.DataFrame({
    'date_list': date_list,
    'value_list1': value_list1,
    'value_list2': value_list2
})
data['date_list'] = pd.to_datetime(data['date_list'])

plt.figure(figsize=(8, 6))
plt.plot(data['date_list'], data['value_list1'], color='red', alpha=0.86, label='value1')
plt.plot(data['date_list'], data['value_list2'], color='blue', alpha=0.86, label='value2')
plt.legend()
plt.xlabel('date')
plt.ylabel('value')
plt.show()

# 使用Plotnine绘制日历图：
df = pd.DataFrame({
    'date': pd.date_range('2021-01-01', '2021-12-31'),
    'value': np.abs(np.random.randn(len(pd.date_range('2021-01-01', '2021-12-31')))*10)
})

df['Year'] = df['date'].dt.year  # 哪一年
df['Month'] = df['date'].dt.month   # 哪个月
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['Month_label'] = df['Month'].replace(np.arange(1, 13, 1), month_list)  # 将月份转化成英文形式
df['Month_label'] = pd.Categorical(df['Month_label'], categories=month_list, ordered=True) # 数据由字符串变成Category格式
df['Week'] = [int(t.strftime('%W')) for t in df['date']]  # 第几周
df['Weekday'] = df['date'].dt.weekday + 1  # 星期几
weekday_list = ['mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df['Weekday_label'] = df['Weekday'].replace(np.arange(1, 8, 1), weekday_list) #将星期几换成英文
df['Weekday_label'] = pd.Categorical(df['Weekday_label'], categories=weekday_list, ordered=True) # 数据由字符串变成Category格式
df['Day'] = df['date'].dt.day  # 几号
df['month_week'] = df.groupby(['Month_label'])['Week'].apply(lambda x: x-x.min()+1)   # 每个月的哪一周
p1 = (
    ggplot(df, aes(x='Weekday_label', y='month_week', fill='value'))+
    geom_tile(colour='white', size=0.1)+  # 画出日历的方框
    scale_fill_cmap(cmap_name='Spectral')+
    geom_text(aes(label='Day'), size=8)+
    facet_wrap('~Month_label', nrow=3)+  # 按照月份分面
    scale_y_reverse()+  #将y周转向，因为越往下周数越大
    xlab('Day')+ylab('value')+
    theme(
        strip_text = element_text(size=11,face="plain",color="black"),
        axis_title =element_text(size=10,face="plain",color="black"),
        axis_text = element_text(size=8,face="plain",color="black"),
        legend_position = 'right',
        legend_background = element_blank(), 
        aspect_ratio=0.85, 
        figure_size=(12,12), 
        dpi=100)
)
print(p1)