# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:54:30 2022

@author: 18721
"""

'''
不同于task-2中使用的sm.ols，
sm.formula.ols与sm.ols不同，其最大的特点是可以指定模型的形式，这非常有利于我们自主的构建模型，此后我们将统一使用该指令。
值得注意的是，sm.formula.ols默认带截距项
因此sm.formula.ols比较省事
'''

# 3. 回归分析的重要任务——推断/假设检验
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from IPython.display import display
import statsmodels.api as sm
# 在回归分析中所提及的“系数显著性”，本质上都是“系数不为0的显著性”。
# 加载数据
gpa1=pd.read_stata('./data/gpa1.dta')
# 在数据集中提取自变量
X2=gpa1[['ACT','hsGPA','skipped']]
# 提取因变量
y=gpa1.colGPA
# 为自变量增添截距项
X2=sm.add_constant(X2)    
#利用OLS来实现模型的最小二乘拟合
gpa_lm2=sm.OLS(y,X2).fit()                                                               #！！！！这儿是sm.ols
gpa_lm2.summary()
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          1.3896      0.332      4.191      0.000       0.734       2.045
# ACT            0.0147      0.011      1.393      0.166      -0.006       0.036
# hsGPA          0.4118      0.094      4.396      0.000       0.227       0.597
# skipped       -0.0831      0.026     -3.197      0.002      -0.135      -0.032
#根据上面的结果，可知
# colGPA=1.390+0.412hsGPA+0.015ACT−0.083skipped+𝑢
#但是skipped算不算显著，需要根据t来看，就知道是否显著了

'''方式一，用t值与接受域进行对比，判断是否接受还是拒绝'''
# 3.1 t检验
'''双边检验，也就是接受域的检验'''
# 单参数检验问题：这类问题的典型问题就是系数的显著性检验 𝐻0:𝛽𝑗=0↔𝐻1:𝛽𝑗≠0
# 参数线性组合检验问题：这类问题的典型问题就是系数间的相等性检验 𝐻0:𝛽𝑖=𝛽𝑗↔𝐻1:𝛽𝑖≠𝛽𝑗
# 回归系数的显著性检验就是“系数是否为0”的检验，也就是第一种检验



'''3.1.1 t检验的思想-从单参数检验说起'''
# 临界值与显著性水平
# 由于抽样的随机性，我们根据 𝛽𝑗^ 判断 𝛽𝑗 的命题，不论拒绝与否，都有概率会犯以下两类错误的其中之一：
# · 第一类错误，即原假设成立但是我们拒绝了它。犯第一类错误的概率称为拒真概率。
# · 第二类错误，即原假设不成立但是我们没有拒绝它。
# 我们定夺临界值的时候，要保证发生第一类错误的概率需要在一个给定的、较小的水平 𝛼 ，这个 𝛼 也被称为显著性水平。如此以来，我们考虑临界值 𝐶 的判准是，
# 原假设 𝐻0 成立但是 |||𝛽̂ 𝑗−𝛽̂ 𝑗0|||>𝐶 （因而拒绝原假设 𝐻0 ）的概率应当恰好为我们人为给定的 𝛼 ，即

# 手动进行假设检验
gpa_lm3=sm.formula.ols('colGPA~hsGPA+ACT+skipped',data=gpa1).fit()                   #！！！！！这儿是sm.formula.ols
gpa_lm3.summary()
## 计算t值
skipped=gpa_lm3.params[3]  #可以对照上面的gpa_lm2.summary()来看
se_skipped=gpa_lm3.bse[3]  #可以对照上面的gpa_lm2.summary()来看
tvalue=skipped/se_skipped  #其实也就是上面的summary报告中的值
## 计算分位点
from scipy.stats import t
'''
ppf:单侧左分位点
isf:单侧右分位点
interval:双侧分位点
'''
T_int=t.interval(0.95,gpa_lm3.df_resid) # 对于双侧检验（双侧分位点），分位点参数应该输入1-a，这里是1-0.05=0.95
print('双侧分位点为：{}'.format(T_int))
print('t值为：{}'.format(tvalue))
print('t值小于左侧分位点，位于拒绝域，因此在0.05的显著性水平可以拒绝原假设，即skipped系数不为0.')
#也就是原来的假设为0，如果在t分布的双侧分位点之内，则接收这个假设，现在是拒绝这个假设，也就是它的系数不应该为0

# 当然，我们检验的问题还可以变为𝐻0:𝛽3=−0.1↔𝐻1:𝛽3≠−0.1   这其实也就是𝛽3+0.1=0的验证
#  我们只需要变更t值而不需要变更t分位点值。
tvalue=(skipped+0.1)/se_skipped
print('双侧分位点为：{}'.format(T_int))
print('t值为：{}'.format(tvalue))
print('此时t值小于右侧分位点但大于左侧分位点，位于接受域，不能拒绝原假设，即skipped系数可为-0.1')


'''单边检验，也就是正效应检验'''
# 有时候我们回归分析中可能还会有这样的问题：某某自变量对因变量是否存在正效应影响呢？这个问题其实等价于下面的假设
# 𝐻0:𝛽𝑗=𝛽𝑗0↔𝐻1:𝛽𝑗>𝛽𝑗0(𝛽𝑗0=0)     
# 单边检验的分析思路和双边检验基本一样，只不过 𝑃(|||𝛽̂ 𝑗−0|||>𝐶) 要变为 𝑃(𝛽̂ 𝑗−0>𝐶) ， 
# 𝐶/se(𝛽̂ 𝑗) 也应从 1−𝛼2 分位点变为 1−𝛼 分位点

# 手动检验，检验问题为
# 𝐻0:𝛽𝑗=0↔𝐻1:𝛽𝑗<0 显著性水平为0.05  也就是检验这个参数是不是小于0的
tvalue=skipped/se_skipped    #t值其实没有发生任何的变化
# 因为是小于，因此看左分位点
T_left=t.ppf(0.05,gpa_lm3.df_resid) # 对于单侧检验，分位点参数应该输入a，这里是0.05，代表的是左侧检验
print('左侧分位点为：{}'.format(T_left))
print('t值为：{}'.format(tvalue))
print('t值小于左侧分位点，位于拒绝域，因此在0.05的显著性水平可以拒绝原假设，即skipped系数小于0.')
#如果要看右侧的分位点，可以T_left=t.ppf(0.95,gpa_lm3.df_resid)就能得到


'''方式二，在t检验的基础上，改成p值'''
# 用临界值 𝐶 与 |||𝛽̂ 𝑗−𝛽̂ 𝑗0||| 作比较有一个缺点，就是分位点值与显著性水平 𝛼 相关的。如果我们要在不同的显著性水平下检验，就需要计算不同的分位点再比较，这样很繁琐。这个时候，我们可以使用p值。
# p值越小越可以拒绝原假设
# 如果p值为0.025，比0.01的显著性水平要大，但小于0.05，则我们认为在0.05的显著性水平下我们可以拒绝原假设，但在0.01显著性水平下不可以拒绝。
# p值本质上是一种累积概率，且对于同一个 𝛽𝑗0 而言，双边检验的p值为单边检验的两倍

# 手动实现 𝐻0:𝛽3=0↔𝐻1:𝛽3≠0的检验
# 计算t值仍然是第一步
tvalue=skipped/se_skipped
print('由于双边检验p值是对单边检验p值乘两倍得来的，我们要根据t值是否大于0来选择左/右尾累积概率，若小于0，则选择左尾；反之右尾。')
'''
sf:右尾累积概率
cdf:左尾累积概率
'''
print(tvalue<0)
pvalue=t.cdf(tvalue,gpa_lm3.df_resid)*2 # 双边p值记得乘2
print('p值为：{:.3f}'.format(pvalue)) # 保留三位小数
print('p值非常小，可见我们可以拒绝原假设')
# summary中的p值，正是系数0值双边检验的p值

# 当然，对于系数的非0值单边检验，我们也可以进行手动检验，考虑下面问题
# 𝐻0:𝛽3=−0.1↔𝐻1:𝛽3>−0.1   备择假设是大于号，因此要用右尾累积概率，且不用乘2，单边的累计概率
# 还是先计算t值！
tvalue=(skipped+0.1)/se_skipped
pvalue=t.sf(tvalue,gpa_lm3.df_resid) # 由于备择假设是大于号，因此要用右尾累积概率，且不用乘2
print('p值为：{:.3f}'.format(pvalue)) # 保留三位小数
print('p值远大于0.1，可见我们不能拒绝原假设')


'''3.1.2 参数线性组合的检验-巧用模型变式'''
# 如果我们要对多个参数之间的关系进行假设检验，也可以用t检验，这个时候，我们本质上是对参数的线性组合进行检验。
# 探讨一个有关薪酬的问题，想看看哪些因素会影响我们的薪酬。经过一番思考，我们先将模型设置为
# log( wage )=𝛽0+𝛽1𝑗𝑐+𝛽2𝑢𝑛𝑖𝑣+𝛽3exper+𝑢 
# 其中，jc表示为大专教育年限，univ为大学教育年限，exper为工作年限。我们想知道：大专学历的边际回报是否不如大学学历的边际回报，这等价于下面的假设检验
# 𝐻0:𝛽1=𝛽2↔𝐻1:𝛽1<𝛽2 
# 而这又可以变形为  𝐻0:𝛽1−𝛽2=0↔𝐻1:𝛽1−𝛽2<0   反正变形使得半边出现0即可
#那么就需要旧可以采用前面的思路，先构造t检验统计量 𝑡=𝛽̂ 1−𝛽̂ 2𝑠𝑒(𝛽̂ 1−𝛽̂ 2) 
# 再根据t分布求得p值即可。问题是 𝑠𝑒(𝛽̂ 1−𝛽̂ 2) 的求解不那么容易，需要使用协方差矩阵 𝐶𝑜𝑣(𝛽̂ ⃗ ) 内的方差与协方差。
# 当线性组合变得复杂的时候，这样的任务将变得更加困难。
'''为了避免上面求两个参数的差的标准误se，我们可以改变模型'''
# 既然假设检验的问题是 𝐻0:𝛽1=𝛽2↔𝐻1:𝛽1<𝛽2
#  我们干脆令 𝜃1=𝛽1−𝛽2 ，于是 𝛽1=𝜃1+𝛽2 ，将其代入到原式中并将带有系数 𝜃1 的一项提出来，得log( wage )=𝛽0+𝜃1𝑗𝑐+𝛽2(𝑗𝑐+ univ )+𝛽3exper+𝑢
#  记 𝑗𝑐+𝑢𝑛𝑖𝑣=𝑡𝑜𝑡𝑐𝑜𝑙𝑙 ，是两个变量之和，此时模型简化为 log( wage )=𝛽0+𝜃1 jc +𝛽2 totcoll +𝛽3 exper +𝑢
#  原检验问题也变为了 𝐻0:𝜃=0↔𝐻1:𝜃<0
#  此时，问题有转化为了对新模型的单个参数的显著性检验问题。注意，这个新模型的意义仅仅只在于做假设检验，虽然两个模型实际上是等价的。
wage1=pd.read_stata('./data/twoyear.dta')   #6763*23

wage1_lm=sm.formula.ols('lwage~jc+I(jc+univ)+exper',data=wage1).fit()   #然后就是求这个模型的参数
# 注意，如果我们要将jc与univ的和当做一个新变量的话，需要使用I()
print(wage1_lm.summary())
# Intercept        1.4723      0.021     69.910      0.000       1.431       1.514
# jc              -0.0102      0.007     -1.468      0.142      -0.024       0.003
# I(jc + univ)     0.0769      0.002     33.298      0.000       0.072       0.081
# exper            0.0049      0.000     31.397      0.000       0.005       0.005
'''注意，上面的jc的p值是0.142，但却是两边的，实际jc的t值小于0，说明它后面的双侧p值是使用左侧累积概率乘两倍得来的
那么单侧的p为0.071，因此0.05置信水平不能拒绝𝜃=0，但是0.1置信水平拒绝𝜃=0，也就是接受𝜃<0的假设'''


# 3.2 F检验
# F检验是回归分析中多个线性假设检验问题的常用检验方法。多个线性假设检验问题可分为如下：
# · 多参数联合显著性检验问题： 𝐻0:𝛽𝑖=⋯=𝛽𝑗=0↔𝐻1:   𝐻0 不成立
# · 一般多参数检验问题： 𝐻0:𝛽𝑛=𝛽𝑛0,𝛽𝑖=⋯=𝛽𝑗=0↔𝐻1:   𝐻0 不成立

#3.2.1 多参数联合显著性检验
# Example5. 考虑美国棒球职业大联盟的运动员薪水问题，假设模型为 log( salary )=𝛽0+𝛽1 years +𝛽2 gamesyr +𝛽3 bavg +𝛽4 hrunsyr +𝛽5 rbisyr +𝑢
# 其中，salary是队员薪水，years为加入联盟的年限，gamesyr为每年参加比赛的次数，bavg是击球率，hrunsyr为本垒打次数，rbisyr表示击球跑垒得分。后面三个指标是运动员的球场表现正向指标（指标越高，代表表现越好），而前面两个指标则为运动员的球场资历指标。
# 我们想弄明白一个问题：运动员的表现正向指标是否对薪水有显著影响。如何理解这一问题？如果这三个指标中至少有一个指标系数显著不为0，我们便可以认为表现正向指标对薪资有显著影响。于是原假设可以设置为
# 𝐻0:𝛽3=0,𝛽4=0,𝛽5=0 对立假设则为：原假设不成立。
'''注意，三个参数做联合显著性检验完全不等价于三个参数分开做显著性t检验！如果我们是出于联合检验的目的但是却做了分开检验，将大大增加拒真概率。'''
# 称原模型为无约束模型(unrestricted model)：log( salary )=𝛽0+𝛽1 years +𝛽2 gamesyr +𝛽3 bavg +𝛽4 hrunsyr +𝛽5 rbisyr +𝑢
# 然后将原假设 𝐻0 成立下的条件代入无约束模型，得到的模型称为有约束模型(restricted model)：log( salary )=𝛽0+𝛽1 years +𝛽2 gamesyr +𝑢

# 回归分析中的F检验统计量 𝐹=(𝑅𝑆𝑆𝑟−𝑅𝑆𝑆𝑢𝑟)/𝑞 /  𝑅𝑆𝑆𝑢𝑟/(𝑛−𝑘−1)  服从∼𝐹𝑞,𝑛−𝑘−1 
# 它服从自由度为 𝑞 与 𝑛−𝑘−1 的F分布，其中 𝑞 为有效约束个数， 𝑛−𝑘−1 为无约束模型自由度。
# F检验拒绝原假设的判别规则非常简单，即𝐹>𝐹𝑞,𝑛−𝑘−1(1−𝛼)   这意思是F越大，越会拒绝是么？
mlb1=pd.read_stata('./data/mlb1.dta')
'''手动F检验'''
#根据上面的F的公式，我们可以进行手动的检验
 # 无约束模型
mlb_ur=sm.formula.ols('lsalary~years+gamesyr+bavg+hrunsyr+rbisyr',data=mlb1).fit()
# 有约束模型
mlb_r=sm.formula.ols('lsalary~years+gamesyr',data=mlb1).fit()

# 计算两个模型的RSS
RSS_mlb_ur=np.sum(np.power(mlb_ur.resid,2))
RSS_mlb_r=np.sum(np.power(mlb_r.resid,2))

# 计算F统计量
Fvalue=((RSS_mlb_r-RSS_mlb_ur)/3)/(RSS_mlb_ur/(mlb_ur.df_resid))                               #这是计算的F值
print('F值为：{}'.format(Fvalue))

# 计算F分布分位点
from scipy.stats import f
# 由于F检验只有大于号的假设，因此只会使用单侧右分位点
F_isf=f.isf(0.05,3,mlb_ur.df_resid) # 注意自由度的顺序不能颠倒,这里显著性水平为0.05              这是计算的分位点，后侧会拒绝，左侧是接受
print('F分位点为：{}'.format(F_isf))
print('F值位列分位点右侧，说明位于拒绝域当中，可以在显著性水平0.05下拒绝原假设')
# 计算p值
# 由于F检验只有大于号的假设，因此只会使用单侧右分位点
pvalue=f.sf(Fvalue,3,mlb_ur.df_resid)
print('p值为：{:.6f}'.format(pvalue))
'''anova_lm函数，它会直接输出F值与p值，也就是自动F检验'''
from statsmodels.stats.anova import anova_lm
anova_lm(mlb_r,mlb_ur) # 注意，是有约束在前，无约束在后
#    df_resid         ssr  df_diff   ss_diff         F    Pr(>F)
# 0     350.0  198.311502      0.0       NaN       NaN       NaN
# 1     347.0  183.186322      3.0  15.12518  9.550272  0.000004     看这儿的F以及P即可，p很小，因此会拒绝


#3.2.2 一般多参数检验问题
# 除了全零假设，原假设还可以更一般地设置为 𝐻0:𝛽𝑛=𝛽𝑛0,𝛽𝑖=⋯=𝛽𝑗=0  即，部分假设可以设置为非0参数。
# 对于这种检验问题，我们的有约束模型需要将假设中非0参数的变量移至因变量一侧。例如，若假设为 𝐻0:𝛽3=1,𝛽4=0,𝛽5=0
# 则有约束模型为 log( salary )−bavg=𝛽0+𝛽1 years +𝛽2 gamesyr +𝑢 
# 这意味着有约束模型的因变量发生了改变。
hprice1=pd.read_stata('./data/hprice1.dta')
hprice_ur=sm.formula.ols('lprice~lassess+llotsize+lsqrft+bdrms',data=hprice1).fit()
hprice_r=sm.formula.ols('I(lprice-lassess)~1',data=hprice1).fit()    #这个右边的1是常数β0
# 注意，将lassess移至因变量后，它们的差应视作一个整体回归元，因此需要添加I()
anova_lm(hprice_r,hprice_ur) # 注意，是有约束在前，无约束在后
print('p值远大于0.1，不能拒绝原假设')


# 4. 更广义的“线性”回归——多种形式自变量
# 实际上，自变量不仅可以是一次的连续变量，还可以是一种定性变量，也可以是某个变量的函数，如二次项 𝑋2 、对数项 𝑙𝑜𝑔(𝑋) 。这是因为，所谓的线性回归模型，线性关系并不是指代被解释变量 𝑦 与解释变量 𝑋 之间的关系，而是指回归函数相对于回归系数是线性的。


# 4.1 带有定性变量的回归分析  就是类别数据的处理方式
# 4.1.1 二分类变量


# 定性变量定量化——虚拟变量，其实也就是0-1变量，
wage1=pd.read_stata('./data/wage1.dta')
wage1_lm=sm.formula.ols('wage~female+educ+exper+tenure',data=wage1).fit()
print(wage1_lm.summary())
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     -1.5679      0.725     -2.164      0.031      -2.991      -0.145
# female        -1.8109      0.265     -6.838小于0      0.000      -2.331      -1.291
# educ           0.5715      0.049     11.584      0.000       0.475       0.668
# exper          0.0254      0.012      2.195      0.029       0.003       0.048
# tenure         0.1410      0.021      6.663      0.000       0.099       0.183
#结果是0/2=0，则拒绝原来的假设


# 交互效应模型——定性变量间的交互效应
'''不考虑交互效应的时候：
· Example7. 承接example.6，我们除了考虑性别的影响外，还决定同时考察婚姻状况对薪资的影响。考虑一下模型
log( wage )=𝛽0+𝛿0 female +𝛾0 married +𝛽1 educ +𝛽2 exper +𝛽3 exper 2+𝛽4 tenure +𝛽5 tenure 2+𝑢
 
在这个模型中，人群被分为四个类别：单身男性、单身女性、已婚男性、已婚女士。他们在薪资上的区别依旧可以用回归函数表示出来
𝐸(log( wage )∣ male , single ,𝑥)=ℎ(𝑥)𝐸(log( wage )∣ female, single ,𝑥)=𝛿0+ℎ(𝑥)𝐸(log( wage )∣ male , married ,𝑥)=𝛾0+ℎ(𝑥)𝐸(log( wage )∣ female , married ,𝑥)=𝛿0+𝛾0+ℎ(𝑥)
 
其中， ℎ(𝑥) 在这里表示模型中不含定性变量的部分。

我们可以清楚地看到，不论是未婚还是已婚，性别差异都是 𝛿0 ；不论是男性还是女性，结婚与否的差异都是 𝛾0 。大家稍加思考一下可能可以发现，这里面暗示着这两个定性因素彼此互不相关。在这个模型下，男性结婚与否的差异，与女性结婚与否的差异是相同的。'''


'''
考虑交互效应的时候：
在现实中，这一假设未必成立。相对于男性而言，婚姻给女性在职场上带来的影响可能相对较大，这意味着这两种定性因素相互之间存在交互效应。而要在模型中体现出这种交互效应，我们需要在原模型的基础上加上它们的交互乘积项
log( wage )=𝛽0+𝛿0 female +𝛾0 married +𝛿𝛾 female ∗ married +𝛽1 educ +𝛽2 exper +𝛽3 exper 2+𝛽4 tenure +𝛽5 tenure 2+𝑢
 
在这个模型下，单身男性、单身女性、已婚男性、已婚女士的薪资区别就变成了
𝐸(log( wage )∣ male, single ,𝑥)=ℎ(𝑥)𝐸(log( wage )∣ female single, 𝑥)=𝛿0+ℎ(𝑥)𝐸(log( wage )∣ male, married, 𝑥)=𝛾0+ℎ(𝑥)𝐸(log( wage )∣ female, married ,𝑥)=𝛿0+𝛾0+𝛿𝑦+ℎ(𝑥)
 
于男性而言，婚姻差异为 𝛾0 ，而对于女性而言，婚姻带来的影响是 𝛾0+𝛿𝛾'''
wage1_lm2=sm.formula.ols('lwage~female+married+educ+I(female*married)+exper+I(exper**2)+tenure+I(tenure**2)',data=wage1).fit()
print(wage1_lm2.summary())

'''考虑交互效应，而且是定性与定量变量的交互效应'''
# 直接根据报告表做t检验
wage1_lm3=sm.formula.ols('lwage~female+educ+I(female*educ)+exper+I(exper**2)+tenure+I(tenure**2)',data=wage1).fit()
print(wage1_lm3.summary())
# 使用anova函数做F检验
wage1_lm3_r=sm.formula.ols('lwage~educ+exper+I(exper**2)+tenure+I(tenure**2)',data=wage1).fit()
anova_lm(wage1_lm3_r,wage1_lm3)


# 4.1.2 多分类变量
# 用多个二值虚拟变量来表示多分类定性变量。具体的，如果一个变量有n个类别，则需要定义n-1个虚拟变量表示它。以季节变量为例，我们定义三个虚拟变量：spring/summer/fall，当它们其中之一等于1时，代表季节为它们本身；而如果它们全都为0，则代表季节为winter。
# · 虚拟变量陷阱——完全共线性

# 之所以需要这样定义多分类定性变量，是因为如果我们如果将winter也纳入模型中时，这四个变量会满足一个恒等关系式
# 𝑠𝑝𝑟𝑖𝑛𝑔+𝑠𝑢𝑚𝑚𝑒𝑟+𝑓𝑎𝑙𝑙+𝑤𝑖𝑛𝑡𝑒𝑟=1
 
# 这说明这四个自变量存在完全共线性，违背了CLM假设中的MLR.4，使得模型完全失效。
# S：年薪，单位是美元；
# X：工作经验，单位是年；
# E：教育，1表示高中毕业，2表示获得学士学位，3表示更高学位；
# M：1表示为管理人员，0表示非管理人员；

# 我们以S为因变量，以X/E/M为自变量进行多元回归。其中：X为定量变量，M为二分类变量（且已经0-1化），它们已经可以直接进行回归处理了。但是E则需要进行0-1处理。
data=pd.read_table('./data/P130.txt')
# One-hot编码。我们使用pandas包的get_dummies函数进行重编码。

# get_dummies这一函数会自动变换所有具有对象类型（如字符串）的列，但是如果某列的变量是数值型变量（哪怕它实际上是分类变量），
# 它将不会为该列创建虚拟变量，除非我们将该列的数据类型从数值转化为字符串。

data['E']=data['E'].astype(str)  #46*4
data_dummies=pd.get_dummies(data,columns=['E'])  ## 指定columns参数，就可以对我们想要虚拟变量化的列进行精准转换 46*6
# 注意：不可以直接将三个虚拟变量同时纳入回归当中，不然就会造成完全共线性
data_lm=sm.formula.ols('S~X+E_2+E_3+M',data=data_dummies).fit()
print(data_lm.summary())


# 4.2 带有自变量函数的回归分析
# 三种常见的自变量函数：对数化、二次项化、交互项化。


# 4.2.1 对数画
'''对数变换是线性回归中非常常见的变量变换，它的作用非常明显：

正如上面的例子所示，对数变换可以方便地计算变换百分比，于“价格”型变量而言，百分比解释比绝对值解释更有经济意义。
当因变量为严格取正的变量，它的分布一般存在异方差性或偏态性，这容易违背CLM假设的同方差/正态性假设。而对数变换可以缓和这种情况。
然而，对数变换并不能滥用，因为在一些情况下对数变换会产生极端值。首先，存在负值的变量不可以对数变换；其次，当原变量 𝑦 有部分取值位于[0,1]区间时， 𝑙𝑜𝑔(𝑦) 的负数值会非常大！而线性模型对极端值是非常敏感的，这会影响模型的效果。

对于变量何时取对数，没有一个准确的标准，但在长久的实践中，我们认为可以遵循以下经验：

对于大数值大区间变量（价格类变量、人口变量等），可取对数变换，如：工资、薪水、销售额、企业市值、人口数量、雇员数量等。
对于小数值小区间变量（时间类变量等），一般不取对数变换，如：受教育年限、工作年限、年龄等。'''
# 对数化对正态化的作用
## 以hprice1的price为例
### 未经对数化的直方图
hprice1.hist(column='price',figsize=(8,6))

### 对数化后的直方图
hprice1.hist(column='lprice',figsize=(8,6))

# 4.2.2 二次项
# 经验告诉我们，随着工作年限的上升，工资提升会逐渐变缓。也就是说，工作年限对工资水平的影响可能不是线性的，而是有一个“弧度”。如何让模型具备这个“弧度”呢？最简单也是最直观的方法就是，在原模型的基础上加入二次项



# 4.2.3 交互项
# 我们在定性变量章节介绍了定性变量之间以及定性与定量变量之间交互项的理解与意义，在这一小节我们介绍定量变量之间的交互项的理解。
#检验1：
# 出勤率对成绩的影响是否显著？对于这一问题，假设需要设置成这样
# 𝐻0:𝛽1=𝛽6=0↔𝐻1:𝐻0不成立 
# 很明显，我们用F检验即可。
attend=pd.read_stata('./data/attend.dta')
attend.head()
# 构建有约束、无约束模型
attend_lm_ur=sm.formula.ols('stndfnl~atndrte+priGPA+ACT+I(priGPA**2)+I(ACT**2)+I(priGPA*atndrte)',data=attend).fit()
attend_lm_r=sm.formula.ols('stndfnl~priGPA+ACT+I(priGPA**2)+I(ACT**2)',data=attend).fit()

# F检验
display(anova_lm(attend_lm_r,attend_lm_ur))
print('结果显著，可以拒绝原假设')

# 当 𝑝𝑟𝑖𝐺𝑃𝐴 为一个具体的值时，出勤率在这个值下对对成绩的影响是否显著？
# 当 𝑝𝑟𝑖𝐺𝑃𝐴=2.59 时，偏效应正好就是 𝛽1 ！我们只需要查看这个模型的 𝛽1 显著性就可以回答这一假设检验问题了。
attend['priGPA2']=attend['priGPA']-2.59
attend_lm2=sm.formula.ols('stndfnl~atndrte+priGPA2+ACT+I(priGPA2**2)+I(ACT**2)+I(priGPA2*atndrte)',data=attend).fit()
print(attend_lm2.summary())
print('结果显著，可以拒绝原假设')
