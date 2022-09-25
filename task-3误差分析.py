# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:19:38 2022

@author: 18721
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
#task-2
# 第一章，我们学习了回归的基本思想，并介绍了本课程中最重要、最常用的模型——多元线性回归模型。
# 第二章，我们介绍了多元线性回归模型的基本六大假设——CLM假设，OLS最小二乘估计法并学习了在这套假设下的性质。需要注意的是，后面两章的内容均建立这些OLS性质上，我们需要时刻牢记这一点。

# task-3
# 第三章，我们学习了回归分析中最重要的分析任务之一——模型的统计推断，介绍了回归分析中各种类型的假设检验。
# 第四章，我们对线性模型进行了拓展，将回归元从定量变量一次项拓展至定性变量、对数项、二次项、交互项，使模型蕴含的信息更丰富。


# 以上内容均建立在模型严格满足CLM假设的前提下，一旦模型违背了CLM假设中的某一条假设，模型估计的精度、
# 假设检验的可信度将受到影响。那么，违反某条CLM假设具体会给模型估计带来多大的影响呢？这就是本节需要讨论的内容。
gpa1=pd.read_stata('./data/gpa1.dta')
# 常规解法的结果
gpa_lm1=sm.formula.ols('colGPA~ACT+hsGPA',data=gpa1).fit()
print('常规解法的结果:')
print(gpa_lm1.params[1])
print('-------------------------------------------')
gpa_lm1.summary()
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      1.2863      0.341      3.774      0.000       0.612       1.960
# ACT            0.0094      0.011      0.875      0.383      -0.012       0.031
# hsGPA          0.4535      0.096      4.733      0.000       0.264       0.643

# 新求解法,也就是先用ACT对hsGPA进行拟合，得到残差，然后用y对残差进行拟合
gpa_lm2_pre=sm.formula.ols('ACT~hsGPA',data=gpa1).fit()
gpa1['resid']=gpa_lm2_pre.resid
gpa_lm2_pre2=sm.formula.ols('colGPA~resid-1',data=gpa1).fit()
print('新解法的效果')
print(gpa_lm2_pre2.params[0])

# 5.2 模型误设的误差分析——违反MLR.1的后果是什么
# 5.2.1 如何理解模型误设
# 在MLR.1中，我们假设自己设置的模型是“正确的”，即对
# 𝑦=𝛽0+𝛽1𝑥1+𝛽2𝑥2+⋯+𝛽𝑘𝑥𝑘+𝑢 
# 的假设上，我们正确地纳入了所有关键的自变量，且没有纳入多余的自变量。而多纳入一个无关的变量，以及少纳入一个关键的变量，都能算是违反MLR.1


# 5.3 随机误差不满足正态性假设
# 频数直方图
crime1=pd.read_stata('./data/crime1.dta')
crime1.hist(column='narr86',figsize=(8,6))
print(crime1.narr86.value_counts())

# 6. 异方差下的回归分析
# 6.1 异方差稳健的t检验与F检验
# 6.1.1 重新估计方差
#方差不再是常数，使用稳健t检验和稳健F检验

# 6.2 异方差诊断
# 主观看图法
# BP一阶法
# white二阶法

# 6.3广义OLS
# 6.3.1 wls 很难知道hx的形式，不考虑
# 6.3.2FGLS   一种估计hx，再用WLS的方法
#下面是FGLS的四个步骤：
# 做回归 𝑦∼𝑥1+⋯𝑥𝑘 ，得到残差 𝑢̂  
# 做回归 log(𝑢̂ 2)∼𝑥1+⋯+𝑥𝑘 ，得到拟合值 𝑔̂  
# 计算函数 ℎ̂ =exp(𝑔̂ ) 
# 以 1/ℎ̂  做权重，用wls估计模型 𝑦∼𝑥1+⋯𝑥𝑘
smoke=pd.read_stata('./data/smoke.dta')
# 第一步，进行ols估计，得到残差
smoke_lm_ols=sm.formula.ols('cigs~np.log(income)+np.log(cigpric)+educ+age+I(age**2)+restaurn',data=smoke).fit()
smoke['resid']=smoke_lm_ols.resid   #这就是残差

# 第二步，残差的平方对自变量进行回归，得到拟合值
smoke_lm_log=sm.formula.ols('np.log(resid**2)~np.log(income)+np.log(cigpric)+educ+age+I(age**2)+restaurn',
                            data=smoke).fit()

#第三步，计算h_hat
h_hat=np.exp(smoke_lm_log.fittedvalues)

#第四步，以 1/ℎ̂  做权重，用wls估计模型 𝑦∼𝑥1+⋯𝑥𝑘
smoke_lm_wls=sm.formula.wls('cigs~np.log(income)+np.log(cigpric)+educ+age+I(age**2)+restaurn',weights=1/h_hat,data=smoke).fit()
print(smoke_lm_wls.summary())

#假设同方差的时候，看估计的标准误
smoke_lm_ols=sm.formula.ols('cigs~np.log(income)+np.log(cigpric)+educ+age+I(age**2)+restaurn',data=smoke).fit()
print(smoke_lm_ols.summary())