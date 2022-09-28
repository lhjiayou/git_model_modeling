# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:10:44 2022

@author: 18721
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


'''作业1：含二次项/对数项模型的讨论'''
data=pd.read_stata('./data/bwght2.dta')
data.shape #(1832, 23)
# 1.1 使用OLS估计方程
# log(𝑏𝑤𝑔ℎ𝑡)=𝛽0+𝛽1𝑛𝑝𝑣𝑖𝑠+𝛽2𝑛𝑝𝑣𝑖𝑠2+𝑢 
# 输出报告表，并回答：自变量npvis的二次项是否显著？自变量npvis是否对因变量有显著影响？
baby_lm=sm.formula.wls('np.log(𝑏𝑤𝑔ℎ𝑡)~𝑛𝑝𝑣𝑖𝑠+I(𝑛𝑝𝑣𝑖𝑠**2)',data=data).fit()
baby_lm.summary()
#                     coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept         7.9579      0.027    291.364      0.000       7.904       8.011
# 𝑛𝑝𝑣𝑖𝑠             0.0189      0.004      5.140      0.000       0.012       0.026
# I(𝑛𝑝𝑣𝑖𝑠 ** 2)    -0.0004      0.000     -3.573      0.000      -0.001      -0.000
#从结果上看npvis的二次项系数很小，但是从p来看其实是显著的
#npvis的正效应比较明显，因此说明母亲产前检查次数是有利的
#不过结合一次项和二次项来看，也是呈现边际递减的效果

#1.2 基于（1）的方程，我们认为最大化log(bwght)的产前检查次数npvis约为24，其理论依据是什么？
#就是二次函数的最值出现的位置，-b/2a
# - 0.0189 /(2*-0.0004 )= 23.625

# 1.3 按照这个模型的结果，在24次产前检查后婴儿出生体重会下降，这是为什么？
# 你认为这有实际意义吗？这蕴含了一个含二次项变量模型的常见陷阱，请仔细思考！
#回答：首先，边际效应递减是肯定的
#其次，我们看一下npvis的分布情况
plt.hist(data['npvis']) 
plt.savefig('./data/npvis_hist.png') #可见大部分是位于15以下的
data['npvis>24']=data['npvis'].mask(data['npvis']<=24,0)
data['npvis>24']=data['npvis>24'].mask(data['npvis>24']>24,1)
data['npvis>24'].value_counts()
# 0.0    1748
# 1.0      16
#那么这16个样本可能没有很好的说服力，存在很大的取样偏差
#另外，就这一个特征，5.2.2节的模型误设也说明了
# 遗漏变量会对我们实际估计模型的系数估计产生有偏影响，R-squared: 0.021也说明了模型的解释性不好


# 1.4在模型中加入母亲年龄变量及其二次形式。回答：保持npvis不变，母亲在什么生育年龄时，
# 孩子出生体重最大？大于这个年龄时，孩子出生体重下降，这是否具有实际意义呢？请结合问题（3）思考这一问题。
baby_lm_addmage=sm.formula.wls('np.log(𝑏𝑤𝑔ℎ𝑡)~𝑛𝑝𝑣𝑖𝑠+I(𝑛𝑝𝑣𝑖𝑠**2)+mage+I(mage**2)',data=data).fit()
baby_lm_addmage.summary()
#                     coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept         7.9344      0.038    208.265      0.000       7.860       8.009
# 𝑛𝑝𝑣𝑖𝑠             0.0185      0.004      4.982      0.000       0.011       0.026
# I(𝑛𝑝𝑣𝑖𝑠 ** 2)    -0.0004      0.000     -3.465      0.001      -0.001      -0.000
# mage              0.0009      0.001      0.878      0.380      -0.001       0.003
# I(mage ** 2)  -9.481e-06   6.66e-05     -0.142      0.887      -0.000       0.000
#从结果来看，这个mage的参数不是很有说服力，两个参数的p都不显著
#如果非要给个结果的话，那么就是：
# -0.0009 /(2*-9.481e-06)=47.46
#此时R2是0.022，也就是模型性能仍然很差

# 1.5（4）中的模型能否解释log(gwght)大部分变异？
#R2是0.022，也就是模型性能仍然很差


'''作业2：异方差模型的讨论'''

#现在是2022.9.28，分类完成了之后再来看这个有点忘了，把异方差的处理方式再回顾一次把
#########################6.3节复习开始########################
data=pd.read_table('./data/P176.txt')
data.shape  #(27, 2)
# 定义一个输出bp检验的函数
def bp_test(res, X):
    result_bp_test = sm.stats.diagnostic.het_breuschpagan(res, X)
    bp_lm_statistic = result_bp_test[0]
    bp_lm_pval = result_bp_test[1]
    bp_F_statistic= result_bp_test[2]
    bp_F_pval = result_bp_test[3]
    bp_test_output=pd.Series(result_bp_test[0:4],index=['bp_lm_statistic','bp_lm_pval','bp_F_statistic','bp_F_pval'])    
    return bp_test_output
# White检验函数在python上的使用与bp检验完全一样
def white_test(res, X):
    result_bp_test = sm.stats.diagnostic.het_white(res, X)
    bp_lm_statistic = result_bp_test[0]
    bp_lm_pval = result_bp_test[1]
    bp_F_statistic= result_bp_test[2]
    bp_F_pval = result_bp_test[3]
    white_test_output=pd.Series(result_bp_test[0:4],index=['white_lm_statistic','white_lm_pval','white_F_statistic','white_F_pval'])    
    return white_test_output
# 直接看Y与X的散点图
fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
plt.scatter(data.X,data.Y,axes=ax1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Y | X')

# 查看ols估计的残差与X的散点图
data_lm=sm.formula.ols('Y~X',data=data).fit()
ax2=fig.add_subplot(1,2,2)
plt.scatter(data.X,data_lm.resid,axes=ax2)
ax2.set_xlabel('X')
ax2.set_ylabel('resid_ols')
ax2.set_title('resid_ols | X')

# 使用BP检验
data_lm_reg=sm.formula.ols('Y~X',data=data)
print(bp_test(data_lm.resid,data_lm_reg.exog))
print('----------------------------------')
# 使用White检验
data_lm_reg=sm.formula.ols('Y~X',data=data)
print(white_test(data_lm.resid,data_lm_reg.exog))

#使用wls估计一下，发现X的std_error明显变小了，但是仍然很主观
data_lm_wls=sm.formula.wls('Y~X',weights=1/data.X**2,data=data).fit()
# 注意：weights传入的是一个数组，不是一个“表达式”。如果方差函数为h(x)，则要传入1/h(x)的数组
print(data_lm.summary())
print(data_lm_wls.summary())

#我们再来看一下6.3.2的FGLS
smoke=pd.read_stata('./data/smoke.dta')
# 第一步，先进行ols估计，得到残差
smoke_lm_ols=sm.formula.ols('cigs~np.log(income)+np.log(cigpric)+educ+age+I(age**2)+restaurn',data=smoke).fit()
smoke['resid']=smoke_lm_ols.resid
# 第二步，回归，得到拟合值g
smoke_lm_log=sm.formula.ols('np.log(resid**2)~np.log(income)+np.log(cigpric)+educ+age+I(age**2)+restaurn',data=smoke).fit()
#第三步，从g变成h_hat
h_hat=np.exp(smoke_lm_log.fittedvalues)
# 第四步，进行wls检验
smoke_lm_wls=sm.formula.wls('cigs~np.log(income)+np.log(cigpric)+educ+age+I(age**2)+restaurn',weights=1/h_hat,data=smoke).fit()
print(smoke_lm_wls.summary())
#########################6.3节复习结束########################
##########接下来看习题，。example就是6.3.1节P176.txt数据

# 2.1 在假设方差形式为 Var(𝑢∣𝑥)=𝜎2𝑥2 并进行wls估计后，比较wls估计与ols估计的残差图，
# 回答：异方差消除了吗？
# 查看ols估计的残差与X的散点图
fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
data_lm=sm.formula.ols('Y~X',data=data).fit()
plt.scatter(data.X,data_lm.resid,axes=ax1)
ax1.set_xlabel('X')
ax1.set_ylabel('resid_ols')
ax1.set_title('resid_ols | X')
# 查看wls估计的残差与X的散点图
ax2=fig.add_subplot(1,2,2)
data_lm_wls=sm.formula.wls('Y~X',weights=1/data.X**2,data=data).fit()
plt.scatter(data.X,data_lm_wls.resid,axes=ax2)
ax2.set_xlabel('X')
ax2.set_ylabel('resid_wls')
ax2.set_title('resid_wls | X')
#保存图
plt.savefig('./data/习题2_1.png')

# 2.2使用FGLS估计对该模型进行重新估计，观察残差图并回答：异方差消除了吗？
# 第一步，先进行ols估计，得到残差
data_lm_ols=sm.formula.ols('Y~X',data=data).fit()
data['resid']=data_lm_ols.resid
# 第二步，回归，得到拟合值g
data_lm_log=sm.formula.ols('np.log(resid**2)~X',data=data).fit()
#第三步，从g变成h_hat
h_hat=np.exp(data_lm_log.fittedvalues)
# 第四步，进行wls检验
data_lm_fgls=sm.formula.wls('Y~X',weights=1/h_hat,data=data).fit()

#绘图
fig=plt.figure(figsize=(18,6))
#普通ols
ax1=fig.add_subplot(1,3,1)
data_lm=sm.formula.ols('Y~X',data=data).fit()
plt.scatter(data.X,data_lm.resid,axes=ax1)
ax1.set_xlabel('X')
ax1.set_ylabel('resid_ols')
ax1.set_title('resid_ols | X')
# wls
ax2=fig.add_subplot(1,3,2)
data_lm_wls=sm.formula.wls('Y~X',weights=1/data.X**2,data=data).fit()
plt.scatter(data.X,data_lm_wls.resid,axes=ax2)
ax2.set_xlabel('X')
ax2.set_ylabel('resid_wls')
ax2.set_title('resid_wls | X')
#fgls
ax3=fig.add_subplot(1,3,3)
plt.scatter(data.X,data_lm_fgls.resid,axes=ax3)
ax3.set_xlabel('X')
ax3.set_ylabel('resid_fgls')
ax3.set_title('resid_fgls | X')
#保存图
plt.savefig('./data/习题2_2.png')


#2.3画出log(Y)与X的散点图，观察方差的状况，说说你的发现；根据散点图的情况，请大胆假设一个你认为正确的模型。
fig=plt.figure()
plt.scatter(data.X,np.log(data.Y))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('log(Y) | X')
plt.savefig('./data/习题2_3.png')
#此时的异方差现象不明显，因此对数化很有用，这也是2.4将要考虑的模型

# 2.4考虑新模型 log(𝑌)=𝛽0+𝛽1𝑋+𝛽2𝑋2+𝑢 使用ols估计该模型，并画出残差散点图，
data_lm_logy=sm.formula.ols('np.log(Y)~X+I(X**2)',data=data).fit()
print(data_lm_logy.summary())
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      2.8516      0.157     18.205      0.000       2.528       3.175
# X              0.0031      0.000      7.803      0.000       0.002       0.004
# I(X ** 2)  -1.102e-06   2.24e-07     -4.925      0.000   -1.56e-06    -6.4e-07
#可见现在系数很显著，而且R2很大
#接下来对比Y~X和logY~X的异方差,明显logY的残差异方差不明显了
# 查看ols估计的残差与X的散点图
fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
data_lm=sm.formula.ols('Y~X',data=data).fit()
plt.scatter(data.X,data_lm.resid,axes=ax1)
ax1.set_xlabel('X')
ax1.set_ylabel('resid_ols')
ax1.set_title('resid_ols | X')
# 查看ols估计log(Y)的残差与X的散点图
ax2=fig.add_subplot(1,2,2)
data_lm_logy=sm.formula.ols('np.log(Y)~X+I(X**2)',data=data).fit()
plt.scatter(data.X,data_lm_logy.resid,axes=ax2)
ax2.set_xlabel('X')
ax2.set_ylabel('resid_log(Y)_ols')
ax2.set_title('resid_log(Y) | X')
plt.savefig('./data/习题2_4.png')

# 2.5综合以上四个问题，谈谈你对纠正模型异方差的见解。
# 异方差模型有可能是因为模型设计不合理造成的。在这种情况下，即使采用wls或者FGLS估计，也不一定能够消除异方差现象。
# 当出现残差的方差随着自变量变大而变大的现象时，可以考虑对因变量取对数后再回归，可以很好地缓解异方差程度。


'''作业3：二分类：信贷风险评估'''
loan=pd.read_stata('./data/loanapp.dta')
loan.shape #(1989, 62)
# 选取要用的变量组成新的数据集
loan=loan[["approve","white","hrat","obrat","loanprc","unem","male","married","dep","sch","cosign","chist","pubrec","mortlat1","mortlat2","vr"]]
loan.shape #(1989, 16)
loan=loan.dropna() #去除含缺失值样本
loan.shape #(1971, 16)
#3.1先考虑一个线性概率模型 𝑎𝑝𝑝𝑟𝑜𝑣𝑒=𝛽0+𝛽1𝑤ℎ𝑖𝑡𝑒+𝑢 如果存在种族歧视，那么 𝛽1 的符号应如何？
#由于 white：种族哑变量（0为黑人，1为白人），因此直觉上存在种族其实的时候，𝛽1 应该是正数

#3.2用OLS估计上述模型，解释参数估计的意义，其显著性如何？在该模型下种族歧视的影响大吗？
loan_lm=sm.formula.ols('approve~white',data=loan).fit()
loan_lm.summary()
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.7030      0.018     38.105      0.000       0.667       0.739
# white          0.2047      0.020     10.208      0.000       0.165       0.244
#从white的p值来看，应该拒绝原假设，因此存在种族歧视，也就是说设定模型下，白人approved的概率比黑人大0.2047

# 3.3 在上述模型中加入数据集中的其他所有自变量，此时white系数发生了什么变化？我们仍然可以认为存在黑人歧视现象吗？
feature=["white","hrat","obrat","loanprc","unem","male","married","dep","sch","cosign","chist","pubrec","mortlat1","mortlat2","vr"]
feature_str='+'.join(feature)
loan_lm_all=sm.formula.ols('approve~'+feature_str,data=loan).fit()
loan_lm_all.summary()
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.9367      0.053     17.763      0.000       0.833       1.040
# white          0.1288      0.020      6.529      0.000       0.090       0.168
#虽然white的系数从0.2047变成了0.1288，但是从p值来看，其实还是会拒绝原假设

# 3.4 允许种族效应与债务占比(obrat)有交互效应，请问交互效应显著吗？请解读这种交互效应。
feature_str_cross=feature_str+'+I(white*obrat)'
loan_lm_cross=sm.formula.ols('approve~'+feature_str_cross,data=loan).fit()
loan_lm_cross.summary()
#可见交互项的系数是0.0081，但是p=0.000，因此拒绝等于0的原假设，交互效应是显著的
#但是现在white的系数是-0.1460，而obrat的系数是-0.0122
#解读：关注white，obrat以及交互项
# 黑人approved=-0.0122*obrat+hx
# 白人approved=-0.1460-0.0122*obrat+0.0081*obrat+hx=-0.1460-0.0041*obrat+hx
#也就是说obrat虽然对approved有副作用，但是白人的副作用更小一点。
# 但是我有一点问题是white的系数是-0.1460，虽然其p是0.069，好像在0.05置信水平下不是显著的

# 3.5 使用logit模型与probit模型重新（3）中的模型，观察变量系数及其显著性的变化。
loan_logit=sm.formula.logit('approve~'+feature_str,data=loan).fit()
loan_logit.summary()
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      3.8017      0.595      6.393      0.000       2.636       4.967
# white          0.9378      0.173      5.424      0.000       0.599       1.277
# hrat           0.0133      0.013      1.030      0.303      -0.012       0.039
# obrat         -0.0530      0.011     -4.701      0.000      -0.075      -0.031
#可见white和obrat的系数都符合直觉，而且p=0，也就是系数很显著
loan_probit=sm.formula.probit('approve~'+feature_str,data=loan).fit()
loan_probit.summary()
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      2.0623      0.313      6.585      0.000       1.449       2.676
# white          0.5203      0.097      5.366      0.000       0.330       0.710
# hrat           0.0079      0.007      1.131      0.258      -0.006       0.022
# obrat         -0.0277      0.006     -4.578      0.000      -0.040      -0.016
#可见white和obrat的系数虽然绝对值变小了，但是都还符合直觉，而且p=0，也就是系数很显著

'''作业4，多分类：鸢尾花分类问题'''
#D:\Anaconda\Lib\site-packages\sklearn\datasets\data 是数据的默认保存位置
from sklearn.datasets import load_iris
iris_dataset=load_iris()

# 提取数据集中的自变量集与标签集
iris_data=iris_dataset['data'] # 自变量
iris_target=iris_dataset['target'] # 标签集

pd.Series(iris_target).value_counts()  #是一个均衡的多酚类问题
# 0    50
# 1    50
# 2    50

# 4.1  将原数据集划分为训练集与测试集，两者样本比例为3:1，是随机抽样，还是按照label进行分层抽样呢？
#先就按照随机抽样来吧
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_data,iris_target,test_size=0.25,random_state=0) 
pd.Series(y_test).value_counts() 
# 1    16  其实不是分层抽样的效果可能不是最好的
# 0    13
# 2     9

# 4.2使用训练集数据训练logistic回归模型，并分别对训练集与测试集数据进行预测，并将预测的结果分别储存在两个自定义的变量中。
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=10000,multi_class='multinomial').fit(X_train,y_train) #max_iter太小会报警
train_pred=model.predict(X_train)
test_pred=model.predict(X_test)

# 4.3使用函数接口计算出：模型对训练集数据的分类正确率、模型对测试集数据的分类正确率，比较它们孰高孰低，并思考为什么会有这样的差异。
from sklearn.metrics import accuracy_score
acc_train=accuracy_score(y_train,train_pred) #0.9821428571428571
acc_test=accuracy_score(y_test,test_pred)  #0.9736842105263158
#结果轻微差异，可能是过拟合了呗，很正常，整体上还是很不错的了

# 4.4给出测试集数据的混淆矩阵以及精确率、召回率、f分数的综合报告。
from sklearn.metrics import classification_report,confusion_matrix
con_mat=confusion_matrix(y_test,test_pred)
# 13	0	0
# 0	15	1
# 0	0	9
report=classification_report(y_test,test_pred) 
#输出为：
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        13
#            1       1.00      0.94      0.97        16
#            2       0.90      1.00      0.95         9

#     accuracy                           0.97        38
#    macro avg       0.97      0.98      0.97        38
# weighted avg       0.98      0.97      0.97        38
