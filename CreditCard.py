# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# 一、数据导入
data = pd.read_csv('E:/python/data/CreditCard/cs_training.csv')

# 二、数据预处理
# 查看数据集的基本情况，对缺失值和异常值进行初步的判断
#data.info()
data.describe().to_csv('E:/python/data/CreditCard/DataDescribe.csv')

# 观察发现变量比较长，将变量名进行简化
columns = ({'SeriousDlqin2yrs':'Bad',
            'RevolvingUtilizationOfUnsecuredLines':'Percentage',
            'NumberOfTime30-59DaysPastDueNotWorse':'30-59',
           'NumberOfOpenCreditLinesAndLoans':'Number_Open',
           'NumberOfTimes90DaysLate':'90-',
           'NumberRealEstateLoansOrLines':'Number_Estate',
           'NumberOfTime60-89DaysPastDueNotWorse':'60-89',
           'NumberOfDependents':'Dependents'}
          )
data.rename(columns=columns,inplace = True)

# 2.1 删除无关的数据
data.drop(data.iloc[:,:1],inplace = True,axis=1)

# 2.2 缺失值的处理
# 用随机森林对缺失值进行预测，用预测值填充MonthlyIncome的缺失值 
def rf_filling(df):
    # 处理数集
    process_miss = df.iloc[:,[5,0,1,2,3,4,6,7,8,9]]
    #分成已知特征与未知特征
    known = process_miss[process_miss.MonthlyIncome.notnull()].as_matrix()
    unknown = process_miss[process_miss.MonthlyIncome.isnull()].as_matrix()
    #X，要训练的特征
    X = known[:,1:]
    #y ,结果标签
    y = known[:,0]
    #训练模型
    rf = RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,n_jobs=-1)
    rf.fit(X,y)
    #预测缺失值
    pred = rf.predict( unknown[:,1:]).round(0)
    #补缺缺失值
    df.loc[df['MonthlyIncome'].isnull(),'MonthlyIncome'] = pred
    return df

data = rf_filling(data)

# Dependents变量缺失值比较少，直接删除
data = data.dropna()
#data.info()
# 2.3 删除重复项
data = data.drop_duplicates()

# 2.4 异常值处理
# 通过箱线图，可以看出30-59，60-89，90-三个变量有异常值
df1 = data.iloc[:,[3,7,9]]
df1.boxplot() #也可用plot.box()
#plt.show()

# 剔除age为0的异常值
data =data[data['age'] > 0]
# 剔除30-59、60-89、90-三个变量的异常值
data = data[data['30-59']<90]
data = data[data['60-89']<90]
data = data[data['90-']<90]
data = data.reset_index(drop=True)

# 三、探索性数据分析
# 单变量分析，年龄通过下图可得大致呈正态分布
sns.distplot(data['age'])
#plt.show()

# 变量相关性分析
corr = data.corr()
#print(corr)
fig = plt.figure(figsize = (12,8))
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow',ax=ax1, \
            annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})#绘制相关性系数热力图
#plt.show()

# 验证模型的拟合效果，将数据随机分成训练集和测试集
from sklearn.model_selection import train_test_split
y = data['Bad']
X=data.iloc[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8,random_state=0)
train = pd.concat([y_train,X_train], axis =1)
test = pd.concat([y_test,X_test], axis =1)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
#保留一份测试数据集，后面生成评分卡
test.to_csv('E:/python/data/CreditCard/origin_test.csv', index=False)

# 四、特征变量选择
# 4.1 测试所有变量的离散化
# 定义分箱函数
import scipy.stats as stats
def monoto_bin(Y, X, n = 20):
    r = 0
    total_bad = Y.sum()
    total_good =Y.count()-total_bad  
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns = ['min_' + X.name])
    d3['min_' + X.name] = d2.min().X
    d3['max_' + X.name] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    #d3[Y.name + '_rate'] = d2.mean().Y
    #好坏比，求woe,证据权重，自变量对目标变量有没有影响，什么影响
    d3['goodattr']=d3[Y.name]/total_good
    d3['badattr']=(d3['total']-d3[Y.name])/total_bad
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])
    #iv，信息值，自变量对于目标变量的影响程度
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min_' + X.name)).reset_index(drop = True)
    print (d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1,n+1):
        qua =X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe

#能用自动优化分箱的变量：age,Percentage,DebtRatio
dfx1,ivx1,cutx1,woex1 = monoto_bin(train['Bad'],train['Percentage'],n = 10)
dfx2,ivx2,cutx2,woex2 = monoto_bin(train['Bad'],train['age'],n = 10)
dfx4,ivx4,cutx4,woex4 = monoto_bin(train['Bad'],train['DebtRatio'],n = 10)

#定义等距分段函数
import scipy.stats as stats
def self_bin(Y, X, cut):
    total_bad = Y.sum()
    total_good =Y.count()-total_bad
    print("total_bad:",total_bad)
    print("total_good:",total_good)
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, cut)})
    d2 = d1.groupby('Bucket', as_index = True)
    d3 = pd.DataFrame(d2.min().X, columns = ['min_' + X.name])
    d3['min_' + X.name] = d2.min().X
    d3['max_' + X.name] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['badattr']=d3[Y.name]/total_bad
    d3['goodattr']=(d3['total']-d3[Y.name])/total_good
    d3['woe'] = np.log(d3['badattr']/d3['goodattr'])
    #iv，信息值，自变量对于目标变量的影响程度
    iv = ((d3['badattr']-d3['goodattr'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min_' + X.name)).reset_index(drop = True)
    print (d4)
    woe = list(d4['woe'].round(3))
    return d4,iv,woe

pinf = float('inf')#正无穷大
ninf = float('-inf')#负无穷大
cutx3 = [ninf, 0, 1, 3, 5, pinf]
cutx5 = [ninf,1000,2000,3000,4000,5000,6000,7500,9500,12000,pinf]
cutx6 = [ninf, 1, 2, 3, 5, pinf]
cutx7 = [ninf, 0, 1, 3, 5, pinf]
cutx8 = [ninf, 0,1,2, 3, pinf]
cutx9 = [ninf, 0, 1, 3, pinf]
cutx10 = [ninf, 0, 1, 2, 3, 5, pinf]
dfx3, ivx3,woex3 = self_bin(train['Bad'],train['30-59'],cutx3)
dfx5, ivx5,woex5 = self_bin(train['Bad'],train['MonthlyIncome'],cutx5)
dfx6, ivx6,woex6 = self_bin(train['Bad'],train['Number_Open'],cutx6) 
dfx7, ivx7,woex7 = self_bin(train['Bad'],train['90-'],cutx7)
dfx8, ivx8,woex8 = self_bin(train['Bad'],train['Number_Estate'],cutx8) 
dfx9, ivx9,woex9 = self_bin(train['Bad'],train['60-89'],cutx9)
dfx10, ivx10,woex10 = self_bin(train['Bad'],train['Dependents'],cutx10)

# 4.2计算每个变量的Infomation Value（IV），确定自变量的预测能力
y=[ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9,ivx10]
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
fig= plt.figure(figsize = (18,8))
ax1 = fig.add_subplot(1, 1, 1)
ax1.bar(range(1,11), y, width=0.4)#生成柱状图
ax1.set_xticks(range(1,11))
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV', fontsize=14)
#在柱状图上添加数字标签
for i, v in enumerate(y):
    plt.text(i+1, v+0.01, '%.4f' % v, ha='center', va='bottom', fontsize=12)
plt.show()

# 五、建立模型
# 5.1WOE转化：将筛选后的变量转换为WoE值，便于信用评分
def change_woe(d,cut,woe):
    list=[]
    i=0
    while i<len(d):
        value=d[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list

#训练集转化
train['Percentage'] = pd.Series(change_woe(train['Percentage'], cutx1, woex1))
train['age'] = pd.Series(change_woe(train['age'], cutx2, woex2))
train['30-59'] = pd.Series(change_woe(train['30-59'], cutx3, woex3))
train['DebtRatio'] = pd.Series(change_woe(train['DebtRatio'], cutx4, woex4))
train['MonthlyIncome'] = pd.Series(change_woe(train['MonthlyIncome'], cutx5, woex5))
train['Number_Open'] = pd.Series(change_woe(train['Number_Open'], cutx6, woex6))
train['90-'] = pd.Series(change_woe(train['90-'], cutx7, woex7))
train['Number_Estate'] = pd.Series(change_woe(train['Number_Estate'], cutx8, woex8))
train['60-89'] = pd.Series(change_woe(train['60-89'], cutx9, woex9))
train['Dependents'] = pd.Series(change_woe(train['Dependents'], cutx10, woex10))

#测试集转化
test['Percentage'] = pd.Series(change_woe(test['Percentage'], cutx1, woex1))
test['age'] = pd.Series(change_woe(test['age'], cutx2, woex2))
test['30-59'] = pd.Series(change_woe(test['30-59'], cutx3, woex3))
test['DebtRatio'] = pd.Series(change_woe(test['DebtRatio'], cutx4, woex4))
test['MonthlyIncome'] = pd.Series(change_woe(test['MonthlyIncome'], cutx5, woex5))
test['Number_Open'] = pd.Series(change_woe(test['Number_Open'], cutx6, woex6))
test['90-'] = pd.Series(change_woe(test['90-'], cutx7, woex7))
test['Number_Estate'] = pd.Series(change_woe(test['Number_Estate'], cutx8, woex8))
test['60-89'] = pd.Series(change_woe(test['60-89'], cutx9, woex9))
test['Dependents'] = pd.Series(change_woe(test['Dependents'], cutx10, woex10))
# 删除对因变量不明显的变量
train_X =train.drop(['Number_Estate','Dependents','Number_Open','DebtRatio','MonthlyIncome'],axis=1)
test_X =test.drop(['Number_Estate','Dependents','Number_Open','DebtRatio','MonthlyIncome'],axis=1)

# 5.2Logistc模型建立
# Logistc模型：对训练数据进行训练，对测试数据进行预测
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
X_train =train_X.drop('Bad',axis =1)
y_train =train_X['Bad']
y_test = test['Bad']
X_train1=sm.add_constant(X_train)
logit=sm.Logit(y_train,X_train1)
lg=logit.fit()
print(lg.summary2())

# 5.3模型验证
from sklearn.metrics import roc_curve, auc
X_test =test_X.drop('Bad',axis =1)
X3 = sm.add_constant(X_test)
pre = lg.predict(X3)
FPR,TPR,threshold =roc_curve(y_test,pre)
ROC_AUC= auc(FPR,TPR)
plt.plot(FPR, TPR, 'b', label='AUC = %0.2f' % ROC_AUC)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

# 六、 信用评分
#个人总分=基础分+各部分得分
import math
B = 20 / math.log(2)
A = 600 - B / math.log(20)
# #基础分
base = round(A+B * coe[0], 0)

#计算分数函数
def compute_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores
# 各项部分分数
x1 = compute_score(coe[1], woex1, B)
x2 = compute_score(coe[2], woex2, B)
x3 = compute_score(coe[3], woex3, B)
x7 = compute_score(coe[4], woex7, B)
x9 = compute_score(coe[5], woex9, B)

# 八、信用卡评分系统
def change_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list

test1 = pd.read_csv('E:/python/data/CreditCard/origin_test.csv')
test2 = pd.DataFrame()
test2['x1'] = pd.Series(change_score(test1['Percentage'], cutx1, x1))
test2['x2'] = pd.Series(change_score(test1['age'], cutx2, x2))
test2['x3'] = pd.Series(change_score(test1['30-59'], cutx3, x3))
test2['x7'] = pd.Series(change_score(test1['90-'], cutx7, x7))
test2['x9'] = pd.Series(change_score(test1['60-89'], cutx9, x9))
test2['Score'] = test2['x1'] + test2['x2'] + test2['x3'] + test2['x7'] +test2['x9']  + base
test2.to_csv('E:/python/data/CreditCard/ScoreData.csv', index=False)

