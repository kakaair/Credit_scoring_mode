基于Python语言的信用评分卡建模分析  
信用评分是指根据客户的信用历史资料，利用一定的信用评分模型，得到不同等级的信用分数。根据客户的信用分数，授信者可以分析客户按时还款的可能性。据此，授信者可以决定是否准予授信以及授信的额度和利率。虽然授信者通过分析客户的信用历史资料，同样可以得到这样的分析结果，但利用信用评分却更加快速、更加客观、更具有一致性。  
一、数据来源  
本案例数据来源kaggles上Give Me Some Credit  
样本总共有150000条数据，11个变量，其中SeriousDlqin2yrs为目标变量。  
二、开发环境  
版本：Python 3.6.3  
三、项目流程  
信用评分卡模型的主要开发流程如下：   
1.数据获取。   
2.数据预处理，主要包括数据清洗、缺失值处理、异常值处理。  
3.探索性数据分析，该步骤主要是获取样本总体的大概情况，描述样本总体情况的指标主要有直方图、箱形图等。  
（1）单变量分析：可以看到年龄变量大致呈正态分布，符合统计分析的假设。月收入也大致呈正态分布，符合统计分析的需要。  
（2）变量相关性分析：各变量之间的相关性是非常小的:，可以初步判断不存在多重共线性问题。  
4.特征选择，该步骤主要是通过统计学的方法，筛选出对违约状态影响最显著的指标。主要有单变量特征选择方法和基于机器学习模型的方法。  
（1）对连续变量进行离散化，首先尝试最优分段，定义一个自动分箱函数；  
（2）尝试所有的变量进行最优分段，只有age,Percentage,DebtRatio可以进行最优分段；  
（3）对于不能用最优分段的变量，采用自定义分箱，进行等距分段（需对业务有深刻的了解）  
5.模型建立，该步骤主要包括变量分段、变量的WOE（证据权重）变换和逻辑回归估算三部分。  
6.模型评估，该步骤主要是评估模型的区分能力、预测能力、稳定性，并形成模型评估报告，得出模型是否可以使用的结论。  
7.信用评分，根据逻辑回归的系数和WOE等确定信用评分的方法。将Logistic模型转换为标准评分的形式。  
8.建立评分系统，根据信用评分方法，建立自动信用评分系统。



