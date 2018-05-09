#垃圾邮件分类
'''
经典的二元分类问题就是垃圾邮件分类（spam classification）。
这里，我们分类垃圾短信。我们用第三章介绍的TF-IDF算法来抽取短信的特征向量，然后用逻辑回归分类。

我们可以用UCI Machine Learning Repository的短信垃圾分类数据集（SMS Spam Classification Data Set）。
http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
首先，我们还是用Pandas做一些描述性统计：
'''
#ham正常邮件
#spam垃圾邮件 
import pandas as pd
df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)
print(df.head())
print('含spam短信数量：', df[df[0] == 'spam'][0].count())
print('含ham短信数量：', df[df[0] == 'ham'][0].count())