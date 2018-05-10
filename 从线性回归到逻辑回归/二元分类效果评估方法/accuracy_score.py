#准确率
#准确率是分类器预测正确性的评估指标
from sklearn.metrics import accuracy_score
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print(accuracy_score(y_true, y_pred))


#LogisticRegression.score()用来计算模型预测的准确率：
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('准确率：',np.mean(scores), scores)
#你的结果可能和这些数字不完全相同，毕竟交叉检验的训练集和测试集都是随机抽取的。
'''
准确率是分类器预测正确性的比例，但是并不能分辨出假阳性错误和假阴性错误。
在有些问题里面，比如第一章的肿瘤预测问题中，假阴性与假阳性要严重得多，其他的问题里可能相反。
另外，有时准确率并非一个有效的衡量指标，如果分类的比例在样本中严重失调。比如，分类器预测信用卡交易是否为虚假交易时，假阴性比假阳性更敏感。为了提高客户满意度，信用卡部门更倾向于对合法的交易进行风险检查，往往会忽略虚假交易。因为绝大部分交易都是合法的，这里准确率不是一个有效的衡量指标。
经常预测出虚假交易的分类器可能有很高的准确率，但是实际情况可能并非如此。
因此，分类器的预测效果还需要另外两个指标：精确率和召回率。
'''