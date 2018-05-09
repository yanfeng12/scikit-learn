#用LogisticRegression类来预测
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split


#用pandas加载数据.csv文件，然后用train_test_split分成训练集（75%）和测试集（25%）：
'''
sep:指定分隔符。如果不指定参数，默认使用逗号分隔。如果分隔符长于一个字符并且不是‘\s+’，将使用python的语法分析器，并忽略数据中的逗号。正则表达式例子：'\r\t'。
delimiter: 定界符，备选分隔符（如果指定该参数，则sep参数失效）。
header:指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。如果明确设定header=0 就会替换掉原来存在列名。
'''
df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],
df[0])
#建一个TfidfVectorizer实例来计算TF-IDF权重：
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
#最后，我们建一个LogisticRegression实例来训练模型。和LinearRegression类似，LogisticRegression同样实现了fit()和predict()方法。
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
for i, prediction in enumerate(predictions[-5:]):
    print('预测类型：%s. 信息：%s' % (prediction, X_test_raw.iloc[i]))
