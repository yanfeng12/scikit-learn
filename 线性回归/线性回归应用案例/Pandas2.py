#拟合与评估模型
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
df = pd.read_csv('winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R-squared:', regressor.score(X_test, y_test))
'''
开始和前面类似，加载数据，然后通过train_test_split把数据集分成训练集和测试集。两个分区的数据比例都可以通过参数设置。默认情况下，25%的数据被分配给测试集。最后，我们训练模型并用测试集测试。

R方值0.38表明38%的测试集数据都通过了测试。
'''
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(), scores)
'''
这里cross_val_score函数可以帮助我们轻松实现交叉检验功能。cv参数将数据集分成了5份。每个分区都会轮流作为测试集使用。cross_val_score函数返回模拟器score方法的结果。R方结果是在0.13到0.37之间，均值0.29，是模拟器模拟出的结果，相比单个训练/测试集的效果要好。
'''
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
plt.scatter(y_test, y_predictions)
plt.xlabel('实际品质',fontproperties=font)
plt.ylabel('预测品质',fontproperties=font)
plt.title('预测品质与实际品质',fontproperties=font)
plt.show()