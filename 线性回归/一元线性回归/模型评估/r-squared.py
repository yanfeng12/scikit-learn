'''
用R方（r-squared）评估匹萨价格预测的效果。R方也叫确定系数（coefficient of determination），表示模型对现实数据拟合的程度。计算R方的方法有几种。一元线性回归中R方等于皮尔逊积矩相关系数（Pearson product moment correlation coefficient或Pearson's r）的平方。

这种方法计算的R方一定介于0～1之间的正数。其他计算方法，包括scikit-learn中的方法，不是用皮尔逊积矩相关系数的平方计算的，因此当模型拟合效果很差的时候R方会是负值。下面我们用scikit-learn方法来计算R方。
'''
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
# 测试集
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model.score(X_test, y_test)
print(model.score(X_test, y_test))