#%matplotlib inline
'''
Matplotlib 是一个 Python 的 2D绘图库，它以各种硬拷贝格式和跨平台的交互式环境生成出版质量级别的图形 [1]  。
通过 Matplotlib，开发者可以仅需要几行代码，便可以生成绘图，直方图，功率谱，条形图，错误图，散点图等。  
'''
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)


def runplt():
    plt.figure()
    plt.title('匹萨价格与直径数据',fontproperties=font)
    plt.xlabel('直径（英寸）',fontproperties=font)
    plt.ylabel('价格（美元）',fontproperties=font)
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt

plt = runplt()
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.plot(X, y, 'k.')
plt.show()


from sklearn.linear_model import LinearRegression
# 创建并拟合模型
model = LinearRegression()
model.fit(X, y)
#print('预测一张12英寸匹萨价格：$%.2f' % model.predict([12])[0]
import numpy as np  
temp =  [12] #an instance  
temp = np.array(temp).reshape((1, -1))  
print('预测一张12英寸匹萨价格：$%.2f' % model.predict(temp))


'''
上述代码中sklearn.linear_model.LinearRegression类是一个估计器（estimator）。
估计器依据观测值来预测结果。在scikit-learn里面，所有的估计器都带有fit()和predict()方法。
fit()用来分析模型参数，predict()是通过fit()算出的模型参数构成的模型，对解释变量进行预测获得的值。
因为所有的估计器都有这两种方法，所有scikit-learn很容易实验不同的模型。
LinearRegression类的fit()方法学习下面的一元线性回归模型：
y = α + βx
y表示响应变量的预测值，本例指匹萨价格预测值，x是解释变量，本例指匹萨直径。截距α和相关系数β是线性回归模型最关心的事情。
'''

#一元线性回归拟合模型的参数估计常用方法是普通最小二乘法（ordinary least squares ）或线性最小 二乘法（linear least squares）。