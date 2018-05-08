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
#plt.show()


from sklearn.linear_model import LinearRegression
# 创建并拟合模型
model = LinearRegression()
model.fit(X, y)
#print('预测一张12英寸匹萨价格：$%.2f' % model.predict([12])[0]
import numpy as np  
temp =  [12] #an instance  
temp = np.array(temp).reshape((1, -1))  
print('预测一张12英寸匹萨价格：$%.2f' % model.predict(temp))


X2 = [[0], [10], [14], [25]]
y2 = model.predict(X2)
plt = runplt()
plt.plot(X, y, 'k.')
y3 = [14.25, 14.25, 14.25, 14.25]
y4 = y2 * 0.5 + 5
model.fit(X[1:-1], y[1:-1])
y5 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-.')
plt.plot(X2, y3, 'r-.')
plt.plot(X2, y4, 'y-.')
plt.plot(X2, y5, 'o-')

# 残差预测值
yr = model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')

plt.show()

'''
对模型的拟合度进行评估的函数称为残差平方和（residual sum of squares）成本函数。就是 让所有训练数据与模型的残差的平方之和最小化

'''
print('残差平方和: %.2f' % np.mean((model.predict(X) - y) ** 2))