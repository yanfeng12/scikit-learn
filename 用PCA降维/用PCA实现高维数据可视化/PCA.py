#PCA（Principal Component Analysis，主成分分析）
#PCA可以把可能具有相关性的高维变量合成线性无关的低维变量，称为主成分（ principal components）。新的低维数据集会尽可能的保留原始数据的变量。
#PCA旋转数据集与其主成分对齐，将最多的变量保留到第一主成分中。
#二主成分必须与第一主成分正交，也就是说第二主成分必须是在统计学上独立（设A，B为随机事件，若同时发生的概率等于各自发生的概率的乘积，则A，B相互独立。）的。



#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
'''
Fisher1936年收集了三种鸢尾花分别50个样本数据（Iris Data）：Setosa、Virginica、Versicolour。
解释变量是花瓣（petals）和萼片（sepals）长度和宽度的测量值，响应变量是花的种类。
鸢尾花数据集经常用于分类模型测试，scikit-learn中也有。
'''
from sklearn.datasets import load_iris


#首先，我们导入鸢尾花数据集和PCA估计器。PCA类把主成分的数量作为超参数，和其他估计器一样，PCA也用fit_transform()返回降维的数据矩阵：
#sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False) 
''' 
n_components:
PCA算法中所要保留的特征个数n，也即保留下来的特征个数n.
int 或者 string，缺省时默认为None，所有成分被保留。
赋值为int，比如n_components=1，将把原始数据降到一个维度。
赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
'''
'''
copy:
类型：bool，True或者False，缺省时默认为True。
表示是否在运行算法时，将原始训练数据复制一份。
若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；
若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算。
'''
'''
whiten:
意义：白化，使得每个特征具有相同的方差。
类型：bool，缺省时默认为False
'''
data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)
#PCA对象的方法
'''
fit(X,y=None)
fit()可以说是scikit-learn中通用的方法，每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。因为PCA是无监督学习算法，此处y自然等于None。
fit(X)，表示用数据X来训练PCA模型。
函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。
'''
'''
fit_transform(X)
用X来训练PCA模型，同时返回降维后的数据。
newX=pca.fit_transform(X)，newX就是降维后的数据。
'''
'''
inverse_transform()
将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)
'''
'''
transform(X)
将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。 
'''


red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
