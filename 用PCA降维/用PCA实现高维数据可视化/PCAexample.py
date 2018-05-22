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


#以一组二维的数据data为例，data如下，一共12个样本（x,y），其实就是分布在直线y=x上的点，并且聚集在x=1、2、3、4上，各3个。
import numpy as np
data = np.array([[ 1.  ,  1.  ],  
       [ 0.9 ,  0.95],  
       [ 1.01,  1.03],  
       [ 2.  ,  2.  ],  
       [ 2.03,  2.06],  
       [ 1.98,  1.89],  
       [ 3.  ,  3.  ],  
       [ 3.03,  3.05],  
       [ 2.89,  3.1 ],  
       [ 4.  ,  4.  ],  
       [ 4.06,  4.02],  
       [ 3.97,  4.01]])  
data2 = np.array([[ 1.  ,  1.  ],  
       [ 0.9 ,  0.95],  
       [ 1.01,  1.03],  
       [ 2.  ,  2.  ],  
       [ 2.03,  2.06],  
       [ 1.98,  1.89],  
       [ 3.  ,  3.  ],  
       [ 3.03,  3.05],  
       [ 2.89,  3.1 ],  
       [ 4.  ,  4.  ],  
       [ 4.06,  4.02],  
       [ 3.97,  4.01]]) 
#data这组数据，有两个特征，因为两个特征是近似相等的，所以用一个特征就能表示了，即可以降到一维。下面就来看看怎么用sklearn中的PCA算法包。


#（1）n_components设置为1，copy默认为True，可以看到原始数据data并未改变，newData是一维的，并且明显地将原始数据分成了四类。
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
newData = pca.fit_transform(data)
print(newData)
print(data)


#（2）将copy设置为False，原始数据data将发生改变。
pca2 = PCA(n_components=1,copy=False)
newData2 = pca2.fit_transform(data2)
print(data2)


#（3）n_components设置为'mle'，看看效果，自动降到了1维。
pca3 = PCA(n_components='mle')
newData3 = pca3.fit_transform(data)
print(newData3)


#（4）对象的属性值
'''
components_ ：返回具有最大方差的成分。
explained_variance_ratio_：返回 所保留的n个成分各自的方差百分比。
n_components_：返回所保留的成分个数n。
'''
print(pca.n_components)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
