#肘部法则
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
import numpy as np
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T
plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
#cdist: 计算两个矩阵行间的所有向量对的距离(Computes distance between each pair of the two collections of inputs.)

K = range(1, 10)
meandistortions = []
for k in K:
    #n_clusters：即我们的k值，一般要多试一些值以获得较好的聚类效果。
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('平均畸变程度',fontproperties=font)
plt.title('用肘部法则来确定最佳的K值',fontproperties=font)
plt.show()
