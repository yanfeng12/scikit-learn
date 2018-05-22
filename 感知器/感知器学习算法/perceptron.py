#感知器是Frank Rosenblatt在1957年就职于Cornell航空实验室(Cornell Aeronautical Laboratory)时发明的，其灵感源自于对人脑的仿真。
#感知器二元分类
'''
下面我们来解决一个分类案例。
假设你想从一堆猫里分辨幼猫（kitten）和成年猫（adult cats）。
数据集只有两个解释变量：用来睡觉的天数比例，闹脾气的天数比例。
'''
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)


import numpy as np

X = np.array([
    [0.2, 0.1],
    [0.4, 0.6],
    [0.5, 0.2],
    [0.7, 0.9]
])

y = [0, 0, 0, 1]

markers = ['.', 'x']
plt.scatter(X[:3, 0], X[:3, 1], marker='.', s=400)
plt.scatter(X[3, 0], X[3, 1], marker='x', s=400)
plt.xlabel('用来睡觉的天数比例',fontproperties=font)
plt.ylabel('闹脾气的天数比例',fontproperties=font)
plt.title('幼猫和成年猫',fontproperties=font)
plt.show()



