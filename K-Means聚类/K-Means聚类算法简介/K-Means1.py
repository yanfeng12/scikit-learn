#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
import numpy as np
#创建数组： np.array 索引从0开始
X0 = np.array([7, 5, 7, 3, 4, 1, 0, 2, 8, 6, 5, 3])
X1 = np.array([5, 7, 7, 3, 6, 4, 0, 2, 7, 8, 5, 7])
plt.figure()
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(X0, X1, 'k.')


#range(start, stop[, step])从start开始不包括stop
#set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
#list() 方法用于将元组转换为列表。
#注：元组与列表是非常类似的，区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。
C1 = [1, 4, 5, 9, 11]
C2 = list(set(range(12)) - set(C1))
#0,2,3,6,7,8,10
#列表索引可以是数组和list。返回的数据不和原来的数据共享内存。索引可以是list和array
'''
x=np.arange(10)  
index=[1,2,3,4,5]  
arr_index=np.array(index)  
print x  
print x[index]  # list索引  
print x[arr_index]  # array索引 

[0 1 2 3 4 5 6 7 8 9]
[1 2 3 4 5]
[1 2 3 4 5]
'''
X0C1, X1C1 = X0[C1], X1[C1]
X0C2, X1C2 = X0[C2], X1[C2]
#X0C2:7,7,3,0,2,8,5
#X1C2:5,7,3,0,2,7,5
plt.figure()
plt.title('第一次迭代后聚类结果',fontproperties=font)
plt.axis([-1, 9, -1, 9])
plt.grid(True)
#第一类用X表示，第二类用点表示
plt.plot(X0C1, X1C1, 'rx')
plt.plot(X0C2, X1C2, 'g.')
#重心位置用稍大的点突出显示。
plt.plot(4,6,'rx',ms=12.0)
plt.plot(5,5,'g.',ms=12.0)

C1 = [1, 2, 4, 8, 9, 11]
C2 = list(set(range(12)) - set(C1))
X0C1, X1C1 = X0[C1], X1[C1]
X0C2, X1C2 = X0[C2], X1[C2]
plt.figure()
plt.title('第二次迭代后聚类结果',fontproperties=font)
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(X0C1, X1C1, 'rx')
plt.plot(X0C2, X1C2, 'g.')
plt.plot(3.8,6.4,'rx',ms=12.0)
plt.plot(4.57,4.14,'g.',ms=12.0)


C1 = [0, 1, 2, 4, 8, 9, 10, 11]
C2 = list(set(range(12)) - set(C1))
X0C1, X1C1 = X0[C1], X1[C1]
X0C2, X1C2 = X0[C2], X1[C2]
plt.figure()
plt.title('第三次迭代后聚类结果',fontproperties=font)
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(X0C1, X1C1, 'rx')
plt.plot(X0C2, X1C2, 'g.')
plt.plot(5.5,7.0,'rx',ms=12.0)
plt.plot(2.2,2.8,'g.',ms=12.0)
plt.show()