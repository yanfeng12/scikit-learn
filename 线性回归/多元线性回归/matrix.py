from numpy.linalg import inv
from numpy import dot, transpose
#直径6、8、10、14、18
#辅料种类2、1、0、2、0
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]
print(dot(inv(dot(transpose(X), X)), dot(transpose(X), y)))
#Numpy也提供了最小二乘法函数来实现这一过程
from numpy.linalg import lstsq
print(lstsq(X, y)[0])

