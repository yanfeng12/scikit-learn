'''
%matplotlib具体作用是当你调用matplotlib.pyplot的绘图函数plot()进行绘图的时候，或者生成一个figure画布的时候，可以直接在你的python console里面生成图像。
而我们在spyder或者pycharm实际运行代码的时候，可以直接注释掉这一句，也是可以运行成功的。

'''
#%matplotlib inline
#通过像素值提取特征
from sklearn import datasets
import matplotlib.pyplot as plt
digits = datasets.load_digits()
print('Digit:', digits.target[0])
print(digits.images[0])
plt.figure()
plt.axis('off')
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()