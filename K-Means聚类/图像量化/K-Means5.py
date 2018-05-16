#图像量化
'''
图像量化（image quantization）是一种将图像中相似颜色替换成同样颜色的有损压缩方法。
图像量化会减少图像的存储空间，由于表示不同颜色的字节减少了。
下面的例子中，我们将用聚类方法从一张图片中找出包含图片大多数颜色的压缩颜色调色板（palette），然后我们用这个压缩颜色调色板重新生成图片。
这个例子需要用mahotas图像处理库.
'''
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)


import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import mahotas as mh


original_img = np.array(mh.imread('6.6 tree.png'), dtype=np.float64) / 255
#Python 元组 tuple() 函数将列表转换为元组。
original_dimensions = tuple(original_img.shape)
width, height, depth = tuple(original_img.shape)
#改变数组的形状
image_flattened = np.reshape(original_img, (width * height, depth))
#然后我们用K-Means算法在随机选择1000个颜色样本中建立64个类。每个类都可能是压缩调色板中的一种颜色。
image_array_sample = shuffle(image_flattened, random_state=0)[:1000]
estimator = KMeans(n_clusters=64, random_state=0)
estimator.fit(image_array_sample)
#之后，我们为原始图片的每个像素进行类的分配。
cluster_assignments = estimator.predict(image_flattened)
#最后，我们建立通过压缩调色板和类分配结果创建压缩后的图片：
compressed_palette = estimator.cluster_centers_
compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_idx = 0
for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[cluster_assignments[label_idx]]
        label_idx += 1
plt.subplot(122)
plt.title('Original Image')
plt.imshow(original_img)
plt.axis('off')
plt.subplot(121)
plt.title('Compressed Image')
plt.imshow(compressed_img)
plt.axis('off')
plt.show()


