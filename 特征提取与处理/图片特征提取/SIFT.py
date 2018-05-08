#尺度不变特征转换（Scale-Invariant Feature Transform，SIFT）是一种特征提取方法，相比前面使用的方法，SIFT对图像的尺寸，旋转，亮度变化更不敏感。
import mahotas as mh
from mahotas.features import surf

image = mh.imread('3.2 xiaobao.png', as_grey=True)
print('第一个SURF描述符：\n{}\n'.format(surf.surf(image)[0]))
print('抽取了%s个SURF描述符' % len(surf.surf(image)))