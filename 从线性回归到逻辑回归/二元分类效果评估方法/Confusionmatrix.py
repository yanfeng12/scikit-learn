'''
二元分类的效果评估方法有很多，常见的包括第一章里介绍的肿瘤预测使用的准确率（accuracy），精确率（precision）和召回率（recall）三项指标，以及综合评价指标（F1 measure）， ROC AUC值（Receiver Operating Characteristic ROC，Area Under Curve，AUC）。
这些指标评价的样本分类是真阳性（true positives），真阴性（true negatives），假阳性（false positives），假阴性（false negatives）。
阳性和阴性指分类，真和假指预测的正确与否。
'''
'''
在我们的垃圾短信分类里，真阳性是指分类器将一个垃圾短信分辨为spam类。
真阴性是指分类器将一个正常短信分辨为ham类。假阳性是指分类器将一个正常短信分辨为spam类。假阴性是指分类器将一个垃圾短信分辨为ham类。
混淆矩阵（Confusion matrix），也称列联表分析（Contingency table）可以用来描述真假与阴阳的关系。矩阵的行表示实际类型，列表示预测类型。
'''
#真阳性(True positive)TP
#假阳性(False positive）FP
#真阴性(True negative)TN
#假阴性(False negative）FN
#%matplotlib inline
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
#混淆矩阵（Confusion matrix）
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('混淆矩阵',fontproperties=font)
plt.colorbar()
plt.ylabel('实际类型',fontproperties=font)
plt.xlabel('预测类型',fontproperties=font)
plt.show()