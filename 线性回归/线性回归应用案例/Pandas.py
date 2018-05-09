import pandas as pd
df = pd.read_csv('winequality-red.csv', sep=';')
df.head()
df.describe()


#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('酒精度（Alcohol）与品质（ Quality）',fontproperties=font)
plt.show()