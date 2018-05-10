'''
现实中有很多问题不只是分成两类，许多问题都需要分成多个类，成为多类分类问题（Multi-class classification）。
比如听到一首歌的样曲之后，可以将其归入某一种音乐风格。这类风格就有许多种。
scikit-learn用one-vs.-all或one-vs.-the-rest方法实现多类分类，就是把多类中的每个类都作为二元分类处理。
分类器预测样本不同类型，将具有最大置信水平的类型作为样本类型。
LogisticRegression()通过one-vs.-all策略支持多类分类。
'''
#利用烂番茄（Rotten Tomatoes）网站影评短语数据对电影进行评价。每个影评可以归入下面5个类项：不给力（negative），不太给力（somewhat negative），中等（neutral），有点给力（somewhat positive）, 给力（positive）。
import zipfile
import pandas as pd
# 压缩节省空间
z = zipfile.ZipFile('train.zip')
#tsv格式也可以
df = pd.read_csv(z.open(z.namelist()[0]), header=0, delimiter='\t')
print(df.head())
print(df.count())  
'''
Sentiment是响应变量，0是不给力（negative），4是给力（positive），其他以此类推。
Phrase列是影评的内容。影评中每句话都被分割成一行。我们不需要考虑PhraseId列和SentenceId列。
'''
print(df.Phrase.head(10))
print(df.Sentiment.describe())
print(df.Sentiment.value_counts())


print(df.Sentiment.value_counts()/df.Sentiment.count())
'''
可以看出，近51%都是评价为2中等（neutral）的电影。
可见，在这个问题里，准确率不是一个有信息量的评价指标，因为即使很烂的分类器预测出中等水平的结果，其准确率也是51%。
3有点给力（somewhat positive）的电影占21%, 4给力（positive）的电影占6%，共占27%。剩下的21%就是不给力（negative），不太给力（somewhat negative）的电影。
'''