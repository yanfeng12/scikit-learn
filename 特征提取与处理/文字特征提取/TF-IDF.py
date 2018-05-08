#带TF-IDF权重的扩展词库
#单词频率

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
#过滤英语停用词the、a，和单字母I
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
#[[2 1 3 1 1]]
#结果中第一行是单词的频率，dog频率为1，sandwich频率为3。注意和前面不同的是，binary=True没有了，因为binary默认是False，这样返回的是词汇表的词频，不是二进制结果[1 1 1 1 1]。