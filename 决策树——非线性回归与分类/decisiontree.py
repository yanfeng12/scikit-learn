'''
我们用互联网广告数据集（Internet Advertisements Data Set）来实现分类器，里面包含了3279张图片。
不过类型的比例并不协调，459幅广告图片，2820幅正常内容。
决策树学习算法可以从比例并不协调的数据集中生成一个不平衡的决策树（biased tree）。
在决定是否值得通过过抽样（over-sampling）和欠抽样（under-sampling）的方法平衡训练集之前，我们将用不相关的数据集对模型进行评估。
本例的解释变量就是图片的尺寸，网址链接里的单词，以及图片标签周围的单词。
响应变量就是图片的类型。解释变量已经被转换成特征向量了。
前三个特征值表示宽度，高度，图像纵横比（aspect ratio）。
剩下的特征是文本变量的二元频率值。
下面，我们用网格搜索来确定决策树模型最大最优评价效果（F1 score）的超参数，然后把决策树用在测试集进行效果评估。
'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

#我们创建了pipeline和DecisionTreeClassifier类的实例，将criterion参数设置成entropy，这样表示使用信息增益启发式算法建立决策树。
pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
])


#我们创建了pipeline和DecisionTreeClassifier类的实例，将criterion参数设置成entropy，这样表示使用信息增益启发式算法建立决策树。
pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
])


#然后，我们确定网格搜索的参数范围。
parameters = {
    'clf__max_depth': (150, 155, 160),
    'clf__min_samples_split': (1, 2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}


import zipfile
if __name__ == '__main__':
    # 压缩节省空间
    z = zipfile.ZipFile('ad.zip')
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)

    explanatory_variable_columns = set(df.columns.values)
    response_variable_column = df[len(df.columns.values)-1]
    # The last column describes the targets
    explanatory_variable_columns.remove(len(df.columns.values)-1)

    y = [1 if e == 'ad.' else 0 for e in response_variable_column]
    X = df.loc[:, list(explanatory_variable_columns)]


    #我们把广告图片设为阳性类型，正文图片设为阴性类型。超过1/4的图片其宽带或高度的值不完整， 用空白加问号（“ ?”）表示，我们用正则表达式替换为-1，方便计算。然后我们用交叉检验对训练集 和测试集进行分割。
    X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    #最后将GridSearchCV的搜索目标scoring设置为f1。
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train, y_train)
    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))
