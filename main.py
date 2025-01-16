import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

tdata = pd.read_csv('customer_data.csv')
del tdata['end_date']   #欠損行を早めに削除
print(tdata.count())
Y = tdata['is_deleted'] #   目的変数を指定
X = tdata
X['class'] = X['class'].map({'C01':1,'C02':2,'C03':3})  #　classを文字列からintとして置き換え
X['mean'] = X['mean'].round(1)  #   平均値の少数を減少
X = tdata.drop(['customer_id','gender','start_date','is_deleted','price'],axis=1)   #   不要（欠損行のある）列を予め削除
print(X.head())

model = DecisionTreeClassifier(max_depth=3,random_state=0)
model.fit(X,Y)
plt.figure(figsize=(16,8))
tree.plot_tree(model,
                feature_names=X.columns, class_names=True,
                filled=True, fontsize=8)
plt.show()
