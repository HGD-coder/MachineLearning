import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

#读取数据
data=pd.read_csv('winequality-red.csv',sep=';') #从文件中读取data
#print(data.isnull().sum())  #统计有几个缺失值，为0所以不做后续处理

#拆分特征列与目标列
X = data.drop(labels='quality',axis=1).copy()	#X是特征列
y = data['quality'].copy()	                    #y是目标列
for i in range(1599):
    y[i]=np.where(y[i]<=5,-1,1)#将Quality大于5为优质，定为1，反之则为劣质，定为-1

#8:2划分训练集与测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)      

#PCA降维
pca=PCA(n_components=7)
pca.fit(X_train)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)

#归一化
X_scaler = MinMaxScaler()
X_train_scaled=X_scaler.fit_transform(X_train_pca)
X_test_scaled=X_scaler.transform(X_test_pca)

#网格搜索交叉验证来调整SVC函数中的超参数
params = {
    'gamma':[0.01, 0.1, 0.5, 1, 2, 10, 100],
    'C': [0.01, 0.1, 0.5, 1, 2, 10, 100]
    }
clf=svm.SVC()#选用SVM模型
grid=GridSearchCV(clf,params,cv=10,n_jobs=-1)
grid.fit(X_train_scaled,y_train)

#结果展示
y_pred=grid.predict(X_test_scaled)
print('最佳分类器：',grid.best_estimator_)
print("混淆矩阵\n",confusion_matrix(y_test,y_pred))
print("分类报告\n",classification_report(y_test,y_pred))

#PR曲线
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recall, precision)
plt.title('Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()




