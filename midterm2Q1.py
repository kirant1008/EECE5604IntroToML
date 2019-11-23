# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:21:56 2019

@author: kiran
"""
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydotplus import graph_from_dot_data
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

import os

def plot_OGData(og_data):
    
    
    for i in range(0,1000):
        if og_data[i,2]>0:
            plt.pyplot.scatter(og_data[i,0],og_data[i,1],marker='x',color='black')
        elif og_data[i,2]<0:
            plt.pyplot.scatter(og_data[i,0],og_data[i,1],marker='o',color='red')
    
    # labeling the x axis 
    plt.pyplot.xlabel('Feature value X1') 
    # labelingthe y axis 
    plt.pyplot.ylabel('Feature value X2')   
    # Title 
    plt.pyplot.title('Plot of Decision Boundary and DataSet')
    plt.pyplot.legend(['-1','+1'])
    plt.pyplot.grid()

def plot_decisionRegion(og_data,clf):
# Plotting decision regions
    x_min, x_max = og_data[:, 0].min() - 1, og_data[:, 0].max() + 1
    y_min, y_max = og_data[:, 1].min() - 1, og_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    
    f, axarr = plt.pyplot.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plot_OGData(og_data)
    
    axarr.contour(xx, yy, Z, alpha=0.5)
    plt.pyplot.show()

def create_Tree(clf_ID3,X):
    dot_data = StringIO()
    export_graphviz(clf_ID3, out_file=dot_data, filled=True,
                    rounded=True,special_characters=True,feature_names=['x1','x2'])
    graph = graph_from_dot_data(dot_data.getvalue())
    graph.write_png(X)
    Image(graph.create_png())
    
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

og_data = np.loadtxt('Q1.csv',delimiter=',')
#Spliting Training and Test DataSet
x_train = og_data[100:999,0:2]
y_train = og_data[100:999,2]

x_test = og_data[0:100,0:2]
y_test = og_data[0:100,2]

#Plotting the dataset
plot_OGData(og_data)

#Creating a ID3 classifier

clf_ID3 = DecisionTreeClassifier(criterion='entropy',max_depth=11,min_impurity_decrease=0.01)

clf_ID3.fit(x_train,y_train)

#Creatinf the tree for ID3 algorithm


#Creating Tree 
create_Tree(clf_ID3,'ID3_tree.png')
y_pred = clf_ID3.predict(x_test)

#Accuracy of ID3
acc_ID3=accuracy_score(y_test, y_pred)
#Confusion Matrix for ID3
cfmatrix_ID3=pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['-1', '1'],index=['-1', '1'])
plot_decisionRegion(og_data,clf_ID3)


#Bagging Algorithm
cart = DecisionTreeClassifier(criterion='entropy',max_depth=11,min_impurity_decrease=0.01)
clf_bagg = BaggingClassifier(base_estimator=cart, n_estimators=7,bootstrap=True,random_state=3)
clf_bagg.fit(x_train,y_train)
bagg_pred = clf_bagg.predict(x_test)
accuracy = accuracy_score(y_test , bagg_pred)
#print(clf_bagg.estimators_)
pred_bagg=np.zeros([100,7])

names = ['tree1.png','tree2.png','tree3.png','tree4.png','tree5.png','tree6.png','tree7.png']
for i, est in enumerate(clf_bagg.estimators_):
    est.fit(x_train,y_train)
    create_Tree(est,names[i])
    pred_bagg[:,i]=est.predict(x_test) 

pred_bagg_avg= np.zeros([100,1])
for i in range(0,100):
    pred_bagg_avg[i] = max(pred_bagg[i,:])
    
cfmatrix_bagg=pd.DataFrame(confusion_matrix(y_test, bagg_pred), columns=['-1', '1'],index=['-1', '1'])
plot_decisionRegion(og_data,clf_bagg)

acc_bagg=accuracy_score(y_test, pred_bagg_avg)

#Adaboost Algorithm
cart2 = DecisionTreeClassifier(criterion='entropy',max_depth=11,min_impurity_decrease=0.01)
clf_adaboost = AdaBoostClassifier(base_estimator= cart2,algorithm='SAMME',n_estimators = 7)
clf_adaboost.fit(x_train,y_train)
weights = clf_adaboost.estimator_weights_
pred_adaboost= clf_adaboost.predict(x_test)
cfmatrix_adaboost=pd.DataFrame(confusion_matrix(y_test, pred_adaboost), columns=['-1', '1'],index=['-1', '1'])
plot_decisionRegion(og_data,clf_adaboost)
acc_adaboost=accuracy_score(y_test, pred_adaboost)
print(clf_adaboost.estimator_weights_)
