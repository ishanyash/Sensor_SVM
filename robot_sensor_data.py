# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:59:57 2019

@author: Ishan Yash
"""

import numpy as np
import pandas as pd
from sklearn import svm
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation
from pyts.metrics import dtw
from pyts.metrics.dtw import (cost_matrix, accumulated_cost_matrix,
                              _return_path, _multiscale_region)
from pyts.classification import BOSSVS
from pyts.transformation import WEASEL
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

PATH = r"G:\Coding\ML\AllData_US readings\sensor_readings_24.data" # Change this value if necessary
df = pd.read_csv(PATH, header = None)
label = df.iloc[:,24]
feature = df.iloc[:,0:-1]
feature = feature.values.tolist()
feature = np.asarray(feature, dtype=np.float64)
label = label.values.tolist()
for i in range(len(label)):
    if label[i] =='Slight-Right-Turn':
        label[i]=0
    elif label[i]=='Sharp-Right-Turn':
        label[i]=1
    elif label[i]=='Move-Forward':
        label[i]=2
    elif label[i]=='Slight-Left-Turn':
        label[i]=3
label = np.asarray(label, dtype=np.float64)

        
#full_data = np.htack((feature,label))     
test_size = 0.3
X_train = feature[:-int(test_size*len(feature))]
X_test = feature[-int(test_size*len(feature)):]
y_train = label[:-int(test_size*len(label))]
y_test = label[-int(test_size*len(label)):]
clf = LogisticRegression(penalty='l2', C=1, fit_intercept=False,
                         solver='liblinear', multi_class='ovr')

#Feature extraction DTW
# Dynamic Time Warping: classic
DTW_Classic_test = []
DTW_Classic_train = []
for i in range(len(X_test)):
    for j in range(len(X_train)):
        dtw_classic, path_classic = dtw(X_test[i], X_train[j], dist='square',
                                method='classic', return_path=True)
        DTW_Classic_test.append(dtw_classic)

for i in range(len(X_train)):
    for j in range(len(X_train)):
        dtw_classic, path_classic = dtw(X_train[i], X_train[j], dist='square',
                                method='classic', return_path=True)
        DTW_Classic_train.append(dtw_classic)

DTW_Classic_train = np.array(DTW_Classic_train)
DTW_Classic_train.resize(y_train.shape[0],int(len(DTW_Classic_train)/y_train.shape[0]))
DTW_Classic_test = np.array(DTW_Classic_test)
DTW_Classic_test.resize(y_test.shape[0],int(len(DTW_Classic_test)/y_test.shape[0]))

train_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(DTW_Classic_train)
test_concat_mv = TimeSeriesScalerMeanVariance().fit_transform(DTW_Classic_test)

train_concat_mv.resize(DTW_Classic_train.shape[0],DTW_Classic_train.shape[1])
test_concat_mv.resize(DTW_Classic_test.shape[0],DTW_Classic_test.shape[1])

#SVM

clf = svm.SVC(gamma='scale')
clf.fit(DTW_Classic_train, y_train)

print('Accuracy: ',clf.score(DTW_Classic_test,y_test))
#WEASEL
weasel_adiac = WEASEL(word_size=5, window_sizes=np.arange(6, X_train.shape[1]))

pipeline_adiac = Pipeline([("weasel", weasel_adiac), ("clf", clf)])

accuracy_adiac = pipeline_adiac.fit(X_train, y_train).score(X_test, y_test)

print("Accuracy on the testing set: {0:.3f}".format(accuracy_adiac))