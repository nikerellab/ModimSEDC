# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 23:24:57 2022

code for training multi-omic classifiers for pancreatic cancer

@author: Ezgi
"""

import pandas as pd
import os
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, classification_report
import pickle
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    # Fit all estimators with their respective feature arrays
    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return label_encoder.inverse_transform(pred)
data_dir = '.../classification/LC_MultiOmic'
# NC data
txNC = np.transpose(np.array(pd.read_csv(join(data_dir,'TX/PC_TX_NC.csv'),sep=',',header=None)))
jxNC = np.transpose(np.array(pd.read_csv(join(data_dir,'JX/PC_JX_NC.csv'),sep=',',header=None)))
pxNC = np.transpose(np.array(pd.read_csv(join(data_dir,'PX/PC_PX_NC.csv'),sep=',',header=None)))
y = np.concatenate((np.tile(0,(21, 1)),np.tile(1, (140, 1))))
Xnc=np.c_[txNC, jxNC]
Xnc=np.c_[Xnc,pxNC]
b = np.isnan(Xnc)
Xnc[b] = 0

params = {'n_estimators': [20,50,100,200],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, 112//2, ],
          'min_samples_leaf': [1, 0.5, 112//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, 22173//2 ],}
          
X_train, X_test, y_train, y_test = train_test_split(Xnc, y,random_state=10,test_size=0.3)
Xtrain1, Xtrain2, Xtrain3 = X_train[:,:28057], X_train[:,28057:41691], X_train[:,41691:]
Xtest1, Xtest2, Xtest3 = X_test[:,:28057], X_test[:,28057:41691], X_test[:,41691:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3]
X_test_list = [Xtest1, Xtest2, Xtest3]
# scaler =StandardScaler().fit(Xtrain1)
# X_train1 = scaler.transform(Xtrain1)
# X_test1 = scaler.transform(Xtest1)
# scaler =StandardScaler().fit(Xtrain2)
# X_train2 = scaler.transform(Xtrain2)
# X_test2 = scaler.transform(Xtest2)
# scaler =StandardScaler().fit(Xtrain3)
# X_train3 = scaler.transform(Xtrain3)
# X_test3 = scaler.transform(Xtest3)
# X_train_list = [X_train1, X_train2, X_train3]
# X_test_list = [X_test1, X_test2, X_test3]

classifiers = [('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=TXGrid.best_params_["max_depth"], 
                               max_features=TXGrid.best_params_["max_features"],
                               min_samples_leaf=TXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=TXGrid.best_params_["min_samples_split"],
                               n_estimators=TXGrid.best_params_["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_
                               ["n_estimators"]))]

fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, np.ravel(y_train))
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

print(classification_report(y_test, y_pred,target_names=['N','C']))



TXGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
TXGrid.fit(Xtrain2,np.ravel(y_train))
TXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=TXGrid.best_params_["max_depth"], 
                               max_features=TXGrid.best_params_["max_features"],
                               min_samples_leaf=TXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=TXGrid.best_params_["min_samples_split"],
                               n_estimators=TXGrid.best_params_
                               ["n_estimators"])
TXclf.fit(Xtrain2, np.ravel(y_train))
y_pred = TXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['N','C']))

JXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
JXGrid.fit(Xtrain2,np.ravel(y_train))
JXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_
                               ["n_estimators"])
JXclf.fit(Xtrain2, np.ravel(y_train))
y_pred = JXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['N','C']))

PXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
PXGrid.fit(Xtrain3,np.ravel(y_train))
PXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=PXGrid.best_params_["max_depth"], 
                               max_features=PXGrid.best_params_["max_features"],
                               min_samples_leaf=PXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=PXGrid.best_params_["min_samples_split"],
                               n_estimators=PXGrid.best_params_
                               ["n_estimators"])
PXclf.fit(Xtrain3, np.ravel(y_train))
y_pred = PXclf.predict(Xtest3)
print(classification_report(y_test, y_pred,target_names=['N','C']))

class_names = ['N', 'C']
explainer = shap.TreeExplainer(TXclf)
shap_values = explainer.shap_values(Xtest1)
explainer = shap.TreeExplainer(JXclf)
shap_values = explainer.shap_values(Xtest2)
explainer = shap.TreeExplainer(PXclf)
shap_values = explainer.shap_values(Xtest3)

NmeanSHAP = np.mean(np.abs(shap_values[0]),axis=0)
TmeanSHAP = np.mean(np.abs(shap_values[1]),axis=0)
# shap.summary_plot(shap_values, Xtest1, plot_type="bar", class_names= class_names, max_display=50)

# Stage data
txS = np.transpose(np.array(pd.read_csv(join(data_dir,'TX/PC_TX_St.csv'),sep=',',header=None)))
jxS = np.transpose(np.array(pd.read_csv(join(data_dir,'JX/PC_JX_St.csv'),sep=',',header=None)))
pxS = np.transpose(np.array(pd.read_csv(join(data_dir,'PX/PC_PX_St.csv'),sep=',',header=None)))
gxS = np.transpose(np.array(pd.read_csv(join(data_dir,'GX/PC_GX_St.csv'),sep=',',header=None)))

samples = pd.read_csv(join(data_dir,'StlabelsPC.csv'),sep=',')
mxS = mxS[:,1:]
X=np.c_[gxS, txS]
X=np.c_[X,jxS]
X=np.c_[X,pxS]
X=np.c_[X,mxS]

y = samples.values[:,0]
b =  pd.isnull(X)
X[b] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=10,test_size=0.3)
Xtrain1, Xtrain2, Xtrain3 , Xtrain4, Xtrain5 = X_train[:,:22173], X_train[:,22173:50230], X_train[:,50230:63864],X_train[:,63864:86037],X_train[:,86037:]
Xtest1, Xtest2, Xtest3, Xtest4, Xtest5 = X_test[:,:22173], X_test[:,22173:50230], X_test[:,50230:63864],X_test[:,63864:86037],X_test[:,86037:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3, Xtrain4, Xtrain5]
X_test_list = [Xtest1, Xtest2, Xtest3, Xtest4, Xtest5]

classifiers = [('rf',  RandomForestClassifier(random_state=3,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=3,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=3,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=3,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=3,bootstrap=True))]

fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

print(classification_report(y_test, y_pred,target_names=['Stage I','Other' ]))

## ----------------- early diagnosis
X2 = Xnc[0:21,:]
Xc = Xnc[21:161,:]
ind = stlabel.values=='Stage I'
b2 = pd.DataFrame(ind)
b3 = b2[0]
Xc2 = Xc[b3,:]
Xe = np.concatenate((X2,Xc2))
y = np.concatenate((np.tile(0,(21, 1)),np.tile(1, (16, 1))))

X_train, X_test, y_train, y_test = train_test_split(Xe, y,random_state=10,test_size=0.3)
Xtrain1, Xtrain2, Xtrain3 = X_train[:,:28057], X_train[:,28057:41691], X_train[:,41691:]
Xtest1, Xtest2, Xtest3 = X_test[:,:28057], X_test[:,28057:41691], X_test[:,41691:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3]
X_test_list = [Xtest1, Xtest2, Xtest3]

classifiers = [('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=TXGrid.best_params_["max_depth"], 
                               max_features=TXGrid.best_params_["max_features"],
                               min_samples_leaf=TXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=TXGrid.best_params_["min_samples_split"],
                               n_estimators=TXGrid.best_params_["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=PXGrid.best_params_["max_depth"], 
                               max_features=PXGrid.best_params_["max_features"],
                               min_samples_leaf=PXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=PXGrid.best_params_["min_samples_split"],
                               n_estimators=PXGrid.best_params_
                               ["n_estimators"]))]
classifiers = [('rf',  RandomForestClassifier(random_state=10,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True))]
fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, np.ravel(y_train))
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

print(classification_report(y_test, y_pred,target_names=['N','S1']))



TXGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
TXGrid.fit(Xtrain2,np.ravel(y_train))
TXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=TXGrid.best_params_["max_depth"], 
                               max_features=TXGrid.best_params_["max_features"],
                               min_samples_leaf=TXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=TXGrid.best_params_["min_samples_split"],
                               n_estimators=TXGrid.best_params_
                               ["n_estimators"])
TXclf = RandomForestClassifier(random_state=10,bootstrap=True)
TXclf.fit(Xtrain1, np.ravel(y_train))
y_pred = TXclf.predict(Xtest1)
print(classification_report(y_test, y_pred,target_names=['N','S1']))

JXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
JXGrid.fit(Xtrain2,np.ravel(y_train))
JXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_
                               ["n_estimators"])
JXclf = RandomForestClassifier(random_state=10,bootstrap=True)
JXclf.fit(Xtrain2, np.ravel(y_train))
y_pred = JXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['N','S1']))

PXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
PXGrid.fit(Xtrain3,np.ravel(y_train))
PXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=PXGrid.best_params_["max_depth"], 
                               max_features=PXGrid.best_params_["max_features"],
                               min_samples_leaf=PXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=PXGrid.best_params_["min_samples_split"],
                               n_estimators=PXGrid.best_params_
                               ["n_estimators"])
PXclf = RandomForestClassifier(random_state=10,bootstrap=True)
PXclf.fit(Xtrain3, np.ravel(y_train))
y_pred = PXclf.predict(Xtest3)
print(classification_report(y_test, y_pred,target_names=['N','S1']))

class_names = ['N','S1']
explainer = shap.TreeExplainer(TXclf)
shap_values = explainer.shap_values(Xtest1)
explainer = shap.TreeExplainer(JXclf)
shap_values = explainer.shap_values(Xtest2)
explainer = shap.TreeExplainer(PXclf)
shap_values = explainer.shap_values(Xtest3)

NmeanSHAP = np.mean(np.abs(shap_values[0]),axis=0)
TmeanSHAP = np.mean(np.abs(shap_values[1]),axis=0)


################################3
S1_Y = list()
for i in range(0,len(y)):
	if y[i]=='Stage I':
		S1_Y.append(0)
	else:
		S1_Y.append(1)

X_train, X_test, y_train, y_test = train_test_split(X, S1_Y,random_state=3,test_size=0.3)
Xtrain1, Xtrain2, Xtrain3 , Xtrain4, Xtrain5 = X_train[:,:22173], X_train[:,22173:50230], X_train[:,50230:63864],X_train[:,63864:86037],X_train[:,86037:]
Xtest1, Xtest2, Xtest3, Xtest4, Xtest5 = X_test[:,:22173], X_test[:,22173:50230], X_test[:,50230:63864],X_test[:,63864:86037],X_test[:,86037:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3, Xtrain4, Xtrain5]
X_test_list = [Xtest1, Xtest2, Xtest3, Xtest4, Xtest5]

classifiers = [('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=GXGrid.best_params_["max_depth"], 
                               max_features=GXGrid.best_params_["max_features"],
                               min_samples_leaf=GXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=GXGrid.best_params_["min_samples_split"],
                               n_estimators=GXGrid.best_params_["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=TXGrid.best_params_["max_depth"], 
                               max_features=TXGrid.best_params_["max_features"],
                               min_samples_leaf=TXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=TXGrid.best_params_["min_samples_split"],
                               n_estimators=TXGrid.best_params_["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_
                               ["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=PXGrid.best_params_["max_depth"], 
                               max_features=PXGrid.best_params_["max_features"],
                               min_samples_leaf=PXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=PXGrid.best_params_["min_samples_split"],
                               n_estimators=PXGrid.best_params_["n_estimators"])),
    ('rf',  RandomForestClassifier(random_state=10,bootstrap=True,max_depth=MuXGrid.best_params_["max_depth"], 
                               max_features=MuXGrid.best_params_["max_features"],
                               min_samples_leaf=MuXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=MuXGrid.best_params_["min_samples_split"],
                               n_estimators=MuXGrid.best_params_["n_estimators"]))]

params = {'n_estimators': [20,50,100,200],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, 96//2, ],
          'min_samples_leaf': [1, 0.5, 96//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, 28057//2 ],}
TXGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
TXGrid.fit(Xtrain2,np.ravel(y_train))
TXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=TXGrid.best_params_["max_depth"], 
                               max_features=TXGrid.best_params_["max_features"],
                               min_samples_leaf=TXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=TXGrid.best_params_["min_samples_split"],
                               n_estimators=TXGrid.best_params_
                               ["n_estimators"])
TXclf.fit(Xtrain2, np.ravel(y_train))
y_pred = TXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['S1','N' ]))

params = {'n_estimators': [20,50,100,200],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, 96//2, ],
          'min_samples_leaf': [1, 0.5, 96//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, 13634//2 ],}
JXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
JXGrid.fit(Xtrain3,np.ravel(y_train))
JXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_
                               ["n_estimators"])
JXclf.fit(Xtrain3, np.ravel(y_train))
y_pred = JXclf.predict(Xtest3)
print(classification_report(y_test, y_pred,target_names=['S1','N' ]))

params = {'n_estimators': [20,50,100,200],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, 96//2, ],
          'min_samples_leaf': [1, 0.5, 96//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, 22173//2 ],}
PXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
PXGrid.fit(Xtrain4,np.ravel(y_train))
PXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=PXGrid.best_params_["max_depth"], 
                               max_features=PXGrid.best_params_["max_features"],
                               min_samples_leaf=PXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=PXGrid.best_params_["min_samples_split"],
                               n_estimators=PXGrid.best_params_
                               ["n_estimators"])
PXclf.fit(Xtrain4, np.ravel(y_train))
y_pred = PXclf.predict(Xtest4)
print(classification_report(y_test, y_pred,target_names=['S1','N' ]))

params = {'n_estimators': [20,50,100,200],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, 96//2, ],
          'min_samples_leaf': [1, 0.5, 96//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, 22173//2 ],}
GXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
GXGrid.fit(Xtrain2,np.ravel(y_train))
GXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=GXGrid.best_params_["max_depth"], 
                               max_features=GXGrid.best_params_["max_features"],
                               min_samples_leaf=GXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=GXGrid.best_params_["min_samples_split"],
                               n_estimators=GXGrid.best_params_
                               ["n_estimators"])
GXclf.fit(Xtrain2, np.ravel(y_train))
y_pred = GXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['S1','N' ]))

params = {'n_estimators': [20,50,100,200],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, 112//2, ],
          'min_samples_leaf': [1, 0.5, 112//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, 22173//2 ],}
MuXGrid = GridSearchCV(RandomForestClassifier(random_state=10,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
MuXGrid.fit(Xtrain2,np.ravel(y_train))
MuXclf = RandomForestClassifier(random_state=10,bootstrap=True,max_depth=MuXGrid.best_params_["max_depth"], 
                               max_features=MuXGrid.best_params_["max_features"],
                               min_samples_leaf=MuXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=MuXGrid.best_params_["min_samples_split"],
                               n_estimators=MuXGrid.best_params_
                               ["n_estimators"])
MuXclf.fit(Xtrain2, np.ravel(y_train))
y_pred = MuXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['S1','N' ]))

class_names = ['N', 'C']
explainer = shap.TreeExplainer(TXclf)
shap_values = explainer.shap_values(Xtest1)
explainer = shap.TreeExplainer(JXclf)
shap_values = explainer.shap_values(Xtest2)
explainer = shap.TreeExplainer(PXclf)
shap_values = explainer.shap_values(Xtest3)
explainer = shap.TreeExplainer(GXclf)
shap_values = explainer.shap_values(Xtest3)
explainer = shap.TreeExplainer(MuXclf)
shap_values = explainer.shap_values(Xtest3)

NmeanSHAP = np.mean(np.abs(shap_values[0]),axis=0)
TmeanSHAP = np.mean(np.abs(shap_values[1]),axis=0)

