# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 01:59:03 2022

codes for training multi-omic classifiers using lung cancer data

@author: Ezgi Tanil
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

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

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
txNC = np.transpose(np.array(pd.read_csv(join(data_dir,'TX/LC_TX_NC.csv'),sep=',',header=None)))
jxNC = np.transpose(np.array(pd.read_csv(join(data_dir,'JX/LC_JX_NC.csv'),sep=',',header=None)))
gxNC = np.transpose(np.array(pd.read_csv(join(data_dir,'GX/LC_GX_NC.csv'),sep=',',header=None)))
samples = pd.read_csv(join(data_dir,'NCsamples.csv'),sep=',',header=None)
X=np.c_[gxNC, txNC]
X=np.c_[X,jxNC]

X_train, X_test, y_train, y_test = train_test_split(X, samples[1],random_state=0)
Xtrain1, Xtrain2, Xtrain3 = X_train[:,:25988], X_train[:,25988:45935], X_train[:,45935:]
Xtest1, Xtest2, Xtest3 = X_test[:,:25988], X_test[:,25988:45935], X_test[:,45935:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3]
X_test_list = [Xtest1, Xtest2, Xtest3]

classifiers = [('rf',  RandomForestClassifier(random_state=0,bootstrap=True,    )),
    ('rf',  RandomForestClassifier(random_state=0,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=0,bootstrap=True))]

fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

print(classification_report(y_test, y_pred,target_names=['N','C']))
binomy_pred = list()
for i in range(0,len(y_pred)):
    if y_pred[i]=='N':
        binomy_pred.append(0)
    else:
        binomy_pred.append(1)
            
y_pred = np.transpose([pred[:, 1] for pred in y_pred])

GXclf = RandomForestClassifier(random_state=0,bootstrap=True)
TXclf = RandomForestClassifier(random_state=0,bootstrap=True)
JXclf = RandomForestClassifier(random_state=0,bootstrap=True)

GXclf.fit(Xtrain1, y_train)
y_pred = GXclf.predict(Xtest1)
print(classification_report(y_test, y_pred,target_names=['N','C']))

TXclf.fit(Xtrain2, y_train)
y_pred = TXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['N','C']))

JXclf.fit(Xtrain3, y_train)
y_pred = JXclf.predict(Xtest3)
print(classification_report(y_test, y_pred,target_names=['N','C']))

class_names = ['N', 'C']
explainer = shap.TreeExplainer(GXclf)
shap_values = explainer.shap_values(Xtest1)
explainer = shap.TreeExplainer(TXclf)
shap_values = explainer.shap_values(Xtest2)
explainer = shap.TreeExplainer(JXclf)
shap_values = explainer.shap_values(Xtest3)

NmeanSHAP = np.mean(np.abs(shap_values[0]),axis=0)
JmeanSHAP = np.mean(np.abs(shap_values[1]),axis=0)
# shap.summary_plot(shap_values, Xtest1, plot_type="bar", class_names= class_names, max_display=50)

# AS data
txAS = np.transpose(np.array(pd.read_csv(join(data_dir,'TX/LC_TX_AS.csv'),sep=',',header=None)))
jxAS = np.transpose(np.array(pd.read_csv(join(data_dir,'JX/LC_JX_AS.csv'),sep=',',header=None)))
gxAS = np.transpose(np.array(pd.read_csv(join(data_dir,'GX/LC_GX_AS.csv'),sep=',',header=None)))
pxAS = np.transpose(np.array(pd.read_csv(join(data_dir,'PX/LC_PX_AS.csv'),sep=',',header=None)))

samples = pd.read_csv(join(data_dir,'ASsamples.csv'),sep=',')

X=np.c_[gxAS, txAS]
X=np.c_[X,jxAS]
X=np.c_[X,pxAS]
b = pd.isnull(samples.values[:,1])
X = np.delete(X,b,0)
y = samples.values[:,1]
y = np.delete(y,b,0)
b = np.isnan(X)
X[b] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
Xtrain1, Xtrain2, Xtrain3 , Xtrain4 = X_train[:,:25988], X_train[:,25988:45935], X_train[:,45935:63884],X_train[:,63884:]
Xtest1, Xtest2, Xtest3, Xtest4 = X_test[:,:25988], X_test[:,25988:45935], X_test[:,45935:63884],X_test[:,63884:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3, Xtrain4]
X_test_list = [Xtest1, Xtest2, Xtest3, Xtest4]

classifiers = [('rf',  RandomForestClassifier(random_state=0,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=0,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=0,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=0,bootstrap=True))]

fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

print(classification_report(y_test, y_pred,target_names=['AD','SC']))

GXclf = RandomForestClassifier(random_state=0,bootstrap=True)
TXclf = RandomForestClassifier(random_state=0,bootstrap=True)
JXclf = RandomForestClassifier(random_state=0,bootstrap=True)
PXclf = RandomForestClassifier(random_state=0,bootstrap=True)

class_names = ['AD','SC']

GXclf.fit(Xtrain1, y_train)
y_pred = GXclf.predict(Xtest1)
print(classification_report(y_test, y_pred,target_names=['AD','SC']))
explainer = shap.TreeExplainer(GXclf)
shap_values = explainer.shap_values(Xtest1)

TXclf.fit(Xtrain2, y_train)
y_pred = TXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['AD','SC']))
explainer = shap.TreeExplainer(TXclf)
shap_values = explainer.shap_values(Xtest2)

JXclf.fit(Xtrain3, y_train)
y_pred = JXclf.predict(Xtest3)
print(classification_report(y_test, y_pred,target_names=['AD','SC']))
explainer = shap.TreeExplainer(JXclf)
shap_values = explainer.shap_values(Xtest3)

PXclf.fit(Xtrain4, y_train)
y_pred = PXclf.predict(Xtest4)
print(classification_report(y_test, y_pred,target_names=['AD','SC']))
explainer = shap.TreeExplainer(PXclf)
shap_values = explainer.shap_values(Xtest4)

NmeanSHAP =     np.mean(np.abs(shap_values[0]),axis=0)

params = {'n_estimators': [20,50,100,200],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, 511//2, ],
          'min_samples_leaf': [1, 0.5, 511//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, 25988//2 ],}

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
                               n_estimators=PXGrid.best_params_
                               ["n_estimators"]))]

fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, np.ravel(y_train))
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)

print(classification_report(y_test, y_pred,target_names=['AD','SC']))

GXGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
GXGrid.fit(Xtrain1,np.ravel(y_train))
GXclf = RandomForestClassifier(random_state=0,bootstrap=True,max_depth=GXGrid.best_params_["max_depth"], 
                               max_features=GXGrid.best_params_["max_features"],
                               min_samples_leaf=GXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=GXGrid.best_params_["min_samples_split"],
                               n_estimators=GXGrid.best_params_
                               ["n_estimators"])
GXclf.fit(Xtrain1, np.ravel(y_train))
y_pred = GXclf.predict(Xtest1)
print(classification_report(y_test, y_pred,target_names=['A','S']))

TXGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
TXGrid.fit(Xtrain2,np.ravel(y_train))
TXclf = RandomForestClassifier(random_state=0,bootstrap=True,max_depth=TXGrid.best_params_["max_depth"], 
                               max_features=TXGrid.best_params_["max_features"],
                               min_samples_leaf=TXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=TXGrid.best_params_["min_samples_split"],
                               n_estimators=TXGrid.best_params_
                               ["n_estimators"])
TXclf.fit(Xtrain2, np.ravel(y_train))
y_pred = TXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['A','S']))

JXGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
JXGrid.fit(Xtrain3,np.ravel(y_train))
JXclf = RandomForestClassifier(random_state=0,bootstrap=True,max_depth=JXGrid.best_params_["max_depth"], 
                               max_features=JXGrid.best_params_["max_features"],
                               min_samples_leaf=JXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=JXGrid.best_params_["min_samples_split"],
                               n_estimators=JXGrid.best_params_
                               ["n_estimators"])
JXclf.fit(Xtrain3, np.ravel(y_train))
y_pred = JXclf.predict(Xtest3)
print(classification_report(y_test, y_pred,target_names=['A','S']))

PXGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=3, verbose=1)
PXGrid.fit(Xtrain4,np.ravel(y_train))
PXclf = RandomForestClassifier(random_state=0,bootstrap=True,max_depth=PXGrid.best_params_["max_depth"], 
                               max_features=PXGrid.best_params_["max_features"],
                               min_samples_leaf=PXGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=PXGrid.best_params_["min_samples_split"],
                               n_estimators=PXGrid.best_params_
                               ["n_estimators"])
PXclf.fit(Xtrain4, np.ravel(y_train))
y_pred = PXclf.predict(Xtest4)
print(classification_report(y_test, y_pred,target_names=['A','S']))

# Staging

X_train, X_test, y_train, y_test = train_test_split(X, samples.values[:,2])
Xtrain1, Xtrain2, Xtrain3 , Xtrain4 = X_train[:,:25988], X_train[:,25988:45935], X_train[:,45935:63884],X_train[:,63884:]
Xtest1, Xtest2, Xtest3, Xtest4 = X_test[:,:25988], X_test[:,25988:45935], X_test[:,45935:63884],X_test[:,63884:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3, Xtrain4]
X_test_list = [Xtest1, Xtest2, Xtest3, Xtest4]

classifiers = [('rf',  RandomForestClassifier(random_state=1,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=1,bootstrap=True)),
    ('rf',  RandomForestClassifier(random_state=1,bootstrap=True))]

#earlystage------------------------------------------------------------------
labels = (samples.values[:,2]=="Stage I")
labelsS = samples.values[labels,2]
X = X[labels,]
y = labelsS
X = X[:,:63884]

#for N

#samplesN = pd.read_csv(join(data_dir,'NCsamples.csv'),sep=',',header=None)
#txNC = np.transpose(np.array(pd.read_csv(join(data_dir,'TX/LC_TX_NC.csv'),sep=',',header=None)))
#jxNC = np.transpose(np.array(pd.read_csv(join(data_dir,'JX/LC_JX_NC.csv'),sep=',',header=None)))
#gxNC = np.transpose(np.array(pd.read_csv(join(data_dir,'GX/LC_GX_NC.csv'),sep=',',header=None)))
Xn=np.c_[gxNC, txNC]
Xn=np.c_[Xn,jxNC]

labelsn = (samplesN.values[:,1]=="N")
labelsSn = samplesN.values[labelsn,1]
Xn = Xn[labelsn,]
yn = labelsSn

Xf = np.concatenate((X, Xn))
yf = np.concatenate((y, yn))

X_train, X_test, y_train, y_test = train_test_split(Xf, yf,random_state=0,train_size=0.8)
Xtrain1, Xtrain2, Xtrain3 = X_train[:,:25988], X_train[:,25988:45935], X_train[:,45935:]
Xtest1, Xtest2, Xtest3 = X_test[:,:25988], X_test[:,25988:45935], X_test[:,45935:]
X_train_list = [Xtrain1, Xtrain2, Xtrain3]
X_test_list = [Xtest1, Xtest2, Xtest3]

fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)
print(classification_report(y_test, y_pred,target_names=['S1','N']))

GXclf = RandomForestClassifier(random_state=0,bootstrap=True)
TXclf = RandomForestClassifier(random_state=0,bootstrap=True)
JXclf = RandomForestClassifier(random_state=0,bootstrap=True)

GXclf.fit(Xtrain1, y_train)
y_pred = GXclf.predict(Xtest1)
print(classification_report(y_test, y_pred,target_names=['S1','N']))
explainer = shap.TreeExplainer(GXclf)
shap_values = explainer.shap_values(Xtest1)

TXclf.fit(Xtrain2, y_train)
y_pred = TXclf.predict(Xtest2)
print(classification_report(y_test, y_pred,target_names=['S1','N']))
explainer = shap.TreeExplainer(TXclf)
shap_values = explainer.shap_values(Xtest2)

JXclf.fit(Xtrain3, y_train)
y_pred = JXclf.predict(Xtest3)
print(classification_report(y_test, y_pred,target_names=['S1','N']))
explainer = shap.TreeExplainer(JXclf)
shap_values = explainer.shap_values(Xtest3)

NmeanSHAP = np.mean(np.abs(shap_values[0]),axis=0)

