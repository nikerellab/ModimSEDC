# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 02:01:25 2022
% Improved classifiers for JX data
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

data_dir = '.../classification/finalClassification'
AD1 = pd.read_csv(join(data_dir,'AD1all.csv'),sep=',',header=None)
AD2 = pd.read_csv(join(data_dir,'AD2all.csv'),sep=',',header=None)
AD3 = pd.read_csv(join(data_dir,'AD3all.csv'),sep=',',header=None)
AD4 = pd.read_csv(join(data_dir,'AD4all.csv'),sep=',',header=None)
SC1 = pd.read_csv(join(data_dir,'SC1all.csv'),sep=',',header=None)
SC2 = pd.read_csv(join(data_dir,'SC2all.csv'),sep=',',header=None)
SC3 = pd.read_csv(join(data_dir,'SC3all.csv'),sep=',',header=None)
SC4 = pd.read_csv(join(data_dir,'SC4all.csv'),sep=',',header=None)
N = pd.read_csv(join(data_dir,'Nall.csv'),sep=',',header=None)
SCLC = pd.read_csv(join(data_dir,'SCLCall.csv'),sep=',',header=None)
AD = np.concatenate((AD1.values,AD2.values,AD3.values,AD4.values),axis=1)
labelsAD = np.concatenate((np.tile(1,(359, 1)),np.tile(2, (155, 1)),np.tile(3,(106, 1)),np.tile(4, (28, 1))),axis=0)
AD_X_train, AD_X_test, AD_Y_train, AD_Y_test = train_test_split(AD.T, labelsAD, train_size=0.8, random_state=1)
SC = np.concatenate((SC1.values,SC2.values,SC3.values,SC4.values),axis=1)
labelsSC = np.concatenate((np.tile(1,(284, 1)),np.tile(2, (206, 1)),np.tile(3,(105, 1)),np.tile(4, (10, 1))),axis=0)
SC_X_train, SC_X_test, SC_Y_train, SC_Y_test = train_test_split(SC.T, labelsSC, train_size=0.8, random_state=1)
Nall = N.values
labelsN = np.concatenate((np.tile(1,(312, 1))),axis=0)
N_X_train, N_X_test, N_Y_train, N_Y_test = train_test_split(Nall.T, labelsN, train_size=0.8, random_state=1)
labelsSCLC = np.concatenate((np.tile(5, (79, 1))),axis=0)
SCLC_X_train, SCLC_X_test, SCLC_Y_train, SCLC_Y_test = train_test_split(SCLC.values.T, labelsSCLC, train_size=0.8, random_state=1)
params = {'n_estimators': [20,50,100],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, n_samples//2, ],
          'min_samples_leaf': [1, 0.5, n_samples//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, n_features//2, ],
         }
Allshaps = list()
features = list()
for i in range(0,len(model.reactions)):
    features.append(model.reactions[i].id)
#NC---------------------------------------------------------
NCtrain_X = np.concatenate((N_X_train,AD_X_train,SC_X_train,SCLC_X_train),axis=0)
n_samples = NCtrain_X.shape[0]
n_features = NCtrain_X.shape[1]
NCtrain_Y=  np.concatenate((np.tile(0,(256, 1)),np.tile(1, (518, 1)),np.tile(1, (484, 1)),np.tile(1, (63, 1))),axis=0)
NCGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=5, verbose=1)
NCGrid.fit(NCtrain_X,np.ravel(NCtrain_Y))
NCtest_X = np.concatenate((N_X_test,AD_X_test,SC_X_test,SCLC_X_test),axis=0)
NCtest_Y=  np.concatenate((np.tile(0,(64, 1)),np.tile(1, (130, 1)),np.tile(1, (121, 1)),np.tile(1, (16, 1))),axis=0)
y_pred = NCGrid.predict(NCtest_X)
print(precision_score(NCtest_Y, y_pred,average='micro'))
print(classification_report(NCtest_Y, y_pred,target_names=['N','C']))

CNclf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=NCGrid.best_params_["max_depth"], 
                               max_features=NCGrid.best_params_["max_features"],
                               min_samples_leaf=NCGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=NCGrid.best_params_["min_samples_split"],
                               n_estimators=NCGrid.best_params_
                               ["n_estimators"])
CNclf.fit(NCtrain_X, np.ravel(NCtrain_Y))
explainer = shap.TreeExplainer(CNclf)
shap_values = explainer.shap_values(NCtest_X)
class_names = ['N', 'C']
shap.summary_plot(shap_values, NCtest_X, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)

filename = 'NC_model.sav'
pickle.dump(NCGrid, open(filename, 'wb'))

#NSCLC-SCLC-------------------------------------------------
NStrain_X = np.concatenate((AD_X_train,SC_X_train,SCLC_X_train),axis=0)
n_samples = NStrain_X.shape[0]
n_features = NStrain_X.shape[1]
NStrain_Y=  np.concatenate((np.tile(0, (518, 1)),np.tile(0, (484, 1)),np.tile(1, (63, 1))),axis=0)
NSGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
NSGrid.fit(NStrain_X,np.ravel(NStrain_Y))
NStest_X = np.concatenate((AD_X_test,SC_X_test,SCLC_X_test),axis=0)
NStest_Y=  np.concatenate((np.tile(0, (130, 1)),np.tile(0, (121, 1)),np.tile(1, (16, 1))),axis=0)
y_pred = NSGrid.predict(NStest_X)
print(precision_score(NStest_Y, y_pred,average='micro'))
print(classification_report(NStest_Y, y_pred,target_names=['NSCLC','SCLC']))

NSclf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=NSGrid.best_params_["max_depth"], 
                               max_features=NSGrid.best_params_["max_features"],
                               min_samples_leaf=NSGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=NSGrid.best_params_["min_samples_split"],
                               n_estimators=NSGrid.best_params_
                               ["n_estimators"])
NSclf.fit(NStrain_X, np.ravel(NStrain_Y))
# y_pred = NSclf.predict(NStest_X)
# print(precision_score(NStest_Y, y_pred,average='micro'))
# print(classification_report(NStest_Y, y_pred,target_names=['NSCLC','SCLC']))
explainer = shap.TreeExplainer(NSclf)
shap_values = explainer.shap_values(NStest_X)
class_names = ['NSCLC', 'SCLC']
shap.summary_plot(shap_values, NStest_X, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)

filename = 'NS_model.sav'
pickle.dump(NSGrid, open(filename, 'wb'))

#AD-SC--------------------------------------------------------
AStrain_X = np.concatenate((AD_X_train,SC_X_train),axis=0)
n_samples = AStrain_X.shape[0]
n_features = AStrain_X.shape[1]
AStrain_Y=  np.concatenate((np.tile(0, (518, 1)),np.tile(1, (484, 1))),axis=0)
ASGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
ASGrid.fit(AStrain_X,np.ravel(AStrain_Y))
AStest_X = np.concatenate((AD_X_test,SC_X_test),axis=0)
AStest_Y=  np.concatenate((np.tile(0, (130, 1)),np.tile(1, (121, 1))),axis=0)
y_pred = ASGrid.predict(AStest_X)
print(precision_score(AStest_Y, y_pred,average='micro'))
print(classification_report(AStest_Y, y_pred,target_names=['AD','SC']))

ASClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=ASGrid.best_params_["max_depth"], 
                               max_features=ASGrid.best_params_["max_features"],
                               min_samples_leaf=ASGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=ASGrid.best_params_["min_samples_split"],
                               n_estimators=ASGrid.best_params_
                               ["n_estimators"])
ASClf.fit(AStrain_X, np.ravel(AStrain_Y))
explainer = shap.TreeExplainer(ASClf)
shap_values = explainer.shap_values(AStest_X)
class_names = ['AD', 'SC']
shap.summary_plot(shap_values, AStest_X, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)

filename = 'AS_model.sav'
pickle.dump(ASGrid, open(filename, 'wb'))

#AD-STAGES--------------------------------------------------------
n_samples = AD_X_train.shape[0]
n_features = AD_X_train.shape[1]
ADsGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
ADsGrid.fit(AD_X_train,np.ravel(AD_Y_train))
y_pred = ADsGrid.predict(AD_X_test)
print(precision_score(AD_Y_test, y_pred,average='micro'))
print(classification_report(AD_Y_test, y_pred,target_names=['AD1','AD2','AD3','AD4']))
ADsClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=ADsGrid.best_params_["max_depth"], 
                               max_features=ADsGrid.best_params_["max_features"],
                               min_samples_leaf=ADsGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=ADsGrid.best_params_["min_samples_split"],
                               n_estimators=ADsGrid.best_params_
                               ["n_estimators"])
ADsClf.fit(AD_X_train,np.ravel(AD_Y_train))
explainer = shap.TreeExplainer(ADsClf)
shap_values = explainer.shap_values(AD_X_test)
class_names = ['AD1', 'AD2','AD3', 'AD4']
shap.summary_plot(shap_values, AD_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'ADs_model.sav'
pickle.dump(ADsGrid, open(filename, 'wb'))

#AD-STAGE 1--------------------------------------------------------
n_samples = AD_X_train.shape[0]
n_features = AD_X_train.shape[1]
AD1Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
AD1_train_Y = list()
for i in range(0,len(AD_Y_train)):
	if AD_Y_train[i,0]==1:
		AD1_train_Y.append(AD_Y_train[i,0])
	else:
		AD1_train_Y.append(0)
AD1_test_Y = list()
for i in range(0,len(AD_Y_test)):
	if AD_Y_test[i,0]==1:
		AD1_test_Y.append(AD_Y_test[i,0])
	else:
		AD1_test_Y.append(0)
AD1Grid.fit(AD_X_train,np.ravel(np.array(AD1_train_Y)))
y_pred = AD1Grid.predict(AD_X_test)
print(precision_score(AD1_test_Y, y_pred,average='micro'))
print(classification_report(AD1_test_Y, y_pred,target_names=['Other','AD1']))
AD1Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=AD1Grid.best_params_["max_depth"], 
                               max_features=AD1Grid.best_params_["max_features"],
                               min_samples_leaf=AD1Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=AD1Grid.best_params_["min_samples_split"],
                               n_estimators=AD1Grid.best_params_
                               ["n_estimators"])
AD1Clf.fit(AD_X_train, np.ravel(AD1_train_Y))
explainer = shap.TreeExplainer(AD1Clf)
shap_values = explainer.shap_values(AD_X_test)
class_names = ['Others', 'AD-Stage 1']
shap.summary_plot(shap_values, AD_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'AD1_model.sav'
pickle.dump(AD1Grid, open(filename, 'wb'))

#AD-STAGE 2--------------------------------------------------------
n_samples = AD_X_train.shape[0]
n_features = AD_X_train.shape[1]
AD2Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
AD2_train_Y = list()
for i in range(0,len(AD_Y_train)):
	if AD_Y_train[i,0]==2:
		AD2_train_Y.append(AD_Y_train[i,0])
	else:
		AD2_train_Y.append(0)
AD2_test_Y = list()
for i in range(0,len(AD_Y_test)):
	if AD_Y_test[i,0]==2:
		AD2_test_Y.append(AD_Y_test[i,0])
	else:
		AD2_test_Y.append(0)
AD2Grid.fit(AD_X_train,np.ravel(np.array(AD2_train_Y)))
y_pred = AD2Grid.predict(AD_X_test)
print(precision_score(AD2_test_Y, y_pred,average='micro'))
print(classification_report(AD2_test_Y, y_pred,target_names=['Other','AD2']))
AD2Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=AD2Grid.best_params_["max_depth"], 
                               max_features=AD2Grid.best_params_["max_features"],
                               min_samples_leaf=AD2Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=AD2Grid.best_params_["min_samples_split"],
                               n_estimators=AD2Grid.best_params_
                               ["n_estimators"])
AD2Clf.fit(AD_X_train, np.ravel(AD2_train_Y))
explainer = shap.TreeExplainer(AD2Clf)
shap_values = explainer.shap_values(AD_X_test)
class_names = ['Others', 'AD-Stage 2']
shap.summary_plot(shap_values, AD_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'AD2_model.sav'
pickle.dump(AD2Grid, open(filename, 'wb'))

#AD-STAGE 3--------------------------------------------------------
n_samples = AD_X_train.shape[0]
n_features = AD_X_train.shape[1]
AD3Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
AD3_train_Y = list()
for i in range(0,len(AD_Y_train)):
	if AD_Y_train[i,0]==3:
		AD3_train_Y.append(AD_Y_train[i,0])
	else:
		AD3_train_Y.append(0)
AD3_test_Y = list()
for i in range(0,len(AD_Y_test)):
	if AD_Y_test[i,0]==3:
		AD3_test_Y.append(AD_Y_test[i,0])
	else:
		AD3_test_Y.append(0)
AD3Grid.fit(AD_X_train,np.ravel(np.array(AD3_train_Y)))
y_pred = AD3Grid.predict(AD_X_test)
print(precision_score(AD3_test_Y, y_pred,average='micro'))
print(classification_report(AD3_test_Y, y_pred,target_names=['Other','AD3']))
AD3Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=AD3Grid.best_params_["max_depth"], 
                               max_features=AD3Grid.best_params_["max_features"],
                               min_samples_leaf=AD3Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=AD3Grid.best_params_["min_samples_split"],
                               n_estimators=AD3Grid.best_params_
                               ["n_estimators"])

AD3Clf.fit(AD_X_train, np.ravel(AD3_train_Y))
y_pred = AD3Clf.predict(AD_X_test)
print(precision_score(AD3_test_Y, y_pred,average='micro'))
print(classification_report(AD3_test_Y, y_pred,target_names=['Others', 'AD-Stage 3']))
explainer = shap.TreeExplainer(AD3Clf)
shap_values = explainer.shap_values(AD_X_test)
class_names = ['Others', 'AD-Stage 3']
shap.summary_plot(shap_values, AD_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'AD3_model.sav'
pickle.dump(AD3Grid, open(filename, 'wb'))

#AD-STAGE 4--------------------------------------------------------
n_samples = AD_X_train.shape[0]
n_features = AD_X_train.shape[1]
AD4Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
AD4_train_Y = list()
for i in range(0,len(AD_Y_train)):
	if AD_Y_train[i,0]==4:
		AD4_train_Y.append(AD_Y_train[i,0])
	else:
		AD4_train_Y.append(0)
AD4_test_Y = list()
for i in range(0,len(AD_Y_test)):
	if AD_Y_test[i,0]==4:
		AD4_test_Y.append(AD_Y_test[i,0])
	else:
		AD4_test_Y.append(0)
AD4Grid.fit(AD_X_train,np.ravel(np.array(AD4_train_Y)))
y_pred = AD4Grid.predict(AD_X_test)
print(precision_score(AD4_test_Y, y_pred,average='micro'))
print(classification_report(AD4_test_Y, y_pred,target_names=['Other','AD4']))
AD4Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=AD4Grid.best_params_["max_depth"], 
                               max_features=AD4Grid.best_params_["max_features"],
                               min_samples_leaf=AD4Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=AD4Grid.best_params_["min_samples_split"],
                               n_estimators=AD4Grid.best_params_
                               ["n_estimators"])
AD4Clf.fit(AD_X_train, np.ravel(AD4_train_Y))
explainer = shap.TreeExplainer(AD4Clf)
shap_values = explainer.shap_values(AD_X_test)
class_names = ['Others', 'AD-Stage 4']
shap.summary_plot(shap_values, AD_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'AD4_model.sav'
pickle.dump(AD4Grid, open(filename, 'wb'))

#SC-STAGES--------------------------------------------------------
n_samples = SC_X_train.shape[0]
n_features = SC_X_train.shape[1]
SCsGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
SCsGrid.fit(SC_X_train,np.ravel(SC_Y_train))
y_pred = SCsGrid.predict(SC_X_test)
print(precision_score(SC_Y_test, y_pred,average='micro'))
print(classification_report(SC_Y_test, y_pred,target_names=['SC1','SC2','SC3','SC4']))
SCsClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=SCsGrid.best_params_["max_depth"], 
                               max_features=SCsGrid.best_params_["max_features"],
                               min_samples_leaf=SCsGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=SCsGrid.best_params_["min_samples_split"],
                               n_estimators=SCsGrid.best_params_
                               ["n_estimators"])
SCsClf.fit(SC_X_train,np.ravel(SC_Y_train))
explainer = shap.TreeExplainer(SCsClf)
shap_values = explainer.shap_values(SC_X_test)
class_names = ['SC1', 'SC2','SC3', 'SC4']
shap.summary_plot(shap_values, SC_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'SCs_model.sav'
pickle.dump(SCsGrid, open(filename, 'wb'))

#SC-STAGE 1--------------------------------------------------------
n_samples = SC_X_train.shape[0]
n_features = SC_X_train.shape[1]
SC1Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
SC1_train_Y = list()
for i in range(0,len(SC_Y_train)):
	if SC_Y_train[i,0]==1:
		SC1_train_Y.append(SC_Y_train[i,0])
	else:
		SC1_train_Y.append(0)
SC1_test_Y = list()
for i in range(0,len(SC_Y_test)):
	if SC_Y_test[i,0]==1:
		SC1_test_Y.append(SC_Y_test[i,0])
	else:
		SC1_test_Y.append(0)
SC1Grid.fit(SC_X_train,np.ravel(np.array(SC1_train_Y)))
y_pred = SC1Grid.predict(SC_X_test)
print(precision_score(SC1_test_Y, y_pred,average='micro'))
print(classification_report(SC1_test_Y, y_pred,target_names=['Others', 'SC-Stage 1']))
SC1Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=SC1Grid.best_params_["max_depth"], 
                               max_features=SC1Grid.best_params_["max_features"],
                               min_samples_leaf=SC1Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=SC1Grid.best_params_["min_samples_split"],
                               n_estimators=SC1Grid.best_params_
                               ["n_estimators"])
SC1Clf.fit(SC_X_train, np.ravel(SC1_train_Y))
explainer = shap.TreeExplainer(SC1Clf)
shap_values = explainer.shap_values(SC_X_test)
class_names = ['Others', 'SC-Stage 1']
shap.summary_plot(shap_values, SC_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'SC1_model.sav'
pickle.dump(SC1Grid, open(filename, 'wb'))

#SC-STAGE 2--------------------------------------------------------
n_samples = SC_X_train.shape[0]
n_features = SC_X_train.shape[1]
SC2Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
SC2_train_Y = list()
for i in range(0,len(SC_Y_train)):
	if SC_Y_train[i,0]==2:
		SC2_train_Y.append(SC_Y_train[i,0])
	else:
		SC2_train_Y.append(0)
SC2_test_Y = list()
for i in range(0,len(SC_Y_test)):
	if SC_Y_test[i,0]==2:
		SC2_test_Y.append(SC_Y_test[i,0])
	else:
		SC2_test_Y.append(0)
SC2Grid.fit(SC_X_train,np.ravel(np.array(SC2_train_Y)))
y_pred = SC2Grid.predict(SC_X_test)
print(precision_score(SC2_test_Y, y_pred,average='micro'))
print(classification_report(SC2_test_Y, y_pred,target_names=['Others', 'SC-Stage 2']))
SC2Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=SC2Grid.best_params_["max_depth"], 
                               max_features=SC2Grid.best_params_["max_features"],
                               min_samples_leaf=SC2Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=SC2Grid.best_params_["min_samples_split"],
                               n_estimators=SC2Grid.best_params_
                               ["n_estimators"])
SC2Clf.fit(SC_X_train, np.ravel(SC2_train_Y))
explainer = shap.TreeExplainer(SC2Clf)
shap_values = explainer.shap_values(SC_X_test)
class_names = ['Others', 'SC-Stage 2']
shap.summary_plot(shap_values, SC_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'SC2_model.sav'
pickle.dump(SC2Grid, open(filename, 'wb'))

#SC-STAGE 3--------------------------------------------------------
n_samples = SC_X_train.shape[0]
n_features = SC_X_train.shape[1]
SC3Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
SC3_train_Y = list()
for i in range(0,len(SC_Y_train)):
	if SC_Y_train[i,0]==3:
		SC3_train_Y.append(SC_Y_train[i,0])
	else:
		SC3_train_Y.append(0)
SC3_test_Y = list()
for i in range(0,len(SC_Y_test)):
	if SC_Y_test[i,0]==3:
		SC3_test_Y.append(SC_Y_test[i,0])
	else:
		SC3_test_Y.append(0)
SC3Grid.fit(SC_X_train,np.ravel(np.array(SC3_train_Y)))
y_pred = SC3Grid.predict(SC_X_test)
print(precision_score(SC3_test_Y, y_pred,average='micro'))
print(classification_report(SC3_test_Y, y_pred,target_names=['Others', 'SC-Stage 3']))
SC3Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=SC3Grid.best_params_["max_depth"], 
                               max_features=SC3Grid.best_params_["max_features"],
                               min_samples_leaf=SC3Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=SC3Grid.best_params_["min_samples_split"],
                               n_estimators=SC3Grid.best_params_
                               ["n_estimators"])
SC3Clf.fit(SC_X_train, np.ravel(SC3_train_Y))
explainer = shap.TreeExplainer(SC3Clf)
shap_values = explainer.shap_values(SC_X_test)
class_names = ['Others', 'SC-Stage 3']
shap.summary_plot(shap_values, SC_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'SC3_model.sav'
pickle.dump(SC3Grid, open(filename, 'wb'))

#SC-STAGE 4--------------------------------------------------------
n_samples = SC_X_train.shape[0]
n_features = SC_X_train.shape[1]
SC4Grid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
SC4_train_Y = list()
for i in range(0,len(SC_Y_train)):
	if SC_Y_train[i,0]==4:
		SC4_train_Y.append(SC_Y_train[i,0])
	else:
		SC4_train_Y.append(0)
SC4_test_Y = list()
for i in range(0,len(SC_Y_test)):
	if SC_Y_test[i,0]==4:
		SC4_test_Y.append(SC_Y_test[i,0])
	else:
		SC4_test_Y.append(0)
SC4Grid.fit(SC_X_train,np.ravel(np.array(SC4_train_Y)))
y_pred = SC4Grid.predict(SC_X_test)
print(precision_score(SC4_test_Y, y_pred,average='micro'))
print(classification_report(SC4_test_Y, y_pred,target_names=['Others', 'SC-Stage 4']))
SC4Clf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=SC4Grid.best_params_["max_depth"], 
                               max_features=SC4Grid.best_params_["max_features"],
                               min_samples_leaf=SC4Grid.best_params_["min_samples_leaf"] ,
                               min_samples_split=SC4Grid.best_params_["min_samples_split"],
                               n_estimators=SC4Grid.best_params_
                               ["n_estimators"])
SC4Clf.fit(SC_X_train, np.ravel(SC4_train_Y))
explainer = shap.TreeExplainer(SC4Clf)
shap_values = explainer.shap_values(SC_X_test)
class_names = ['Others', 'SC-Stage 4']
shap.summary_plot(shap_values, SC_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = 'SC4_model.sav'
pickle.dump(SC4Grid, open(filename, 'wb'))

pd.DataFrame(AD_X_test).to_csv('ADXtest.csv')
pd.DataFrame(AD_X_train).to_csv('ADXtrain.csv')
pd.DataFrame(AD_Y_train).to_csv('ADYtrain.csv')
pd.DataFrame(AD_Y_test).to_csv('ADYtest.csv')
pd.DataFrame(SC_X_test).to_csv('SCXtest.csv')
pd.DataFrame(SC_X_train).to_csv('SCXtrain.csv')
pd.DataFrame(SC_Y_train).to_csv('SCYtrain.csv')
pd.DataFrame(SC_Y_test).to_csv('SCYtest.csv')
pd.DataFrame(N_X_test).to_csv('NXtest.csv')
pd.DataFrame(N_X_train).to_csv('NXtrain.csv')
pd.DataFrame(N_Y_train).to_csv('NYtrain.csv')
pd.DataFrame(N_Y_test).to_csv('NYtest.csv')
pd.DataFrame(SCLC_X_test).to_csv('SCLCXtest.csv')
pd.DataFrame(SCLC_X_train).to_csv('SCLCXtrain.csv')
pd.DataFrame(SCLC_Y_train).to_csv('SCLCYtrain.csv')
pd.DataFrame(SCLC_Y_test).to_csv('SCLCYtest.csv')

ShapNC= pd.DataFrame (np.vstack(Allshaps[0]).T).to_csv('NCshap.csv')
ShapNS= pd.DataFrame (np.vstack(Allshaps[1]).T).to_csv('NSshap.csv')
ShapAS= pd.DataFrame (np.vstack(Allshaps[2]).T).to_csv('ASshap.csv')
ShapADs= pd.DataFrame (np.vstack(Allshaps[3]).T).to_csv('ADsshap.csv')
ShapAD1= pd.DataFrame (np.vstack(Allshaps[4]).T).to_csv('AD1shap.csv')
ShapAD2= pd.DataFrame (np.vstack(Allshaps[5]).T).to_csv('AD2shap.csv')
ShapAD3= pd.DataFrame (np.vstack(Allshaps[6]).T).to_csv('AD3shap.csv')
ShapAD4= pd.DataFrame (np.vstack(Allshaps[7]).T).to_csv('AD4shap.csv')
ShapSCs= pd.DataFrame (np.vstack(Allshaps[8]).T).to_csv('SCsshap.csv')
ShapSC1= pd.DataFrame (np.vstack(Allshaps[9]).T).to_csv('SC1shap.csv')
ShapSC2= pd.DataFrame (np.vstack(Allshaps[10]).T).to_csv('SC2shap.csv')
ShapSC3= pd.DataFrame (np.vstack(Allshaps[11]).T).to_csv('SC3shap.csv')
ShapSC4= pd.DataFrame (np.vstack(Allshaps[12]).T).to_csv('SC4shap.csv')