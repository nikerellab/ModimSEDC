# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:30:03 2022

@author: Ezgi
"""

import pandas as pd
# import cobra
# import cobamp
import pickle
# import gurobipy
import numpy as np
import os
from os.path import join
from sklearn.preprocessing import RobustScaler,StandardScaler 
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pickle
import shap
data_dir = '.../classification/LC_JX_model/NCBI_JX'
model_dir = '.../classification/LC_JX_model' 
dfgenes = pd.read_csv('.../classification/LC_JX_model/NCBI_JX/probe2ENT.csv',sep=',',header=None)
#for cell line
dfgenes = pd.read_csv('.../classification/LC_JX_model/NCBI_JX/cellLineEnt2sym.csv',header=None,sep=',')
probes = pd.read_csv(join(data_dir,'probes.csv'),sep=',',header=None)
model = cobra.io.read_sbml_model('.../classification/LC_JX_model/modelIrrev.xml')
gprbase = cobamp.wrappers.external_wrappers.COBRAModelObjectReader(model=model)
IDdict = dict(zip(list(dfgenes.values[:,0]),list(dfgenes.values[:,1])))
media = list(['EX_gly[e]_b', 'EX_arg_L[e]_b', 'EX_asp_L[e]_b', 'EX_asn_L[e]_b', 'EX_cys_L[e]_b','EX_glu_L[e]_b', 'EX_gln_L[e]_b', 'EX_his_L[e]_b', 'EX_4hpro[e]_b', 'EX_ile_L[e]_b', 'EX_leu_L[e]_b', 'EX_lys_L[e]_b', 'EX_met_L[e]_b', 'EX_phe_L[e]_b', 'EX_pro_L[e]_b', 'EX_ser_L[e]_b', 'EX_thr_L[e]_b', 'EX_trp_L[e]_b', 'EX_tyr_L[e]_b', 'EX_val_L[e]_b', 'EX_btn[e]_b', 'EX_chol[e]_b', 'EX_pnto_R[e]_b', 'EX_fol[e]_b', 'EX_ncam[e]_b', 'EX_bz[e]_b', 'EX_pydxn[e]_b', 'EX_ribflv[e]_b', 'EX_thm[e]_b', 'EX_inost[e]_b', 'CAt7r_b', 'EX_so4[e]_b', 'EX_k[e]_b', 'NKCCt_b', 'EX_hco3[e]_b', 'EX_pi[e]_b', 'EX_gthrd[e]_b', 'EX_h2o[e]_b', 'EX_h[e]_b', 'EX_glc_D[e]_b'])
subs = pd.read_csv('.../classification/LC_JX_model/NCBI_JX/modelIrrevSubs.csv',sep=',',header=None)
for i in range(0,len(list(model.reactions))):
    model.reactions[i].subsystem = subs.values[i,0]
    for file in csv_files:
        df = pd.read_csv(join(data_dir,'arrayData.csv'),sep=',')
        df.drop('Probes', inplace=True, axis=1)
    solutionAll = list()
    ExpRxnsAll = list()
    for k in range(0,df.shape[1],1):
        explist = list()
        explist = np.array(df.values[:,k])
        ind = list(probes.values).index("201252_at")
        #ind = list(probes.values).index("NM_002654_at")
        #ind = list(dfgenes.values).index("ENSG00000067225.18")
        pkmvalue = explist[ind]
        #create expression array for each gene,entrez ids can be keys of dictionary or symbols
        dataDict = dict()   
        dataDict = dict(zip(list(datagenes.values),list(explist)))
        geneExp = list()
        genes = list()
        for i in range(0,len(model.genes)):
            genes.append(model.genes[i].id)
        geneExp = list()
        for i in range(0,len(model.genes)):
            if pd.isna(IDdict[int(model.genes[i].id)]):
                geneExp.append(np.nan)
            else:
                if IDdict[int(model.genes[i].id)] in dataDict:
                    geneExp.append(dataDict[IDdict[int(model.genes[i].id)]])
                else:
                    geneExp.append(np.nan)
        geneExp = np.array(geneExp)/pkmvalue
        geneExpDict = dict(zip(genes,geneExp))
        rxnExp = list()
        for i in range(0,len(model.reactions)):
            gprList = gprbase.gene_protein_reaction_rules.get_gpr_as_lists(i)
            exptemp = 0
            if any(gprList):
                expsList = x = [[] for a in range(len(gprList))]
                for j in range(0,len(gprList)):
                    for k in range(0,len(gprList[j])):
                        expsList[j].append(geneExpDict[gprList[j][k]])
                for j in range(0,len(expsList)):
                    exptemp_2 = np.nanmin(expsList[j])
                    exptemp = exptemp+exptemp_2
            else:
                exptemp = np.nan
            rxnExp.append(exptemp)
        ExpRxnsAll.append(rxnExp)
        for i in range(0,len(model.reactions),1):
            if np.isnan(rxnExp[i]):
                model.reactions[i].bounds = (0,1000)
            else:
                model.reactions[i].bounds = (0,rxnExp[i])
        #Setting other constraints - media/oxygen requirement/drug met inactivation
        for i in range(0,len(model.reactions)):
            if model.reactions[i].id.startswith('EX_') & model.reactions[i].id.endswith('_b'):
                model.reactions[i].bounds = (0,1000)
            if model.reactions[i].subsystem.startswith('Drug'):
                model.reactions[i].bounds = (0,0)
            if model.reactions[i].subsystem.startswith('IDH1'):
                model.reactions[i].bounds = (0,0)
            if model.reactions[i].subsystem.startswith('MODULE'):
                model.reactions[i].bounds = (0,0)
            if model.reactions[i].id in media:
                model.reactions[i].bounds = (0,1000)
            if model.reactions[i].id == 'EX_o2[e]_r':
                model.reactions[i].bounds = (0,1000)
        model.objective = model.reactions.biomass_maintenance
        solution = model.optimize('max')   
        solutionAll.append(solution.fluxes.values)
    solutionAll2 = np.array(np.vstack(solutionAll)).T
    ExpRxnsAll2 = np.array(np.vstack(ExpRxnsAll)).T
    pd.DataFrame(solutionAll2).to_csv('.../classification/LC_JX_model/NCBI_JX/OptRes_cellLine.csv')
    pd.DataFrame(ExpRxnsAll2).to_csv('.../classification/LC_JX_model/NCBI_JX/ExpRes_cellLine.csv')

#May file import
data_dir = '.../classification/LC_JX_model/NCBI_JX'
df1 = pd.read_csv(join(data_dir,'X_train_flux_py.csv'),sep=',',header=None)
df2 = pd.read_csv(join(data_dir,'X_test_flux_py.csv'),sep=',',header=None)
df =  pd.concat([df1, df2], axis=1)
dfgenes = pd.read_csv('.../classification/LC_JX_model/NCBI_JX/IrrevFeatures.csv',sep=',',header=None)

data = np.array(df.values).T
scaler = StandardScaler()
scaler.fit(data)
data_S = scaler.transform(data)
X_train = data_S[0:937,:]
X_test =  data_S[937:1172,:]
y_train = pd.read_csv(join(data_dir,'y_train.csv'),sep=',',header=None)
y_test = pd.read_csv(join(data_dir,'y_test.csv'),sep=',',header=None)

#Stepbystep --------
#NC


#for SVM
# pipe = Pipeline([('scaler', StandardScaler()), ('classifier',SVC())])
# param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000],
#               'classifier__gamma': ["scale",1, 0.1, 0.01, 0.001, 0.0001],
#               'classifier__kernel': ['rbf','linear']}

# param_grid = {'classifier__n_estimators': [20,50,100],
#               'classifier__max_depth': [None, 2, 5,],
#               'classifier__min_samples_split': [2, 0.5, n_samples//2, ],
#               'classifier__min_samples_leaf': [1, 0.5, n_samples//2, ],
#               'classifier__max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, n_features//2, ]}



# 0 - ad, 1 - LaC, 2 - N, 3 - SC 4 - SCLC
 

NCClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=NCGrid.best_params_["max_depth"], 
                               max_features=NCGrid.best_params_["max_features"],
                               min_samples_leaf=NCGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=NCGrid.best_params_["min_samples_split"],
                               n_estimators=NCGrid.best_params_
                               ["n_estimators"])
# NCClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=NCGrid.best_params_["classifier__max_depth"], 
#                                max_features=NCGrid.best_params_["classifier__max_features"],
#                                min_samples_leaf=NCGrid.best_params_["classifier__min_samples_leaf"] ,
#                                min_samples_split=NCGrid.best_params_["classifier__min_samples_split"],
#                                n_estimators=NCGrid.best_params_
#                                ["classifier__n_estimators"])

NCClf.fit(NC_train_X, np.ravel(NC_train_Y))
y_pred = NCClf.predict(NC_X_test)
print(precision_score(NC_test_Y, y_pred,average='micro'))
print(classification_report(NC_test_Y, y_pred,target_names=['Normal', 'Cancer']))
explainer = shap.TreeExplainer(NCClf)
# scaled_train_data = scaler.transform(NC_train_X)
# sub_sampled_train_data = shap.sample(scaled_train_data, 250, random_state=1) # use 600 samples of train data as background data
# scaler = NCClf["scaler"]
# explainer = shap.KernelExplainer(NCClf.named_steps['classifier'].predict_proba, sub_sampled_train_data)
shap_values = explainer.shap_values(NC_X_test)
class_names = ['Normal', 'Cancer']
shap.summary_plot(shap_values, NC_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps = list()
Allshaps.append(shap_values)
filename = '.../classification/LC_JX_model/NCBI_JX/NC_model.sav'
pickle.dump(NCGrid, open(filename, 'wb'))

# NS

# 0 - ad, 1 - LaC, 2 - N, 3 - SC, 4 - SCLC
# pipe = Pipeline([('scaler', StandardScaler()), ('classifier',SVC())])
# param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000],
#               'classifier__gamma': ["scale",1, 0.1, 0.01, 0.001, 0.0001],
#               'classifier__kernel': ['rbf','linear']}


a = list()
for i in range(0,len(y_train)):
    if y_train.values[i,0] == 2:
        a.append(i)
b = list()
for i in range(0,len(y_test)):
    if y_test.values[i,0] == 2:
        b.append(i)
   
NS_train_X = np.array(pd.DataFrame(X_train).drop(index=a))
NS_Y_train = np.array(pd.DataFrame(y_train).drop(index=a))
NS_Y_test =np.array(pd.DataFrame(y_test).drop(index=b))
NS_X_test= np.array(pd.DataFrame(X_test).drop(index=b))

NS_train_Y = list()
for i in range(0,len(NS_Y_train)):
	if NS_Y_train[i,0]==4:
		NS_train_Y.append(0)
	else:
		NS_train_Y.append(1)
NS_test_Y = list()
for i in range(0,len(NS_Y_test)):
	if NS_Y_test[i,0]==4:
		NS_test_Y.append(0)
	else:
		NS_test_Y.append(1)     
n_samples = NS_train_X.shape[0]
n_features = NS_train_X.shape[1]
NSGrid = GridSearchCV(RandomForestClassifier(random_state=2,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
# pipe = Pipeline([('scaler', StandardScaler()), ('classifier',RandomForestClassifier(random_state=2,bootstrap=True))])
# NSGrid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, verbose=1)
NSGrid.fit(NS_train_X,np.ravel(np.array(NS_train_Y)))
y_pred = NSGrid.predict(NS_X_test)
print(precision_score(NS_test_Y, y_pred,average='micro'))
print(classification_report(NS_test_Y, y_pred,target_names=['SCLC', 'NSCLC']))
NSpr = list()
for i in range(0,8):
    test = np.array(testMatlab[i,:])
    # test = np.array(scalednews2[i,:])
    NSpr.append(NSGrid.predict_proba(test.reshape(1, -1)))
    
NSClf = RandomForestClassifier(random_state=2,bootstrap=True,max_depth=NSGrid.best_params_["max_depth"], 
                               max_features=NSGrid.best_params_["max_features"],
                               min_samples_leaf=NSGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=NSGrid.best_params_["min_samples_split"],
                               n_estimators=NSGrid.best_params_
                               ["n_estimators"])
# NSClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=NSGrid.best_params_["classifier__max_depth"], 
#                                max_features=NSGrid.best_params_["classifier__max_features"],
#                                min_samples_leaf=NSGrid.best_params_["classifier__min_samples_leaf"] ,
#                                min_samples_split=NSGrid.best_params_["classifier__min_samples_split"],
#                                n_estimators=NSGrid.best_params_
#                                ["classifier__n_estimators"])

# NS_train_X_t = scaler2.transform(NS_train_X)
# NS_test_X_t = scaler2.transform(NS_X_test)
NSClf.fit(NS_train_X, np.ravel(NS_train_Y))
explainer = shap.TreeExplainer(NSClf)
shap_values = explainer.shap_values(NS_X_test)
class_names = ['SCLC', 'NSCLC']
shap.summary_plot(shap_values, NS_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = '.../classification/LC_JX_model/NCBI_JX/NS_model.sav'
pickle.dump(NSGrid, open(filename, 'wb'))

# Subt

# 0 - ad, 1 - LaC, 2 - N, 3 - SC, 4 - SCLC
a2 = list()
for i in range(0,len(y_train)):
    if y_train.values[i,0] == 2 or y_train.values[i,0] == 4:
        a2.append(i)
b2 = list()
for i in range(0,len(y_test)):
    if y_test.values[i,0] == 2 or y_test.values[i,0] == 4:
        b2.append(i)        
Subt_train_X = np.array(pd.DataFrame(X_train).drop(index=a2))
Subt_Y_train = np.array(pd.DataFrame(y_train).drop(index=a2))
Subt_Y_test =np.array(pd.DataFrame(y_test).drop(index=b2))
Subt_X_test= np.array(pd.DataFrame(X_test).drop(index=b2))
n_samples = Subt_train_X.shape[0]
n_features = Subt_train_X.shape[1]

# SubtGrid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, verbose=1)
Subt_train_Y = list()
for i in range(0,len(Subt_Y_train)):
    if Subt_Y_train[i,0]==0:
        Subt_train_Y.append(0)
    if Subt_Y_train[i,0]==1:
        Subt_train_Y.append(1)
    if Subt_Y_train[i,0]==3:
        Subt_train_Y.append(2)

Subt_test_Y = list()
for i in range(0,len(Subt_Y_test)):
    if Subt_Y_test[i,0]==0:
        Subt_test_Y.append(0)
    elif Subt_Y_test[i,0]==1:
        Subt_test_Y.append(1)
    else:
        Subt_test_Y.append(2)
SubtGrid = GridSearchCV(RandomForestClassifier(random_state=0,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
# pipe = Pipeline([('scaler', StandardScaler()), ('classifier',RandomForestClassifier(random_state=1,bootstrap=True))])
# SubtGrid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, verbose=1)
SubtGrid.fit(Subt_train_X,np.ravel(np.array(Subt_train_Y)))
y_pred = SubtGrid.predict(Subt_X_test)
print(precision_score(Subt_test_Y, y_pred,average='micro'))
print(classification_report(Subt_test_Y, y_pred,target_names=['AD', 'LaC', 'SC' ]))
Subpr = list()
for i in range(0,8):
    test = np.array(testMatlab[i,:])
    # test = np.array(scalednews2[i,:])
    Subpr.append(SubtGrid.predict_proba(test.reshape(1, -1)))

SubtClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=SubtGrid.best_params_["max_depth"], 
                               max_features=SubtGrid.best_params_["max_features"],
                               min_samples_leaf=SubtGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=SubtGrid.best_params_["min_samples_split"],
                               n_estimators=SubtGrid.best_params_
                               ["n_estimators"])
SubtClf.fit(Subt_train_X, np.ravel(Subt_train_Y))
explainer = shap.TreeExplainer(SubtClf)
shap_values = explainer.shap_values(Subt_X_test)
class_names = ['AD', 'LaC', 'SC' ]
shap.summary_plot(shap_values, Subt_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = '.../classification/LC_JX_model/NCBI_JX/Subt_model.sav'
pickle.dump(SubtGrid, open(filename, 'wb'))

ShapNC= pd.DataFrame (np.vstack(Allshaps[0]).T).to_csv('.../classification/LC_JX_model/NCBI_JX/NCshap_array.csv')
ShapNS= pd.DataFrame (np.vstack(Allshaps[1]).T).to_csv('.../classification/LC_JX_model/NCBI_JX/NSshap_array.csv')
ShapAS= pd.DataFrame (np.vstack(Allshaps[2]).T).to_csv('.../classification/LC_JX_model/NCBI_JX/Subtshap_array.csv')