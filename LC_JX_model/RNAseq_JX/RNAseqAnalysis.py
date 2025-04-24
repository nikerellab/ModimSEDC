# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:11:58 2022
codes for RNAseq data integration to irreversible Recon3D and classification
@author: Ezgi Tanil
"""
import pandas as pd
import cobra
import cobamp
import pickle
import gurobipy
import numpy as np
import os
from os.path import join
import GEOparse
from sklearn.preprocessing import MinMaxScaler

data_dir  = '.../classification/LC_JX_model/RNAseq_JX'
subs      = pd.read_csv(join(data_dir,'modelsubsystems.csv'),sep=',',header=None)
df        = pd.read_csv(join(data_dir,'FPKMdata_2704.csv'),sep=',',header=None)
dfgenes   = pd.read_csv(join(model_dir,'modelgenesTPM.csv'),sep=',')
model_dir = '../classification/LC_JX_model'


model = cobra.io.read_sbml_model(join(model_dir, "modelIrrev.xml"))
modelgenes = pd.read_csv(join(model_dir,'modelGenesENS.csv'),sep=',',header=None)
for i in range(0,17949):
    model.reactions[i].subsystem = subs.values[i,0]
IDdict = dict(zip(list(modelgenes[0]),list(modelgenes[1])))
gprbase = cobamp.wrappers.external_wrappers.COBRAModelObjectReader(model=model)
media = list(['EX_gly[e]', 'EX_arg_L[e]', 'EX_asp_L[e]', 'EX_asn_L[e]', 'EX_cys_L[e]','EX_glu_L[e]', 'EX_gln_L[e]', 'EX_his_L[e]', 'EX_4hpro[e]', 'EX_ile_L[e]', 'EX_leu_L[e]', 'EX_lys_L[e]', 'EX_met_L[e]', 'EX_phe_L[e]', 'EX_pro_L[e]', 'EX_ser_L[e]', 'EX_thr_L[e]', 'EX_trp_L[e]', 'EX_tyr_L[e]', 'EX_val_L[e]', 'EX_btn[e]', 'EX_chol[e]', 'EX_pnto_R[e]', 'EX_fol[e]', 'EX_ncam[e]', 'EX_bz[e]', 'EX_pydxn[e]', 'EX_ribflv[e]', 'EX_thm[e]', 'EX_inost[e]', 'CAt7r', 'EX_so4[e]', 'EX_k[e]', 'NKCCt', 'EX_hco3[e]', 'EX_pi[e]', 'EX_gthrd[e]', 'EX_h2o[e]', 'EX_h[e]', 'EX_glc_D[e]'])
genes = list()
ensid = list()
for i in range(0,len(model.genes)):
    genes.append(model.genes[i].id)
for i in range(0,65217):
    ensid.append(dfgenes.iloc[i,0].split(sep='.')[0])
ensdata = dfgenes["gene_id"]
for i in range(0,len(ensdata)):
    ensid.append(ensdata[i].split(sep='.')[0])
solutionAll = list()
ExpRxnsAll = list()
for k in range(0,86,1):
    explist = list()
    explist = np.array(df.values[:,k])
    # index for PKM gene
    ind = 1007
    pkmvalue = explist[ind]
    #create expression array for each gene,entrez ids can be keys of dictionary or symbols
    dataDict = dict()   
    dataDict = dict(zip(ensid,explist))
    geneExp = list()
    for i in range(0,len(model.genes)):
        if pd.isna(IDdict[int(model.genes[i].id)]):
            geneExp.append(np.nan)
        else:
            if IDdict[int(model.genes[i].id)] in dataDict:
                geneExp.append(dataDict[IDdict[int(model.genes[i].id)]])
            else:
                geneExp.append(np.nan)
    geneExpN = np.array(geneExp)/pkmvalue
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
                # model.reactions[i].bounds = (-1000,1000)
                model.reactions[i].bounds = (0,1000)
            if model.reactions[i].id == 'EX_o2[e]_r':
                model.reactions[i].bounds = (0,1000)
    model.objective = model.reactions.biomass_maintenance
    solution = model.optimize('max')
    solutionAll.append(solution.fluxes.values)
    maxGeneExp.append(max(geneExpN))
    geneExps.append(geneExpN)

    
solutionAll2 = np.vstack(solutionAll)
solutionAll2 = np.array(solutionAll2).T
ExpRxnsAll2 = np.vstack(ExpRxnsAll)
ExpRxnsAll2 = np.array(ExpRxnsAll2).T
pd.DataFrame(solutionAll2).to_csv('../classification/OptResRNAseq.csv')
pd.DataFrame(ExpRxnsAll2).to_csv('../classification/RxnExpRNAseq.csv')

#Classification -------------------------------------------------------------------
from sklearn.preprocessing import RobustScaler,StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import shap
    

df = pd.read_csv('../classification/FPKMdata_2704.csv',sep=',',header=None)
df2 = pd.read_csv('../classification/SCLC_FPKM.csv',sep=',',header=None)
data2 = pd.concat([df.T,df2.T])
y = pd.read_csv('../classification/fpkm_y.csv',header=None,sep=',')
sclcy = pd.DataFrame(np.tile(1, (7, 1)))
sclcy2 = pd.DataFrame(np.tile(4, (79, 1)))
Y = pd.concat([y, sclcy, sclcy2])

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(data2, Y.values, train_size=0.8, random_state=1)

#NC
NC_train_X = np.array(X_train)
NC_Y_train = np.array(Y_train)
NC_Y_test =np.array(Y_test)
NC_X_test= np.array(X_test)
NC_train_Y = list()
for i in range(0,len(NC_Y_train)):
	if NC_Y_train[i,0]==1:
		NC_train_Y.append(0)
	else:
		NC_train_Y.append(1)
NC_test_Y = list()
for i in range(0,len(NC_Y_test)):
	if NC_Y_test[i,0]==1:
		NC_test_Y.append(0)
	else:
		NC_test_Y.append(1)

n_samples = NC_train_X.shape[0]
n_features = NC_train_X.shape[1]
params = {'n_estimators': [20,50,100],
          'max_depth': [None, 2, 5,],
          'min_samples_split': [2, 0.5, n_samples//2, ],
          'min_samples_leaf': [1, 0.5, n_samples//2, ],
          'max_features': [None, 'sqrt', 'auto', 'log2', 0.4,0.5, n_features//2, ],
          }
NCGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
NCGrid.fit(NC_train_X,np.ravel(np.array(NC_train_Y)))
y_pred = NCGrid.predict(NC_X_test)
print(precision_score(NC_test_Y, y_pred,average='micro'))
print(classification_report(NC_test_Y, y_pred,target_names=['Normal', 'Cancer']))
NCClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=NCGrid.best_params_["max_depth"], 
                               max_features=NCGrid.best_params_["max_features"],
                               min_samples_leaf=NCGrid.best_params_["min_samples_leaf"] ,
                               min_samples_split=NCGrid.best_params_["min_samples_split"],
                               n_estimators=NCGrid.best_params_
                               ["n_estimators"])
NCClf.fit(NC_train_X, np.ravel(NC_train_Y))
y_pred = NCClf.predict(NC_X_test)
print(precision_score(NC_test_Y, y_pred,average='micro'))
print(classification_report(NC_test_Y, y_pred,target_names=['Normal', 'Cancer']))
explainer = shap.TreeExplainer(NCClf)
shap_values = explainer.shap_values(NC_X_test)
class_names = ['Normal', 'Cancer']
shap.summary_plot(shap_values, NC_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = '../classification/NCmodel.sav'
pickle.dump(NCGrid, open(filename, 'wb'))

#SCLC-NSCLC
a = list()
for i in range(0,len(Y_train)):
    if Y_train[i,0] == 1:
        a.append(i)
b = list()
for i in range(0,len(Y_test)):
    if Y_test[i,0] == 1:
        b.append(i)
      
NS_train_X = np.array(pd.DataFrame(X_train).drop(index=a))
NS_Y_train = np.array(pd.DataFrame(Y_train).drop(index=a))
NS_Y_test =np.array(pd.DataFrame(Y_test).drop(index=b))
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
NSGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
NSGrid.fit(NS_train_X,np.ravel(np.array(NS_train_Y)))
y_pred = NSGrid.predict(NS_X_test)
print(precision_score(NS_test_Y, y_pred,average='micro'))
print(classification_report(NS_test_Y, y_pred,target_names=['SCLC', 'NSCLC']))
NSClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=NSGrid.best_params_["max_depth"], 
                                max_features=NSGrid.best_params_["max_features"],
                                min_samples_leaf=NSGrid.best_params_["min_samples_leaf"] ,
                                min_samples_split=NSGrid.best_params_["min_samples_split"],
                                n_estimators=NSGrid.best_params_
                                ["n_estimators"])
NSClf.fit(NS_train_X, np.ravel(NS_train_Y))
explainer = shap.TreeExplainer(NSClf)
shap_values = explainer.shap_values(NS_X_test)
class_names = ['SCLC', 'NSCLC']
shap.summary_plot(shap_values, NS_X_test, plot_type="bar", class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = '../classification/NSmodel.sav'
pickle.dump(NSGrid, open(filename, 'wb'))
y_score1 = NSfpkm.predict_proba(NS_X_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(NS_test_Y, y_score1)
aucNS = roc_auc_score(NS_test_Y, y_score1)
#AS
a2 = list()
for i in range(0,len(Y_train)):
    if Y_train[i,0] == 1 or Y_train[i,0] == 4:
        a2.append(i)
b2 = list()
for i in range(0,len(Y_test)):
    if Y_test[i,0] == 1 or Y_test[i,0] == 4:
        b2.append(i)        
Subt_train_X = np.array(pd.DataFrame(X_train).drop(index=a2))
Subt_Y_train = np.array(pd.DataFrame(Y_train).drop(index=a2))
Subt_Y_test =np.array(pd.DataFrame(Y_test).drop(index=b2))
Subt_X_test= np.array(pd.DataFrame(X_test).drop(index=b2))
n_samples = Subt_train_X.shape[0]
n_features = Subt_train_X.shape[1]

Subt_train_Y = list()
for i in range(0,len(Subt_Y_train)):
    if Subt_Y_train[i,0]==2:
        Subt_train_Y.append(0)
    if Subt_Y_train[i,0]==3:
        Subt_train_Y.append(1)


Subt_test_Y = list()
for i in range(0,len(Subt_Y_test)):
    if Subt_Y_test[i,0]==2:
        Subt_test_Y.append(0)
    if Subt_Y_test[i,0]==3:
        Subt_test_Y.append(1)

SubtGrid = GridSearchCV(RandomForestClassifier(random_state=1,bootstrap=True), param_grid=params, n_jobs=-1, cv=4, verbose=1)
SubtGrid.fit(Subt_train_X,np.ravel(np.array(Subt_train_Y)))
y_pred = SubtGrid.predict(Subt_X_test)
print(precision_score(Subt_test_Y, y_pred,average='micro'))
print(classification_report(Subt_test_Y, y_pred,target_names=['AD', 'SC' ]))
SubtClf = RandomForestClassifier(random_state=1,bootstrap=True,max_depth=SubtGrid.best_params_["max_depth"], 
                                max_features=SubtGrid.best_params_["max_features"],
                                min_samples_leaf=SubtGrid.best_params_["min_samples_leaf"] ,
                                min_samples_split=SubtGrid.best_params_["min_samples_split"],
                                n_estimators=SubtGrid.best_params_
                                ["n_estimators"])
SubtClf.fit(Subt_train_X, np.ravel(Subt_train_Y))
explainer = shap.TreeExplainer(SubtClf)
shap_values = explainer.shap_values(Subt_X_test)

class_names = ['AD', 'SC' ]
shap.summary_plot(shap_values, data, class_names= class_names, max_display=50)
Allshaps.append(shap_values)
filename = '../classification/Subtmodel.sav'
pickle.dump(SubtGrid, open(filename, 'wb'))

ShapNC= pd.DataFrame (np.vstack(Allshaps[0]).T).to_csv('../classification/NCshap.csv')
ShapNS= pd.DataFrame (np.vstack(Allshaps[1]).T).to_csv('../classification/NSshap.csv')
ShapAS= pd.DataFrame (np.vstack(Allshaps[2]).T).to_csv('../classification/Subtshap.csv')
data = pd.DataFrame(Subt_X_test,columns=dfgenes.values)
shapObj = explainer(data)
shap_values2 = copy.deepcopy(shapObj)
shap_values2.values = shap_values2.values[:,:,1]
shap_values2.base_values = shap_values2.base_values[:,1]
shap.plots.beeswarm(shap_values2,max_display=50)
shap.waterfall_plot(explainer.expected_value,explainer.shap_values(data))
shap.bar_plot(shap_values[0])