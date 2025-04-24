# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 01:30:55 2022

@author: Ezgi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:11:58 2022

@author: Ezgi
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

data_dir = 'C:\\Users\Ezgi\Desktop\classification' #for sclc
subs = pd.read_csv('C:\\Users\Ezgi\Desktop\classification\TPM\modelsubsystems.csv',sep=',',header=None)
df = pd.read_csv('C:\\Users\Ezgi\Desktop\classification\sclcFPKM.csv',sep=',',header=None)
dfgenes = pd.read_csv('C:\\Users\Ezgi\Desktop\classification\sclcCount.csv',sep=',')
model_dir = 'C:\\Users\Ezgi\Desktop\classification'

model = cobra.io.read_sbml_model(join(model_dir, "classifiermodel.xml"))
modelgenes = pd.read_csv(join(model_dir,'modelGenesENS.csv'),sep=',',header=None)
for i in range(0,13634):
    model.reactions[i].subsystem = subs.values[i,1]
IDdict = dict(zip(list(modelgenes[0]),list(modelgenes[1])))
gprbase = cobamp.wrappers.external_wrappers.COBRAModelObjectReader(model=model)
media = list(['EX_gly[e]', 'EX_arg_L[e]', 'EX_asp_L[e]', 'EX_asn_L[e]', 'EX_cys_L[e]','EX_glu_L[e]', 'EX_gln_L[e]', 'EX_his_L[e]', 'EX_4hpro[e]', 'EX_ile_L[e]', 'EX_leu_L[e]', 'EX_lys_L[e]', 'EX_met_L[e]', 'EX_phe_L[e]', 'EX_pro_L[e]', 'EX_ser_L[e]', 'EX_thr_L[e]', 'EX_trp_L[e]', 'EX_tyr_L[e]', 'EX_val_L[e]', 'EX_btn[e]', 'EX_chol[e]', 'EX_pnto_R[e]', 'EX_fol[e]', 'EX_ncam[e]', 'EX_bz[e]', 'EX_pydxn[e]', 'EX_ribflv[e]', 'EX_thm[e]', 'EX_inost[e]', 'CAt7r', 'EX_so4[e]', 'EX_k[e]', 'NKCCt', 'EX_hco3[e]', 'EX_pi[e]', 'EX_gthrd[e]', 'EX_h2o[e]', 'EX_h[e]', 'EX_glc_D[e]'])
genes = list()
ensid = list()
for i in range(0,len(model.genes)):
    genes.append(model.genes[i].id)
# for i in range(0,60659):
#     ensid.append(dfgenes.iloc[i,0].split(sep='.')[0])
ensdata = dfgenes.values[:,0]
solutionAll = list()
ExpRxnsAll = list()
maxGeneExp = list()
geneExps = list()
for k in range(0,86,1):
# for k in range(0,10,1):
    explist = list()
    explist = np.array(df.values[:,k])
    pkmvalue = explist[1006]
#create expression array for each gene,entrez ids can be keys of dictionary or symbols
    dataDict = dict()   
    dataDict = dict(zip(ensdata,explist))
    geneExp = list()
    for i in range(0,len(model.genes)):
        if pd.isna(IDdict[int(model.genes[i].id)]):
            geneExp.append(np.nan)
        else:
            if IDdict[int(model.genes[i].id)] in dataDict:
                geneExp.append(dataDict[IDdict[int(model.genes[i].id)]])
            else:
                geneExp.append(np.nan)
    geneExpN = geneExp/pkmvalue
    # geneExpN = (geneExpN - np.nanmin(geneExpN)) / (np.nanmax(geneExpN) - np.nanmin(geneExpN)) * (1000 - 0) + 0  
    # geneExpScaled = scaler.fit_transform(np.array(geneExpN).reshape(-1,1))
    # df_n = df_n.T
    geneExpDict = dict(zip(genes,geneExpN))
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
            if model.reactions[i].reversibility:
                model.reactions[i].bounds = (-1000,1000)
            else:
                model.reactions[i].bounds = (0,1000)
        else:
            if model.reactions[i].reversibility:
                model.reactions[i].bounds = (-1*rxnExp[i],rxnExp[i])
            else:
                model.reactions[i].bounds = (0,rxnExp[i])
        #Setting other constraints - media/oxygen requirement/drug met inactivation
    for i in range(0,len(model.reactions)):
        if model.reactions[i].id.startswith('EX_'):
            model.reactions[i].bounds = (0,1000)
        if model.reactions[i].subsystem.startswith('Drug'):
            model.reactions[i].bounds = (0,0)
        if model.reactions[i].subsystem.startswith('IDH1'):
            model.reactions[i].bounds = (0,0)
        if model.reactions[i].subsystem.startswith('MODULE'):
            model.reactions[i].bounds = (0,0)
        if model.reactions[i].id in media:
            model.reactions[i].bounds = (-1000,1000)
        if model.reactions[i].id == 'EX_o2[e]':
            model.reactions[i].bounds = (-1000,0)
    model.objective = model.reactions.biomass_maintenance
    solution = model.optimize('max')   
    solutionAll.append(solution.fluxes.values)
    maxGeneExp.append(max(geneExpN))
    geneExps.append(geneExpN)

    
solutionAll2 = np.vstack(solutionAll)
solutionAll2 = np.array(solutionAll2).T
ExpRxnsAll2 = np.vstack(ExpRxnsAll)
ExpRxnsAll2 = np.array(ExpRxnsAll2).T
pd.DataFrame(solutionAll2).to_csv(join(data_dir,'OptResFPKMsclc_pkm.csv'))
pd.DataFrame(ExpRxnsAll2).to_csv(join(data_dir,'RxnExpFPKMsclc_pkm.csv'))
# pd.DataFrame(np.arraymaxGeneExp.T).to_csv(join(data_dir,'maxGeneExpTPM_pkm_scaled.csv'))


#Stats -------------------------------------------------------------------


