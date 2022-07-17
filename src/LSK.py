import numpy as np
import matplotlib.pyplot as plt

import io 
import pandas as pd  
import pyreadr

from load_data import *
from pred_score import *
from overlap_genes import *

#Fixing seed to get reproducible results
np.random.seed(3)

def prediction_onlibrary(name:str, norm_path:str, family_info_path:str, flip:bool):
    
    norm_path = '../data/family_datasets/Weinreb_libraries_norm_lifted/' + norm_path
    norm  = pyreadr.read_r(norm_path)
    norm = norm[None]

    family_info_path = '../data/family_datasets/Family_info/' + family_info_path
    family_info = pyreadr.read_r(family_info_path)
    family_info = np.array(family_info['WORK_clones'])
    if flip == True:
        family_info[:,[0,1]] = family_info[:,[1,0]]
    
    families, count = np.unique(family_info[:,0], return_counts=True)
    family_interest = families[np.logical_and(count > 1, count < 6)]
    
    #Norm data with only the cells belonging to the family of interest
    cells_interest = []
    for fam in family_interest:
        cell = family_info[fam == family_info[:,0]][:,1]
        cells_interest.append(cell)
    cells_interest = [item for sublist in cells_interest for item in sublist]
        
    norm = norm.loc[:,cells_interest]
    y = pd.DataFrame(np.zeros((norm.shape[1],)), index= norm.columns)
    family_info = pd.DataFrame(family_info[:,0], index = family_info[:,1])
    y.loc[cells_interest] = (family_info.loc[cells_interest])
    y = np.squeeze(np.array(y))   

    print(name)
    
    norm.to_csv('../data/merged_data/' + name + '.csv', index=True)
    pd.DataFrame(y).to_csv('../data/merged_data/y_' + name + '.csv', index=False)
    
    p_value_path = p_value_path = '../data/Characteristics_masterfiles/Memory_genes/P_value_estimate_CV2_ofmeans_' + name + '.txt'
    memory = read_memory_genes(p_value_path)
    memory_genes = get_memory_genes(memory) 

    pd.DataFrame(memory_genes).to_csv('../data/CV2genes/' + name + '_CV2mean.csv', index = False)
    
    norm = pd.read_csv ('../data/merged_data/' + name + '.csv')
    norm = norm.set_index('Unnamed: 0')
    y = np.squeeze(np.array(pd.read_csv ('../data/merged_data/y_' + name + '.csv')))
    print(norm.shape, y.shape)
    
    gene_ML = np.squeeze(pd.read_csv ('../data/optimized_subsets/' + name + 'genes_best.csv'))
    norm_ML = np.array(norm.loc[gene_ML]).T
    subset = np.ones((len(gene_ML),))
    subsets_ML = subsampling_genes(subset, 100, 0.25)
    print(norm_ML.shape)
 
    gene_CV2 = np.squeeze(pd.read_csv ('../data/CV2genes/' + name + '_CV2mean.csv'))
    norm_CV2 = np.array(norm.loc[gene_CV2]).T
    subset = np.ones((len(gene_CV2),))
    subsets_CV2 = subsampling_genes(subset, 100, 0.25)
    print(norm_CV2.shape)

    #Predict family using ML and CV2 sets
    model_ML = EnsemblingHierarchical(np.unique(y),compute_precision,True, subsets = subsets_ML, ensembling='voting', threshold_voting = 0.5)
    result_ML = model_ML.fit_predict(X = norm_ML, y= y)

    model_CV2 = EnsemblingHierarchical(np.unique(y),compute_precision,True, subsets = subsets_CV2, ensembling='voting', threshold_voting = 0.5)
    result_CV2 = model_CV2.fit_predict(X = norm_CV2, y= y)

    #Compute scores
    score = [model_ML.score_, compute_sensitivity(y, result_ML), model_CV2.score_, compute_sensitivity(y, result_CV2)] 
    
    return score

name_library = ['LSK_D2_exp1_library_LSK_d2_1','LSK_D2_exp1_library_LSK_d2_2','LSK_D2_exp1_library_LSK_d2_3','LSK_D2_exp2_library_d2A_1','LSK_D2_exp2_library_d2A_2','LSK_D2_exp2_library_d2A_3','LSK_D2_exp2_library_d2A_4','LSK_D2_exp2_library_d2A_5',
                'LSK_D2_exp2_library_d2B_1','LSK_D2_exp2_library_d2B_2','LSK_D2_exp2_library_d2B_3','LSK_D2_exp2_library_d2B_4','LSK_D2_exp2_library_d2B_5']

libraries_LSK = ['Weinreb_LSK_D2_exp1_library_LSK_d2_1_norm.rds','Weinreb_LSK_D2_exp1_library_LSK_d2_2_norm.rds','Weinreb_LSK_D2_exp1_library_LSK_d2_3_norm.rds','Weinreb_LSK_D2_exp2_library_d2A_1_norm.rds','Weinreb_LSK_D2_exp2_library_d2A_2_norm.rds',
                 'Weinreb_LSK_D2_exp2_library_d2A_3_norm.rds','Weinreb_LSK_D2_exp2_library_d2A_4_norm.rds','Weinreb_LSK_D2_exp2_library_d2A_5_norm.rds','Weinreb_LSK_D2_exp2_library_d2B_1_norm.rds','Weinreb_LSK_D2_exp2_library_d2B_2_norm.rds','Weinreb_LSK_D2_exp2_library_d2B_3_norm.rds',
                 'Weinreb_LSK_D2_exp2_library_d2B_4_norm.rds', 'Weinreb_LSK_D2_exp2_library_d2B_5_norm.rds']
family_info_LSK = ['family_info_Weinreb_LSK_D2_exp1_library_LSK_d2_1.RData','family_info_Weinreb_LSK_D2_exp1_library_LSK_d2_2.RData','family_info_Weinreb_LSK_D2_exp1_library_LSK_d2_3.RData','family_info_Weinreb_LSK_D2_exp2_library_d2A_1.RData',
                   'family_info_Weinreb_LSK_D2_exp2_library_d2A_2.RData','family_info_Weinreb_LSK_D2_exp2_library_d2A_3.RData','family_info_Weinreb_LSK_D2_exp2_library_d2A_4.RData','family_info_Weinreb_LSK_D2_exp2_library_d2A_5.RData','family_info_Weinreb_LSK_D2_exp2_library_d2B_1.RData',
                   'family_info_Weinreb_LSK_D2_exp2_library_d2B_2.RData','family_info_Weinreb_LSK_D2_exp2_library_d2B_3.RData','family_info_Weinreb_LSK_D2_exp2_library_d2B_4.RData','family_info_Weinreb_LSK_D2_exp2_library_d2B_5.RData']

scores = []
lib_name = []
for i in range(3,len(libraries_LSK)):
    score  = prediction_onlibrary(name=name_library[i], norm_path=libraries_LSK[i], family_info_path=family_info_LSK[i], flip=True)
    scores.append(score)
    
names_scores = ['ML precision', 'ML recovery', 'CV2 precision', 'CV2 recovery']; 
scores = pd.DataFrame(scores, columns = names_scores, index = name_library[3:])
scores.to_csv('LSK_pred.csv', index = True)