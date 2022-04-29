#Import library
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr

from load_data import open_charac
from binaryclass_memory import *
import sys

#Set the seed for reproducibility
np.random.seed(1)
random.seed(1)

#Load data
general_charac = pyreadr.read_r('../data/Characteristics_masterfiles/General_characteristics/EPFL_gene_master_matrix.RData')['gene_master_matrix']

names = ['AE3', 'AE4', 'AE7', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2',
        'LK_D2_exp1_library_d2_1', 'LK_D2_exp1_library_d2_2', 'LK_D2_exp1_library_d2_3', 'LK_LSK_D2_exp3_library_d2_1', 
        'LK_LSK_D2_exp3_library_d2_2', 'LK_LSK_D2_exp3_library_d2_3', 'LK_LSK_D2_exp3_library_d2_4', 
        'LK_LSK_D2_exp3_library_d2_5', 'LSK_D2_exp1_library_LSK_d2_1', 'LSK_D2_exp1_library_LSK_d2_2', 'LSK_D2_exp1_library_LSK_d2_3',
       'LSK_D2_exp2_library_d2A_1', 'LSK_D2_exp2_library_d2A_2', 'LSK_D2_exp2_library_d2A_3' , 'LSK_D2_exp2_library_d2A_4', 'LSK_D2_exp2_library_d2A_5', 
       'LSK_D2_exp2_library_d2B_1','LSK_D2_exp2_library_d2B_2', 'LSK_D2_exp2_library_d2B_3', 'LSK_D2_exp2_library_d2B_4', 'LSK_D2_exp2_library_d2B_5']

charac_matrix = []
norm_matrix = []
families_matrix = []
for name in names:
    #Open characteristics file
    charac_out_path = '../data/Characteristics_masterfiles/Dataset_specific_characteristics/' + name + '__characteristics_output.txt'
    p_value_path = '../data/Characteristics_masterfiles/Memory_genes/P_value_estimate_CV2_ofmeans_' + name + '.txt'
    charac_matrix.append(open_charac(charac_out_path, p_value_path, 200))
    
    #Open normalized data
    norm_path = '../data/merged_data/' + name + '.csv'
    fam_path = '../data/merged_data/y_' + name + '.csv'
    norm = pd.read_csv (norm_path)
    norm = norm.set_index('Unnamed: 0')
    families= np.squeeze(np.array(pd.read_csv(fam_path)))
    
    norm_matrix.append(norm)
    families_matrix.append(families)
    
#Select the features with highest clustering precision
indices_max = []
for i, data in enumerate(charac_matrix):
    feat_sets =[[4,5],[3,5],[1,5],[2,5],[1,2,5],[3,4,5],[1,3,5],[2,4,5],[1,2,3,4,5]]
    name_feat = [list(data[0].iloc[: , feat].columns) for feat in feat_sets] 
    name_feat = [', '.join(names) for names in name_feat]
    scores = ['accuracy', 'recovery', 'FP', 'Clustering precision', 'Clustering recovery']
    
    score_sets = []
    for feat in feat_sets:
        score_sets.append(fit_evaluate(data[0], norm_matrix[i], families_matrix[i], 'svm', feat = feat, kernel = 'rbf', verbose =False))

    scores_df = pd.DataFrame(score_sets, index = name_feat, columns= scores)
    scores_df.to_csv('../data/binaryClass_scores/featSelecRBFSVM/' + names[i] + '.csv', index=True)
    
    #Get best features indices
    prec_rec = scores_df['Clustering precision'] + scores_df['Clustering recovery']
    ind_max = np.squeeze(np.argmax(np.array(prec_rec)))
    indices_max.append(feat_sets[ind_max])
    
pd.DataFrame(indices_max).to_csv('../data/binaryClass_scores/featSelecRBFSVM/bestFeat.csv', index=False)

#Grid search of penalty values
#L2 regularization
plot = False
feat = [4,5]
scores = ['accuracy', 'recovery', 'FP', 'Clustering precision', 'Clustering recovery']

for i, data in enumerate(charac_matrix):
    C = np.logspace(-10, 3, 14)
    scores_grid = []
    for lamb in C:
        scores_grid.append(fit_evaluate(data[0], norm_matrix[i], families_matrix[i], 'svm', feat = feat, lamb = lamb, kernel = 'rbf', verbose =False))
    
    scores_df = pd.DataFrame(scores_grid, index = C, columns= scores)
    scores_df.to_csv('../data/binaryClass_scores/RegRBFSVM/' + names[i] + '.csv', index=True)
    
    #Get best best reg indices
    prec_rec = scores_df['Clustering precision'] + scores_df['Clustering recovery']
    ind_max = np.max(np.array(prec_rec))
    indices_max.append(ind_max)
    
    if plot:
        print(names[i])
        plt.plot(C,scores_df['Clustering precision'])
        plt.title('Grid search for C regularization parameter for L2 regulatization.')
        plt.xlabel('regularization parameter, C')
        plt.xscale('log')
        plt.ylabel('clustering precision')
        plt.show()
        print('------')
        
pd.DataFrame(indices_max).to_csv('../data/binaryClass_scores/RegRBFSVM/bestreg.csv', index=False)

#Grid search of gamma parameter for rbf kernel
plot = False
feat = [4,5]
best_reg =  pd.read_csv('../data/binaryClass_scores/featSelecRBFSVM/bestreg.csv')
scores = ['accuracy', 'recovery', 'FP', 'Clustering precision', 'Clustering recovery']

indices_max = []
for i, data in enumerate(charac_matrix):
    C = np.logspace(-10, 3, 14)
    scores_grid = []
    for lamb in C:
        scores_grid.append(fit_evaluate(data[0], norm_matrix[i], families_matrix[i], 'svm', feat = feat, lamb = best_reg[i], kernel = 'rbf', gamma = lamb,  verbose =False))
    
    scores_df = pd.DataFrame(scores_grid, index = C, columns= scores)
    scores_df.to_csv('../data/binaryClass_scores/gammaRBFSVM/' + names[i] + '.csv', index=True)
    
    #Get best best gamma indices
    prec_rec = scores_df['Clustering precision'] + scores_df['Clustering recovery']
    ind_max = np.max(np.array(prec_rec))
    indices_max.append(ind_max)
    
    if plot:
        print(names[i])
        plt.plot(C,scores_df['Clustering precision'])
        plt.title('Grid search for C regularization parameter for L2 regulatization.')
        plt.xlabel('regularization parameter, C')
        plt.xscale('log')
        plt.ylabel('clustering precision')
        plt.show()
        print('------')
        
pd.DataFrame(indices_max).to_csv('../data/binaryClass_scores/gammaRBFSVM/bestFeat.csv', index=False)