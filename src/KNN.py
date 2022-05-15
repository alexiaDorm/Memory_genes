#Import library
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

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
    
#Add general characteristic
for i in range(0,len(charac_matrix)):
    charac_matrix[i] = add_general_charac(charac_matrix[i], general_charac)
    charac_matrix[i] = charac_matrix[i].drop(['CV2ofmeans_residuals','cell_cycle_dependence', 'skew', 'CV2ofmeans', 'exon_expr_median', 'exon_expr_mean'], axis=1)
    charac_matrix[i] = charac_matrix[i].dropna()
    
#Remove AE7, also keep BIDDYD15_2 for validation
val = [8]
data_to_fuse = [0,1,3,4,5,6,7]

for data in charac_matrix:
    #Normalize skew_residuals, same for mean_expression after removing outliers
    charac_matrix[i], outliers = remove_extreme_values(charac_matrix[i], k=200)
    charac_matrix[i]['skew_residuals'], charac_matrix[i]['mean_expression'] = normalize(charac_matrix[i]['skew_residuals']), normalize(charac_matrix[i]['mean_expression'])

val_charac =  []
for i in val:
    val_charac.append(charac_matrix[i])
fused_charac = []
name_fused = []
for i in data_to_fuse:
    fused_charac.append(charac_matrix[i])
    name_fused.append(names[i])
    
fused = pd.concat(fused_charac)
X = np.array(fused.drop(columns=['memory_gene']))
y = np.array(fused['memory_gene'])

#Oversample data
ros = RandomOverSampler(random_state=42)    
#X, y = ros.fit_resample(X,y)

#Undersample data
rus = RandomUnderSampler(random_state=42)
X,y = rus.fit_resample(X,y)

#KNN - hyperparameters tuning
#Define model, grid search space, CV
model = KNeighborsClassifier()
cv = KFold(n_splits=5, shuffle=True, random_state=1)

grid = {'n_neighbors': np.arange(2,31,1),
 'weights' : ['uniform', 'distance'],
 'leaf_size' : np.arange(5,51,5),
 'p' : np.arange(1,5,1)}

grid_search = RandomizedSearchCV(estimator = model, param_distributions = grid, n_iter = 200, cv = cv, scoring='accuracy', random_state=42, n_jobs = -1)
grid_search.fit(X, y)

#Get best param
best_acc, best_params = grid_search.best_score_, grid_search.best_params_
print('The best hyperparameters are: ', best_params, 'with accuracy: ', best_acc) 
    
'''#Fit KNN with best params and evaluate clustering
model = KNeighborsClassifier(n_neighbors = best_params['n_neighbors'], weights = best_params['weights'], leaf_size = best_params['leaf_size'], p = best_params['p'])
model = rf.fit(X,y)

clust_score = []
for i in data_to_fuse:
    clust_score.append(predict_evaluate(charac_matrix[i], norm_matrix[i], families_matrix[i], model, mult_pred=True))
    
#Save individual clustering results
scores_df = pd.DataFrame(clust_score, index = name_fused, columns= ['precision', 'recovery','100 precision', '100 recovery'])
scores_df.to_csv('../data/binaryClass_scores/KNNover.csv', index=True)'''