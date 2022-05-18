#Import library
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import KFold, GridSearchCV
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
val = [0,8]
data_to_fuse = [1,3,4,5,6,7]

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
X = fused.drop(columns=['memory_gene'])
y = fused['memory_gene']     

#Select relevant features
best_acc, best_param, best_feat = 0, 0, []
for i in range (2, X.shape[1] + 1):
    selector = SelectKBest(mutual_info_classif, k=i)
    X_redu = selector.fit_transform(X, y)

    #Grid search around best found parameters during random grid search
    model = LogisticRegression(class_weight = "balanced_subsample")
    grid = {'C': np.logspace(-9,5,14)}

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

    #Grid search
    grid_result = grid_search.fit(X_redu, y)

    #Get cv accuracy
    acc = grid_result.best_score_
    
    if acc > best_acc:
        best_acc, best_param = acc, grid_result.best_params_
        cols = selector.get_support(indices=True)
        best_feat = X.iloc[:,cols].columns.tolist()

print('Mutual info FS: features used = ', best_feat, 'with C = ', best_param['C'], 'and accuracy = ', best_acc)
#Fit Logreg with best params and evaluate clustering
model = LogisticRegression(C = best_param['C'], class_weight = "balanced_subsample")
model = model.fit(X,y)

clust_score = []
for i in data_to_fuse:
    clust_score.append(predict_evaluate(charac_matrix[i], norm_matrix[i], families_matrix[i], model, mult_pred=True))
    
#Save individual clustering results
scores_df = pd.DataFrame(clust_score, index = name_fused, columns= ['precision', 'recovery','100 precision', '100 recovery'])
scores_df.to_csv('../data/binaryClass_scores/LogFS_MI.csv', index=True)