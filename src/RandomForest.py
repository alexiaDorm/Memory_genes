#Import library
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


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

grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

#Random forest - hyperparameters tuning
#Define model, random grid search space, CV
rf = RandomForestRegressor(class_weight = "balanced_subsample")
cv = KFold(n_splits=5, shuffle=True, random_state=1)
random_search = RandomizedSearchCV(estimator = rf, param_distributions = grid, n_iter = 100, cv = cv, scoring='accuracy', random_state=42, n_jobs = -1)

#Get best param
random_search.fit(X, y)
best_acc, best_params = random_search.best_score_, random_search.best_params
print('The best hyperparameters are: ', best_param, 'with accuracy: ', best_acc) 

#------------------------------------------------------------------------
'''rf = RandomForestRegressor(class_weight = "balanced_subsample")
grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
cv = KFold(n_splits=5, shuffle=True, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
    
    #Grid search
    grid_result = grid_search.fit(X, y)
    
    #Get best scores
    best_acc, best_param = grid_result.best_score_, grid_result.best_params_
    print('The best hyperparameters are: ', best_param, 'with accuracy: ', best_acc)  
    
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    
    #Fit ADAboost with best params and evaluate clustering
    base_tree = DecisionTreeClassifier(max_depth = i, class_weight = 'balanced')
    model = AdaBoostClassifier(base_estimator = base_tree, n_estimators = best_param['n_estimators'], learning_rate= best_param['learning_rate'])
    model = model.fit(X,y)

    clust_score = []
    for i in data_to_fuse:
        clust_score.append(predict_evaluate(charac_matrix[i], norm_matrix[i], families_matrix[i], model, mult_pred=True))
    
    #Save individual clustering results
    scores_df = pd.DataFrame(clust_score, index = name_fused, columns= ['precision', 'recovery','100 precision', '100 recovery'])
    scores_df.to_csv('../data/binaryClass_scores/ADAboost_tree/ADA' + str(i) +'.csv', index=True)'''