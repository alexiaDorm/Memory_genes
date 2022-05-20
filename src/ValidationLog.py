import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr


from load_data import open_charac, add_general_charac
from binaryclass_memory import *

general_charac = pyreadr.read_r('../data/Characteristics_masterfiles/General_characteristics/EPFL_gene_master_matrix.RData')['gene_master_matrix']

names = ['AE3', 'AE4', 'AE7', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2', 'CD8', 'L1210', 'LK_D2_exp1_library_d2_1', 'LK_D2_exp1_library_d2_2', 'LK_D2_exp1_library_d2_3', 'LK_LSK_D2_exp3_library_d2_1', 
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
val = np.arange(8,32,1)
val = np.append(val,0)
data_to_fuse = [1,3,4,5,6,7] 

outliers = []
for data in charac_matrix:
    #Normalize skew_residuals, same for mean_expression after removing outliers
    charac_matrix[i], outlier_temp = remove_extreme_values(charac_matrix[i], k=200)
    outliers.append(outlier_temp)
    charac_matrix[i]['skew_residuals'], charac_matrix[i]['mean_expression'] = normalize(charac_matrix[i]['skew_residuals']), normalize(charac_matrix[i]['mean_expression'])

val_charac =  []
names_val = []
for i in val:
    val_charac.append(charac_matrix[i])
    names_val.append(names[i])

fused_charac = []
names_fused = []
for i in data_to_fuse:
    fused_charac.append(charac_matrix[i])
    names_fused.append(names[i])
    
fused = pd.concat(fused_charac)

#Best parameters
lamb = 100
FS = ['skew_residuals', 'mean_expression', 'length', 'GC', 'Myc', 'Nanog', 'Sox2', 'H3K4me3', 'H3K27ac', 'Ctnnb1', 'Srebf1']

X = np.array(fused[FS])
Y = np.array(fused['memory_gene'])

clf = LogisticRegression(C = lamb, class_weight = 'balanced', max_iter=10000).fit(X,Y)

#Evaluate clustering
scores = []
for i in data_to_fuse:
    X = np.array(charac_matrix[i][FS])
    Y = np.array(charac_matrix[i]['memory_gene'])
    
    #Evaluate fitted classifier
    acc = clf.score(X, Y)
    
    y = clf.predict(X)
    non_memory_gene = list(charac_matrix[i][Y == False].index)
    memory_gene = list(charac_matrix[i][Y == True].index)
    y = pd.DataFrame(y, index = charac_matrix[i].index, columns = ['pred'])

    y_non_mem = y.loc[non_memory_gene]
    y_mem = y.loc[memory_gene]
    recovery = np.sum(y_mem['pred'])/len(memory_gene)
    false_pos = np.sum(y_non_mem['pred'])
    
    score = [acc, recovery, false_pos]
    
    score.extend(predict_evaluate(charac_matrix[i], norm_matrix[i], families_matrix[i], clf, mult_pred=True, outliers = outliers[i]))
    scores.append(score)

#Evaluate clustering on validation set 
for i in val:
    X = np.array(charac_matrix[i][FS])
    Y = np.array(charac_matrix[i]['memory_gene'])
    
    #Evaluate fitted classifier
    acc = clf.score(X, Y)
    
    y = clf.predict(X)
    non_memory_gene = list(charac_matrix[i][Y == False].index)
    memory_gene = list(charac_matrix[i][Y == True].index)
    y = pd.DataFrame(y, index = charac_matrix[i].index, columns = ['pred'])

    y_non_mem = y.loc[non_memory_gene]
    y_mem = y.loc[memory_gene]
    recovery = np.sum(y_mem['pred'])/len(memory_gene)
    false_pos = np.sum(y_non_mem['pred'])
    
    score = [acc, recovery, false_pos]
    
    score.extend(predict_evaluate(charac_matrix[i], norm_matrix[i], families_matrix[i], clf, mult_pred=True, outliers = outliers[i]))
    scores.append(score)
    
scores_df = pd.DataFrame(scores, index = names_fused + names_val, columns= ['accuracy', 'recovery memory gene', 'FP', 'precision', 'recovery', 'ensembling precision', 'ensembling recovery'])
scores_df.to_csv('../data/binaryClass_scores/bestLogFS.csv', index=True)
