import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr

from load_data import open_charac, add_general_charac
from binaryclass_memory import *

general_charac = pyreadr.read_r('../data/Characteristics_masterfiles/General_characteristics/EPFL_gene_master_matrix.RData')['gene_master_matrix']

names = ['AE3', 'AE4', 'AE7', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2']
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
    charac_matrix[i] = charac_matrix[i].drop(['CV2ofmeans_residuals','cell_cycle_dependence', 'skew', 'CV2ofmeans'], axis=1)
    charac_matrix[i] = charac_matrix[i].dropna()
    
#Remove AE7, also keep BIDDYD15_2 for validation
val = [8]
data_to_fuse = [0,1,3,4,5,6,7] 

outliers = []
for data in charac_matrix:
    #Normalize skew_residuals, same for mean_expression after removing outliers
    charac_matrix[i], outlier_temp = remove_extreme_values(charac_matrix[i], k=200)
    outliers.append(outlier_temp)
    charac_matrix[i]['skew_residuals'], charac_matrix[i]['mean_expression'] = normalize(charac_matrix[i]['skew_residuals']), normalize(charac_matrix[i]['mean_expression'])

val_charac =  []
for i in val:
    val_charac.append(charac_matrix[i])

fused_charac = []
names_fused = []
for i in data_to_fuse:
    fused_charac.append(charac_matrix[i])
    names_fused.append(names[i])
    
fused = pd.concat(fused_charac)

#Best parameters
lamb = 0.0001

X = np.array(fused.drop(columns=['memory_gene']))
Y = np.array(fused['memory_gene'])

clf = SVC(C=lamb, kernel = 'rbf', class_weight = 'balanced').fit(X,Y)

#Evaluate clustering
scores = []
for i in data_to_fuse:
    X = np.array(charac_matrix[i].drop(columns=['memory_gene']))
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
    
scores_df = pd.DataFrame(scores, index = names_fused, columns= ['accuracy', 'recovery memory gene', 'FP', 'precision', 'recovery', 'ensembling precision', 'ensembling recovery'])
scores_df.to_csv('../data/binaryClass_scores/bestSVMfew.csv', index=True)

mean_score = np.mean(scores, axis = 0)
print('accuracy: ', mean_score[0])
print('recovery memory genes: ', mean_score[1])
print('false postive: ', mean_score[2])
print('100 clustering precision: ', mean_score[3])
print('100 clustering recovery: ', mean_score[4])

X = np.array(charac_matrix[8].drop(columns=['memory_gene']))
Y = np.array(charac_matrix[8]['memory_gene'])
    
#Evaluate fitted classifier
acc = clf.score(X, Y)
    
y = clf.predict(X)
non_memory_gene = list(charac_matrix[8][Y == False].index)
memory_gene = list(charac_matrix[8][Y == True].index)
y = pd.DataFrame(y, index = charac_matrix[8].index, columns = ['pred'])

y_non_mem = y.loc[non_memory_gene]
y_mem = y.loc[memory_gene]
recovery = np.sum(y_mem['pred'])/len(memory_gene)
false_pos = np.sum(y_non_mem['pred'])
    
score = [acc, recovery, false_pos]
    
score.extend(predict_evaluate(charac_matrix[8], norm_matrix[8], families_matrix[8], clf, mult_pred=True, outliers=outliers[8]))

print('accuracy: ', score[0])
print('recovery memory genes: ', score[1])
print('false postive: ', score[2])
print('100 clustering precision: ', score[3])
print('100 clustering recovery: ', score[4])