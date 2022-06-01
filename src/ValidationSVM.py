import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr

from load_data import open_charac, add_general_charac
from binaryclass_memory import *

#Load data
fused, charac_matrix, norm_matrix, families_matrix, names_val, names_fused, data_to_fuse, val, outliers = load_all_data()

#Train model
params = {  'C': X, 'gamma' : X}

X = np.array(fused[['CV2ofmeans_residuals','mean_expression']])
Y = np.array(fused['memory_gene'])

model = SVC(C=params['C'], kernel = 'rbf', gamma = params['gamma'], class_weight = 'balanced').fit(X,Y)

FS

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