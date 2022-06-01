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
params = {  'learning_rate': 1e-4,
            'weight_decay' : 1e-6,
            'n1': 29,
            'n2': 21,
            'n3': 23,
            'nb_features' : 15}


X = np.array(fused[['CV2ofmeans_residuals','mean_expression']])
Y = np.array(fused['memory_gene'])

model = LogisticRegression(C = lamb, class_weight = 'balanced', max_iter=10000).fit(X,Y)

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
    
    score.extend(predict_evaluate(charac_matrix[i][all_feat], norm_matrix[i], families_matrix[i], clf, mult_pred=True, outliers = outliers[i]))
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
    
    score.extend(predict_evaluate(charac_matrix[i][all_feat], norm_matrix[i], families_matrix[i], clf, mult_pred=True, outliers = outliers[i]))
    scores.append(score)
    
scores_df = pd.DataFrame(scores, index = names_fused + names_val, columns= ['accuracy', 'recovery memory gene', 'FP', 'precision', 'recovery', 'ensembling precision', 'ensembling recovery'])
scores_df.to_csv('../data/binaryClass_scores/bestLogFS.csv', index=True)
