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
lamb = 1000

X = np.array(fused[['CV2ofmeans_residuals','mean_expression']])
Y = np.array(fused['memory_gene'])

clf = LogisticRegression(C = lamb, class_weight = 'balanced', max_iter=10000).fit(X,Y)

FS = ['CV2ofmeans_residuals','mean_expression']

#Evaluate clustering
scores = []
for i in data_to_fuse:
    X = np.array(charac_matrix[i][FS])
    y = np.array(charac_matrix[i]['memory_gene'])
    
    #Evaluate fitted classifier
    acc = clf.score(X, y)
    yhat = clf.predict(X)
    
    #Compute enrichment of gene set (percentage of memory genes recovered + number of non-memory genes in set)
    recovery, false_pos = compute_enrichment(charac_matrix[i], y, yhat)
    score = [acc, recovery, false_pos]
    
    score.extend(predict_evaluate(charac_matrix[i].index, yhat, norm_matrix[i], families_matrix[i], mult_pred=True, outliers = outliers[i]))
    scores.append(score)

#Evaluate clustering on validation set 
for i in val:
    X = np.array(charac_matrix[i][FS])
    y = np.array(charac_matrix[i]['memory_gene'])
    
    #Evaluate fitted classifier
    acc = clf.score(X, y)
    yhat = clf.predict(X)
    
    #Compute enrichment of gene set (percentage of memory genes recovered + number of non-memory genes in set)
    recovery, false_pos = compute_enrichment(charac_matrix[i], y, yhat)
    score = [acc, recovery, false_pos]
    
    score.extend(predict_evaluate(charac_matrix[i].index, yhat, norm_matrix[i], families_matrix[i], mult_pred=True, outliers = outliers[i]))
    scores.append(score)
    
scores_df = pd.DataFrame(scores, index = names_fused + names_val, columns= ['accuracy', 'recovery memory gene', 'FP', 'precision', 'recovery', 'ensembling precision', 'ensembling recovery'])
scores_df.to_csv('../data/binaryClass_scores/bestLog.csv', index=True)
