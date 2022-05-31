import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr


from load_data import open_charac, add_general_charac
from binaryclass_memory import *


#Evaluate clustering
scores = []
for i in data_to_fuse:
    X = charac_matrix[i][FS]
    y = np.expand_dims(charac_matrix[i]['memory_gene'], axis=1)
    dataset = Dataset(np.array(X), y)
    data = DataLoader(dataset, batch_size = 200000, shuffle=False)
    
    #compute accuracy and predict
    acc = evaluate_model(data,model)
    yhat = predict(np.array(X),model)
    
    #Compute enrichment of gene set (percentage of memory genes recovered + number of non-memory genes in set)
    recovery, false_pos = compute_enrichment(charac_matrix[i], y, yhat)
    score = [acc, recovery, false_pos]
    
    score.extend(predict_evaluate_NN(charac_matrix[i].index, yhat, norm_matrix[i], families_matrix[i], mult_pred=True, outliers = outliers[i]))
    scores.append(score)

#Evaluate clustering on validation set 
for i in val:
    X = charac_matrix[i][FS]
    y = np.expand_dims(charac_matrix[i]['memory_gene'], axis=1)
    dataset = Dataset(np.array(X), y)
    data = DataLoader(dataset, batch_size = 200000, shuffle=False)
    
    #compute accuracy and predict
    acc = evaluate_model(data,model)
    yhat = predict(np.array(X),model)
    
    #Compute enrichment of gene set (percentage of memory genes recovered + number of non-memory genes in set)
    recovery, false_pos = compute_enrichment(charac_matrix[i], y, yhat)
    score = [acc, recovery, false_pos]
    
    score.extend(predict_evaluate_NN(charac_matrix[i].index, yhat, norm_matrix[i], families_matrix[i], mult_pred=True, outliers = outliers[i]))
    scores.append(score)
    
scores_df = pd.DataFrame(scores, index = names_fused + names_val, columns= ['accuracy', 'recovery memory gene', 'FP', 'precision', 'recovery', 'ensembling precision', 'ensembling recovery'])
scores_df.to_csv('../data/binaryClass_scores/bestNN.csv', index=True)