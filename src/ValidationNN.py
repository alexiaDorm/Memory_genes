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
params = {  'learning_rate': 0.00027,
            'weight_decay' : 1e-5,
            'n1': 27,
            'n2': 47,
            'n3': 33,
            'nb_features' : 10}

model = train_best_model(fused, params)

#Get the N top features according to mutual information
X, y = fused.drop(columns=['memory_gene']), fused['memory_gene']
FS = feature_selection(X,y,params)

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
    
    score.extend(predict_evaluate(charac_matrix[i].index, yhat, norm_matrix[i], families_matrix[i], mult_pred=True, outliers = outliers[i]))
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
scores_df.to_csv('../data/binaryClass_scores/bestNNpos3.csv', index=True)