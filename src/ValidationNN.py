import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr
from load_data import load_all_data
from binaryclass_memory import *


#Load data
fused, charac_matrix, norm_matrix, families_matrix, names_val, names_fused, data_to_fuse, val, outliers = load_all_data()

#Train model
params = {  'learning_rate': 0.0085,
            'weight_decay' : 1.41e-7,
            'n1': 37}

model = train_best_model(fused, params)

#Evaluate clustering
scores = []
for i in range(0,len(charac_matrix)):
    X = charac_matrix[i][['mean_expression','CV2ofmeans_residuals']]
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
    
scores_df = pd.DataFrame(scores, index = names_fused + names_val, columns= ['accuracy', 'recovery memory gene', 'FP', 'precision', 'recovery', 'ensembling precision', 'ensembling recovery'])
scores_df.to_csv('../data/NN.csv', index=True)