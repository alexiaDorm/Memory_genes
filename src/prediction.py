import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import math 
import random
from pred_score import *

random.seed(3)
np.random.seed(3)

#Define name of all dataset to analyze, CD8 and L1210 missing add after
names = ['AE3', 'AE4' 'D0_exp1', 'D0_exp2', 'D6_exp1', 'D6_exp2', 'D15_exp1', 'D15_exp2']

scores = []; names_scores = ['ML precision', 'ML recovery', 'CV2 precision', 'CV2 recovery']
for name in names:
    
    print(name)
    #Load preprocess data
    norm = pd.read_csv ('../data/merged_data/' + name + '.csv')
    norm = norm.set_index('Unnamed: 0')
    y = np.squeeze(np.array(pd.read_csv ('../data/merged_data/y_' + name + '.csv')))

    #Get optmized gene set, for best ML and CV2 
    gene_ML = np.squeeze(pd.read_csv ('../data/optimized_subsets/' + name + 'genes_best.csv'))
    norm_ML = np.array(norm.loc[gene_ML]).T
    subset = np.ones((len(gene_ML),))
    subsets_ML = subsampling_genes(subset, 100, 0.25)

    gene_CV2 = np.squeeze(pd.read_csv ('../data/CV2genes/' + name + '_CV2mean.csv'))
    norm_CV2 = np.array(norm.loc[gene_CV2]).T
    subset = np.ones((len(gene_CV2),))
    subsets_CV2 = subsampling_genes(subset, 100, 0.25)
    
    #Predict family using ML and CV2 sets
    model_ML = EnsemblingHierarchical(np.unique(y),compute_precision,True, subsets = subsets_ML, ensembling='voting', threshold_voting = 0.5)
    result_ML = model_ML.fit_predict(X = norm_ML, y= y)
    
    model_CV2 = EnsemblingHierarchical(np.unique(y),compute_precision,True, subsets = subsets_CV2, ensembling='voting', threshold_voting = 0.5)
    result_CV2 = model_CV2.fit_predict(X = norm_CV2, y= y)
    
    #Compute scores
    score = [model_ML.score_, compute_sensitivity(y, result_ML), model_CV2.score_, compute_sensitivity(y, result_CV2)] 
    scores.append(score)
    
scores = pd.DataFrame(scores, columns = names_scores, index = names)
scores.to_csv('prediction_values', index = True)