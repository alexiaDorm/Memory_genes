#Import library
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd  
from pred_score import *
from Filter_FS import MIM,ANOVA
import sys

#Set the seed for reproducibility
np.random.seed(1)
random.seed(1)

#Get input for script
input_norm_path = sys.argv[0]
input_y_path = sy.argv[1]
if not (os.path.exists(input_norm_path) and os.path.exists(input_y_path)):
    print(input_path, ' File not found.')
    sys.exit 1
    
#Load data 
norm = np.array(pd.read_csv (input_norm_path).T)
y = np.squeeze(np.array(pd.read_csv(intput_y_path)))

#Run optimization on mixed datasets
#Define parameters for MIM method
print('MIM optimization')
N = np.arange(200,3000,25)
kwargs = {'Model': FamiliesClusters, 'Scoring': compute_precision,'maximize': True,'N': N, 'n_neighbors': 3, 'plot': True} 

subset, score_training, score_testing = cross_validation(y,norm, FamiliesClusters, compute_precision,True, 5,  MIM, **kwargs)

mean_score_test, std_score_test = np.mean(score_testing), np.std(score_testing)    
print('test', mean_score_test, std_score_test)


#Predict and evaluate
norm_subset = norm[:,subset]
print('Predicting once')
model = FamiliesClusters(np.unique(y),compute_precision,True)
pred = model.fit_predict(norm_subset,y)
print(model.score_, model.recovery)

print('Ensembling clustering')
subset = np.ones((norm_subset.shape[1],))
subsets = subsampling_genes(subset, 101, 0.25)
model = EnsemblingHierarchical(np.unique(y),compute_precision,True,subsets = subsets, ensembling='voting', threshold_voting = 0.5)
result  = model.fit_predict(X = norm_subset, y = y)
print(model.score_, model.recovery)


print('ANOVA optimization')
N = np.arange(200,3000,50)
kwargs = {'Model': FamiliesClusters, 'Scoring': compute_precision,'maximize': True,'N': N, 'n_neighbors': 3, 'plot': True} 

subset, score_training, score_testing = cross_validation(y,norm, FamiliesClusters, compute_precision,True, 5,  MIM, **kwargs)

mean_score_test, std_score_test = np.mean(score_testing), np.std(score_testing)    
print('test', mean_score_test, std_score_test)


#Predict and evaluate
norm_subset = norm[:,subset]
print('Predicting once')
model = FamiliesClusters(np.unique(y),compute_precision,True)
pred = model.fit_predict(norm_subset,y)
print(model.score_, model.recovery)

print('Ensembling clustering')
subset = np.ones((norm_subset.shape[1],))
subsets = subsampling_genes(subset, 101, 0.25)
model = EnsemblingHierarchical(np.unique(y),compute_precision,True,subsets = subsets, ensembling='voting', threshold_voting = 0.5)
result  = model.fit_predict(X = norm_subset, y = y)
print(model.score_, model.recovery)