#Import library
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd  
from pred_score import *
from Filter_FS import MIM,ANOVA
import sys

#Get input for script
#input_path = sys.argv[0]
#if not os.path.exists(input_path):
    #print(input_path, ' File not found.')
    #sys.exit 1
    
AE3 = np.array(pd.read_csv ('../data/processed_data/AE3.csv'))
y = np.array(AE3[:,-1],dtype=int)
AE3 = AE3[:,0:-1]

#Set the seed for reproducibility
np.random.seed(1)
random.seed(1)

#Predict and evaluate
model = FamiliesClusters(np.unique(y),compute_precision,True)
pred = model.fit_predict(AE3,y)
print(model.score_, model.recovery)

subset = np.ones((AE3.shape[1],))
subsets = subsampling_genes(subset, 101, 0.25)
model = EnsemblingHierarchical(np.unique(y),compute_precision,True,subsets = subsets, ensembling='voting', threshold_voting = 0.5)
result  = model.fit_predict(X = AE3, y = y)
print(model.score_, model.recovery)