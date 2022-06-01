#Import library
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV

from load_data import open_charac
from binaryclass_memory import *
import sys

#Set the seed for reproducibility
np.random.seed(1)
random.seed(1)

fused = load_charac()

#Get the N top features according to mutual information
params = {'nb_features' : 18}
X, y = fused.drop(columns=['memory_gene']), fused['memory_gene']
FS = feature_selection(X,y,params)
    
X = fused[FS]
y = fused['memory_gene']     

#Grid search around best found parameters during random grid search
model = SVC(kernel='rbf', class_weight='balanced', random_state=1)
grid = {'C': np.logspace(-5, 5, 11),
        'gamma': ['scale', 'auto']}

cv = KFold(n_splits=5, shuffle=True, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='balanced_accuracy')

#Grid search
grid_result = grid_search.fit(X, y)

#Get cv accuracy
acc = grid_result.best_score_
param = grid_result.best_params_

#Print result, best param and score
print("The best accuracy (", acc, ") is obtained with param= ", param)
