import numpy as np
import pandas as pd
import sklearn
import math 
from typing import AnyStr, Callable

from pred_score import *
from Filter_FS import *
from Wrapper_FS import *

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Few functions for cross-validation
def split_data_cv(x:np.array, y:np.array, kfold:int):
    """ Split the data for CV. Keep cells of same family together.
  
      parameters:
      y: np.array,
        family of each data points
      x: np.array,
        gene expression of each data points
      kfold: int,
        number of fold for CV
        

      returns:
      split_x : np.array of np.array,
        split normalized data, split_x[k] = kth split
      split_y : np.array of np.array,
        split families data   """
    
    #Get all families indices
    ind_families = np.unique(y)
    N_families = len(ind_families)
    N_fold_family = math.ceil(N_families/kfold)
    
    
    #Randomly choose the families in each split    
    np.random.shuffle(ind_families)
    
    fam_split = np.empty(kfold, dtype=object)
    for i in range (0,kfold):
        fam_split[i] = []
        
    for i in range (0, kfold):
        temp = ind_families[i*N_fold_family:((i+1)*N_fold_family)]
        fam_split[i].append(temp)
      
    #Get the data from splitted families: split_x, split_y
    split_x = np.empty(kfold, dtype=object)
    split_y = np.empty(kfold, dtype=object)
    for i in range (0,kfold):
        split_x[i] = []
        split_y[i] = []
        
    for i in range (0,kfold):
        ind = np.squeeze(fam_split[i])
     
        mask = np.isin(y, ind)
        temp_x = x[mask]
        temp_y = y[mask]
        
        split_x[i].append(temp_x)
        split_y[i].append(temp_y)
            
        
    return split_x, split_y

def getTrainTest(split_y, split_x, i:int, kfold:int):
    x_test = np.squeeze(split_x[i])
    y_test = np.squeeze(split_y[i])
    x_train_temp = np.squeeze(np.delete(split_x, i))
    y_train_temp = np.squeeze(np.delete(split_y, i))

    #Fuse the training fold
    for j in range(0,kfold-1):
        x_train_temp[j] = np.squeeze(np.array(x_train_temp[j]))
        y_train_temp[j] = np.squeeze(np.array(y_train_temp[j]))

    x_train = x_train_temp[0].T
    y_train = y_train_temp[0]
    for j in range(1,kfold-1):
        x_train = np.c_[x_train,x_train_temp[j].T]
        y_train = np.append(y_train,y_train_temp[j])

    x_train = x_train.T
    
    return x_train, y_train, x_test, y_test
#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Feature selction methods with CV 
def cross_validation(y:np.array, x:np.array, Model_test: Callable, Scoring_test: Callable, maximize_test:bool, kfold:int, func: Callable, **kwargs: dict):
    """ Cross validate any feature selection method.
  
      parameters:
      y : np.array,
        family of each data points
      x : np.array,
        features of each data points
      Model_test : Callable,
        the model is fitted using this method
      Scoring_test: Callable,
        scoring function use to evaluate the model
      maximize_test: bool,
        if True the scoring function is maximize, else it is minimize
      kfold: int,
        number of folds for CV
      func: Callable,
        feature selection function, should return seleted subset and associated score
      kwargs: **kwargs : dict,
        dictionnary of parameters and their values (kwargs = {'param_name' : value}) to pass to the given method (func)
        

      returns:
      final_subset : np.array,
        subset of features with the best score
      best_test_score : float,
        test score obtained with the best subset of features """
    
    #Store score training and best subset
    score_training = []
    score_testing = []
    final_subset = []
    
    #Split the data in kfold
    split_x, split_y = split_data_cv(x,y,kfold)
    
    for i in range(0,kfold):
        #Get split data
        x_train, y_train, x_test, y_test = getTrainTest(split_y, split_x, i,kfold)
        
        #Run feature selection on training set
        subset, score = func(y_train, x_train, **kwargs)
        
        #Evaluate subset on test set
        model_test = FamiliesClusters(np.unique(y_test),compute_ratio,True)
        pred_test = model_test.fit_predict(x_test[:, subset],y_test)
        test_score = model_test.score(x_test[:, subset],y_test)
        
        #Store best score on current folds
        score_training.append(score)
        score_testing.append(test_score)
        if (len(final_subset) == 0 or np.argmax(score_testing) == i): #if the last best test score is best overall keep subset as the finals subset
            final_subset = subset
        
    return final_subset, score_training, score_testing