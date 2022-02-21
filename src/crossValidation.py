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
#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Feature selction methods with CV 