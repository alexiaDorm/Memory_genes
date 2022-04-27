from scipy.cluster.hierarchy import cophenet
import numpy as np
from functional import *

def computeTP(pred: np.array, family:np.array):
    """ Compute the TP, FP, TN, and FN of a given clustering.
  
      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells

      returns:
      TP: int,
        true positive, 2 cells in the same cluster also have same barcode
      FP: int,
        false positive, 2 cells in the same cluster have different barcode
      TN: int,
        true negative,  2 cells in different clusters also have different barcodes
      FN: int,
        false negative = 2 cells in different clusters have the same barcode
        """
    
    TP,FP,TN,FN = 0,0,0,0 
    for i in range (min(pred)+1, max(pred)+1):
        #Get which cells are in the ith predicted cluster
        cluster = np.squeeze(np.argwhere(pred==i))
        
        #Get the true family of the cells of the cluster
        true = family[cluster]
        #If only one cell in cluster put in an array
        if type(true) == np.int32 or type(true) == np.int64:
            true = np.array([true])
        
        
        #Compute TP and FP
        # Compare the true family of each cells in the observed cluster, only keep the upper without diagonal
        #-> comparaisons not repeated or with themself
        mask = upper_tri_indexing(np.squeeze(true[:,None]  == true))
        TP += np.sum(mask)
        FP += len(mask) - np.sum(mask)
        
        for j in range (i+1,max(pred)+1):
            #Get which cells are in the compered predicted cluster
            diff_cluster = np.argwhere(pred==j)
            
            #Get the true family of the cells of the cluster
            diff_true = family[diff_cluster]
           
            
            if type(diff_true) == np.int32:
                diff_true = np.array([diff_true])
            
            #Compute TN, FN
            # Compare the true family of each cells in the first and second clusters
            mask = np.squeeze(true[:,None]  == diff_true)
            FN += np.sum(mask)
            TN += len(np.ravel(mask)) - np.sum(mask)
    return TP,FP,TN,FN

def compute_statTP(family:np.array,pred: np.array):
    """ Compute the TP,FP and ratio, sensitivity, specificity, precision, NPV, FDR, FNR.

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
        
      returns:
      sensitivity: float,
          computed sensitivity
      specificity: float,
          computed specificity
      precision: float,
          computed precision
      NPV: float,
          computed NPV
      FDR: float,
          computed FDR
      FNR: float,
          computed FNR
          
        """
    #Compute TP,FP,TN,FN
    TP,FP,TN,FN = computeTP(pred, family)
    ratio = TP/FP
    sensitivity, specificity, precision,NPV,FDR,FNR = 0,0,0,0,0,0
    
    sensitivity = TP/(TP+FN) #maybe most important
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    if max(pred) != 0:
        NPV = TN/(TN+FN)
    FDR = FP/(FP+TP)
    FNR = FN/(FN+TP)
    
    
    return TP, FP, ratio, sensitivity, specificity, precision, NPV, FDR, FNR

def compute_ratio(family:np.array, pred: np.array):
    """ Compute the TP/FP ratio.

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
     """
    #Compute TP,FP,TN,FN
    TP,FP,TN,FN = computeTP(pred, family)
    
    return TP/FP

def compute_RI(family:np.array, pred: np.array):
    """ Compute the rand index.

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
     """
    #Compute TP,FP,TN,FN
    TP,FP,TN,FN = computeTP(pred, family)
    RI = (TP+TN)/(TP+FP+TN+FN)
    
    return RI

def compute_precision(family:np.array, pred: np.array):
    """ Compute the sensitivity.

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
     """
    #Compute TP,FP,TN,FN
    TP,FP,TN,FN = computeTP(pred, family)
    
    pre = 0
    if TP + FP != 0:
        pre = TP/(TP+FP)
    
    return pre

def compute_sensitivity(family:np.array, pred: np.array):
    """ Compute the sensitivity.

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
     """
    #Compute TP,FP,TN,FN
    TP,FP,TN,FN = computeTP(pred, family)
    
    return TP/(TP+FN)

def compute_accuracy(family:np.array, pred: np.array):
    """ Compute the accuracy.

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
     """
    #Compute TP,FP,TN,FN
    TP,FP,TN,FN = computeTP(pred, family)
    
    return (TP+TN)/(TP+TN+FP+FN)

def compute_frac_only_same_family(family:np.array,pred: np.array):
    """ Fraction of clusters that contain only cells with the same barcode e.g. all clusters feature only a single cell so by 
    definition all cells in each cluster have the same barcode -> 100%; there is only one cluster with all barcodes -> 0%; 

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
        
     return:
     frac: float,
       fraction of clusters that contain only cells with the same barcode (i.e same family)
     """
    frac = 0
    for i in np.unique(pred):
        #Get which cells are in the predicted cluster
        cluster = np.argwhere(pred==i)
        
        #Get the true family of the cells of the cluster
        true = family[cluster]
        
        #Test if all cells in same cluster have sames barcode 
        frac += (np.max(true) == np.min(true))
    
    frac = frac/(max(pred)+1)
    
    return frac

def compute_frac_each_family_same_cluster(family:np.array, pred: np.array):
    """ Compute mean of the fractions of cells for each barcode that are in the same cluster e.g. 4 cells with barcode A are distributed 
    over 4 different clusters-> 25%; 4 cells with barcode A are equally distributed over 2 different clusters (2, 2) -> 50%; 4 cells with 
    barcode A are unequally distributed over 2 different clusters (1, 3) -> 75%

      parameters:
      pred: np.array,
        predicted families(cluster)
      family: np.array,
        true families of the cells
        
     return:
     fract_same_family: float,
       
     """
    frac = []
    for i in np.unique(family):
        #Get which cells are in the ith family
        cluster = np.argwhere(family==i)
        
        #Get the prediction of the cells of the given cluster
        prediction = pred[cluster]
        
        #Compute distribution of cells in clusters
        uni, distribution = np.unique(prediction,return_counts=True)
        
        frac.append(np.max(distribution)/len(prediction))
    
    return np.mean(frac)

def compute_recovery(final_label:np.array = None):
    '''Compute the recovery of the clustring
        parameters:
        -------
        r
    
        Returns
        -------
        recovery: float [0,1]',
            computed recovery, percentage of cell with family assignment
        '''
    return len(np.nonzero(final_label)[0])/len(final_label)


def compute_cophe_coeff(orign_dists:np.array, Z:np.array):
    '''Compute the Cophenetic coefficient from the original distance matrix and generated dendrogram of clustering
        parameters:
        -------
        orign_dists : np.array,
            original distance matrix before clustering
        Z : np.array,
            dendrogram of the clustering to evaluate
    
        Returns
        -------
        corr_coeff:float [0,1],
            computed Cophenetic coefficient '''
    cophe_dists = cophenet(Z) 
    corr_coef = np.corrcoef(orign_dists, cophe_dists)[0,1]
    
    return corr_coef