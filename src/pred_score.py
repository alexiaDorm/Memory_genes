import numpy as np
import pandas as pd
import sklearn
import math 
from typing import AnyStr, Callable
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.base import ClusterMixin,BaseEstimator
from scipy.cluster.hierarchy import ward, cut_tree
from sklearn.metrics import make_scorer
from scipy_cut_tree_balanced import cut_tree_balanced

#-----------------------------------------------------------------------------
#Few functions for scoring

def upper_tri_indexing(A:np.array):
    """ Return the upper triangle without diagonal.
  
      parameters:
      A: np.array,
        matrix to transform

      returns:
      up: np.array,
        values of the upper triangle without the diagonal
    """
    #If only one cell in a predicted cluster
    if A.shape == ():
        return []
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]
def lower_tri_indexing(A:np.array):
    """ Return the lower triangle without diagonal.
  
      parameters:
      A: np.array,
        matrix to transform

      returns:
      up: np.array,
        values of the lower triangle without the diagonal
    """
    #If only one cell in a predicted cluster
    if A.shape == ():
        return []
    m = A.shape[0]
    r,c = np.tril_indices(m,-1)
    return A[r,c]
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
    for i in range (min(pred), max(pred)+1):
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
    
    return TP/(TP+FP)

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

#---------------------------------------------------------------------------------------------------------------
#New sklearn class for prediction of families
class FamiliesClusters(ClusterMixin, BaseEstimator):
    '''
    Families clustering
    Hierachical clustering with the ward2 criterion, use the spearman correlation as the distance measure.
    Parameters
    ----------
    family_interest: np.array,
        list of family of interest
    Scoring : Callable,
        scoring function use to evaluate the model
    maximize: bool,
        if True the scoring function is maximize, else it is minimize
        
    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.
    family_interest: np.array,
        list of family of interest.
    Scoring : Callable,
        scoring function use to evaluate the model.
    maximize: bool,
        if True the scoring function is maximize, else it is minimize.
    
    '''
    
    def __init__(self, family_interest_:np.array, Scoring_:Callable, maximize_:bool):
        super().__init__()
        self.family_interest_ = family_interest_
        self.Scoring_ = Scoring_
        self.maximize_ = maximize_
        
    def fit(self, X:np.array, y:np.array, NmaxCluster:int = None):
        '''Fit data using hierachical clustering with the ward2 criterion and use the spearman correlation as the distance measure and predict.
        
        parameters:
        -------
        x : np.array,
            features of each data points
        y : np.array,
            family of each data points
        NmaxCluster : int,
            max number of cells in a cluster
        

        Returns
        -------
        self,
            return fitted self'''
        
        #Compute the spearman correlation of X
        X_pd = pd.DataFrame(X.T)
        corr_expr= X_pd.corr(method= 'spearman')
        corr_expr = np.array((1 - corr_expr)/2)
        corr_expr = upper_tri_indexing(corr_expr)**2 #Squared for ward2 criterion
        
        if(np.shape(X.T)[0] == 1):
            corr_expr.fill(1)
        
        #Create clustering tree using hierarchical clustering with spearmann correlation and ward2 criterion
        Z = ward(corr_expr)
        
        #Cut the tree into clusters of maximum size equal to the number of cells in the largest family in data set
        if NmaxCluster == None:
            Nmax = np.max(np.unique(y,return_counts=True)[1])
        else:
            Nmax = NmaxCluster
        clustering = np.squeeze(cut_tree_balanced(Z, max_cluster_size = Nmax)[0])
        
        #Score the cluster and determine the number of clusters
        score = self.Scoring_(y,clustering)
        N = len(np.unique(clustering))
   
        self.n_clusters_, self.labels_, self.score_ = N, clustering, score
        return self
    
    def fit_predict(self, X:np.array, y:np.array,NmaxCluster:int = None):
        self.fit(X,y,NmaxCluster)
        
        return self.labels_      
    
    def score(self, X, y_true):
        #Error come from here y_true and X not same size as self.labels_ -> function fit_as
        return self.score_
    
def outer_equal(x:np.array):
    out = np.zeros((len(x),len(x)))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            out[i,j] = x[i] == x[j]
    
    return out

def iterative_clustering(X:np.array, y:np.array, N:int =2, iterations:int =20):
    '''Fit data iteratively using hierachical clustering with the ward2 criterion and use the spearman correlation as the distance measure and predict.
        
        parameters:
        -------
        x : np.array,
            features of each data points
        y : np.array,
            family of each data points
        N : int,
            max number of 
        iterations : int,
            number of iterative clustering
    

        Returns
        -------
        cell_cluster: np.array,
            output of the clustering algorithm at each iteration
        co_clustering: np.array,
            matrix containing the number of times each cell were clustered with the other cells
        cell_clustering_correlation: np.array,
            blabla      '''
        #Compute the pearson's correlation of X
    X_pd = pd.DataFrame(X)
    corr_expr_raw = X_pd.corr(method= 'spearman') #Which correlation mesure used ???
    corr_expr_raw = np.array(corr_expr_raw)
    corr_expr = np.array((1 - corr_expr_raw)/2)
        
    if(np.shape(X.T)[0] == 1):
        corr_expr.fill(1)
            
    #Create empty matrix of size (#cells x iteration) 
    cell_clusters = np.zeros((len(y), iterations))
    
    #Create empty matrix of size (#cells x iteration)
    cell_clusters_correlation = np.empty((len(y), iterations))
    cell_clusters_correlation[:] = np.NaN

    #Put 1 in all first iteration column
    cell_clusters[:,0] = 1
    #Put the mean correlation in all first iteration column
    cell_clusters_correlation[:,0] = np.mean(upper_tri_indexing(corr_expr))
        
    for i in range(1,iterations) :
            
        #Loop over the clusters that are not zero
        id_cluster = np.unique(cell_clusters[:,(i-1)])
        non_zero_clusters = id_cluster[id_cluster!=0]
            
        for cluster in non_zero_clusters :
                
            #Get the name of cells in current cluster 
            cells_in_cluster = np.where(cell_clusters[:,(i-1)] == cluster)[0]

            if len(cells_in_cluster) >= 3:
                #Get only correlation for cells in given cluster
                corr = corr_expr_raw[cells_in_cluster,:]
                corr = corr[:,cells_in_cluster]
                correlation = np.mean(upper_tri_indexing(corr))
                cell_clusters_correlation[cells_in_cluster,(i-1)] = correlation
                corr_expr_subset = corr_expr[cells_in_cluster, :]
                corr_expr_subset = corr_expr_subset[:,cells_in_cluster]
                #Squared for ward2 criterion
                corr_expr_subset = upper_tri_indexing(corr_expr_subset)**2
                    
                #Cluster the cells in current cluster hierarchical clustering with pearson's correlation and ward2 criterion
                Z = ward(corr_expr_subset)
                #Cut the linkage matrix into N clusters
                clustering = np.squeeze(cut_tree(Z, n_clusters=N)) + 1
                
                cell_clusters[cells_in_cluster,i] = clustering + max(cell_clusters[:,i])
                    
            else:
                cell_clusters[cells_in_cluster,i] = 0
        
    #Create matrix with zeros of size (#cells, #cells)
    co_clustering = np.zeros((len(y), len(y)))
        
    #Remplace 0 values by NaN in cell_cluster
    cell_clusters[cell_clusters==0] = np.NaN
        
    for i in range (0,iterations):
        to_add = outer_equal(cell_clusters[:,i])
        to_add[np.isnan(to_add)] = 0
        co_clustering = co_clustering + to_add
        
    #Score the cluster and determine the number of clusters
    #score = self.Scoring_(y,clustering)
    #N = len(np.unique(clustering))
   
    #self.n_clusters_, self.labels_, self.score_ = N, clustering, score
    return cell_clusters,co_clustering,cell_clusters_correlation 

class IterativeClustering(ClusterMixin, BaseEstimator):
    '''
    Iterative Families clustering
    Hierachical clustering with the ward2 criterion, use the pearson's correlation as the distance measure.
    Parameters
    ----------
    family_interest: np.array,
        list of family of interest
    Scoring : Callable,
        scoring function use to evaluate the model
    maximize: bool,
        if True the scoring function is maximize, else it is minimize
        
    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.
    family_interest: np.array,
        list of family of interest.
    Scoring : Callable,
        scoring function use to evaluate the model.
    maximize: bool,
        if True the scoring function is maximize, else it is minimize.
    
    '''
    
    def __init__(self, family_interest_:np.array, Scoring_:Callable, maximize_:bool):
        super().__init__()
        self.family_interest_ = family_interest_
        self.Scoring_ = Scoring_
        self.maximize_ = maximize_
        
    def fit(self, X:np.array, y:np.array, N:int =2, iterations:int =20):
        '''Fit data using hierachical clustering with the ward2 criterion and use the spearman correlation as the distance measure and predict.
        
        parameters:
        -------
        x : np.array,
            features of each data points
        y : np.array,
            family of each data points
        N : int,
            max number of 
        iterations : int,
            number of iterative clustering
    

        Returns
        -------
        self,
            return fitted self'''
        
        #Iterative clustering algorithm
        cell_clusters,co_clustering,cell_clusters_correlation = iterative_clustering(X, y, N, iterations)
        
        #Score the cluster and determine the number of clusters
        #score = self.Scoring_(y,clustering)
        #N = len(np.unique(clustering))
   
        #self.n_clusters_, self.labels_, self.score_ = N, clustering, score
        return cell_clusters,co_clustering,cell_clusters_correlation   
    
    def fit_predict(self, X:np.array, y:np.array, N:int =2, iterations:int =20):
        self.fit(X,y,N,iterations)
        
        return self.labels_      
    
    def score(self, X, y_true):
        return self.score_

#---------------------------------------------------------------------------------------------------------------
#Function to evaluate subset of feature
def evaluate(y:np.array, x:np.array, Model:Callable, Scoring: Callable, maximize:bool):
    """ Fit the data to the model and evaluate using given scoring method
  
      parameters:
      y : np.array,
        family of each data points
      x : np.array,
        features of each data points
      Model : Callable,
        the model is fitted using this method
      Scoring: Callable,
        scoring function use to evaluate the model
      maximize: bool,
        if True the scoring function is maximize, else it is minimize

      returns:
      score : float,
        score computed
    """
    
    #Fit the model on the given data, evaluate the prediction
    model = Model(np.unique(y),Scoring,maximize)
    pred = model.fit_predict(x,y)
    score = model.score_

    return score