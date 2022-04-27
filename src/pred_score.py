import numpy as np
import pandas as pd
import sklearn
import math 
from typing import AnyStr, Callable, Tuple
from sklearn.base import ClusterMixin,BaseEstimator
from scipy.cluster.hierarchy import ward, cut_tree
from scipy_cut_tree_balanced import cut_tree_balanced

from evaluation_measure import *
from functional import *

#---------------------------------------------------------------------------------------------------------------
#New sklearn class for prediction of families
class FamiliesClusters(ClusterMixin, BaseEstimator):
    '''
    Families clustering
    Hierachical clustering with the ward2 criterion, use the spearman correlation as the distance measure.
    Parameters
    ----------
    family_interest: np.array,
        list of family of interest, if unsupervised provided
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
        
    def fit(self, X:np.array, y:np.array=None, NmaxCluster:int = None):
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
            Nmax = round(np.mean(np.unique(y,return_counts=True)[1]))
        else:
            Nmax = NmaxCluster
        
        clustering = np.squeeze(cut_tree_balanced(Z, max_cluster_size = Nmax)[0])
        
        #Assign all cells predicted alone in a cluster the label 0
        clustering += 1
        values, counts = np.unique(clustering, return_counts=True)
        onecell_family = values[np.where(counts==1)]
        for fam in onecell_family:
            clustering[clustering == fam] = 0

        self.recovery = compute_recovery(clustering)
    
        #Score the cluster and determine the number of clusters
        if self.Scoring_ != compute_cophe_coeff:
            score = self.Scoring_(y,clustering)
        else:
            score = self.Scoring_(corr_expr,Z)
            
        N = len(np.unique(clustering))
   
        self.n_clusters_, self.labels_, self.score_ = N, clustering, score
        return self
    
    def fit_predict(self, X:np.array, y:np.array=None,NmaxCluster:int = None):
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

#-----------------------------------------------------------------------------------------
#Iterative custering 

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
    corr_expr_raw = X_pd.corr(method= 'pearson') #Which correlation mesure used ???
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
        

    return cell_clusters,co_clustering,cell_clusters_correlation 

def process_result_iterative(result:Tuple[np.array,...], y:np.array):
    ''' Preprocessed the result from the iterative clustering before analysis of results together.
        
        parameters:
        -------
        result: Tuple[np.array,...],
            result from the iterative clustering (cell_clusters,co_clustering,cell_clusters_correlation)
        y : np.array,
            family of each data points 
    

        Returns
        -------
        '''
    cell_clusters = result[0]
    
    cell_clusters_unique_name = cell_clusters
    cell_clusters_unique_name =  cell_clusters_unique_name.astype(str)
    for i in range (0, np.shape(cell_clusters)[1]):
        for j, value in enumerate(cell_clusters_unique_name [:,i]):
            if not(value == 'nan'):
                cell_clusters_unique_name [j,i] =  (str(i + 1) + '_' + value)[0:-2]

    values, counts = np.unique(cell_clusters_unique_name, return_counts=True)
    clustersize_dict = ([values[0:-1], counts[0:-1]])
  
    smallest_clusters = (clustersize_dict[0])[clustersize_dict[1] == 2]
    smallest_clusters = np.append(smallest_clusters,(clustersize_dict[0])[clustersize_dict[1] == 3])
    
    best_prediction = outer_equal(y)
    best_prediction[best_prediction == True] = False
    for cluster in smallest_clusters:
        cell_in_cluster = np.nonzero(np.sum(cell_clusters_unique_name==cluster,axis=1))[0]
        for i in cell_in_cluster:
            for j in cell_in_cluster:
                best_prediction[i,j] = True
                
    np.fill_diagonal(best_prediction, False)

    return best_prediction

def test_prediction_multiple_overlap_3(result1:Tuple[np.array,...], result2:Tuple[np.array,...], result3:Tuple[np.array,...], y:np.array):
    ''' Take the prediction from three iterative clustering and blabla
        
        parameters:
        -------
        result1,2,3: Tuple[np.array,...],
            result from the iterative clustering (cell_clusters,co_clustering,cell_clusters_correlation)
        y : np.array,
            family of each data points 
    

        Returns
        -------
        '''

    best_prediction1 = process_result_iterative(result1, y)
    best_prediction2 = process_result_iterative(result2, y)
    best_prediction3 = process_result_iterative(result3, y)
    real_family_matrix = outer_equal(y)
    np.fill_diagonal(real_family_matrix, False)

  
    best_prediction = outer_equal(y)
    best_prediction[best_prediction == True] = False
    
    for cell in range(0, np.shape(best_prediction)[1]):
        for other_cell in range (0, np.shape(best_prediction)[1]):
            if ((best_prediction1[cell,other_cell]==True and best_prediction2[cell,other_cell]==True and best_prediction3[cell,other_cell]==True)): 
                best_prediction[cell,other_cell] = True
    np.fill_diagonal(best_prediction, False)
                    
  
    return [np.sum(np.logical_and(best_prediction, real_family_matrix)), np.sum(np.logical_and(best_prediction, 1 - real_family_matrix)), int(np.sum(real_family_matrix))]

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
    recovery = model.recovery

    return score, recovery

#---------------------------------------------------------------------------------------------------------------
#Repetitive clustering

class EnsemblingHierarchical(ClusterMixin, BaseEstimator):
    '''
    Hierachical clustering with the ward2 criterion, use the spearmann's correlation as the distance measure, on N subsets of genes. 
    Then, use ensembling method to give final cluster assignments.
    Parameters
    ----------
    family_interest: np.array,
        list of family of interest
    Scoring : Callable,
        scoring function use to evaluate the model
    maximize: bool,
        if True the scoring function is maximize, else it is minimize
    subsets: list,
        list of the different subsets of genes
    ensembling_: str,
        ensembling method to produce final clustering
        
    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.
    family_interest: np.array,
        list of family of interest.
    Scoring_ : Callable,
        scoring function use to evaluate the model.
    maximize: bool,
        if True the scoring function is maximize, else it is minimize.
    subsets_: list,
        list of the different subsets of genes
    ensembling_: str,
        ensembling method to produce final clustering
    threshold: float[0,1],
        two cells need to be predicted in at least <thereshold> of the individual clusterings. Required when ensembling function is 'voting'.
    
    '''
    
    def __init__(self, family_interest_:np.array, Scoring_:Callable, maximize_:bool, subsets: list, ensembling: str, threshold_voting:float=None):
        super().__init__()
        self.family_interest_ = family_interest_
        self.Scoring_ = Scoring_
        self.maximize_ = maximize_
        self.subsets_ = subsets
        self.ensembling_ = ensembling
        self.thereshold_voting = threshold_voting
        
    def fit(self, X:np.array, y:np.array, NmaxCluster:int=None):
        '''Fit data using hierachical clustering with the ward2 criterion and use the spearman correlation as the distance measure and predict 
        on provided subsets.
        
        parameters:
        -------
        x : np.array,
            features of each data points
        y : np.array,
            family of each data points
        NmaxCluster : int,
            max number of cells in a cluster

        return
        -------
        self,
            return fitted self'''
        
        clustering = []
        #Cluster data using the different subsets of features
        for subset in self.subsets_:
            model = FamiliesClusters(self.family_interest_, self.Scoring_, self.maximize_)
            pred = model.fit_predict(X[:,subset],y)
            if (self.ensembling_ != 'voting'):
                pred = [np.nan if x == 0 else x for x in pred]
            clustering.append(pred)
            
        #Get the final clustering from the individual clustering result
        if self.ensembling_ == 'voting':
            final_ensembling = ensembling_voting(clustering, self.thereshold_voting)
        else:
            final_ensembling = CE.cluster_ensembles(np.array(clustering), solver = self.ensembling_)
        
        #Score the cluster and determine the number of clusters
        score = self.Scoring_(y,final_ensembling)
        N = len(np.unique(final_ensembling))
   
        self.n_clusters_, self.labels_, self.score_ = N, final_ensembling, score
        self.recovery = compute_recovery(final_ensembling)
    
        return self
    
    def fit_predict(self, X:np.array, y:np.array,NmaxCluster:int = None):
        self.fit(X,y,NmaxCluster)
        
        return self.labels_ 
    
    def score(self, X, y_true):
        return self.score_
    

def outer_equal_ignore_zero(x:np.array):
    out = np.zeros((len(x),len(x)))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            if x[i] != 0 or x[j] != 0:
                 out[i,j] = x[i] == x[j]
    
    return out


def ensembling_voting(clustering:list, threshold:float):
    """ Compute the final clustering from given individual clustering. 
    Two cells that are the most of the clusterings predict together are together in the final clustering. 
  
      parameters:
      clustering: list,
        list of the independent clusterings from which the final clustering is computed
      threshold: float[0,1],
        two cells need to be predicted in at least <thereshold> of the individual clusterings

      return:
      final_label: np.array,
        final computed 
    """
    #Compute co_occurrence matrix of the clustering
    co_occurrence = np.zeros((len(clustering[0]), len(clustering[0])))
    for cluster in clustering:
        co_occurrence += outer_equal_ignore_zero(cluster)

    
    #Compute final clustering with majority voting
    N_vote = math.ceil(len(clustering)*threshold) #Vote necessary to consider two cells same family
    same_family = co_occurrence >= N_vote 
    for i in range(0,len(clustering[0])):
        for j in range(0,i+1):
            same_family[i,j] = False
    #Get indexes of cell together
    ind_together = np.nonzero(same_family)
    final_label = np.zeros((len(clustering[0],)))
    
    for i in range(0,len(ind_together[0])):
        first_cell, second_cell = ind_together[0][i], ind_together[1][i]
        if final_label[first_cell] == 0 or  final_label[second_cell] == 0:
            if final_label[first_cell] == 0 and  final_label[second_cell] == 0:
                final_label[first_cell], final_label[second_cell] = np.max(final_label) + 1, np.max(final_label) + 1
            else:
                final_label[first_cell], final_label[second_cell] = np.max([final_label[first_cell], final_label[second_cell]]), np.max([final_label[first_cell], final_label[second_cell]])
        else:
            final_label[final_label == final_label[second_cell]] = final_label[first_cell]
    
    return final_label.astype(int)

def subsampling_genes(subset:np.array, N:int, p_mutate:float):
    subsets = []
    
    for i in range(0,N):
        subsets.append((mutate(subset,p_mutate)))
    
    for i in range(0,len(subsets)):
        subsets[i] = subsets[i].astype(bool)
        
    return subsets

def getTrainTestAll(y:np.array, x:np.array, ind_dataset:list, i:int):
    """Split the merged data, one data set is keept for testing, the others for training.
  
      parameters:
      y : np.array,
        family of each data points
      x : np.array,
        features of each data points
      ind_dataset : list,
        list of indices where each data set is stored
      i: int, 
        ind of the data set to keep for testing
        

      returns:
      x_train : np.array,
        norm data without the test dataset for training
      y_train : np.array,
        families of each data point without the test dataset for training
      x_test : np.array,
        norm data of the test dataset
      y_test : np.array,
        families of each data point of the test dataset"""
    
    ind_i = None
    if i == 0:
        ind_i = np.arange(0,ind_dataset[i],1)
    else:
        ind_i = np.arange(ind_dataset[i-1], ind_dataset[i], 1)
    
    
    x_train = np.delete(x, ind_i, axis=0)
    y_train = np.delete(y, ind_i)
    x_test = x[ind_i,:]
    y_test = y[ind_i]
    
    return x_train, y_train, x_test, y_test


def optimization_on_allsets(y:np.array, x:np.array, ind_dataset:list, Model_test: Callable, Scoring_test: Callable, maximize_test:bool, func: Callable, **kwargs: dict):
    """ 
  
      parameters:
      y : np.array,
        family of each data points
      x : np.array,
        features of each data points
      ind_dataset : list,
        list of indices where each data set is stored
      Model_test : Callable,
        the model is fitted using this method
      Scoring_test: Callable,
        scoring function use to evaluate the model
      maximize_test: bool,
        if True the scoring function is maximize, else it is minimize
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
    
    for i in range(0,len(ind_dataset)):
        #Get split data
        x_train, y_train, x_test, y_test = getTrainTestAll(y, x, ind_dataset, i)
        print(y.shape, y_train.shape, y_test.shape)
        print(x.shape, x_train.shape, x_test.shape)
        
        #Run feature selection on training set
        subset, score = func(y_train, x_train, **kwargs)
        
        #Evaluate subset on test set
        model_test = Model_test(np.unique(y_test),Scoring_test,True)
        pred_test = model_test.fit_predict(x_test[:, subset],y_test)
        test_score = model_test.score(x_test[:, subset],y_test)
        
        #Store best score and best subset on current folds
        score_training.append(score)
        score_testing.append(test_score)
        final_subset.append(subset)
        
    return final_subset, score_training, score_testing