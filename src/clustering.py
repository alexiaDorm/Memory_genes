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
        

    return cell_clusters,co_clustering,cell_clusters_correlation 