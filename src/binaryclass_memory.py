import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr
import math 
from typing import AnyStr, Callable
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

from load_data import open_charac, add_general_charac, normalize, remove_extreme_values
from pred_score import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
import optuna

#------------------------------------------------------------------------------------------------------------------------------------------------------------
def visualize_charac(data:pd.DataFrame):
    copy = data
    #Remove the outliers of the data before normalization of mean expression
    data, outliers = remove_extreme_values(data, k=200)
    mem_genes_perc_outliers = 100 *np.sum(copy.loc[outliers]['memory_gene'])/len(outliers)
    print(mem_genes_perc_outliers, '% of the outliers are memory genes')
    
    #Shift the skeness values by substracting them with their mean
    corrected_skewRes = normalize(data['CV2ofmeans_residuals'])
    corrected_skewRes -= np.mean(corrected_skewRes)
    
    #Look at skenness and mean expression of all genes
    colors = ['grey','red']
    plt.scatter(corrected_skewRes, normalize(data['mean_expression']), marker='o', c= data['memory_gene'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel("skew")
    plt.ylabel("mean expression")
    plt.xlim(-40,60)
    plt.yscale('log')
    plt.title("All genes")
    plt.show()

    
    '''#Only non memory genes
    non_mem = list(data.index[np.where(data['memory_gene'] == False)[0]])
    data_non_mem = data.loc[non_mem]
    plt.scatter(data_non_mem['skew'], data_non_mem['mean_expression'])
    plt.xlabel("skew")
    plt.ylabel("mean expression")
    plt.ylim(0,np.max(data['mean_expression']))
    plt.title("Only non-memory genes")
    plt.show()

    #Only memory genes
    mem = list(data.index[np.where(data['memory_gene'] == True)[0]])
    data_mem = data.loc[mem]
    plt.scatter(data_mem['skew'], data_mem['mean_expression'])
    plt.xlabel("skew")
    plt.ylabel("mean expression")
    plt.yscale('log')
    plt.title("Only memory genes")
    plt.show()'''
    
def fit_logistic_reg(X:np.array, y:np.array, penalty:str, lamb:float, solver:str='lbfgs'):
    clf = LogisticRegression(penalty = penalty, C = lamb, class_weight = 'balanced', solver =solver, max_iter=1000).fit(X,y)
    scores = cross_val_score(clf, X, y, cv=5)
    
    return clf, scores.mean()

def fit_svm(X:np.array, y:np.array, lamb:float= 1, kernel:str = 'linear', degree:int = 3, gamma:int='scale'):
    clf = SVC(C=lamb, kernel = kernel, degree = degree, gamma = gamma, class_weight = 'balanced').fit(X,y)
    scores = cross_val_score(clf, X, y, cv=5)
    
    return clf, scores.mean()

def fit_evaluate(data_charac:pd.DataFrame, norm:pd.DataFrame, family:np.array, fit_func:str, feat:list, penalty:str =None, lamb:float=1, solver:str='lbfgs', kernel:str='rbf', degree:int=3, gamma:int='scale', verbose:bool=True):
    data_charac = data_charac.dropna(subset=['skew_residuals', 'mean_expression'])
    X = np.array(data_charac.iloc[: , feat])
    Y = np.array(data_charac['memory_gene'])
    
    #Fit classifier to charac data
    if fit_func == 'logreg':
        clf, score = fit_logistic_reg(X, Y, penalty, lamb, solver)
    if fit_func == 'svm':
        clf, score = fit_svm(X, Y, lamb, kernel, degree, gamma)

    #Evaluate fitted classifier
    y = clf.predict(X)
    non_memory_gene = list(data_charac[data_charac['memory_gene'] == False].index)
    memory_gene = list(data_charac[data_charac['memory_gene'] == True].index)
    y = pd.DataFrame(y, index = data_charac.index, columns = ['pred'])
    y['true_label'] = Y

    y_non_mem = y.loc[non_memory_gene]
    y_mem = y.loc[memory_gene]
    recovery = np.sum(y_mem['pred'])/len(memory_gene)
    false_pos = np.sum(y_non_mem['pred'])
    
    #Evaluate extracted subset on RNAseq Data
    gene_subset = list(y[y['pred']==True].index)
    
    if gene_subset:
        norm_subset = np.array(norm.loc[gene_subset].T)

        model = FamiliesClusters(np.unique(family),compute_precision,True)
        pred = model.fit_predict(norm_subset,family)
        precision, recovery_clust = model.score_, model.recovery
    else:
        precision, recovery_clust = np.NaN, np.NaN
    
    scores = [score, recovery, false_pos, precision, recovery_clust]
    
    if verbose:
        print('accuracy: ',score)
        print('recovery memory genes: ', recovery)
        print('false postive: ', false_pos)
        print('Precision and recovery clustering: ', precision, recovery_clust)
        
    return scores

def fit_evaluate_all(data_charac:pd.DataFrame, fit_func:str, lamb:float=1, kernel:str='rbf', verbose:bool=True):
    X = np.array(data_charac.drop(columns=['memory_gene']))
    Y = np.array(data_charac['memory_gene'])
    
    #Fit classifier to charac data
    if fit_func == 'logreg':
        clf, score = fit_logistic_reg(X, Y, 'l2', lamb)
    if fit_func == 'svm':
        clf, score = fit_svm(X, Y, lamb, kernel)

    #Evaluate fitted classifier
    y = clf.predict(X)
    non_memory_gene = list(data_charac[Y == False].index)
    memory_gene = list(data_charac[Y == True].index)
    y = pd.DataFrame(y, index = data_charac.index, columns = ['pred'])
    y['true_label'] = Y

    y_non_mem = y.loc[non_memory_gene]
    y_mem = y.loc[memory_gene]
    recovery = np.sum(y_mem['pred'])/len(memory_gene)
    false_pos = np.sum(y_non_mem['pred'])
      
    scores = [score, recovery, false_pos]
    
    if verbose:
        print('accuracy: ',score)
        print('recovery memory genes: ', recovery)
        print('false postive: ', false_pos)
        
    return clf, scores

def predict_evaluate(data_charac:pd.DataFrame, norm:pd.DataFrame, family:np.array, clf, mult_pred:bool = False, outliers:list = []):
    #Evaluate extracted subset on RNAseq Data
    X = np.array(data_charac.drop(columns=['memory_gene']))
    Y = np.array(data_charac['memory_gene'])
    
    #Get gene set
    y = clf.predict(X)
    y = pd.DataFrame(y, index = data_charac.index, columns = ['pred'])
    gene_subset = list(y[y['pred']==True].index)
    if outliers:
        gene_subset.extend(outliers)
    
    precision, recovery_clust, mult_precision, mult_recovery_clust = np.NaN, np.NaN, np.NaN, np.NaN
    if gene_subset:
        norm_subset = np.array(norm.loc[gene_subset].T)

        model = FamiliesClusters(np.unique(family),compute_precision,True)
        pred = model.fit_predict(norm_subset,family)
        precision, recovery_clust = model.score_, model.recovery
    
    scores = [precision, recovery_clust]
    if (mult_pred and gene_subset):
        subset = np.ones((len(gene_subset),))
        subsets = subsampling_genes(subset, 101, 0.25)
        
        model = EnsemblingHierarchical(np.unique(family),compute_precision,True,subsets = subsets, ensembling='voting', threshold_voting = 0.5)
        result  = model.fit_predict(norm_subset, family)
        mult_precision, mult_recovery_clust = model.score_, model.recovery
        
        scores.extend([mult_precision,mult_recovery_clust])
    
    return scores

#--------------------------------------------------------------------------------
#A few functions for Neural network training
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, charac, labels):
        'Initialization'
        self.labels = labels
        self.charac = charac

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.charac)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.charac[index]
        y = self.labels[index]

        return X, y