import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import math 
from typing import AnyStr, Callable

from load_data import open_charac
from pred_score import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#------------------------------------------------------------------------------------------------------------------------------------------------------------
def visualize_charac(data:pd.DataFrame):
    #Look at all genes
    colors = ['grey','red']
    plt.scatter(data['skew'], data['mean_expression'], marker='o', c= data['memory_gene'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel("skew")
    plt.ylabel("mean expression")
    plt.yscale('log')
    plt.title("All genes")
    plt.show()
    
    #Only non memory genes
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
    plt.show()
    
def fit_logistic_reg(X:np.array, y:np.array, penalty:str, lamb:float, solver:str='lbfgs'):
    clf = LogisticRegression(penalty = penalty, C = lamb, class_weight = 'balanced', solver =solver, max_iter=1000).fit(X,y)
    scores = cross_val_score(clf, X, y, cv=5)
    
    return clf, scores.mean()

def fit_svm(X:np.array, y:np.array, kernel = 'linear'):
    clf = SVC(kernel = kernel, class_weight = 'balanced').fit(X,y)
    scores = cross_val_score(clf, X, y, cv=5)
    
    return clf, scores.mean()

def fit_evaluate(data_charac:pd.DataFrame, norm:pd.DataFrame, family:np.array, fit_func:str, feat:list, penalty:str =None, lamb:float=None, solver:str='lbfgs', kernel:str=None, verbose:bool=True):
    data_charac = data_charac.dropna(subset=['skew_residuals', 'mean_expression'])
    X = np.array(data_charac.iloc[: , feat])
    #X[:,-1] = np.log10(X[:,-1]) #log scale for mean expression
    Y = np.array(data_charac['memory_gene'])
    
    #Fit classifier to charac data
    if fit_func == 'logreg':
        clf, score = fit_logistic_reg(X, Y, penalty, lamb, solver)
    if fit_func == 'svm':
        clf, score = fit_svm(X, Y, kernel)

    #Evaluate fitted classifier
    y = clf.predict(X)
    non_memory_gene = list(data_charac[data_charac['memory_gene'] == False].index)
    memory_gene = list(data_charac[data_charac['memory_gene'] == True].index)
    y = pd.DataFrame(y, index = data_charac.index, columns = ['pred'])
    y['true_label'] = Y

    y_non_mem = y.loc[non_memory_gene]
    y_mem = y.loc[memory_gene]
    recovery = np.sum(y_mem['pred'])/np.sum(y_mem['true_label'])
    false_pos = np.sum(y_non_mem['pred'])
    
    #Evaluate extracted subset on RNAseq Data
    gene_subset = list(y[y['pred']==True].index)

    norm_subset = np.array(norm.loc[gene_subset].T)
    
    model = FamiliesClusters(np.unique(family),compute_precision,True)
    pred = model.fit_predict(norm_subset,family)
    
    scores = [score, recovery, false_pos, model.score_, model.recovery]
    
    if verbose:
        print('accuracy: ',score)
        print('recovery memory genes: ', recovery)
        print('false postive: ', false_pos)
        print('Precision and recovery clustering: ',model.score_, model.recovery)
        
    return scores