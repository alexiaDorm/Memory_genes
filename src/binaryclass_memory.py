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
from torch.optim import SGD
from torch.utils.data import random_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif


#------------------------------------------------------------------------------------------------------------------------------------------------------------
def visualize_charac(data:pd.DataFrame):
    copy = data
    #Remove the outliers of the data before normalization of mean expression
    data, outliers = remove_extreme_values(data, k=200)
    mem_genes_perc_outliers = 100 *np.sum(copy.loc[outliers]['memory_gene'])/len(outliers)
    print(mem_genes_perc_outliers, '% of the outliers are memory genes')
    
    #Shift the skeness values by substracting them with their mean
    corrected_skewRes = normalize(data['CV2ofmeans_residuals'])
    #corrected_skewRes -= np.mean(corrected_skewRes)
    
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
    
def load_data (fused:pd.DataFrame, params):
    X = np.array(fused.drop(columns=['memory_gene']))
    y = np.expand_dims((fused['memory_gene']*1), axis=1)
    dataset = Dataset(np.array(X), y)

    N = len(y)
    train, test = random_split(dataset, [math.floor(N*0.8), math.ceil(N*0.2)], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train, batch_size= 32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    
    return train_dl, test_dl


def load_charac():
    #Load data
    general_charac = pyreadr.read_r('../data/Characteristics_masterfiles/General_characteristics/EPFL_gene_master_matrix.RData')['gene_master_matrix']

    names = ['AE3', 'AE4', 'AE7', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2']

    charac_matrix = []
    for name in names:
        #Open characteristics file
        charac_out_path = '../data/Characteristics_masterfiles/Dataset_specific_characteristics/' + name + '__characteristics_output.txt'
        p_value_path = '../data/Characteristics_masterfiles/Memory_genes/P_value_estimate_CV2_ofmeans_' + name + '.txt'
        charac_matrix.append(open_charac(charac_out_path, p_value_path, 200))

    #Add general characteristic
    for i in range(0,len(charac_matrix)):
        charac_matrix[i] = add_general_charac(charac_matrix[i], general_charac)
        charac_matrix[i] = charac_matrix[i].drop(['skew_residuals','cell_cycle_dependence', 'skew', 'CV2ofmeans', 'exon_expr_median', 'exon_expr_mean'], axis=1)
        charac_matrix[i] = charac_matrix[i].dropna()

    #Remove AE7, also keep BIDDYD15_2 for validation
    val = [8]
    data_to_fuse = [0,1,3,4,5,6,7]

    for i in range(0,len(charac_matrix)):
        #Normalize skew_residuals, same for mean_expression after removing outliers
        charac_matrix[i], outlier_temp = remove_extreme_values(charac_matrix[i], k=200)
        outliers.append(outlier_temp)
        charac_matrix[i]['CV2ofmeans_residuals'], charac_matrix[i]['mean_expression'] = normalize(charac_matrix[i]['CV2ofmeans_residuals']), normalize(charac_matrix[i]['mean_expression'])
        charac_matrix[i]['length'], charac_matrix[i]['GC'] = normalize(charac_matrix[i]['length']), normalize(charac_matrix[i]['GC'])

    val_charac =  []
    for i in val:
        val_charac.append(charac_matrix[i])
    fused_charac = []
    name_fused = []
    for i in data_to_fuse:
        fused_charac.append(charac_matrix[i])
        name_fused.append(names[i])

    fused = pd.concat(fused_charac)
    
    return fused

class NN_1l(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN, self).__init__()
        self.layer = nn.Linear(n_inputs, 1)
        nn.init.xavier_uniform_(self.layer.weight)
 
    def forward(self, X):
        X = self.layer(X)
        return X
    
def train_model(train_dl, model, criterion, optimizer):
    for epoch in range(200):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            #Convert double to float
            inputs, targets = inputs.to(torch.float32),  targets.to(torch.float32)

            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(test_dl, model):
    with torch.no_grad():
        acc = []
        for i, (inputs, targets) in enumerate(test_dl):
            #Convert double to float
            inputs, targets = inputs.to(torch.float32),  targets.to(torch.float32)

            # evaluate the model on the test set
            yhat = torch.sigmoid(model(inputs))
            yhat[yhat >= 0.5] = 1; yhat[yhat < 0.5] = 0

            acc.append(balanced_accuracy_score(y_true = targets, y_pred = yhat))

    return np.mean(acc)

def predict(inputs, model):
    inputs =torch.Tensor([inputs])
    yhat = torch.sigmoid(model(inputs))
    yhat[yhat >= 0.5] = 1; yhat[yhat < 0.5] = 0
    
    return yhat.detach().numpy()

def obj(trial, fused):
    #Set hyperparamters to tune
    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 100),
              'weight_decay' : trial.suggest_loguniform('weight_decay', 1e-5, 1),
              'n': trial.suggest_int("n", 4, 50),
              #'batch_size': trial.suggest_int("batch_size", 5, 8), #2^i
              'nb_features' : trial.suggest_int("nb_features", 2, 18)
              }
    #Load data
    X = fused.drop(columns=['memory_gene'])
    y = fused['memory_gene']*1
    
    #Get the N top features according to mutual information
    selector = SelectKBest(mutual_info_classif, k=params['nb_features'])
    X_redu = selector.fit_transform(X, y)
    cols = selector.get_support(indices=True)
    FS = X.iloc[:,cols].columns.tolist(); FS.append('memory_gene')

    train_dl, test_dl = load_data(fused[FS],params)

    model = NN_1l(len(FS)-1, params)

    #Optmization criterion and optimizer
    num_positives= np.sum(y); num_negatives = len(y) - num_positives
    pos_weight  = torch.as_tensor(num_negatives / num_positives, dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=params['weight_decay'])

    #Train and evaluate the NN
    train_model(train_dl, model, criterion, optimizer)
    acc = evaluate_model(test_dl,model)
    
    return acc

class Objective(object):
    def __init__(self, fused):
        self.fused = fused

    def __call__(self, trial):
        
        acc = obj(trial, self.fused)
        
        return acc