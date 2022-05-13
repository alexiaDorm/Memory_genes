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
    corrected_skewRes = normalize(data['skew_residuals'])
    corrected_skewRes -= np.mean(corrected_skewRes)
    
    #Look at skenness and mean expression of all genes
    colors = ['grey','red']
    plt.scatter(corrected_skewRes, normalize(data['mean_expression']), marker='o', c= data['memory_gene'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel("skew")
    plt.ylabel("mean expression")
    plt.xlim(-40,60)
    #plt.yscale('log')
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

def reset_weights(m):
    '''Try resetting model weights to avoid weight leakage.'''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            
def compute_scores(network, inputs, targets):
    #Compute prediction of network
    outputs = torch.sigmoid(network(inputs))

    #Compute accuracy
    accuracy = ((outputs >= 0.5) == targets).float().mean()
    
    #Compute recovery and false positive rate
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    targets, outputs = targets.detach().numpy(), outputs.detach().numpy()
    
    recovery, false_pos_rate = np.nan, np.nan
    if np.sum(targets) != 0:
        recovery = np.sum(targets*outputs)/np.sum(targets)
        
    FP = np.logical_and(outputs == 1, targets*outputs == 0)
    if np.sum(outputs) != 0:
        false_pos_rate = np.sum(FP)/np.sum(outputs)
    
    return [accuracy, recovery, false_pos_rate]

def objective(trial):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 100),
              'n_fl': trial.suggest_int("n_fl", 4, 50),
              'n_sl': trial.suggest_int("n_sl", 4, 50),
              'n_tl': trial.suggest_int("n_tl", 4, 50),
              'batch_size': trial.suggest_int("batch_size", 32, 256)
              }
    
    model = build_model(params)
    
    #Load data
    general_charac = pyreadr.read_r('../data/Characteristics_masterfiles/General_characteristics/EPFL_gene_master_matrix.RData')['gene_master_matrix']

    names = ['AE3', 'AE4', 'AE7', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2']

    charac_matrix = []
    norm_matrix = []
    families_matrix = []
    for name in names:
        #Open characteristics file
        charac_out_path = '../data/Characteristics_masterfiles/Dataset_specific_characteristics/' + name + '__characteristics_output.txt'
        p_value_path = '../data/Characteristics_masterfiles/Memory_genes/P_value_estimate_CV2_ofmeans_' + name + '.txt'
        charac_matrix.append(open_charac(charac_out_path, p_value_path, 200))

        #Open normalized data
        norm_path = '../data/merged_data/' + name + '.csv'
        fam_path = '../data/merged_data/y_' + name + '.csv'
        norm = pd.read_csv (norm_path)
        norm = norm.set_index('Unnamed: 0')
        families= np.squeeze(np.array(pd.read_csv(fam_path)))

        norm_matrix.append(norm)
        families_matrix.append(families)

    #Add general characteristic
    for i in range(0,len(charac_matrix)):
        charac_matrix[i] = add_general_charac(charac_matrix[i], general_charac)
        charac_matrix[i] = charac_matrix[i].drop(['CV2ofmeans_residuals','cell_cycle_dependence', 'skew', 'CV2ofmeans', 'exon_expr_median', 'exon_expr_mean'], axis=1)
        charac_matrix[i] = charac_matrix[i].dropna()

    #Remove AE7, also keep BIDDYD15_2 for validation
    val = [8]
    data_to_fuse = [0,1,3,4,5,6,7]

    for data in charac_matrix:
        #Normalize skew_residuals, same for mean_expression after removing outliers
        charac_matrix[i], outliers = remove_extreme_values(charac_matrix[i], k=200)
        charac_matrix[i]['skew_residuals'], charac_matrix[i]['mean_expression'] = normalize(charac_matrix[i]['skew_residuals']), normalize(charac_matrix[i]['mean_expression'])

    val_charac =  []
    for i in val:
        val_charac.append(charac_matrix[i])
    fused_charac = []
    name_fused = []
    for i in data_to_fuse:
        fused_charac.append(charac_matrix[i])
        name_fused.append(names[i])

    fused = pd.concat(fused_charac)

    accuracy = train_and_evaluate(params, model, fused)

    return accuracy

# Build neural network model
def build_model(params):
    
    return nn.Sequential(
        nn.Linear(18, params['n_fl']),
        nn.ReLU(),
        nn.Linear(params['n_fl'], params['n_sl']),
        nn.ReLU(),
        nn.Linear(params['n_sl'],params['n_tl']),
        nn.ReLU(),
        nn.Linear(params['n_tl'],1)
        
    )

# Train and evaluate the neural network model with CV
def train_and_evaluate(param, model, data):
    
    #Configuration of kfold and NN optimization 
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    num_epochs = 100
    results = [] #store fold result

    #Load data into torch.Dataset
    labels = np.expand_dims((data['memory_gene']*1), axis=1)
    X = data.drop(['memory_gene'], axis=1)
    #Oversample the memory genes data points
    ros = RandomOverSampler(random_state=42)
    X, labels = ros.fit_resample(X, labels)
    labels = np.expand_dims(labels, axis =1)
    dataset = Dataset(np.array(X), labels)
    
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        # Sample elements randomly 
        train_data = torch.utils.data.SubsetRandomSampler(train_ids)
        test_data = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=param['batch_size'], sampler=train_data)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=param['batch_size'], sampler=test_data)

        # Init the neural network
        network = model
        network.apply(reset_weights)

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(network.parameters(), lr=param['learning_rate'])
        #Determine ratio #non-memory genes/#memory genes to for pos_weight for loss function to deal with unbalanced data.
        #num_positives= np.sum(dataset.labels)
        #num_negatives = len(dataset.labels) - num_positives
        #pos_weight  = torch.as_tensor(num_negatives / num_positives, dtype=torch.float)
        loss_function = nn.BCEWithLogitsLoss() 

        # Run the training loop for defined number of epochs
        accuracy = []
        for epoch in range(0, num_epochs):
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(torch.float32),  targets.to(torch.float32)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()
                
        with torch.no_grad():
            
            # Iterate over the test data and generate predictions
            correct, total = 0, 0
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)

                #Compute accuracy on testing fold
                outputs = torch.sigmoid(network(inputs))
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                targets, outputs = targets.detach().numpy(), outputs.detach().numpy()

                correct += np.sum(targets == outputs)
                total += targets.shape[0]
             
        results.append(correct/total)

    return np.mean(results)