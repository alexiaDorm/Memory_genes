import numpy as np
import pandas as pd
import pyreadr
from typing import AnyStr, Callable
from pred_score import *

def get_best_genes_names(subset:np.array, path_gene_interest:str, output_path: str):
    '''Get the name of genes from a subset and create a .csv with these names
    
    parameters:
    subset: np.array
        indices of genes that we want the name pulled
    path_gene_interest: str
        path of the file where genes name are
    output_path: str
        path of output file 
    
    '''

    #Load gene_names
    name_genes = np.array(pd.read_csv (path_gene_interest))

    #Get their name
    genes = pd.DataFrame(name_genes[subset])

    #Create data with genes name in subset
    genes.to_csv(output_path, index=False)
    
def genes_present_at_least(list_all:list, at_least:int):
    '''Return all genes present in at least N optimization subset.
    
    parameters:
    list_all: list of list,
        name of the optimal genes for each data set
    at_least: int,
        number of optimization in which the gene need to present
    
    return:
    overlap: list,
        list of the name of the genes in the overlap subset
    '''
    #Concatenate all the genes together    
    all_genes = np.sum(list_all)
    
    #Count how many time each gene appears
    genes, count = np.unique(all_genes, return_counts=True)
    
    #Get indices of the genes in at least N optimization
    ind = np.argwhere(count >= at_least)
    
    return np.squeeze(genes[ind])

def evaluate_overlap_genes(list_all:list, data:list, y:list, at_least:int=1):
    '''Evaluate the optimal overlap subset over all data sets and return the score for each set.
    
    parameters:
    list_all: list,
        name of the optimal genes for each data set
    data: list,
        list of all normalized data from each data set
    y: list,
        list of correct families for each data set
    at_least: int,
        number of optimization in which the gene need to present
        
    return:
    score: list,
        list of score of each data set with the overlaped optimal subset
    overlap: list
        list of the name of the genes in the overlap subset
    
    '''
    #Compute overlap accrording to overlap_function   
    overlap = list(np.squeeze(genes_present_at_least(list_all, at_least)))

    #Predict and evaluate overlap on each data set
    score = []
    for i, x in enumerate(data):
        #Remove genes not in current data x 
        overlap_x = list(np.squeeze(genes_present_at_least([overlap, list(x.columns)], 2)))
        
        y[i] = np.array(np.squeeze(y[i]))

        model = FamiliesClusters(np.unique(y[i]),compute_precision,True)
        pred = model.fit_predict(x[overlap_x],y[i])

        score.append(compute_statTP(y[i],pred)) 

    return score, overlap
