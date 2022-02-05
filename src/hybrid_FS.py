import numpy as np
import pandas as pd
import sklearn
import math 
from typing import AnyStr, Callable

from pred_score import *
from Filter_FS import *
from Wrapper_FS import *

 
def MI_stimulated_annealing(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:int, n_neighbors: int, n_iter:int, p_mutate:float, c:float=1, plot:bool = False):
    """ Choose the top N features with largest MI, and run stimulated annealing on this subset. 
  
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
      N: np.array,
          number of features to keep, the top N features with the highest mutual information will be keept
      n_neighbors : int,
        number of neighbors to use for MI estimation
      n_iters : int,
        number of iterations of the stimulated annealing algorithm
      c : float (default = 1),
        controls the amount of perturbation happenning
      plot : bool (defaul = False),
        if true plot the accuracy vs the number of selected features
        

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of features """
    
    #Find top N features with the highest mutual information
    subsetMI , _ = MIM(y, x, Model, Scoring, maximize, N, n_neighbors)
    xMI = x[:, subsetMI]
    
    #Run stimulated annealing on MI subset
    subset, best_score = stimulated_annealing(y, xMI, Model, Scoring, maximize, n_iter, p_mutate, c, plot)
    
    best_subset = subsetMI[subset]
        
    return best_subset, best_score

def MI_genetic_fs(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:int, n_neighbors: int,  n_population : int, crossover_proba : float, mutation_proba : float, n_generations : int, tournament_size : int,plot:bool = False):
    """ Choose the top N features with largest MI, and run genetic algorithm on this subset. 
  
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
      N: np.array,
          number of features to keep, the top N features with the highest mutual information will be keept
      n_neighbors : int,
        number of neighbors to use for MI estimation
      crossover_proba : float,
        probability of crossover
      mutation_proba : float,
        probability of mutation
      n_generations : int,
        number of generations
      tournament_size : int,
         tournament size
      plot : bool (defaul = False),
        if true plot the accuracy vs the number of selected features
        

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of features """
    
    #Find top N features with the highest mutual information
    subsetMI , _ = MIM(y, x, Model, Scoring, maximize, N, n_neighbors)
    xMI = x[:, subsetMI]
    
    #Run gentic algorithm on MI subset
    subset, best_score = genetic_fs(y,xMI, Model, Scoring, maximize, n_population, crossover_proba, mutation_proba, n_generations, tournament_size, plot)
    
    best_subset = subsetMI[subset]
        
    return best_subset, best_score