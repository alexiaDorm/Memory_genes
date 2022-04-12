import numpy as np
import itertools 
import sklearn
import matplotlib.pyplot as plt
import random
from typing import AnyStr, Callable
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

from pred_score import *

#-----------------------------------------------------------------------------
#Filter feature selection: the selection is independent from any ML algorithm. The features are selected according to their scores in various statistical tests.

def MIM(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, n_neighbors: int, plot:bool = False):
    """ Calculate the mutual information and select the ones which has more information gain. 
  
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
          number of features to keep, will chose the best number out of entry in the N array
      n_neighbors : int,
        number of neighbors to use for MI estimation
      plot : bool (defaul = False),
        if true plot the accuracy vs the number of selected features
        

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of features """
    
    #Store score for plotting 
    plot_score = []
    
    #Compute mutual information
    mutual_information = mutual_info_classif(x,y,n_neighbors = n_neighbors, random_state = 3)
    index_sorted  = np.argsort(mutual_information)
    
    best_subset, best_score= None, None
    for i in N:
        #Define the subset with i features
        subset = index_sorted[-i :]
        
        #Evaluate the error on given subset
        x_subset = x[:, subset]
        score = evaluate(y,x_subset,Model,Scoring,maximize)
        if ((best_score == None) or (maximize==True and score > best_score) or (maximize ==False and score < best_score)):
            #Convert best_subset into features indices
            best_subset = np.sort(subset)
            best_score = score
        
        plot_score.append(score)
    
    #Plot the score vs the number of selected features
    if(plot == True):
        plt.plot(N,plot_score)
        plt.xlabel("number of selected features")
        plt.ylabel("score")
        plt.title("Best training score obtained for each number of selected features")
        
    return best_subset, best_score

def ANOVA(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, plot:bool = False):
    """ Calculate the ANOVA test and select the ones with larger score. 
  
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
          number of features to keep, will chose the best number out of entry in the N array
      plot : bool (defaul = False),
        if true plot the accuracy vs the number of selected features
        

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of features """
    
    #Store score for plotting 
    plot_score = []
    
    #Compute Anova test
    anova, _ = f_classif(x,y)
    index_sorted  = np.argsort(anova)
    
    best_subset, best_score= None, None
    for i in N:
        #Define the subset with i features
        subset = index_sorted[-i :]
        
        #Evaluate the error on given subset
        x_subset = x[:, subset]
        score = evaluate(y,x_subset,Model,Scoring,maximize)
        if ((best_score == None) or (maximize==True and score > best_score) or (maximize ==False and score < best_score)):
            #Convert best_subset into features indices
            best_subset = np.sort(subset)
            best_score = score
        
        plot_score.append(score)
    
    #Plot the score vs the number of selected features
    if(plot == True):
        plt.plot(N,plot_score)
        plt.xlabel("number of selected features")
        plt.ylabel("score")
        plt.title("Best score obtained for each number of selected features")
        
    return best_subset, best_score


    