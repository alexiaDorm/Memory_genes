import numpy as np
import itertools 
import sklearn
import matplotlib.pyplot as plt
import random
from typing import AnyStr, Callable
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from skfeature.function.information_theoretical_based.MIFS import mifs
from skfeature.function.information_theoretical_based.JMI import jmi
from skfeature.function.information_theoretical_based.DISR import disr
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based.lap_score import  lap_score
from skfeature.utility.construct_W import construct_W
from skrebate import ReliefF
from skfeature.function.statistical_based.CFS import cfs
from skfeature.function.information_theoretical_based.FCBF import fcbf
from pyHSICLasso import HSICLasso

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
        plt.title("Best score obtained for each number of selected features")
        
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
    anova = f_classif(x,y)
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

def MIFS(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, beta: int,  plot:bool = False):
    """ Compute the mutual information and select the ones which has more information gain and select non-redundant features thanks to a penalization term.
  
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
      beta : int,
        constant before penalization term
      plot : bool (defaul = False),
        if true plot the accuracy vs the number of selected features
        

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of features
      """
    
    #Store score for plotting 
    plot_score = []
    
    #Rank the features using cfs algorithm
    kwargs = {"n_selected_features": max(N), "beta": beta}
    features_ranked = mifs(x, y, **kwargs)
                                         
    best_subset, best_score = None, None
    for i in N:
        #Define the subset with i features
        subset = features_ranked[0 : i]
        
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

def JMI(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array,  plot:bool = False):
    """ Compute the joint mutual information and select features complementary with the already included ones. 
  
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
        score obtained with the best subset of features
      """
    
    #Store score for plotting 
    plot_score = []
    
    #Rank the features using cfs algorithm
    kwargs = {"n_selected_features": max(N)}
    features_ranked = jmi(x, y, **kwargs)
    
    best_subset, best_score = None, None
    for i in N:
        #Define the subset with i features
        subset = features_ranked[0 : i]
        
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

def DISR(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, plot:bool = False):
    """ Compute a measure of variable complementarity. Select features according to this computed score. 
  
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
        score obtained with the best subset of features
       """
    
    #Store score for plotting 
    plot_score = []
    
    #Rank the features using cfs algorithm
    kwargs = {"n_selected_features": max(N)}
    features_ranked = disr(x, y, **kwargs)
    
    best_subset, best_score= None, None
    for i in N:
        #Define the subset with i features
        subset = features_ranked[0 : i]
        
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


def fishers_score(y:np.array,x:np.array, Model: Callable,Scoring: Callable, maximize:bool, N:np.array, plot:bool=False):
    """ Calculate the Fisher's score and chose features with highest score.
  
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
          number of features to keep, will chose the best number out of  entry in the N array
      plot : bool (defaul = False),
        if true plot the score vs the number of selected features
        

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of fetures
     """
    #Store the score for plotting
    plot_score = []
    
    #Compute fisher's score
    fish_score = fisher_score.fisher_score(x,y)
    index_sorted  = np.argsort(fish_score)
    
    best_subset, best_score= None, None
    for i in N:
        #Define the subset with i features
        subset = index_sorted[-i :]
        #Evaluate error on given subset
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
    
def laplacian_score(y:np.array,x:np.array, Model: Callable,Scoring: Callable, maximize:bool, N:np.array, plot:bool=False):
    """ Compute the Laplacian's score and select features according to it.
  
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
        if true plot the score vs the number of selected features
        

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of fetures
     """
    #Store the score for plotting
    plot_score = []
    
    #Compute Laplacian's score
    W = construct_W(x)
    kwargs = {"W": W}
    laplacian_score = lap_score(x, **kwargs)
    features_ranked = np.argsort(laplacian_score,0)
    
    best_subset, best_score = None, None
    for i in N:
        #Define the subset with i features
        subset = features_ranked[0 : i]
        
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

def variance_threshold(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, max_threshold:float, plot:bool=False):
    """ Select all features with a variance higher than a threshold. 
    The threshold is choosen using grid search(between 0 and max_threshold).
  
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
      max_threshold : float,
        max threshold to consider on grid serach
      plot : bool (defaul = False),
        if true plot the score vs the number of selected features
        
        
      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        accurancy obtained with the best subset of fetures
      """
    #Store the score for each threshold for plotting 
    plot_score = []
    
    best_subset, best_score = None, None
    
    #Define grid search for threshold
    threshold = np.linspace(0,max_threshold,100)
    for thres in threshold:
        #Find features with variance higher than threshold
        v_thres = VarianceThreshold(thres)
        v_thres.fit(x)
        subset = v_thres.get_support()
        
        #Evaluate the error on given subset
        x_subset = x[:, subset]
        score = evaluate(y,x_subset,Model,Scoring,maximize)
        
        if ((best_score == None) or (maximize==True and score > best_score) or (maximize ==False and score < best_score)):
            #Convert best_subset into fetures indices
            best_subset = [i for i, x in enumerate(subset) if x]
            best_score = score
            
        plot_score.append(score)
        
    #Plot the score vs the number of selected features
    if(plot == True):
        plt.plot(threshold,plot_score)
        plt.xlabel("variance threshold")
        plt.ylabel("score")
        plt.title("Best score obtained for the different variance threshold")
        
    return best_subset, best_score

def reliefF(y:np.array,x:np.array, Model: Callable,Scoring: Callable, maximize:bool, N:np.array, n_neighbors: int,  plot:bool=False):
    """ Apply relief algorithm and select features based on the computed score.
  
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
        number of neighbors to use in ReliefF algorithm
      plot : bool (defaul = False),
        if true plot the score vs the number of selected features
        
        
      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of features
       """
    #Store the score for each threshold for plotting 
    plot_score = []
    
    #Compute importance of features according to ReliefF algorithm
    fs = ReliefF()
    fs.fit(x, y)
    imp = fs.feature_importances_
    index_sorted  = np.argsort(imp)
    
    best_subset, best_score = None, None
    for i in N:
        #Define the subset with i features
        subset = index_sorted[-i :]
        
        #Evaluate the error on given subset
        x_subset = x[:, subset]
        score= evaluate(y,x_subset,Model,Scoring,maximize)
        
        if ((best_score == None) or (maximize==True and score > best_score) or (maximize ==False and score < best_score)):
            #Convert best_subset into fetures indices
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

def CFS(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, plot:bool = False):
    """ Determine the best subset accoring to CFS.
  
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
        score obtained with the best subset of features
     """
    
    #Store score for plotting 
    plot_score = []
    
    #Rank the features using cfs algorithm
    features_ranked = cfs(x,y)
    
    best_subset, best_score = None, None
    for i in N:
        #Define the subset with i features
        subset = features_ranked[0 : i]
        
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


def FCBF(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, plot:bool = False):
    """ Determine the best subset accoring to CFS.
  
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
        score obtained with the best subset of features
      """
    
    #testing if setting seed makes it reproducible, ça marche pas
    random.seed(3)
    np.random.seed(3)
    
    #Store score for plotting 
    plot_score = []
    
    #Rank the features using cfs algorithm
    features_ranked = fcbf(x,y)
    
    best_subset, best_score = None, None
    for i in  N:
        #Define the subset with i features
        subset = features_ranked[0 : i]
        
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
    
def HsicLasso(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, N:np.array, plot:bool = False):
    """ Determine the best subset accoring to the HSIC Lasso algorithm. CV the number of features in the final subset.
  
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
        score obtained with the best subset of features
       """
    
    #Store score for plotting 
    plot_score = []
    
    #Define HSICLasso object
    hsic_lasso = HSICLasso()
    hsic_lasso.input(x,y)
    
    best_subset, best_score = None, None
    for i in N:
        #Define the subset with i features with algorithm
        hsic_lasso.classification(i)
        subset = hsic_lasso.get_features()
        subset = [int(i)- 1 for i in subset]
        
        #Cross-validate the error on given subset
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
    