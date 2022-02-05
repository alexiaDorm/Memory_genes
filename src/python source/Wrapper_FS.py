import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import math 
import itertools 
import random
from typing import AnyStr, Callable

from genetic_selection import GeneticSelectionCV

from pred_score import *

#-----------------------------------------------------------------------------
#Wrapper features selection methods:
#In wrapper methods, the features are selected by evalating subsets of features using a ML algorithm fitted on the data. Each subset is scored using a evaluation criteria. Finally, the best subset is choosen according to this criteria.
def naive_features_selection(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool,N:np.array):
    """ Enumerate all possible subsets of features and select the best subset according to the score
  
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

      returns:
      best_subset : np.array,
        subset of features with the best score
      best_score : float,
        score obtained with the best subset of features"""

    best_subset, best_score = None, None, 0.0 
    # Enumerate all possible subset of features, only keep the subset with at least 2 features
    possible_subset = np.array([list(x) for x in itertools.product([True, False], repeat=x.shape[1])])
    ind_least2 = np.where(np.sum(possible_subset, axis = 1)>= 2)
    possible_subset = np.squeeze(possible_subset[ind_least2,:])
    
    for subset in possible_subset:
        #Define subset and compute score
        x_subset = x[:, subset]
        score = evaluate(y,x_subset,Model,Scoring,True)
        
        #Check if the score is better than in previous subsets   
        if ((best_score == None) or (maximize==True and score >= best_score) or (maximize ==False and score <= best_score)):
          #Convert best_subset into fetures indices
            best_subset = [i for i, x in enumerate(subset) if x]
            best_score = score
  
    return best_subset, best_score


def mutate(subset:np.array, p_mutate:float):
    """ Flip features with probability p_mutate 
    
      parameters:
      subset: np.array,
        subset to flip 
      p_mutate : float[0,1],
        probability to flip a feature

      return 
      flipped: np.array,
        transformed subset
        """
    flipped = subset.copy()
    for i in range(len(flipped)):
        if np.random.rand() < p_mutate:
            flipped[i] = not flipped[i]
    return flipped
 
def at_least2(subset:np.array):
    '''Check if the subset has at least 2 features, if not one of the not-selected features is flipped.'''
    
    while np.sum(subset) < 2:
        #Get the not used features
        false_features = np.where(subset==False)
        #Randomly Select the one that is flipped
        flip = random.randint(0,len(false_features))
        subset[false_features[flip]] = True
    
    return subset
        
def hillclimbing(y:np.array, x:np.array, Model : Callable, Scoring: Callable, maximize:bool, n_iter:int, p_mutate:float, plot:bool = False):
    """ Randomly flip the features and keep this new set if it improves score. 
    
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
      n_iters : int,
        number of iterations of the hill climbing algorithm
      p_mutate : float[0,1],
        probability to flip a feature
      plot : bool,
          if True the score at each iteration is plotted

      return 
      solution : np.array,
        subset of features with the best score
      solution_score : float,
        score obtained with the best subset of fetures
        """

	#Initial subset (at least 2 features), evaluate it
    solution = np.random.choice([True, False], size=x.shape[1])
    solution = at_least2(solution)
    
    solution_score = evaluate(y, x, Model, Scoring,True)
    
    #Store the score at each iterations for plotting
    plot_score = []
    plot_score.append(solution_score)
 
    for i in range(n_iter):
		#Flip features with probability p_mutate and evaluate new subset (with at least 2 features keept)
        candidate = mutate(solution, p_mutate)
        candidate = at_least2(candidate)
        
        x_subset = x[:, candidate]
        candidate_score= evaluate(y, x_subset, Model, Scoring, True)

		#Check if subset is better than previous one
        if (maximize== True and candidate_score >= solution_score) or (maximize== False and candidate_score <= solution_score):
            solution, solution_score = candidate, candidate_score

        #Store the solution accuracy at each step
        plot_score.append(solution_score)
    
    #Convert the solution into a array of the indices of the keept features
    solution = [i for i, x in enumerate(solution) if x]
    
    if(plot == True):
        plt.plot(plot_score)
        plt.xlabel("iteration")
        plt.ylabel("score")
        plt.title("Evolution of the score")
        plt.show()

    return solution, solution_score



def stimulated_annealing(y:np.array, x:np.array, Model : Callable, Scoring: Callable, maximize:bool, n_iter:int, p_mutate:float, c:float=1, plot:bool = False):
    """ Apply stimulated annealing algorithm for feature selection.
    
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
      n_iters : int,
        number of iterations of the stimulated annealing algorithm
      c : float (default = 1),
        controls the amount of perturbation happenning
      plot : bool,
          if True the score at each iteration is plotted

      returns
      solution : np.array,
        final subset of features 
      solution_score : float,
        score obtained with the best subset of features
        """
    #Fix seed for reproducibility
    np.random.seed(3)
    random.seed(3)
    
    #Initial subset, evaluate it
    solution = np.random.choice([True, False], size=x.shape[1])
    solution = at_least2(solution)
    solution_score = evaluate(y, x, Model, Scoring,True)
    
    #Store the score at each iterations for plotting
    plot_score = []
    plot_score.append(solution_score)
 
    for i in range(n_iter):
		#Flip features with probability p_mutate and evaluate new subset
        candidate = mutate(solution, p_mutate)
        candidate = at_least2(candidate)
        x_subset = x[:, candidate]
        candidate_score = evaluate(y, x_subset, Model, Scoring, True)

		#Check if subset is better than previous one
        if (maximize==True and candidate_score >= solution_score) or (maximize ==False and candidate_score <= solution_score):
            solution, solution_score = candidate, candidate_score
        
        #If the score is not better, compute acceptance probability
        else:
            p_acceptance = np.exp(-(i/c) * ((solution_score - candidate_score)/solution_score))
            #Random number drawn, if smaller than the acceptance, then the new subset is keept. 
            #Otherwise, we reject the new subset
            rand_num = np.random.uniform(0,1)
            if rand_num <= p_acceptance:
                solution, solution_score = candidate, candidate_score

        #Store the solution score at each step
        plot_score.append(solution_score)
    
    #Convert the solution into a array of the indices of the keept features
    solution = [i for i, x in enumerate(solution) if x]
    
    if(plot == True):
        plt.plot(plot_score)
        plt.xlabel("iteration number")
        plt.ylabel("score")
        plt.title("Evolution of the score")
        plt.show()

    return solution, solution_score

def genetic_fs(y:np.array,x:np.array, Model: Callable, Scoring: Callable, maximize:bool, n_population : int, crossover_proba : float, mutation_proba : float, 
               n_generations : int, tournament_size : int, plot:bool=False):
    """ Implement the genetic feature selection algorithm.
  
      parameters:
      y : np.array,
        family of each data points
      x : np.array,
        features of each data points
      Model : Callable,
        the model is fitted and the score computed using this method
      Scoring: Callable,
        scoring function use to evaluate the model
      maximize: bool,
        if True the scoring function is maximize, else it is minimize
        number of population
      crossover_proba : float,
        probability of crossover
      mutation_proba : float,
        probability of mutation
      n_generations : int,
        number of generations
      tournament_size : int,
         tournament size
      plot : bool (default false),
        if true plot the best score in each population

      returns:
      subset : np.array,
        subset found by the algorithm
      score : float,
        score obtained with the subset of features
     """

    #Fix seed for reproducibility
    np.random.seed(3)
    random.seed(3)
    
    #Define model use to fit and evaluate accuracy
    model = Model(np.unique(y),Scoring,maximize)

    #Define Boruta feature selection method and find all relevant features
    fs = GeneticSelectionCV(estimator = model, n_jobs=-1, n_population=n_population, crossover_proba=crossover_proba, mutation_proba=mutation_proba, n_generations=n_generations, crossover_independent_proba=0.1,mutation_independent_proba=0.05, tournament_size=tournament_size, caching=True)
    
    fs.fit(x, y)

    #Get the selected features and final score
    subset = fs.support_
    subset = [i for i, x in enumerate(subset) if x]
    score = fs.generation_scores_[-1]
    
    if(plot == True):
        plot_score = fs.generation_scores_
        x = np.arange(1, len(plot_score), 1)
        
        plt.plot(plot_score)
        plt.xlabel("generation number")
        plt.ylabel("score")
        plt.title("Evolution of the score at each generation")
        plt.show()
        

    return subset, score