import numpy as np
import pandas as pd
import pyreadr
#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Few functions processing the data
def store_family_code_into_family_number(families_info:np.array, families_interest:np.array) :
    """Return name of the cells that belong to the families of interest.
  
      parameters:
      families_info : np.array,
        contains association of cell to family
      families_interest : np.array,
        contains all families of interest (containing right number of cells)

      returns:
      cells_interest : np.array,
          names of the cells that belong to the families of interest"""
    
    cells_interest = (families_info[np.isin(families_info[:,0], families_interest)])
    
    return cells_interest

def select_family_interest_norm_data(families_info:np.array, family_interest:np.array, norm_data:pd.DataFrame) :
    """Return the norm data with only the cells belonging to the family of interest.
  
      parameters:
      families_info : np.array,
        contains association of cell to family
      families_interest : np.array,
        contains all families of interest (containing right number of cells)

      returns:
      norm_data : pd.DataFrame,
          norm_data with only the cells belonging to families of interest
      cells_interest : np.array,
          names of the cells that belong to the families of interest
    """
    
    #Find the names of cells of interest
    code = store_family_code_into_family_number(families_info, np.squeeze(family_interest.astype(np.int)))
    
    norm_data = ((norm_data[code[:,1]]))
    
    return norm_data, code

def filter_norm_data(norm_data : np.array, percentage : float = 0.5):
    """ Finds which genes are expressed in at least 50% (default) of the cells and returns a binary table 
    
    parameters:
    norm_data : np.array,
        normalized data where each gene is a row and each cell is a column, the numbers correspond to a certain expression of the gene in the cell
    percentage : float = 0.5
        percentage threshold which indicates if a gene is expressed enough in a collection of cells
    
    returns:
    tab_good : np.array,
        binary array which indicates which gene is expressed enough (True) or not (False)
    
    """
    
    N_gen = norm_data.shape[0]   #number of genes
    N_cel = norm_data.shape[1]   #number of cells
    
    #Binary result table indicating which gene is expressed at least ..% in the cell data set
    tab_good = np.zeros(N_gen, dtype = bool)
    
    #Number of times val > 0 for each gene
    tab_count = (norm_data > 0).sum(axis = 1)
    
    #Ratio of val bigger than 0 = Number of times val > 0 for each gene/Number of cells
    tab_perc = tab_count/N_cel
    
    #Changing value to True for each gene if tab_perc >= percentage
    tab_good[tab_perc >= percentage] = True
    
    return tab_good

def filter_norm_all(norm_data_sets : list, percentage : float = 0.5):
    """ Finds for each set of the data_sets which genes are expressed in at least 50% of the cells 
    
    parameters:
    norm_data_sets : list,
    list of all the normalized data sets
    percentage : float = 0.5,
    percentage threshold which indicates if a gene is expressed enough in a collection of cells
    
    returns:
    tab_filtered : list,
    list of binary array which indicates for each set which gene are expressed enough

    """ 
    tab_filtered = []
    N_set = len(norm_data_sets)
    for i in range(N_set):
        res = filter_norm_data(norm_data_sets[i], percentage)
        tab_filtered.append(res)
        
    return tab_filtered

def sort_features_pval(data : np.array, p_val : float = 0.05):
    """ Finds all the genes that have at least one feature with p-value with a value smaller than 0.05

      parameters:
      data : np.array,
      features of the genes
      p_val : float,
      threshold for determining if a feature is significant

      returns:
      tab : np.array,
      binary table which indicates which gene seems to be a potential good feature (True) or not (False)

    """
    #transform data into array for future manipulations
    #data = np.array(data)   doit être fait avant de donner le data en paramètre

    #creating result table
    tab = np.zeros(data.shape[0], dtype = bool) #Initializing each value of the table to False


    #loop over parameters that are p-values
    #If a value seems to be intersting for the predcition, the boolean value True is affected to it.
    for i in range(7):
        tab[data[:,i] < p_val] = True

    return tab

def sort_features(data : np.array, cond_FM : np.array, p_val : float = 0.05):
    """ Finds all the genes that have at least one feature with p-value with a value smaller than 0.05 or an optimal FindMarker value

    parameters:
    data : np.array,
    features of the genes
    p_val : float,
    threshold for determining if a feature is significant
    cond_FM : np.array,
    optimal value of FindMarker definded for each feature

    returns:
    tab : np.array,
    binary table which indicates which gene seems to be a potential good feature (1) or not (0)

    """

    #transform data into array for future manipulations
    #data = np.array(data)    need to be done before giving it to the parameter

    #creating result table
    tab = np.zeros(data.shape[0], dtype = bool) #Initializing each value of the table to False

    #loop over parameters that are p-values
    for i in range(7):
        tab[data[:,i] < p_val] = True

    #loop over parameters that are FindMarkers
    #need to find the optimal number for FindMarkers
    #writing the range that way fixes the problem of FindMarkerMAST not present in BIDDY_D0_2_20, it will iterate 7 or 8 times depeding on the file
    for i in range(data.shape[1] - 7):   
        tab[data[:,i+7] >= cond_FM[i]] = True

    return tab

def compare_feat_files(data_sets : list, cond_FM: np.array, p_val : float = 0.05):
    """ Finds for each data_set which genes seems to be good features and the number of these good features

    parameters:
    data_sets : list,
    contains the data of all the data sets through the form of list of arrays, each array corresponds to a data set
    cond_FM : np.array,
    optimal value of FindMarker definded for each feature
    p_val : float,
    threshold for determining if a feature is significant
    
    returns:
    nbrs_good : np.array,
    table which indicates which the number of potential good features for each data_set
    tab_good : list,
    list of arrays, each array corresponds to a data_set and its binary content indicates which gene seems to be a potential good feature 
    """

    N = len(data_sets)    #number of different data sets
    nbrs_good = np.zeros(N)   #number of good features for each file
    tab_good = [] #multi-binary indicating for each data set which gene seems to be a good feature

    #iterate over each data_set
    for i in range(N):
        tab = sort_features(data_sets[i], cond_FM, p_val)
        tab_good.append(tab)
        nbrs_good[i] = tab.sum()    #number of good features

    return nbrs_good, tab_good
