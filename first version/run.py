import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd  
import pyreadr

from pred_score import *
from hybrid_FS import *
from overlap_genes import *

#Set the seed for reproducibility
np.random.seed(1)
random.seed(1)

#Example of feature selection on the Biddy D6_2
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#Load preprocess data
D62 = np.array(pd.read_csv ('../data/processed_data/D62csv_process.csv'))
y = np.array(D62[:,-1],dtype=int)
D62 = D62[:,0:-1]

#Evaluate the baseline 
model = FamiliesClusters(np.unique(y),compute_precision,True)
pred = model.fit_predict(D62,y)

print("sensitivity, specificity, precision, NPV, FDR, FNR = ", compute_statTP(y,pred))


#Run best feature selection method: MI/SA
D62_best_subset, best_score = MI_stimulated_annealing(y,D62, FamiliesClusters, compute_precision,True, np.array([400]), 3, 5000, 0.05, 1)

print('The subset has ',len(D62_best_subset), ' genes.')
model = FamiliesClusters(np.unique(y),compute_precision,True)
pred = model.fit_predict(D62[:, D62_best_subset],y)

print("sensitivity, specificity, precision, NPV, FDR, FNR = ", compute_statTP(y,pred))

#Create the csv file with optimal subset
D62_best = get_best_genes_names(D62_best_subset, '../data/processed_data/D62genes_interest.csv' ,'../data/optimized_subsets/D62genes_best.csv')

#Run final overlap on all cells
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#Load AE3 data
AE3 = pd.read_csv ('../data/processed_data/AE3csv.csv')
AE3_y = np.squeeze(pd.read_csv ('../data/processed_data/AE3_familiescsv.csv'))
AE3_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/AE3genes_best.csv')))
print(np.shape(AE3), np.shape(AE3_y))

#Load AE4 data
AE4 = pd.read_csv ('../data/processed_data/AE4csv.csv')
AE4_y =  np.squeeze(pd.read_csv ('../data/processed_data/AE4_familiescsv.csv'))
#AE4_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/AE4genes_best.csv')))
print(np.shape(AE4), np.shape(AE4_y))

#Load BIDDY DO data
DO = pd.read_csv ('../data/processed_data/DOcsv.csv')
DO_y =  np.squeeze(pd.read_csv ('../data/processed_data/DO_familiescsv.csv'))
DO_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/DOgenes_best.csv')))
print(np.shape(DO), np.shape(DO_y))

#Load BIDDY DO2 data
DO2 = pd.read_csv ('../data/processed_data/DO2csv.csv')
DO2_y =  np.squeeze(pd.read_csv ('../data/processed_data/DO2_familiescsv.csv'))
DO2_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/DO2genes_best.csv')))
print(np.shape(DO2), np.shape(DO2_y))

#Load BIDDY D6 data
D6 = pd.read_csv ('../data/processed_data/D6csv.csv')
D6_y =  np.squeeze(pd.read_csv ('../data/processed_data/D6_familiescsv.csv'))
D6_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/D6genes_best.csv')))
print(np.shape(D6), np.shape(D6_y))

#Load BIDDY D62 data
D62 = pd.read_csv ('../data/processed_data/D62csv.csv')
D62_y =  np.squeeze(pd.read_csv ('../data/processed_data/D62_familiescsv.csv'))
D62_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/D62genes_best.csv')))
print(np.shape(D62), np.shape(D62_y))

#Load BIDDY D15 data
D15 = pd.read_csv ('../data/processed_data/D15csv.csv')
D15_y =  np.squeeze(pd.read_csv ('../data/processed_data/D15_familiescsv.csv'))
D15_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/D15genes_best.csv')))
print(np.shape(D15), np.shape(D15_y))

#Load BIDDY D152 data
D152 = pd.read_csv ('../data/processed_data/D152csv.csv')
D152_y =  np.squeeze(pd.read_csv ('../data/processed_data/D152_familiescsv.csv'))
D152_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/D152genes_best.csv')))
print(np.shape(D152), np.shape(D152_y))

#Load Kimmerling CD8 data
CD8 = pd.read_csv ('../data/processed_data/CD8csv.csv')
CD8_y =  np.squeeze(pd.read_csv ('../data/processed_data/CD8_familiescsv.csv'))
CD8_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/CD8genes_best.csv')))
print(np.shape(CD8), np.shape(CD8_y))
      
#Load Kimmerling L1210 data
L1210 = pd.read_csv ('../data/processed_data/L1210csv.csv')
L1210_y =  np.squeeze(pd.read_csv ('../data/processed_data/L1210_familiescsv.csv'))
L1210_op_genes = np.squeeze(np.array(pd.read_csv('../data/optimized_subsets/L1210genes_best.csv')))
print(np.shape(L1210), np.shape(L1210_y))

#Load Weinreb LK data
LK = pd.read_csv ('../data/processed_data/LKcsv.csv')
LK_y =  np.squeeze(pd.read_csv ('../data/processed_data/LK_familiescsv.csv'))
print(np.shape(L1210), np.shape(L1210_y))

#Load Weinreb LSK data
LSK = pd.read_csv ('../data/processed_data/LSKcsv.csv')
LSK_y =  np.squeeze(pd.read_csv ('../data/processed_data/LSK_familiescsv.csv'))
print(np.shape(L1210), np.shape(L1210_y))

#Making list of all the data
data = [AE3,AE4,DO,DO2,D6,D62,D15,D152,CD8,L1210,LK,LSK]
y = [AE3_y,AE4_y,DO_y,DO2_y,D6_y,D62_y,D15_y,D152_y,CD8_y,L1210_y, LK_y,LSK_y]
list_ = [list(AE3_op_genes),list(DO_op_genes),list(DO2_op_genes),list(D6_op_genes),list(D62_op_genes),list(D15_op_genes),list(D152_op_genes),list(CD8_op_genes),list(L1210_op_genes)]
#Name of the data sets
name = ['AE3','AE4', 'DO', 'DO2', 'D6', 'D62','D15', 'D152', 'CD8', 'L1210', 'LK','LSK']

#Evaluate the subset
score_atleast, subset_atleast = evaluate_overlap_genes(list_, data, y, 2)
print('Taking the  genes present in at least 2 individual optimization, we get ' + str(len(subset_atleast)) + ' genes.')

for i in range (len(name)):
    print('The scores on the ' + name[i] + ' data set is ', score_atleast[i])
    
subset_atleast = pd.DataFrame(subset_atleast)    
subset_atleast.to_csv('../data/final_subset/final.csv', index=False)

