import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pyreadr

from load_data import open_charac, add_general_charac
from binaryclass_memory import *

#Load data
#---------------------------------------------------
general_charac = pyreadr.read_r('../data/Characteristics_masterfiles/General_characteristics/EPFL_gene_master_matrix.RData')['gene_master_matrix']

names = ['AE3', 'AE4', 'AE7', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2',
        'LK_D2_exp1_library_d2_1', 'LK_D2_exp1_library_d2_2', 'LK_D2_exp1_library_d2_3', 'LK_LSK_D2_exp3_library_d2_1', 
        'LK_LSK_D2_exp3_library_d2_2', 'LK_LSK_D2_exp3_library_d2_3', 'LK_LSK_D2_exp3_library_d2_4', 
        'LK_LSK_D2_exp3_library_d2_5', 'LSK_D2_exp1_library_LSK_d2_1', 'LSK_D2_exp1_library_LSK_d2_2', 'LSK_D2_exp1_library_LSK_d2_3',
       'LSK_D2_exp2_library_d2A_1', 'LSK_D2_exp2_library_d2A_2', 'LSK_D2_exp2_library_d2A_3' , 'LSK_D2_exp2_library_d2A_4', 'LSK_D2_exp2_library_d2A_5', 
       'LSK_D2_exp2_library_d2B_1','LSK_D2_exp2_library_d2B_2', 'LSK_D2_exp2_library_d2B_3', 'LSK_D2_exp2_library_d2B_4', 'LSK_D2_exp2_library_d2B_5']
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
    
# Train and evaluate the neural network model with CV
def train_and_evaluate(param, model):
    
    #Configuration of kfold and NN optimization 
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    num_epochs = 100
    results = [] #store fold result

    #Load data into torch.Dataset
    data = charac_matrix[0]
    labels = np.expand_dims((data['memory_gene']*1), axis=1)
    X = data.drop(['memory_gene'], axis=1)
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
        num_positives= np.sum(dataset.labels)
        num_negatives = len(dataset.labels) - num_positives
        pos_weight  = torch.as_tensor(num_negatives / num_positives, dtype=torch.float)
        loss_function = nn.BCEWithLogitsLoss(pos_weight = pos_weight) 

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
            accuracy = []
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)

                #Compute accuracy on testing fold
                accuracy.append(compute_scores(network, inputs, targets))

            # Store memory genes recovery
            results.append((100*np.mean(accuracy, axis = 0)[1]))
            mean_recovery = np.mean(results)#return the average memory genes recovery across folds

    return mean_recovery

torch.manual_seed(1)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)