from basic_import import *
from load_data import open_charac, normalize, remove_extreme_values
from pred_score import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import optuna
from torch.optim import SGD
from torch.utils.data import random_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

#Visualization functions
#--------------------------------------------------------------------------------------------------------------------------
def visualize_charac(data:pd.DataFrame):
    copy = data
    #Remove the outliers of the data before normalization of mean expression
    data, outliers = remove_extreme_values(data, k=200)
    mem_genes_perc_outliers = 100 *np.sum(copy.loc[outliers]['memory_gene'])/len(outliers)
    print(mem_genes_perc_outliers, '% of the outliers are memory genes')
    
    #Shift the skeness values by substracting them with their mean
    corrected_skewRes = data['CV2ofmeans_residuals']
    #corrected_skewRes -= np.mean(corrected_skewRes)
    
    #Look at skenness and mean expression of all genes
    colors = ['grey','red']
    plt.scatter(corrected_skewRes, data['mean_expression'], marker='o', c= data['memory_gene'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel("CV2 of mean residuals")
    plt.ylabel("mean expression")
    #plt.xlim(0,1)
    plt.yscale('log')
    plt.title("All genes")
    plt.show()

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
    test_dl = DataLoader(test, batch_size=256, shuffle=True)
    
    return train_dl, test_dl

def load_data_val (fused:pd.DataFrame, params):
    X = np.array(fused.drop(columns=['memory_gene']))
    y = np.expand_dims((fused['memory_gene']*1), axis=1)
    dataset = Dataset(np.array(X), y)

    train_dl = DataLoader(dataset, batch_size = 256, shuffle=True)
    
    return train_dl


def load_charac():
    #Load data
    names = ['AE3', 'AE4', 'AE7', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2']

    charac_matrix = []
    for name in names:
        #Open characteristics file
        charac_out_path = '../data/Characteristics_masterfiles/Dataset_specific_characteristics/' + name + '__characteristics_output.txt'
        p_value_path = '../data/Characteristics_masterfiles/Memory_genes/P_value_estimate_CV2_ofmeans_' + name + '.txt'
        charac_matrix.append(open_charac(charac_out_path, p_value_path, 200))

    #Only keep mean_exp + Cv2 residual
    for i in range(0,len(charac_matrix)):
        charac_matrix[i] = charac_matrix[i][['mean_expression','CV2ofmeans_residuals', 'memory_gene']]
        charac_matrix[i] = charac_matrix[i].dropna()

    #Remove AE7, also keep BIDDY D15_2 for validation
    val = [8]
    data_to_fuse = [0,1,3,4,5,6,7]

    outliers = []
    for i in range(0,len(charac_matrix)):
        #Normalize mean expression, same for mean_expression after removing outliers
        charac_matrix[i], outlier_temp = remove_extreme_values(charac_matrix[i], k=200)
        outliers.append(outlier_temp)
        charac_matrix[i]['CV2ofmeans_residuals'] = normalize(charac_matrix[i]['CV2ofmeans_residuals'])
        charac_matrix[i]['mean_expression'] =  normalize(charac_matrix[i]['mean_expression'])

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
        super(NN_1l, self).__init__()
        self.layer = nn.Linear(n_inputs, 1)
        nn.init.xavier_uniform_(self.layer.weight)
 
    def forward(self, X):
        X = self.layer(X)
        return X
    
class NN_2l(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_2l, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']),
        nn.ReLU(),
        nn.Linear(params['n1'], 1)
        )
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_2lRBN(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_2lRBN, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']), 
        nn.ReLU(),
        nn.BatchNorm1d(params['n1']),
        nn.Linear(params['n2'], 1)
        )
        
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_3l(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_3l, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']), 
        nn.ReLU(),
        nn.Linear(params['n1'], params['n2']),
        nn.ReLU(),
        nn.Linear(params['n2'], 1)
        )
        
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_4l(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_4l, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']) ,
        nn.ReLU(),
        nn.Linear(params['n1'], params['n2']),
        nn.ReLU(),
        nn.Linear(params['n2'], params['n3']),
        nn.ReLU(),
        nn.Linear(params['n3'], 1)
        )
        
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_5l(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_5l, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']), 
        nn.ReLU(),
        nn.Linear(params['n1'], params['n2']),
        nn.ReLU(),
        nn.Linear(params['n2'], params['n3']), 
        nn.ReLU(),
        nn.Linear(params['n3'], params['n4']), 
        nn.ReLU(),
        nn.Linear(params['n4'], 1)
        )
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_3lBN(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_3lBN, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']), 
        nn.BatchNorm1d(params['n1']),
        nn.ReLU(),
        nn.Linear(params['n1'], params['n2']),
        nn.BatchNorm1d(params['n2']),
        nn.ReLU(),
        nn.Linear(params['n2'], 1)
        )
        
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_3lRBN(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_3lRBN, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']), 
        nn.ReLU(),
        nn.BatchNorm1d(params['n1']),
        nn.Linear(params['n1'], params['n2']),
        nn.ReLU(),
        nn.BatchNorm1d(params['n2']),
        nn.Linear(params['n2'], 1)
        )
        
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_4lBN(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_4lBN, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']), 
        nn.BatchNorm1d(params['n1']),
        nn.ReLU(),
        nn.Linear(params['n1'], params['n2']),
        nn.BatchNorm1d(params['n2']),
        nn.ReLU(),
        nn.Linear(params['n2'], params['n3']),
        nn.BatchNorm1d(params['n3']),
        nn.ReLU(),
        nn.Linear(params['n3'], 1)
        )
        
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
class NN_4lRBN(nn.Module):
    def __init__(self, n_inputs, params=None):
        super(NN_4lRBN, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(n_inputs, params['n1']), 
        nn.ReLU(),
        nn.BatchNorm1d(params['n1']),
        nn.Linear(params['n1'], params['n2']),
        nn.ReLU(),
        nn.BatchNorm1d(params['n2']),
        nn.Linear(params['n2'], params['n3']),
        nn.ReLU(),
        nn.BatchNorm1d(params['n3']),
        nn.Linear(params['n3'], 1)
        )
        
 
    def forward(self, X):
        X = self.layers(X)
        
        return X
    
def train_model(train_dl, model, criterion, optimizer):
    for epoch in range(70):
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
    inputs = torch.from_numpy(inputs); inputs = inputs.to(torch.float32)
    yhat = torch.sigmoid(model(inputs))
    yhat[yhat >= 0.5] = 1; yhat[yhat < 0.5] = 0
    
    return np.squeeze(yhat.detach().numpy())

def obj(trial, fused):
    #Set hyperparamters to tune
    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 0.001),
              'weight_decay' : trial.suggest_loguniform('weight_decay', 1e-10, 1e-5),
              'n1': trial.suggest_int("n1", 20, 50),
              'n2' : trial.suggest_int("n2", 20, 50), 
              'n3' : trial.suggest_int("n3", 15, 40),
              }
    #Load data
    train_dl, test_dl = load_data(fused,params)

    model = NN_2l(2, params)

    #Optmization loss and optimizer
    pos_weight = torch.as_tensor(4., dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

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
    
def train_best_model(fused, params):
    #Load data
    train_dl= load_data_val(fused,params)
    
    model = NN_2l(2, params)

    #Optmization criterion and optimizer
    pos_weight = torch.as_tensor(4., dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=params['weight_decay'])

    #Train and evaluate the NN
    train_model(train_dl, model, criterion, optimizer)
    
    return model

def feature_selection(X,y,params):
    #Get the N top features according to mutual information
    selector = SelectKBest(mutual_info_classif, k=params['nb_features'])
    X_redu = selector.fit_transform(X, y)
    cols = selector.get_support(indices=True)
    FS = X.iloc[:,cols].columns.tolist();FS.append('CV2ofmeans_residuals'); FS = np.unique(FS)
    
    return FS

def compute_enrichment(charac, y, yhat):
    
    non_memory_gene = list(charac[y == False].index)
    memory_gene = list(charac[y == True].index)
    yhat = pd.DataFrame(yhat, index = charac.index, columns = ['pred'])

    y_non_mem = yhat.loc[non_memory_gene]
    y_mem = yhat.loc[memory_gene]
    recovery = np.sum(y_mem['pred'])/len(memory_gene)
    false_pos = np.sum(y_non_mem['pred'])
    
    return recovery, false_pos

def predict_evaluate(genes:list, yhat:np.array, norm:pd.DataFrame, family:np.array, mult_pred:bool = False, outliers:list = []):
    
    y = pd.DataFrame(yhat, index = genes, columns = ['pred'])
    
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