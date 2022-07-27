from basic_import import *
from load_data import load_data_thres
from binaryclass_memory import predict_evaluate

def thresholds_method(X:pd.DataFrame, y:np.array):
    
    #Find memory genes by appling threshold
    cond1 = np.logical_and(X['mean_expression'] >= 80, X['CV2ofmeans_residuals'] >= 80)
    cond2 = X['mean_expression'] >= 98
    cond3 = np.logical_and(X['mean_expression'] >= 90, X['CV2ofmeans_residuals'] >= 40)
    cond4 = np.logical_and(X['mean_expression'] >= 60, X['CV2ofmeans_residuals'] >= 90)
    cond = np.logical_or(np.logical_or(np.logical_or(cond1, cond2), cond3), cond4)
    
    memory_gene = np.zeros(y.shape)
    memory_gene[cond] = True
    
    acc = balanced_accuracy_score(y, memory_gene)
    return memory_gene, acc

#Load data
charac_matrix, norm_matrix, families_matrix, names = load_data_thres()

#Evaluate clustering
scores = []
for i in range(0,len(charac_matrix)):
    X = charac_matrix[i][['mean_expression','CV2ofmeans_residuals']]
    y = np.expand_dims(charac_matrix[i]['memory_gene'], axis=1)
    
    yhat, acc = thresholds_method(X,y)
    
    #Compute enrichment of gene set (percentage of memory genes recovered + number of non-memory genes in set)
    recovery, false_pos = compute_enrichment(charac_matrix[i], y, yhat)
    score = [acc, recovery, false_pos]
    
    score.extend(predict_evaluate(charac_matrix[i].index, yhat, norm_matrix[i], families_matrix[i], mult_pred=True))
    scores.append(score)
    
scores_df = pd.DataFrame(scores, index = names, columns= ['accuracy', 'recovery memory gene', 'FP', 'precision', 'recovery', 'ensembling precision', 'ensembling recovery'])
scores_df.to_csv('../data/thresholds.csv', index=True)