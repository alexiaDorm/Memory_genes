from basic_import import *
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

def read_charac_output(file_path:str):
    #Read .txt file 
    lines = []
    with open(file_path) as f:
        lines = f.readlines()

    #Get the name of the column on first line
    colnames = lines[0].split("\t")
    colnames[-1] = colnames[-1][0:-1]
    colnames = [name.strip('"') for name in colnames]
    #Get data in rest of line
    for i in range (1,len(lines)):
            
        lines[i] = lines[i].split("\t")
        #Remove double quote around gene name
        lines[i][1] = lines[i][1].strip('"')
        #Remplace 'NA' by float np.NAN
        ind = np.squeeze(np.where([(i == 'NA' or i == 'NA\n') for i in lines[i]]))
        if ind.size == 1:
            lines[i][ind] = np.NAN
        elif ind.size > 1:
            for j in ind:
                lines[i][j] = np.NAN
                
        lines[i][2:8] = [float(i) for i in lines[i][2:8]]
        lines[i] = lines[i][1:8]
    
    #Define a pandas DataFrame with all the data
    data = pd.DataFrame(lines[1:],columns=colnames)
    data = data.set_index(colnames[0])
    
    return data

def read_memory_genes(file_path:str):
    #Read .txt file 
    lines = []
    with open(file_path) as f:
        lines = f.readlines()

    #Get data in rest of line
    NA_val = 'NA\n'
    for i in range (1,len(lines)):
        lines[i] = lines[i].split("\t")

        #Remplace 'NA' by float np.NAN
        ind = np.squeeze(np.where([i == NA_val for i in lines[i]]))
        if ind.size > 0:
            lines[i][ind] = np.NAN
        
        lines[i][1] = float(lines[i][1])
    
    #Define a pandas DataFrame with all the data
    data = pd.DataFrame(lines[1:])
    data = data.set_index(0)
    
    return data

def get_memory_genes (memory_pvalue:pd.DataFrame):
    
    return list(memory_pvalue.index[np.where(memory_pvalue <= 0.05)[0]])

def add_general_charac(charac:pd.DataFrame, general_charac:pd.DataFrame):
    genes_interest = (set(charac.index)).intersection(list(general_charac.index))
    general = general_charac.loc[genes_interest]
    general = general.drop(columns=['gene_ID', 'transcript_ID', 'name', 'chr', 'start', 'end', 'TSS', 'strand'])
    
    #Encode features as binary
    TF = general.drop(columns=['length','GC'])
    TF =  np.where(TF > 0, 1, 0)
    TF = pd.DataFrame(TF, index=general.index, columns= (general.drop(columns=['length','GC'])).columns)
    
    general = general[['length', 'GC']]
    
    charac = pd.concat([charac,general,TF], axis=1)
   
    return charac

def open_charac(charac_output_path:str, p_value_path:str, k:float):
    
    #Load data
    charac = read_charac_output(charac_output_path)
    memory = read_memory_genes(p_value_path)
    memory_genes = get_memory_genes(memory) 
        
    #Add to characterectics matrix if gene is a memory gene as bool
    memory_bin = np.zeros((len(charac),))
    memory_bin[np.where([gene in memory_genes for gene in list(charac.index)])] = 1
    memory_bin = [bool(mem) for mem in memory_bin]
    
    charac['memory_gene'] = memory_bin
    
    return charac

def normalize(data:np.array):
    #Normlize into 0 to 1 range
    return ((data - min(data))/ (max(data) - min(data)))

def remove_extreme_values(data:pd.DataFrame, k:float=3):
    '''Remove all extreme values of gene expression using Interquartile Range Method. Return the name of the gene with extreme values.
    
    parameters:
    ---------------
    data:pd.DataFrame,
        characteristic matrix NxM with N, number of genes and M, the number of characteristics
    k:float,
        constant to determine the cut-off of values to remove.
        
    return:
    -------------
    outliers_removed:pd.DataFrame,
        data without outliers 
    outliers_gene: list,
        list of the gene with extreme mean expression values
    '''
    q25, q75 = np.percentile(data["mean_expression"], 25), np.percentile(data["mean_expression"], 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    
    # identify outliers
    outliers =  list(data.iloc[np.where([x>upper for x in data["mean_expression"]])[0]].index)
    not_outliers = np.where([x<upper for x in data["mean_expression"]])[0]
    # remove outliers
    data = data.iloc[not_outliers]
                                                                        
    return data, outliers

def compute_quantile(values:np.array):
    
    quantile = np.quantile(values[values != 0], q= np.linspace(start=0, stop=1, num=101))
    val_quantile = pd.cut(x=values, bins=quantile, labels= np.arange(1,101,1), include_lowest=False)
    
    return np.array(val_quantile)

def load_all_data():

    names = ['AE3', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2', 'CD8', 'L1210', 'LK_D2_exp1_library_d2_1', 'LK_D2_exp1_library_d2_2', 'LK_D2_exp1_library_d2_3', 'LK_LSK_D2_exp3_library_d2_1', 'LK_LSK_D2_exp3_library_d2_2', 'LK_LSK_D2_exp3_library_d2_3', 'LK_LSK_D2_exp3_library_d2_4', 'LK_LSK_D2_exp3_library_d2_5', 'LSK_D2_exp1_library_LSK_d2_1', 'LSK_D2_exp1_library_LSK_d2_2', 'LSK_D2_exp1_library_LSK_d2_3','LSK_D2_exp2_library_d2A_1', 'LSK_D2_exp2_library_d2A_2', 'LSK_D2_exp2_library_d2A_3' , 'LSK_D2_exp2_library_d2A_4', 'LSK_D2_exp2_library_d2A_5','LSK_D2_exp2_library_d2B_1','LSK_D2_exp2_library_d2B_2', 'LSK_D2_exp2_library_d2B_3', 'LSK_D2_exp2_library_d2B_4','LSK_D2_exp2_library_d2B_5', 'Hamange1', 'Hamange2', 'Hamange3', 'Hamange4', 'Hamange5', 'Hamange6', 'Hamange7', 'Hamange8', 'Wehling1', 'Wehling2']
    
    val = np.arange(6,len(names),1)
    data_to_fuse = np.arange(0,6,1)
    
    charac_matrix = []
    norm_matrix = []
    families_matrix = []
    
    #Load charac for each dataset (load memory genes only for training set)
    for i in range(0,len(names)):
        #Open characteristics file
        charac_out_path = '../data/Characteristics_masterfiles/Dataset_specific_characteristics/' + names[i] + '__characteristics_output.txt'
        p_value_path = '../data/Characteristics_masterfiles/Memory_genes/P_value_estimate_CV2_ofmeans_' + names[i] + '.txt'
        charac_matrix.append(open_charac(charac_out_path, p_value_path, 200))
        
    '''for i in val:
        #Open characteristics file
        charac_out_path = '../data/Characteristics_masterfiles/Dataset_specific_characteristics/' + names[i] + '__characteristics_output.txt'
        charac_matrix.append(read_charac_output(charac_out_path))
        charac_matrix[i]['memory_gene'] = np.zeros(charac_matrix[i]['mean_expression'].shape)'''
        
    for name in names:
        #Open normalized data
        norm_path = '../data/merged_data/' + name + '.csv'
        fam_path = '../data/merged_data/y_' + name + '.csv'
        norm = pd.read_csv (norm_path)
        norm = norm.set_index('Unnamed: 0')
        families= np.squeeze(np.array(pd.read_csv(fam_path)))

        norm_matrix.append(norm)
        families_matrix.append(families)

    #Only keep mean_exp + Cv2 residual
    for i in range(0,len(charac_matrix)):
        charac_matrix[i] = charac_matrix[i][['mean_expression','CV2ofmeans_residuals', 'memory_gene']]
        charac_matrix[i] = charac_matrix[i].dropna()

    outliers = []
    for i in range(0,len(charac_matrix)):
        #Normalize skew_residuals, same for mean_expression after removing outliers
        charac_matrix[i], outlier_temp = remove_extreme_values(charac_matrix[i], k=200)
        outliers.append(outlier_temp)
        
        charac_matrix[i]['CV2ofmeans_residuals'], charac_matrix[i]['mean_expression'] = normalize(charac_matrix[i]['CV2ofmeans_residuals']), normalize(charac_matrix[i]['mean_expression'])

    val_charac =  []
    names_val = []
    for i in val:
        val_charac.append(charac_matrix[i])
        names_val.append(names[i])

    fused_charac = []
    names_fused = []
    for i in data_to_fuse:
        fused_charac.append(charac_matrix[i])
        names_fused.append(names[i])

    fused = pd.concat(fused_charac)
    
    return fused, charac_matrix, norm_matrix, families_matrix, names_val, names_fused, data_to_fuse, val, outliers

def load_data_thres():

    names = ['AE3', 'BIDDY_D0', 'BIDDY_D0_2', 'BIDDY_D6', 'BIDDY_D6_2', 'BIDDY_D15', 'BIDDY_D15_2', 'CD8', 'L1210', 'LK_D2_exp1_library_d2_1', 'LK_D2_exp1_library_d2_2', 'LK_D2_exp1_library_d2_3', 'LK_LSK_D2_exp3_library_d2_1', 'LK_LSK_D2_exp3_library_d2_2', 'LK_LSK_D2_exp3_library_d2_3', 'LK_LSK_D2_exp3_library_d2_4', 'LK_LSK_D2_exp3_library_d2_5', 'LSK_D2_exp1_library_LSK_d2_1', 'LSK_D2_exp1_library_LSK_d2_2', 'LSK_D2_exp1_library_LSK_d2_3','LSK_D2_exp2_library_d2A_1', 'LSK_D2_exp2_library_d2A_2', 'LSK_D2_exp2_library_d2A_3' , 'LSK_D2_exp2_library_d2A_4', 'LSK_D2_exp2_library_d2A_5','LSK_D2_exp2_library_d2B_1','LSK_D2_exp2_library_d2B_2', 'LSK_D2_exp2_library_d2B_3', 'LSK_D2_exp2_library_d2B_4','LSK_D2_exp2_library_d2B_5', 'Hamange1', 'Hamange2', 'Hamange3', 'Hamange4', 'Hamange5', 'Hamange6', 'Hamange7', 'Hamange8', 'Wehling1', 'Wehling2']
    
    charac_matrix = []
    norm_matrix = []
    families_matrix = []
    #Load charac for each dataset (load memory genes only for training set)
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

    for i in range(0,len(charac_matrix)):
        #Only keep mean expression + CV2of means + memory gene status, remove gene with nan values
        charac_matrix[i] = charac_matrix[i][['mean_expression','CV2ofmeans_residuals', 'memory_gene']]
        charac_matrix[i] = charac_matrix[i].dropna()
        
        #Compute quantile of mean expression and CV2of mean residuals
        charac_matrix[i]['mean_expression'] = compute_quantile(charac_matrix[i]['mean_expression'])
        charac_matrix[i]['CV2ofmeans_residuals'] = compute_quantile(charac_matrix[i]['CV2ofmeans_residuals'])
        charac_matrix[i] = charac_matrix[i].dropna()
    
    return charac_matrix, norm_matrix, families_matrix, names
     