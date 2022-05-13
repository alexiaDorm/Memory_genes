import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def compute_CAI(sequence:str, codon_freq:pd.DataFrame):
    """Compute the Codon Adaptation Index (CAI) of a given protein/mRNA sequence
  
      parameters:
      sequence: str,
          sequence of the protein/mRNA transcript of interest
      codon_freq: pd.DataFrame,
          frequencies of each codon and corresponding amino acid (species specific)

      returns:s
      CAI : foat,
          computed Codon adaptation index (CAI), returned values are between 0 and 1"""
    #Convert T to U
    sequence = sequence.translate(str.maketrans("T", "U"))
    
    L = int(len(sequence)/3)
    CAI = 0
    for i in range(0,L):
        codon = sequence[3*i:3*i+3]
        if 'N' in codon:
            continue #ignore the codon containing a N (uncertain nucleotide)
        freq = codon_freq.T[codon][1]
        aa = codon_freq.T[codon][0]
        freq_max = np.max((codon_freq[codon_freq['amino acid']==aa])['freq'])
        
        w = np.log(freq/freq_max)
        CAI += w
        
    CAI = np.exp(1/L * CAI)
    
    return CAI

def read_fasta(fasta_file):
    "Parse fasta file"
    count = 0
    headers = []
    sequences = []
    aux = []
    with open(fasta_file, 'r') as infile:
        for line in infile:
            record = line.rstrip()
            if record and record[0] == '>':
                headers.append(record[1:])
                if count > 0:
                    sequences.append(''.join(aux))
                    aux = []
            else:
                aux.append(record)
            count += 1

    sequences.append(''.join(aux))
    return headers, sequences 

#Load data
codon_freq = pd.read_csv('../data/mouse_codon_freq.csv')
codon_freq =codon_freq.set_index('Unnamed: 0')

#Load all the coding sequence from the file
genes, cds = read_fasta('../data/mart_export.txt')
for i, gene in enumerate(genes):
    genes[i] = (gene).split('|')[0]
genes = pd.DataFrame(cds, index = genes, columns = ['cds'])
genes = genes[~genes.index.duplicated()]

#Only keep the one that are in the AE3/AE4 datasets
AE3 = pd.read_csv ('../data/merged_data/AE3.csv')
AE3 = AE3.set_index('Unnamed: 0')
genes = genes.loc[np.unique(AE3.index.intersection(genes.index))]

#Remove not defined sequence genes
error = genes.iloc[3][0]
genes = genes[genes.cds != error]

#Compute the CAI for each gene
genes['CAI'] = genes.apply(lambda x: compute_CAI(x['cds'], codon_freq), axis=1)
genes.to_csv('../data/CAI.csv', index=True)