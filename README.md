# Optimization of a memory gene selection for annotating cell-families in scRNAseq data, a machine learning approach

In the present project, feature selection was performed  to  extract  gene  sets  able  to  predict  cell  families  inscRNAseq  data.  It  has  previously  been  shown  that  some  genes are  more  stably  inherited  over  cell  divisions  then  others.  From data sets  in  which  lineage  tracing  was  combined  with  scRNAseq these can now be assessed widely and in different cell types. Thus, the  identification  of  these  memory  genes  could  lead  the  way  for a  novel  method  of  lineage  tracing.

## Description

Single cell RNA sequencing enables to asses gene expression genome-wide in single cells at high throughput. Different scRNAseq methods exist, among which plate (ex.SMART-seq)and microfluidic based (10X CHormium, InDrop) systems, and all  are  widely  used.  ScRNAseq  profiles  are  a  snapshot  view of  gene  expression  and  a  high  interest  exist  to  add  temporal information  to  such  data.  [1]  One  way  to  do  so  today  is through  assessing  the  relation  of  cells  qua  family  belonging(e.g. information on the part of cells) that can be done through lineage tracing techniques by cellular barcoding of cells before scRNAseq. A unique nucleotide sequence is introduced in each cell.  Since  this  barcode  is  introduced  in  the  genome  of  the cells,  all  progenitor  cells  inherited  this  sequence.  Thus,  cells with the same barcode are identified as a family. This  has  recently  been  implemented  in  BIDDY  [2]  and Weinreb  [3]  papers,  giving  insights  into  reprogramming  and hematopoietic differentiation respectively. The technique however requires the previous isolation, lentiviral transduction andculture  of  cells-and  is  thereby  restricted  to  few,  mainly  invitro, experimental setups. A way to assess lineage relation in scRNAseq data without prior handling would be very valuable. An  analysis  by  Eisele  et  al.  of  scRNAseq  with  annotated family  information  revealed  that  in  different  cell  types  the levels  of  some  genes  are  more  stable  than  of  others  within cell  families.Furthermore  an  overlap of these stable genes was found between different cell types-opening up the possibility to predict cell families using these genes. We will attempt to further optimize this memory gene set. To do so, firstly,  the  set  of  genes was  optimized  on  each  data  set  independently  using  featureselection.  The  aim  was  to  find  a  subset  of  genes  that  work well  on  all  the  data  sets.  Thus,  the  final  subset  of  genes contains all the genes that were considered as a memory gene in at least 2 of the individual optimization.

## Getting Started

### Dependencies

The following libraries were used: 
* Standard python library numpy, matplotlib.pyplot, random, itertools
* scipy [4]
* pandas [5]
* pyreadr [6]
* sklearn [7]
* skfeature-chappers [8]
* skrebate [9]
* sklearn-genetic [10]
* scipy_cut_tree_balanced [11]
* pyHSICLasso [12]
* matplotlib-venn [13]


All of them need to be installed by the user beforehand. This can be done simply by running the following cell in a juypter notebook:
```sh
  pip install *library_name* 
  ```


### Executing program
# To run all the project
1. Install all mentioned libraries
 ```sh
  pip install *library_name* 
  ```
2. Clone the repository
```sh
   git clone https://github.com/CS-433/ml-project-2-ati
   ```
3. Download the data from this address:  Placed it at the same level as the src folder in a folder named "data". Note that to have access to the data the user needs to be connected to the Google services with their epfl address.
4. Run the generate_csv notebook to generate the processed data sets.
5. The optimization of each data set is done in different notebooks, that can be run to generate the optimal genes set files. These files are also provided on the drive as running this notebooks can take some time.
6. Run the overlap notebook to determine the final optimal memory gene set accross data sets.
# Quick overview of the project: run.py
1. Install all mentioned libraries
 ```sh
  pip install *library_name* 
  ```
2. Clone the repo
```sh
   git clone https://github.com/CS-433/ml-project-2-ati
   ```
3. Download the data from this address: https://drive.google.com/drive/folders/1xxONi388bhkJneT1JrNE7K9soaMtorS1?usp=sharing. Placed it at the same level as the src folder in a folder named "data". Note that to have access to the data the user need to be connected to the Google services with their epfl address.
4. Run the generate_csv notebook to generate the processed data sets.
5. Run the run.py file, it produces a file that can be found in data/final_subset which contains the name of the final subset. It also print out the score of this subset on each data set.

## Code Architechture 
All the code can be found in the src folder. 
* It contains a notebook for each of the data sets on which feature selection was performed: AE3.ipynb, AE4.ipynb, D0.ipynb, .... 
The generate_csv notebook generate all the preprocess data that is then used. Whereas, the overlap notebook compute the final overlap subset and evaluate it on all the data sets.
* Filter.py, Wrapper.py, Hybrid.py contain all functions implementing the different feature selection techniques that were used. 
* pred_score.py contains a few functions to evaluate the performance of the clustering. Furthermore, it also contains a class for the clustering of the data into families.
* As its name suggests load_data contains a few functions that help load all the data easily.
* In overlap_genes.py, function to determine the overlap between set of genes were implemented.
* Finally, run.py give a overview of the present project to the reader that would not like to run all the before mentionned notebooks. It gives back the name of the genes present in the final subset.

In the data folder, that the user can dowload following this link: https://drive.google.com/drive/folders/1xxONi388bhkJneT1JrNE7K9soaMtorS1?usp=sharing. One can found multiple files for each data sets.



## Authors

* Alexia Dormann @alexiaDorm 
* Theo Maffei @maffeitheo
* Imane Ben M'Rad @ibenmrad

#References
* [1]  Alex R Lederer and Gioele La Manno.  The emergence and promise ofsingle-cell temporal-omics approaches.Current Opinion in Biotechnol-ogy, 63:70–78, 2020.  Nanobiotechnology   Systems Biology.
* [2]  Kong W. Kamimoto K. et al Biddy, B.A. Single-cell mapping of lineageand identity in direct reprogramming.Nature, 564:219–224, 2018.
* [3]  Camargo  FD  Klein  AM.  Weinreb  C,  Rodriguez-Fraticelli  A.   Lineagetracing on transcriptional landscapes links state to fate during differen-tiation.Science, 2020
* [4] https://github.com/scipy/scipy
* [5] https://github.com/pandas-dev/pandas
* [6] https://github.com/ofajardo/pyreadr
* [7] https://github.com/scikit-learn/scikit-learn
* [8] https://github.com/charliec443/scikit-feature
* [9] https://github.com/EpistasisLab/scikit-rebate
* [10] https://github.com/manuel-calzolari/sklearn-genetic
* [11] https://github.com/vreyespue/scipy_cut_tree_balanced
* [12] https://github.com/riken-aip/pyHSICLasso
* [13] https://github.com/konstantint/matplotlib-venn
