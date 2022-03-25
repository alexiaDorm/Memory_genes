import numpy as np

def upper_tri_indexing(A:np.array):
    """ Return the upper triangle without diagonal.
  
      parameters:
      A: np.array,
        matrix to transform

      returns:
      up: np.array,
        values of the upper triangle without the diagonal
    """
    #If only one cell in a predicted cluster
    if A.shape == ():
        return []
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]
def lower_tri_indexing(A:np.array):
    """ Return the lower triangle without diagonal.
  
      parameters:
      A: np.array,
        matrix to transform

      returns:
      up: np.array,
        values of the lower triangle without the diagonal
    """
    #If only one cell in a predicted cluster
    if A.shape == ():
        return []
    m = A.shape[0]
    r,c = np.tril_indices(m,-1)
    return A[r,c]