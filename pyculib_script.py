from pyculib.sparse import Sparse
import pyculib.sparse as sp
import numpy as np
import scipy
import time

import pandas as pd 
import os 


# sparseCSR x dense
temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        
        s = sp.Sparse()
        a_desc = s.matdescr(indexbase=0,matrixtype='G')
        
        
        a = scipy.sparse.random(N, N, p, "csr")
        ap = sp.csr_matrix(a)
        
        transA = 'N'
        transB = transA
        b = np.random.rand(N, N).astype(np.float64)
        #y = scipy.sparse.random(N,N, p, "csr")
        c = np.zeros((N,N))
        d = c[:]
        

        start = time.time()

        sp.Sparse.csrmm2(s, transA, transB, N, N, N, len(a.data), 1.0, a_desc, ap.data, ap.indptr, ap.indices, b, N, 1.0, c, N)

        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        
ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cusparse_sparseCSR_dense.csv")




# sparseCSR x denseVector
temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        
        s = sp.Sparse()
        a_desc = s.matdescr(indexbase=0,matrixtype='G')
        
        
        a = scipy.sparse.random(N, N, p, "csr")
        ap = sp.csr_matrix(a)
        
        transA = 'N'
    
        b = np.random.rand(N).astype(np.float64)
    
        

        start = time.time()

        
        sp.Sparse.csrmv(s, transA, N, N, len(a.data), 1.0, a_desc, ap.data, ap.indptr, ap.indices, b, 0.0, b)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        
ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cusparse_sparseCSR_denseVector.csv")




os.system("shutdown")