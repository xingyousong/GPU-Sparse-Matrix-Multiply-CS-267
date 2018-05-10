import cupy as cp 
import numpy as np
from scipy.sparse import coo_matrix
import scipy
import time
import pandas as pd


import os 
"""

#----- sparseD x sparseD

temp_dict = {}

for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = scipy.sparse.random(N,N, p, "csr")
        x = x.todense()
        y = y.todense()
        
        start = time.time()
        x.dot(y)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cpu_sparseD_sparseD.csv")




#------- sparseD x dense
temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p)
        x = x.todense()
        y = np.random.rand(N,N)
   
        
        start = time.time()
        x.dot(y)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cpu_sparseD_dense.csv")






temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = np.random.rand(N,1)
        x = x.todense()
      
        
        start = time.time()
        x.dot(y)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cpu_sparseD_denseVector.csv")


"""

#-------- sparseCSR x dense vector 

temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = np.random.rand(N,1)
        
        
        start = time.time()
        x.dot(y)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cpu_sparseCSR_denseVector.csv")


#----- sparseCSR x dense 


temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = np.random.rand(N,N)

        start = time.time()
        x.dot(y)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cpu_sparseCSR_dense.csv")




#----- sparseCSR x sparseCSR

temp_dict = {}

for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = scipy.sparse.random(N,N, p, "csr")
  
        start = time.time()
        x.dot(y)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cpu_sparseCSR_sparseCSR.csv")


os.system("shutdown")
