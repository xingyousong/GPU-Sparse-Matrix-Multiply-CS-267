import cupy as cp 
import numpy as np
from scipy.sparse import coo_matrix
import scipy
import time
import pandas as pd


import os 

#----- sparseCSR x sparseCSR

temp_dict = {}

for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = scipy.sparse.random(N,N, p, "csr")
        x_gpu = cp.sparse.csr_matrix(x)
        y_gpu = cp.sparse.csr_matrix(y)
        
        start = time.time()
        x_gpu.dot(y_gpu)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cublas_sparseCSR_sparseCSR.csv")



#----- sparseD x sparseD

temp_dict = {}

for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = scipy.sparse.random(N,N, p, "csr")
        x = x.todense()
        y = y.todense()
        x_gpu = cp.array(x)
        y_gpu = cp.array(y)
        
        start = time.time()
        x_gpu.dot(y_gpu)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cublas_sparseD_sparseD.csv")



#----- sparseCSR x dense 


temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = np.random.rand(N,N)
        x_gpu = cp.sparse.csr_matrix(x)
        y_gpu = cp.array(y)
        
        start = time.time()
        x_gpu.dot(y_gpu)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cublas_sparseCSR_dense.csv")



#------- sparseD x dense
temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p)
        x = x.todense()
        y = np.random.rand(N,N)
        x_gpu = cp.array(x)
        y_gpu = cp.array(y)
        
        start = time.time()
        x_gpu.dot(y_gpu)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cublas_sparseD_dense.csv")



#-------- sparseCSR x dense vector 

temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = np.random.rand(N,1)
        x_gpu = cp.sparse.csr_matrix(x)
        y_gpu = cp.array(y)
        
        start = time.time()
        x_gpu.dot(y_gpu)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cublas_sparseCSR_denseVector.csv")



#-------- sparseD x dense vector 

temp_dict = {}
for i in range(4, 15):
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: 
        N = pow(2, i)
        x = scipy.sparse.random(N, N, p, "csr")
        y = np.random.rand(N,1)
        x = x.todense()
        x_gpu = cp.array(x)
        y_gpu = cp.array(y)
        
        start = time.time()
        x_gpu.dot(y_gpu)
        end = time.time()
        t = end - start
        
        if N in temp_dict:
            temp_dict[N][p] = t
        else:
            temp_dict[N] = {p: t}
    print(i, t)
        

ssd = pd.DataFrame(data=temp_dict)
ssd.to_csv("cublas_sparseD_denseVector.csv")

os.system("shutdown")