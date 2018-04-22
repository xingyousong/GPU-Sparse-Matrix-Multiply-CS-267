# Block-Sparse-GPU-CS267-Project
Testing, Analysis and Applications of OpenAI's GPU sparse block kernel: https://blog.openai.com/block-sparse-gpu-kernels/


# Links:

#### Laplacian: https://sites.google.com/a/yale.edu/laplacian/#TOC-Code
#### SpGEMM: https://github.com/bhSPARSE/Benchmark_SpGEMM_using_CSR
#### CuSPARSE: https://developer.nvidia.com/cusparse
#### Block-Sparse-GPU: https://github.com/openai/blocksparse
#### CUSP: https://github.com/cusplibrary/cusplibrary

# Python Wrappers:

#### cuSPARSE - pyculib: https://github.com/numba/pyculib
#### cuBLAS - cupy: https://github.com/cupy/cupy

## Failed Wrappers:
#### cuSPARSE - julia: https://github.com/JuliaGPU/CUSPARSE.jl
#### cuSPARSE - skcuda: https://github.com/lebedov/scikit-cuda/tree/master/skcuda



# Old Ideas:

#### TorchMPI: https://github.com/facebookresearch/TorchMPI



# Instructions:

For each i in [4,..., 14], let N = 2^i.
Let p range from [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
For each N x N matrix that is SPARSE, generate it using Erdos Reyni (each element is non-zero and random with probability p). 

Perform (sparse) matrix- (dense)matrix multiplication or (sparse) matrix vector multiplication, or (sparse) matrix -(sparse) matrix multiplication with square matrices of size N x N. 

Do this for CuSPARSE, CUSP, CuBLAS. 

