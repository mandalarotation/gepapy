from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class FitnessSM0001():
    def __init__(self):
        None

    def _fitnessSM0001(self,X,p,d,w,digits,n_samples):
        def fitnessSMC0001():
            @cuda.jit
            def kernel(X,y,p,d,w):
                row = cuda.grid(1)
                if row < n_samples: 
                    ptime=0
                    tardiness=0
                    for j in range(digits):
                        ptime=ptime+p[row,int(math.ceil(X[row][j]))]
                        tardiness=tardiness+w[row,int(math.ceil(X[row][j]))]*max(ptime-d[row,int(math.ceil(X[row][j]))],0)
                        y[row]=tardiness
                cuda.syncthreads()  

            y = cp.zeros(n_samples,dtype=cp.float32)
            threadsperblock = 16
            blockspergrid_x = int(math.ceil(n_samples / threadsperblock))
            blockspergrid = blockspergrid_x

            cuda.synchronize()
            kernel[blockspergrid, threadsperblock](X,y,p,d,w)
            cuda.synchronize()

            return y
        return fitnessSMC0001()