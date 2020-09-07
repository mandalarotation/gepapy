from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class SortA0001():
    def __init__(self):
        None


    def _sortA0001(self,X,y,digits,repetitions,n_samples):
        def sortAC0001():
            @cuda.jit
            def kernel(X,X_AUX,Y_SORT,digits,n_samples):
                row,col = cuda.grid(2)
                if row < n_samples and col < digits: 
                    X_AUX[int(math.ceil(Y_SORT[row,col,row])),col] = X[row,col]

            x_dim_1 = digits*repetitions
            Y_SORT = cp.array(cp.repeat(cp.expand_dims(cp.repeat(cp.expand_dims(cp.argsort(y),axis=0),x_dim_1,axis=0),axis=0),n_samples,axis=0),dtype=cp.float32)
            X_AUX = cp.zeros([n_samples,digits])

            threadsperblock = (16, 16)
            blockspergrid_x = int(math.ceil(n_samples / threadsperblock[0]))
            blockspergrid_y = int(math.ceil(digits / threadsperblock[1]))
            blockspergrid = (blockspergrid_x, blockspergrid_y)


            cuda.synchronize()
            kernel[blockspergrid, threadsperblock](X,X_AUX,Y_SORT,digits,n_samples)
            cuda.synchronize()

            return X_AUX,cp.sort(y)
        return sortAC0001()

    