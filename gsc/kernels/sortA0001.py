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

            X_AUX = cp.copy(X)     
            return X_AUX[y.argsort()],cp.sort(y)
        return sortAC0001()

    