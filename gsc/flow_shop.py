from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

from gsc.kernels.permutationA0001 import PermutationA0001
from gsc.kernels.crossA0001 import CrossA0001
from gsc.kernels.mutationA0001 import MutationA0001

class Flow_Shop(PermutationA0001,CrossA0001,MutationA0001):
    def __init__(self):
        self.n_samples = None
        self.n_machines = None
        
        None

    def get_permutationA0001(self):
        None
    
    def get_crossA0001(self):
        None

    def get_mutation(self):
        None