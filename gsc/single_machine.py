from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

from gsc.kernels.permutationA0001 import PermutationA0001
from gsc.kernels.crossA0001 import CrossA0001
from gsc.kernels.mutationA0001 import MutationA0001

class Single_Machine(PermutationA0001,CrossA0001,MutationA0001):
    def __init__(self,n_samples,n_machines,processing_time,due_date,weigths):
        self._n_samples = None
        self._n_machines = None
        self._processing_time = None
        self._self.due_date = None
        self._weights = None
        self._poulation = None
        self._percent_cross = None
        self._percent_mutation = None
        self._fitness = None


    def set_n_samples(self):
        None

    def _set_n_machines(self):
        None

    def _set_processing_time(self):
        None

    def _set_due_date(self):
        None
    
    def _set_weights(self):
        None
    
    def _set_population(self):
        None

    def _set_percent_cross(self):
        None

    def _set_percent_mutation(self):
        None
    
    def _set_fitness(self):
        None

    def get_permutationA0001(self):
        None
    
    def get_crossA0001(self):
        None

    def get_mutation(self):
        None