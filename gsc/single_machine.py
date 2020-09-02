from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

from gsc.kernels.permutationA0001 import PermutationA0001
from gsc.kernels.crossA0001 import CrossA0001
from gsc.kernels.mutationA0001 import MutationA0001

class Single_Machine(PermutationA0001,CrossA0001,MutationA0001):
    def __init__(self,n_samples,n_machines,processing_time,due_date,weights,percent_cross,percent_mutation):
        self._n_samples = self.set_n_samples(n_samples)
        self._n_machines = self._set_n_machines(n_machines)
        self._processing_time = self._set_processing_time(processing_time)
        self._due_date = self._set_due_date(due_date)
        self._weights = self._set_weights(weights)
        self._poulation = self._set_population()
        self._percent_cross = self._set_percent_cross(percent_cross)
        self._percent_mutation = self._set_percent_mutation(percent_mutation)
        self._fitness = None


    def set_n_samples(self,n_samples):
        if self._poulation == None:
            return n_samples
        else:
            if n_samples < self._n_samples.shape[0]:
                self._n_samples = self._n_samples[0:n_samples]
            else:
                None

    def _set_n_machines(self,n_machines):
        if self._poulation == None:
            return n_machines
        else:
            None

    def _set_processing_time(self,processing_time):
        if self._poulation == None:
            return processing_time
        else:
            None

    def _set_due_date(self,due_date):
        if self._poulation == None:
            return due_date
        else:
            None
    
    def _set_weights(self,weights):
        if self._poulation == None:
            return weights
        else:
            None
    
    def _set_population(self):
        if self._poulation == None:
            return self.get_permutationA0001()
        else:
            None

    def _set_percent_cross(self,percent_cross):
        if self._poulation != None:
            return percent_cross
        else:
            None

    def _set_percent_mutation(self,percent_mutation):
        if self._poulation == None:
            return percent_mutation
        else:
            None
    
    def _set_fitness(self,fitness):
        if self._poulation == None:
            return fitness
        else:
            None

    def get_permutationA0001(self):
        return self._permutationA0001(self._n_machines,1,self._n_samples)
    
    def get_crossA0001(self):
        None

    def get_mutation(self):
        None