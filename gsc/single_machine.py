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
        self._percent_cross = self._set_percent_cross(percent_cross)
        self._percent_mutation = self._set_percent_mutation(percent_mutation)
        self._fitness = None
        self._poulation = self._set_population()


    def set_n_samples(self,n_samples):
        try:
            type(self._population)
            if n_samples < self._n_samples.shape[0]:
                self._n_samples = self._poulation[0:n_samples]
            else:
                None
        except NameError:
            return n_samples


    def _set_n_machines(self,n_machines):
        try:
            type(self._poulation)
        except NameError:
            return n_machines

    def _set_processing_time(self,processing_time):
        try:
            type(self._poulation)
        except NameError:
            return processing_time

    def _set_due_date(self,due_date):
        try:
            type(self._poulation)
        except NameError:
            return due_date
    
    def _set_weights(self,weights):
        try:
            type(self._poulation)
        except NameError:
            return weights
    
    def _set_population(self):
        try:
            type(self._poulation)
        except NameError:
            return None

    def _set_percent_cross(self,percent_cross):
        try:
            type(self._poulation)
        except NameError:
            return percent_cross

    def _set_percent_mutation(self,percent_mutation):
        try:
            type(self._poulation)
        except NameError:
            return percent_mutation
    
    def _set_fitness(self,fitness):
        try:
            type(self._poulation)
        except NameError:
            return fitness

    def get_permutationA0001(self):
        return self._permutationA0001(self._n_machines,1,self._n_samples)
    
    def get_crossA0001(self):
        None

    def get_mutation(self):
        None