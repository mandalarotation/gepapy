from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

from gsc.kernels.permutationA0001 import PermutationA0001
from gsc.kernels.crossA0001 import CrossA0001
from gsc.kernels.mutationA0001 import MutationA0001

class Single_Machine(PermutationA0001,CrossA0001,MutationA0001):
    def __init__(self,n_samples,n_machines,processing_time,due_date,weights,percent_cross,percent_intra_cross,percent_mutation,percent_intra_mutation,percent_migration,percent_selection):
        self._n_samples = self.set_n_samples(n_samples)
        self._n_machines = self._set_n_machines(n_machines)
        self._processing_time = self._set_processing_time(processing_time)
        self._due_date = self._set_due_date(due_date)
        self._weights = self._set_weights(weights)        
        self._percent_cross = self.set_percent_cross(percent_cross)
        self._percent_intra_cross = self.set_percent_intra_cross(percent_intra_cross)
        self._percent_mutation = self.set_percent_mutation(percent_mutation)
        self._percent_intra_mutation = self.set_percent_intra_mutation(percent_intra_mutation)
        self._percent_migration = self.set_percent_migration(percent_migration)
        self._percent_selection = self.set_percent_selection(percent_selection)
        self._fitness = None
        self._population = None
        self._population = self._set_population()


    def set_n_samples(self,n_samples):
        return n_samples

    def _set_n_machines(self,n_machines):
        return n_machines

    def _set_processing_time(self,processing_time):
        return processing_time

    def _set_due_date(self,due_date):
        return due_date
    
    def _set_weights(self,weights):
        return weights
    
    def _set_population(self):
        return self.get_permutationA0001()

    def set_percent_cross(self,percent_cross):
        return percent_cross

    def set_percent_intra_cross(self,percent_intra_cros):
        return percent_intra_cros

    def set_percent_mutation(self,percent_mutation):
        return percent_mutation
    
    def set_percent_intra_mutation(self,percent_intra_mutation):
        return percent_intra_mutation
    
    def set_percent_migration(self,percent_selection):
        return percent_selection
    
    def set_percent_selection(self,percent_migration):
        return percent_migration

    def _set_fitness(self,fitness):
        return fitness

    def _set_population(self):
        if self._population == None:
            return self.exec_permutationA0001()
        else:
            self._population = None

    def get_n_samples(self):
        return self._n_samples 

    def get_n_machines(self):
        self._n_machines

    def get_processing_time(self):
        self._processing_time

    def get_due_date(self):
        self._due_date
    
    def get_weights(self):
        self._weights

    def get_percent_cross(self):
        self._percent_cross

    def get_percent_intra_cross(self):
        self._percent_intra_cross

    def get_percent_mutation(self):
        self._percent_mutation

    def get_percent_intra_mutation(self):
        self.get_percent_intra_mutation

    def get_percent_migration(self):
        self._percent_migration

    def get_fitness(self):
        self._fitness

    def get_population(self): 
        self._population



    def exec_permutationA0001(self):
        return self._permutationA0001(self._n_machines,1,self._n_samples)
    
    def exec_crossA0001(self):
        x_population = cp.copy(self.get_population())
        cp.random.shuffle(x_population)
        index_cross = x_population.shape[0]*self._percent_cross
        if index_cross%2 == 0:
            y_population = _crossA0001(self,x_population[0:index_cross,:],self._n_machines,1,self._n_samples, self._percent_intra_cros)
            x_population[0:index_cross,:] = y_population
            self._population = x_population
        else: 
            index_cros -= 1
        

    def exec_mutation(self):
        None