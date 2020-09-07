from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

from gsc.kernels.permutationA0001 import PermutationA0001
from gsc.kernels.crossA0001 import CrossA0001
from gsc.kernels.mutationA0001 import MutationA0001
from gsc.kernels.sortA0001 import SortA0001

class Operations(PermutationA0001,CrossA0001,MutationA0001,SortA0001):

    def __init__(self):
        None

    def set_n_samples(self,n_samples):
        return n_samples

    def _set_n_jobs(self,n_jobs):
        return n_jobs

    def _set_n_operations(self,n_operations):
        return n_operations

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

    def _set_population(self,population=None):
        if self._population == None:
            return self.exec_permutationA0001()
        else:
            self._population = population

    def get_n_samples(self):
        return self._n_samples 

    def get_n_jobs(self):
        return self._n_jobs
    
    def get_n_operations(self):
        return self._n_operations

    def get_percent_cross(self):
        return self._percent_cross

    def get_percent_intra_cross(self):
        return self._percent_intra_cross

    def get_percent_mutation(self):
        return self._percent_mutation

    def get_percent_intra_mutation(self):
        return self._percent_intra_mutation

    def get_percent_migration(self):
        return self._percent_migration

    def get_percent_selection(self):
        return self._percent_selection

    def get_fitness(self):
        return self._fitness

    def get_population(self): 
        return self._population

    def exec_permutationA0001(self):
        return self._permutationA0001(self.get_n_jobs(),self.get_n_operations(),self.get_n_samples())
    
    def exec_crossA0001(self):
        x_population = cp.copy(self.get_population())
        x_aux = cp.copy(self.get_population())
        index_selection = int(self.get_n_samples()*self.get_percent_selection())
        index_cross = int(self.get_n_samples()*self.get_percent_cross())
        cp.random.shuffle(x_population)
        if index_cross%2 == 0:
            y_population = self._crossA0001(x_population[index_selection:,:][0:index_cross,:],self.get_n_jobs(),self.get_n_operations(),index_cross, self.get_percent_intra_cross())
            x_aux[index_selection:,:][0:index_cross,:] = y_population
            self._population = x_aux
        else: 
            index_cros -= 1
            y_population = self._crossA0001(x_population[index_selection:,:][0:index_cross,:],self.get_n_jobs(),self.get_n_operations(),self.get_n_samples(), self.get_percent_intra_cross())
            x_aux[index_selection:,:][0:index_cross,:] = y_population
            self._set_population(x_aux)
        
    def exec_mutationA0001(self):
        x_population = cp.copy(self.get_population())
        x_aux = cp.copy(self.get_population())
        index_selection = int(self.get_n_samples()*self.get_percent_selection())
        index_mutation = int(self.get_n_samples()*self.get_percent_mutation())
        cp.random.shuffle(x_population)
        y_population = self._mutationA0001(x_population[index_selection:,:][0:index_mutation,:],self.get_n_jobs(),self.get_n_operations(),index_mutation,self.get_percent_intra_mutation())
        x_aux[index_selection:,:][0:index_mutation,:] = y_population
        self._population = x_aux

    def exec_migrationA0001(self):
        x_population = cp.copy(self.get_population())
        x_aux = cp.copy(self.get_population())
        index_selection = int(self.get_n_samples()*self.get_percent_selection())
        index_migration = int(self.get_n_samples()*self.get_percent_migration())
        cp.random.shuffle(x_population)
        y_population = self._permutationA0001(self.get_n_jobs(),self.get_n_operations(),index_migration)
        x_aux[index_selection:,:][0:index_migration,:] = y_population
        self._population = x_aux

    def exec_sortA0001(self):
        x_population = cp.copy(self.get_population())
        x_sort = cp.copy(self.get_fitness())
        y_population,y_sort = self._sortA0001(self.get_population(),self.get_fitness(),self.get_n_jobs(),self.get_n_operations(),self.get_n_samples())
        self._population = y_population
        self._fitness = y_sort
    