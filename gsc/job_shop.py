from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

from gsc.kernels.permutationA0001 import PermutationA0001
from gsc.kernels.crossA0001 import CrossA0001
from gsc.kernels.mutationA0001 import MutationA0001

class Job_Shop(PermutationA0001,CrossA0001,MutationA0001):
    def __init__(self,n_samples=10,n_jobs=8,n_operations=3,processing_time=[1,2,3],due_date=[1,2,3],weights=[1,2,3],percent_cross=0.2,percent_intra_cross=0.5,percent_mutation=0.2,percent_intra_mutation=0.5,percent_migration=0.1,percent_selection=0.1):
        self._n_samples = self.set_n_samples(n_samples)
        self._n_jobs = self._set_n_jobs(n_jobs)
        self._n_operations = self._set_n_operations(n_operations)
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

    def _set_n_jobs(self,n_jobs):
        return n_jobs

    def _set_n_operations(self,n_operations):
        return n_operations

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

    def get_processing_time(self):
        return self._processing_time

    def get_due_date(self):
        return self._due_date
    
    def get_weights(self):
        return self._weights

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