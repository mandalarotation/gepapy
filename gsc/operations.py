from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

from gsc.kernels.permutationA0001 import PermutationA0001
from gsc.kernels.crossA0001 import CrossA0001
from gsc.kernels.mutationA0001 import MutationA0001
from gsc.kernels.sortA0001 import SortA0001
from gsc.kernels.fitnessA0001 import FitnessA0001

class Operations(PermutationA0001,CrossA0001,MutationA0001,SortA0001,FitnessA0001):

    def __init__(self):
        None

    def set_n_samples(self,n_samples):
        return n_samples

    def _set_n_jobs(self,n_jobs):
        return n_jobs
    
    def _set_n_machines(self,n_machines):
        return n_machines

    def _set_n_operations(self,n_operations):
        return n_operations
    
    def _set_fitness_type(self,fitness_type):
        return fitness_type

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
        self._fitness = fitness

    def _set_population(self,population=None):
        if self._population.shape[0] == 0:
            return self.exec_permutationA0001()
        else:
            self._population = population
    
    def _set_processing_time(self,processing_time):
        return cp.array(processing_time,dtype=cp.float32)

    def _set_machine_sequence(self,machine_sequence):
        return cp.array(machine_sequence, dtype=cp.float32)   

    def _set_due_date(self,due_date):
        due_date = cp.array(due_date,dtype=cp.float32)
        return due_date
    
    def _set_weights(self,weights):
        weights = cp.array(weights,dtype=cp.float32) 
        return weights 

    def get_n_samples(self):
        return self._n_samples 

    def get_n_jobs(self):
        return self._n_jobs

    def get_n_machines(self):
        return self._n_machines
    
    def get_n_operations(self):
        return self._n_operations

    def get_fitness_type(self):
        return self._fitness_type

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

    def get_processing_time(self):
        return self._processing_time

    def get_machine_sequence(self):
        return self._machine_sequence
    
    def get_due_date(self):
        return self._due_date
    
    def get_weights(self):
        return self._weights

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
            self._set_population(x_aux)
        else: 
            index_cross -= 1
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
        self._set_population(x_aux)

    def exec_migrationA0001(self):
        x_population = cp.copy(self.get_population())
        x_aux = cp.copy(self.get_population())
        index_selection = int(self.get_n_samples()*self.get_percent_selection())
        index_migration = int(self.get_n_samples()*self.get_percent_migration())
        cp.random.shuffle(x_population)
        y_population = self._permutationA0001(self.get_n_jobs(),self.get_n_operations(),index_migration)
        x_aux[index_selection:,:][0:index_migration,:] = y_population
        self._set_population(x_aux)

    def exec_sortA0001(self):
        x_population = cp.copy(self.get_population())
        x_sort = cp.copy(self.get_fitness())
        y_population,y_sort = self._sortA0001(x_population,x_sort,self.get_n_jobs(),self.get_n_operations(),self.get_n_samples())
        self._set_population(y_population)
        self._set_fitness(y_sort)

    def exec_fitnessA0001(self):
        x_population = cp.copy(self.get_population())
        fitness = self._fitnessA0001(x_population,self.get_due_date(),self.get_weights(),self.get_processing_time(),self.get_machine_sequence(),self.get_n_jobs(),self.get_n_samples(),self.get_n_machines())
        self._set_fitness(fitness[self.get_fitness_type()])

    def get_plan(self):
        x_population = cp.copy(self.get_population())
        return = self._get_planA0001(row,x_population,self.get_processing_time(),self.get_machine_sequence(),self.get_n_jobs(),self.get_n_machines())