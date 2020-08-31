from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math
from gsc.single_machine.kernels.pop_init import Pop_Init
from gsc.single_machine.kernels.cross import Cross
from gsc.single_machine.kernels.fitness import Fitness
from gsc.single_machine.kernels.migration import Migration
from gsc.single_machine.kernels.mutation import Mutation

class Main_Kernel(Pop_Init,Cross,Fitness,Migration,Mutation):

    def init():
        return 
    
    def _main_pop_init(self):
      _kernel_gen_matrix_permutations()

    def _main_cross(self):
      _kernel_crossover_population()
    
    def _main_migration(self):
      self._kernel_migration()
    
    def _main_fitness(self):
      self._kernel_calculate_fitness()
      self._kernel_sort_population()
        
    def main_mutation(self):
      self._kernel_mutation()