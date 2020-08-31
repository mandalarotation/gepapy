from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math
from gsc.kernels_single_machine.pop_init import Pop_Init
from gsc.kernels_single_machine.cross import Cross
from gsc.kernels_single_machine.fitness import Fitness
from gsc.kernels_single_machine.migration import Migration
from gsc.kernels_single_machine.mutation import Mutation

class Main_Kernel(Pop_Init,Cross,Fitness,Migration,Mutation):

    def init():
        return 
    
    def _main_pop_init(self):
      self._kernel_gen_matrix_permutations()

    def _main_cross(self):
      self._kernel_crossover_population()
    
    def _main_migration(self):
      self._kernel_migration()
    
    def _main_fitness(self):
      self._kernel_calculate_fitness()
      self._kernel_sort_population()
        
    def main_mutation(self):
      self._kernel_mutation()