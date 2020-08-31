from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class Migration():
    def __init__(self):
        None    
        
    def _kernel_migration(self):
      """
      Método que se vale el generador de poblaciones para generar nuevos individuos en la población
      y evitar el sindrome de las galapagos.
      """
      x = self.population
      mutation_rate = self.crossover_mutation_rate
      parent = x[0:int(x.shape[0]*mutation_rate),:]
      rows = x.shape[0] - (int(x.shape[0]*mutation_rate) + int(int(x.shape[0]*mutation_rate)/2) + parent.shape[0])
      cols = self.crom_size
      P,AL= GSC.GeneralFunctions.special_gen_matrix_permutations(10,cols)
      x[x.shape[0] - P.shape[0]:x.shape[0],:] = P
      self.population = x
      x = None 
      mutation_rate = None
      parent = None 
      rows = None
      cols = None
      P = None 
      AL = None