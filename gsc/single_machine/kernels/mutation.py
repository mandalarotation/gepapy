from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class Mutation():
    def __init__(self):
        None
        
    def _kernel_mutation(self):
      """
      Metodo que realiza mutaciones aleatorias en cierto cromosomas escogidos.
      """
      @cuda.jit
      def mutation(x,al,p):
        row,col = cuda.grid(2)
        if row < x.shape[0] and col < x.shape[1]:
          aux = 0
          if al[row,col] < 0.5:
            aux = x[row,col]
            x[row,col] = x[row,int(p[row,col])]
            x[row,int(p[row,col])] = aux
        cuda.syncthreads()

      x = self.population
      mutation_rate = self.crossover_mutation_rate
      crom_size = self.crom_size
      parent = x[0:int(x.shape[0]*mutation_rate),:]
      P,AL= GSC.GeneralFunctions.special_gen_matrix_permutations(parent.shape[0],parent.shape[1])

      # Configure the blocks
      threadsperblock = 16
      blockspergrid_x = int(math.ceil(parent.shape[0] / threadsperblock))
      blockspergrid = blockspergrid_x
      cuda.synchronize()
      mutation[blockspergrid, threadsperblock](parent,AL,P)
      cuda.synchronize()

      x[int(x.shape[0]*mutation_rate) + int(int(x.shape[0]*mutation_rate)/2):int(x.shape[0]*mutation_rate) + int(int(x.shape[0]*mutation_rate)/2) + parent.shape[0],:] = parent
      self.population = x

      x = None
      P = None
      AL = None
      parent = None
