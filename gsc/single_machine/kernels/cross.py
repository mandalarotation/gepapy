from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class Cross():
    def __init__(self):
        None


    def crossover_population(self):
      """
      Metodo que realiza el cruce de dos cromosomas, funciona, pero todavia esta en fase experimental 
      y puede ser mejorado con el tiempo
      """
      @cuda.jit
      def crossover_one(parent_one,cross_one):
        row,col = cuda.grid(2)
        if row < parent_one.shape[0] and col >= int(parent_one.shape[1]/2) and col < parent_one.shape[1] : 
          cross_one[row,col] = parent_one[row,col]
        cuda.syncthreads()

      @cuda.jit
      def crossover_two(parent_two,cross_two):
        row = cuda.grid(1)
        if row < parent_two.shape[0]: 
          activate = 0
          col = 0
          aux_col = 0
          while activate == 0:
            aux = 0
            for k in range(int(parent_two.shape[1]/2),parent_two.shape[1],1):
              if parent_two[row,col] == cross_two[row,k]:
                aux += 1
            if aux == 0:
              cross_two[row,aux_col] = parent_two[row,col]
              aux_col += 1
              col += 1
            if aux != 0:
              col += 1
            if aux_col == int(parent_two.shape[1]/2):
              activate = 1
        cuda.syncthreads()

      x = self.population
      crossover_rate = self.crossover_mutation_rate
      crom_size = self.crom_size
      parents = x[0:int(x.shape[0]*crossover_rate),:]
      parent_one = parents[0:int(parents.shape[0]/2),:]
      parent_two = parents[int(parents.shape[0]/2):parents.shape[0],:]
      z = cp.zeros([int(parents.shape[0]/2),crom_size])

      threadsperblock = (16, 16)
      blockspergrid_x = int(math.ceil(parent_one.shape[0] / threadsperblock[0]))
      blockspergrid_y = int(math.ceil(parent_one.shape[1] / threadsperblock[1]))
      blockspergrid = (blockspergrid_x, blockspergrid_y)
      
      cuda.synchronize()
      crossover_one[blockspergrid, threadsperblock](parent_one,z)
      cuda.synchronize()
      
      # Configure the blocks
      threadsperblock = 16
      blockspergrid_x = int(math.ceil(parent_two.shape[0] / threadsperblock))
      blockspergrid = blockspergrid_x

      cuda.synchronize()
      crossover_two[blockspergrid, threadsperblock](parent_two,z)
      cuda.synchronize()

      x[int(x.shape[0]*crossover_rate):int(x.shape[0]*crossover_rate) + parent_one.shape[0],:] = z

      self.population = x

      x = None
      crossover_rate = None
      parents = None
      parent_one = None
      parent_two = None
      z = None
