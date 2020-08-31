from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class Fitness():
    def __init__(self):
        None

    def _kernel_calculate_fitness(self):
      """
      Metodo que permite calcular el fitnes de manera paralela para toda la población
      """
      @cuda.jit
      def fitness(x,y,p,d,w):
        row = cuda.grid(1)
        if row < x.shape[0]: 
          ptime=0
          tardiness=0
          for j in range(x.shape[1]):
            ptime=ptime+p[row,int(x[row][j])]
            tardiness=tardiness+w[row,int(x[row][j])]*max(ptime-d[row,int(x[row][j])],0)
          y[row]=tardiness
        cuda.syncthreads()

      # Configure the blocks
      x = self.population
      y = self.fitness
      p = self.processing_time
      d = self.due_date
      w = self.weights
      threadsperblock = 16
      blockspergrid_x = int(math.ceil(self.population.shape[0] / threadsperblock))
      blockspergrid = blockspergrid_x

      cuda.synchronize()
      fitness[blockspergrid, threadsperblock](x,y,p,d,w)
      cuda.synchronize()
      print("aqui")
      self.fitness = y

      x = None
      y = None
      p = None
      d = None
      w = None

    def _kernel_sort_population(self):
      """
      Metodo que permite organizar la población de mayor a menor desempeno según si fitness obtenido
      """

      @cuda.jit
      def sort(x,z,o):
        row,col = cuda.grid(2)
        if row < x.shape[0] and col < x.shape[1]: 
          z[row,col] = x[int(o[row,col]),col]
        cuda.syncthreads()

      y = self.fitness
      x = self.population
      pop_size = self.pop_size
      crom_size = self.crom_size

      o = cp.repeat(cp.expand_dims(cp.argsort(y),axis=1),crom_size,axis=1)
      z = cp.zeros([pop_size,crom_size])

      # Configure the blocks
      threadsperblock = (16, 16)
      blockspergrid_x = int(math.ceil(x.shape[0] / threadsperblock[0]))
      blockspergrid_y = int(math.ceil(x.shape[1] / threadsperblock[1]))
      blockspergrid = (blockspergrid_x, blockspergrid_y)

      cuda.synchronize()
      sort[blockspergrid, threadsperblock](x,z,o)
      cuda.synchronize()

      self.fitness = cp.sort(y)
      self.population = z

      y = None
      x = None
      pop_size = None
      crom_size = None
      o = None
      z = None