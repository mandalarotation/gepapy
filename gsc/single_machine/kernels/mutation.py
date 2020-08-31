from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class Mutation():
    def __init__(self):
        None


    def _special_gen_matrix_permutations(rows,cols):
      """
      Una variante del generador de poblaciones que puede ser implementado para algunos
      casos en un futuro
      """
      @cuda.jit
      def gen_permutations(X,AL):
        row = cuda.grid(1)
        if row < X.shape[0]:
          for j in range(0,X.shape[1],1):
            tmp = X[row,int(AL[row,j]*j+0.5)]
            X[row,int(AL[row,j]*j+0.5)] = X[row,j]
            X[row,j] = tmp
        cuda.syncthreads()
      

      X = cp.repeat(cp.expand_dims(cp.arange(cols,dtype=cp.float32),axis=0),rows,axis=0)
      AL = cp.array(cp.random.rand(rows,cols),dtype=cp.float32)

      # Configure the blocks
      threadsperblock = 16
      blockspergrid_x = int(math.ceil(rows / threadsperblock))
      blockspergrid = blockspergrid_x

      cuda.synchronize()
      gen_permutations[blockspergrid, threadsperblock](X,AL) 
      cuda.synchronize()

      return X,AL
        
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
      P,AL= self._special_gen_matrix_permutations(parent.shape[0],parent.shape[1])

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
