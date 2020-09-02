from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math

class Pop_Init():
    def __init__(self):
        None
    
    def _kernel_gen_matrix_permutations(self):
      """
      Implementacion en cuda que permite generar poblaciones que estan representadas 
      por matrices donde cada fila corresponde a un cromosona y las columnas estan asociadas con 
      el tamano del cromosoma.
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
      
      rows = self.pop_size
      cols = self.crom_size
      X = cp.repeat(cp.expand_dims(cp.arange(cols,dtype=cp.float32),axis=0),rows,axis=0)
      AL = cp.array(cp.random.rand(rows,cols),dtype=cp.float32)

      # Configure the blocks
      threadsperblock = 16
      blockspergrid_x = int(math.ceil(rows / threadsperblock))
      blockspergrid = blockspergrid_x

      cuda.synchronize()
      gen_permutations[blockspergrid, threadsperblock](X,AL) 
      cuda.synchronize()

      self.population = X

      AL = None
      X = None

    def _special_gen_matrix_permutations(self,rows,cols):
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