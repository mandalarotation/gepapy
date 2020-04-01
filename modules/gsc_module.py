from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math


class GSC():
  def __init__():
    """
    GSC es la clase principal de GenSchedulingCuda la cual en un futuro contendra todos los modulos y metodos
    necesarios para resolver diversos problemas de algoritmos geneticos, por el momento solo cuenta con 
    un modulo para flow-shop que cuenta con los metodos mas fundamentales de un algoritmo genetico, la generación de 
    la población, el cruce la mutación y el calculo del fitness, todos ellos están paralelizados con CUDA.
    """
    None


  def create_object_FlowShop(pop_size,crom_size,processing_time,due_date,weights,crossover_mutation_rate,select_mutation_rate):
    """
    Es un metodo de la clase principal GSC cuyo objetivo es gestionar las subclases o modulos según el problema tipo de problema que se 
    requiera resolver, por el momento solamente inciailiza objetos a partir de la clase hija FlowShop
    """
    p=processing_time
    d=due_date
    w=weights
    OBJ = GSC.FlowShop(pop_size,crom_size,p,d,w,crossover_mutation_rate,select_mutation_rate)
    return OBJ



  class FlowShop():
    """
    Es la clase que crea objetos de tipo Flow_Shop los cuales luego serán transformados por los metodos existentes en la subclase GeneralFunctions
    """
    def __init__(self,pop_size,crom_size,processing_time,due_date,weights,crossover_mutation_rate,select_mutation_rate):

        self.pop_size = pop_size   # parametro que define el tamano de la población
        self.crom_size = crom_size  # parametro que define el tamano de los cromosomas
        self.processing_time = cp.repeat(cp.expand_dims(cp.array(processing_time,dtype=cp.float32),axis=0),self.pop_size ,axis=0)  #array 1D  
        self.due_date = cp.repeat(cp.expand_dims(cp.array(due_date,dtype=cp.float32),axis=0),self.pop_size  ,axis=0)  #array 1D
        self.weights = cp.repeat(cp.expand_dims(cp.array(weights,dtype=cp.float32),axis=0),self.pop_size  ,axis=0)     #array 1D
        self.population = GSC.GeneralFunctions.gen_matrix_permutations(self)  # Se usa el metodo gen_matrix_permutations de la subclase GeneralFunctions para crear la problación inicial
        self.crossover_mutation_rate = crossover_mutation_rate  # parametro que define la tasa de mutación
        self.select_mutation_rate = select_mutation_rate  # parametro que difene cuantos cromosomas de la población se mutarán
        self.fitness = cp.zeros([self.pop_size])  # se inicializa un array 1D de ceros que mas adelante será llenado con los fitness de cada cromosoma
        self.evolution_list = []  

  class GeneralFunctions():

    """
    Es una subclase o clase hija de la clase principal GSC y es accesible por los objetos creados a partir de la clase FlowShop
    """

    def __init__():
      None

    def gen_matrix_permutations(OBJ):
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
      
      rows = OBJ.pop_size
      cols = OBJ.crom_size
      X = cp.repeat(cp.expand_dims(cp.arange(cols,dtype=cp.float32),axis=0),rows,axis=0)
      AL = cp.array(cp.random.rand(rows,cols),dtype=cp.float32)

      # Configure the blocks
      threadsperblock = 16
      blockspergrid_x = int(math.ceil(rows / threadsperblock))
      blockspergrid = blockspergrid_x

      cuda.synchronize()
      gen_permutations[blockspergrid, threadsperblock](X,AL) 
      cuda.synchronize()

      AL = None

      return X

    def special_gen_matrix_permutations(rows,cols):
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

    def calculate_fitness(OBJ):
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
      x = OBJ.population
      y = OBJ.fitness
      p = OBJ.processing_time
      d = OBJ.due_date
      w = OBJ.weights
      threadsperblock = 16
      blockspergrid_x = int(math.ceil(OBJ.population.shape[0] / threadsperblock))
      blockspergrid = blockspergrid_x

      cuda.synchronize()
      fitness[blockspergrid, threadsperblock](x,y,p,d,w)
      cuda.synchronize()

      OBJ.fitness = y

      x = None
      y = None
      p = None
      d = None
      w = None

      return OBJ

    def sort_population(OBJ):
      """
      Metodo que permite organizar la población de mayor a menor desempeno según si fitness obtenido
      """

      @cuda.jit
      def sort(x,z,o):
        row,col = cuda.grid(2)
        if row < x.shape[0] and col < x.shape[1]: 
          z[row,col] = x[int(o[row,col]),col]
        cuda.syncthreads()

      y = OBJ.fitness
      x = OBJ.population
      pop_size = OBJ.pop_size
      crom_size = OBJ.crom_size

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

      OBJ.fitness = cp.sort(y)
      OBJ.population = z

      y = None
      x = None
      pop_size = None
      crom_size = None
      o = None
      z = None

      return OBJ

    def crossover_population(OBJ):
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

      x = OBJ.population
      crossover_rate = OBJ.crossover_mutation_rate
      crom_size = OBJ.crom_size
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

      OBJ.population = x

      x = None
      crossover_rate = None
      parents = None
      parent_one = None
      parent_two = None
      z = None

      return OBJ
    
    def mutation(OBJ):
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

      x = OBJ.population
      mutation_rate = OBJ.crossover_mutation_rate
      crom_size = OBJ.crom_size
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
      OBJ.population = x

      x = None
      P = None
      AL = None
      parent = None

      return OBJ

    def migration(OBJ):
      """
      Método que se vale el generador de poblaciones para generar nuevos individuos en la población
      y evitar el sindrome de las galapagos.
      """
      x = OBJ.population
      mutation_rate = OBJ.crossover_mutation_rate
      parent = x[0:int(x.shape[0]*mutation_rate),:]
      rows = x.shape[0] - (int(x.shape[0]*mutation_rate) + int(int(x.shape[0]*mutation_rate)/2) + parent.shape[0])
      cols = OBJ.crom_size
      P,AL= GSC.GeneralFunctions.special_gen_matrix_permutations(10,cols)
      x[x.shape[0] - P.shape[0]:x.shape[0],:] = P
      OBJ.population = x
      x = None 
      mutation_rate = None
      parent = None 
      rows = None
      cols = None
      P = None 
      AL = None
      return OBJ
