from __future__ import division
from numba import cuda
import numpy as np
import cupy as cp
import math


class SingleMachine():
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

