import time 
from gsc.single_machine import Single_Machine
from gsc.job_shop import Job_Shop
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()


T_ = cp.expand_dims(cp.array([10,10,13,4,9,4,8,15,7,1,9,3,15,9,11,6,5,14,18,3],dtype=cp.float32),axis=1)
d_ = cp.array([50,38,49,12,20,105,73,45,6,64,15,6,92,43,78,21,15,50,150,99],dtype=cp.float32)
w_ = cp.array([10,5,1,5,10,1,5,10,5,1,5,10,10,5,1,10,5,5,1,5],dtype=cp.float32)
M_ = cp.expand_dims(cp.zeros(20,dtype=cp.float32),axis=1)

p = Single_Machine(n_samples=100000,n_jobs=20,processing_time=T_,machine_sequence=M_,due_date=d_,weights=w_,percent_cross=0.8,percent_mutation=0.8,percent_migration=0.1,fitness_type="E_LTw")


fitness = []
fitness2 = []

for i in range(100):
    p.exec_crossA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    fitness2.append(p.get_fitness()[0])
    p.exec_mutationA0001()
    p.exec_fitnessA0001
    p.exec_sortA0001()
    fitness2.append(p.get_fitness()[0])
    p.exec_migrationA0001()
    p.exec_fitnessA0001
    p.exec_sortA0001()
    fitness2.append(p.get_fitness()[0])
    fitness.append(p.get_fitness()[0])
    print(p.get_fitness()[0])
    print(p.get_population()[0])
print('the elapsed time:%s'% (time.time() - start_time))



