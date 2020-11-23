# gepapy

The objective of this project is to develop a library that allows to solve various production planning problems in a flexible and fast way by means of modules that the user can configure and join to create their own implementations; Each module is implemented in cuda and runs directly on the GPU with the help of the numba and cupy libraries, which guarantees a parallel execution and much shorter waiting times than if the execution were done on the CPU.

![schema](https://github.com/mandalarotation/gepapy/blob/master/assets/shchema.png)

# Installation

## Requirements

* GPU NVIDIA con soporte para cuda 10 +
* typing
* numpy
* cupy
* numba 
* python 3.5 +  


### Google Colab


In the case of Google Colab, the cuda environment is already configured, so it is enough to activate a session with gpu support and execute the following commands to install the library and be able to view the gantt charts associated with the results.

```
!pip install gepapy --no-deps 
!pip install chart_studio
```


### Personal Computer with NVIDIA GPU


It is recommended to use docker technology and use a preconfigured image by Nvidia so that you do not have to manually install all the cuda libraries and do not have to resolve possible incompatibilities one by one in the installation process, which usually becomes quite tedious. Since docker is cross-platform so it works on a wide range of operating systems and it is enough to refer to the following nvidia repository https://github.com/NVIDIA/nvidia-docker and follow the instructions, then the image can be modified to install python and their respective libraries, as well as a jupyter server if you want to use it from a notebook https://docs.docker.com/. Another alternative is to install an environment with ANCONDA that supports cuda https://www.anaconda.com/. After configuring the environment with support for cuda, just run the following commands to install the library automatically from the pypi repositories.

```
# If the libraries, typing, numpy, cupy, and numba are already installed and the environment configured for cuda
pip install gepapy --no-deps 
```

```
# If any of the libraries are not installed, typing, numpy, cupy, and numba
pip install gepapy 
```

```
# if you want to view Gantt charts
pip install chart_studio
```

## Examples

### Single Machine


The ** single machine ** problem is a scheduling case in which there is only one machine and a number of operations greater than one that must be performed on it, so the objective is to optimize the order of execution of did many operations based on priority weights assigned to each machine and expected delivery times.


The following code can be divided into 4 parts:


> * ** Import of libraries. **
> * ** Import or definition of the problem data. **
> * ** Instantiation of the SingleMachine class object. **
> * ** Definition of the loop that will be repeated over each epoch. **



** Import of libraries. **

```
import time 
from gen_scheduling_cuda.single_machine import Single_Machine
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
```

The time library is only to measure the execution time, so it is not strictly necessary to use it, the gen_scheduling_cuda library contains the different classes that allow us to create objects for the different types of problems that are supported in the library, for example in this case SingleMachine , FlowShop and JobShop. It should be said that Single Machine and FlowShop are specific cases of JobShop, therefore the JobShop type objects can also solve this type of more specific problems, but the user must make the respective modifications in the input data manually, so for ease of use You can create SingleMachine and FlowShop objects and thus not have to worry about the adequacy of the problem. In general, the JobShop type objects implemented in the library can be used to solve any problem that is encoded as sequences of repeated numbers, for example [1, 2, 3, 4, 3, 1, 1, 2, 3, 2 , 4, 4]

The cupy library is an extension of numpy for nvidia GPUs and it is necessary to import it since it allows the handling of all kinds of arrays that are stored in the GPU memory, in turn it is also necessary to import numpy to use these arrays when they are require passing to the CPU memory, either to view them, to save them or to use them in another library that does not have GPU support.

Finally we import the matplotlib library to be able to make graphs and visualize the optimization curve.

** Import or definition of the problem data. **

```
T_ = cp.array([10,10,13,4,9,4,8,15,7,1,9,3,15,9,11,6,5,14,18,3],dtype=cp.float32)
d_ = cp.array([50,38,49,12,20,105,73,45,6,64,15,6,92,43,78,21,15,50,150,99],dtype=cp.float32)
w_ = cp.array([10,5,1,5,10,1,5,10,5,1,5,10,10,5,1,10,5,5,1,5],dtype=cp.float32)
```

For this specific ** Single Machine ** problem, the following data must be defined to solve the problem; T_, d_, w_. Each position in the vector represents an operation, so T_ [0] -> 10 says that the zero operation time is 10, then d_ [0] -> 50, says that the expected delivery time for the zero operation is 50 and W_ [0] -> 10, says that the delivery priority for the zero operation is 1/10, so then the problem conditions are defined.


** SingleMachine class object instantiation. **

```
p = Single_Machine(n_samples=100000,
                   n_jobs=20,
                   processing_time=T_,
                   due_date=d_,
                   weights=w_,
                   percent_cross=0.8,
                   percent_mutation=0.8,
                   percent_migration=0.1,
                   fitness_type="E_LTw")
```

The instantiation of an object is the way to obtain a set of tools to solve the problem associated with said object, once it is created and initialized with the parameters of the problem, then the different compatible operations can be applied, in terms of the Some initialization parameters can be modified afterwards at any moment of the execution and others cannot, among which they can be changed are, for example, the percentages of crossing, mutation and migration, probation size, fitness and a few more. Actually, in this example, not all the possible parameters of a SingleMachine type problem are presented, since they were not necessary for this case, but these will be exposed in the complete documentation of the library.



** Definition of the loop that will be repeated over each epoch. **

```
fitness = []
fitness2 = []

start_time = time.time()

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
```


This loop can be built according to the wishes of the user and the order that he considers pertinent, within the loop at any time he can decide to change any of the object's parameters or even if he has sufficient expertise to modify the population at some point with another code tool or library, tie everything and then continue with the training. For the specific example, a crossing was defined to be made in each epoch, then the fitness is calculated, then the population is rearranged according to the fitness, then a mutation, then again the fitness and the rearrangement and finally a migration and a redenomination and so on. 100 generations or epochs.

```
plt.plot(fitness)
```


![sm_fitness](https://github.com/mandalarotation/gepapy/blob/master/assets/smp_fitness.png)


### Job Shop Problem

The JobShop problem is somewhat more general and interesting than the SingleMachine case, here it is necessary to optimize the order of execution of several jobs, several operations and several machines, for which we have certain restrictions of presence and concurrence in the execution of certain operations on certain machines, which are represented by a pair of matrices, one that defines the execution times in each machine-job combination and a third that defines the order in which each job must be executed in the different machines for each respective operation. There may be several optimization criteria and the library supports several that will be explained in the complete documentation, however for this example we will use the criterion of minimizing the C_max which would be minimizing the time in which the last required operation is completed.

```
import time 
from IPython.display import clear_output
from gepapy.job_shop import Job_Shop
import cupy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pt_tmp =[[29, 78,  9, 36, 49, 11, 62, 56, 44, 21],
       [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
       [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
       [81, 95, 71, 99,  9, 52, 85, 98, 22, 43],
       [14,  6, 22, 61, 26, 69, 21, 49, 72, 53],
       [84,  2, 52, 95, 48, 72, 47, 65,  6, 25],
       [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
       [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
       [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
       [85, 13, 61,  7, 64, 76, 47, 52, 90, 45]]

ms_tmp = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
       [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
       [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
       [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
       [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
       [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
       [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
       [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
       [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
       [1, 0, 2, 6, 8, 9, 5, 3, 4, 7]]

T_ = cp.array(pt_tmp,dtype=cp.float32)
d_ = cp.zeros(10,dtype=cp.float32)
w_ = cp.zeros(10,dtype=cp.float32)
M_ = cp.array(ms_tmp,dtype=cp.float32)




p = Job_Shop(n_samples=1000000,
             n_jobs=10,
             n_operations=10,
             n_machines=10,
             processing_time=T_,
             machine_sequence=M_,
             due_date=d_,
             weights=w_,
             percent_cross=0.5,
             percent_mutation=0.1,
             percent_intra_mutation=0.5,
             percent_migration=0.5,
             percent_selection=0.5,
             fitness_type="max_C")



fitness = []

start_time = time.time()

for i in range(200):

    p.exec_crossA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    p.exec_mutationA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    p.exec_migrationA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    fitness.append(p.get_fitness()[0])
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    clear_output(wait=True)
    print(i,p.get_fitness()[0])
print('the elapsed time:%s'% (time.time() - start_time))
```

```
plt.plot(fitness)
```

![jsp_fitness](https://github.com/mandalarotation/gepapy/blob/master/assets/jsp_fitness_.png)

```
plt.plot(cp.asnumpy(p.get_fitness()))
```


![jsp_all_fitness](https://github.com/mandalarotation/gepapy/blob/master/assets/jsp_all_fitness.png)

```
import chart_studio.plotly as py
import plotly.figure_factory as ff

plan = p.get_plan(0,60,1604868407175) # (#number sequence,conversion factor to seconds, timestap)

fig = ff.create_gantt(plan, show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
fig.show()
```

![gantt](https://github.com/mandalarotation/gepapy/blob/master/assets/gantt%20jsp.png)

The following code presents a possible strategy to avoid premature convergence, giving the opportunity to enter new chromosomes through migration every certain epoch and with a high probability allowing them to remain active for some time even though they are not initially competitive. this makes the algorithm optimize slower, but makes it more stable and less prone to getting stuck.

```
import time 
from IPython.display import clear_output
from gepapy.job_shop import Job_Shop
import cupy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pt_tmp =[[29, 78,  9, 36, 49, 11, 62, 56, 44, 21],
       [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
       [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
       [81, 95, 71, 99,  9, 52, 85, 98, 22, 43],
       [14,  6, 22, 61, 26, 69, 21, 49, 72, 53],
       [84,  2, 52, 95, 48, 72, 47, 65,  6, 25],
       [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
       [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
       [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
       [85, 13, 61,  7, 64, 76, 47, 52, 90, 45]]

ms_tmp = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
       [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
       [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
       [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
       [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
       [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
       [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
       [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
       [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
       [1, 0, 2, 6, 8, 9, 5, 3, 4, 7]]



T_ = cp.array(pt_tmp.values,dtype=cp.float32)
d_ = cp.zeros(10,dtype=cp.float32)
w_ = cp.zeros(10,dtype=cp.float32)
M_ = cp.array(ms_tmp.values -1,dtype=cp.float32)




p = Job_Shop(n_samples=1000000,
             n_jobs=10,
             n_operations=10,
             n_machines=10,
             processing_time=T_,
             machine_sequence=M_,
             due_date=d_,
             weights=w_,
             percent_cross=0.9,
             percent_mutation=0.01,
             percent_intra_mutation=0.1,
             percent_migration=0.01,
             percent_selection=0.1,
             fitness_type="max_C")



fitness = []

start_time = time.time()

for i in range(1,100,1):

    if i%10 == 0:
          p.set_percents_c_m_m_s(
          percent_cross=0.9,
          percent_mutation=0.01,
          percent_migration=0.5,
          percent_selection=0.1)
          p.exec_migrationA0001()
          p.exec_fitnessA0001()
          p.exec_sortA0001()
          p.set_percents_c_m_m_s(
          percent_cross=0.9,
          percent_mutation=0.01,
          percent_migration=0.01,
          percent_selection=0.1)       

    p.exec_crossA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    p.exec_mutationA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    p.exec_migrationA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    fitness.append(p.get_fitness()[0])
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    clear_output(wait=True)
    print(i,p.get_fitness()[0])
print('the elapsed time:%s'% (time.time() - start_time))

```
Example using two populations that are mutually supportive, in this case a main population evolves with 1 million individuals, then a second population consisting of 500,000 individuals acts as a seedbed allowing the laggards already seen before in the first implementation proposal to Job Shop develop and be competitive with the already more developed ones, thus contributing more to diversity and avoiding an elitist degeneration that leads the algorithm to get stuck in a local minimum.
 

```
import time 
from IPython.display import clear_output
from gepapy.job_shop import Job_Shop
import cupy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pt_tmp =[[29, 78,  9, 36, 49, 11, 62, 56, 44, 21],
       [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
       [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
       [81, 95, 71, 99,  9, 52, 85, 98, 22, 43],
       [14,  6, 22, 61, 26, 69, 21, 49, 72, 53],
       [84,  2, 52, 95, 48, 72, 47, 65,  6, 25],
       [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
       [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
       [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
       [85, 13, 61,  7, 64, 76, 47, 52, 90, 45]]

ms_tmp = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
       [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
       [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
       [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
       [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
       [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
       [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
       [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
       [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
       [1, 0, 2, 6, 8, 9, 5, 3, 4, 7]]


T_ = cp.array(pt_tmp.values,dtype=cp.float32)
d_ = cp.zeros(10,dtype=cp.float32)
w_ = cp.zeros(10,dtype=cp.float32)
M_ = cp.array(ms_tmp.values -1,dtype=cp.float32)




p = Job_Shop(n_samples=1000000,
             n_jobs=10,
             n_operations=10,
             n_machines=10,
             processing_time=T_,
             machine_sequence=M_,
             due_date=d_,
             weights=w_,
             percent_cross=0.5,
             percent_mutation=0.5,
             percent_intra_mutation=0.1,
             percent_migration=0.5,
             percent_selection=0.5,
             fitness_type="max_C")


p_aux = Job_Shop(n_samples=100000,
             n_jobs=10,
             n_operations=10,
             n_machines=10,
             processing_time=T_,
             machine_sequence=M_,
             due_date=d_,
             weights=w_,
             percent_cross=0.5,
             percent_mutation=0.5,
             percent_intra_mutation=0.1,
             percent_migration=0.5,
             percent_selection=0.5,
             fitness_type="max_C")



fitness = []
fitness2 = []

start_time = time.time()

for i in range(100):
    if i%10 == 0:
        p_aux.set_population(p.get_population()[900000:1000000])
        for j in range(10):
            p_aux.exec_crossA0001()
            p_aux.exec_fitnessA0001()
            p_aux.exec_sortA0001()
            fitness2.append(p_aux.get_fitness()[0])
            clear_output(wait=True)
            print("población auxiliar",j,p_aux.get_fitness()[0])
        p.get_population()[900000:1000000] = p_aux.get_population()
        p.exec_fitnessA0001()
        p.exec_sortA0001()

    p.exec_crossA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    p.exec_mutationA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    p.exec_migrationA0001()
    p.exec_fitnessA0001()
    p.exec_sortA0001()
    fitness.append(p.get_fitness()[0])
    clear_output(wait=True)
    print("población principal",i,p.get_fitness()[0])
print('the elapsed time:%s'% (time.time() - start_time))

```




# developers

Jean Carlo Jimenez Giraldo 
Student of industrial engineering from the National University of Colombia Medellin headquarters

Elkin Rodriguez Velasquez 
Profesor Professor of industrial engineering from the National University of Colombia Medellin headquarters

Yubar Daniel Marin Benjumea 
Student of statistics from the National University of Colombia Medellin headquarters


