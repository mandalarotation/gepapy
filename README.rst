gepapy
==============

The objective of this project is to develop a library that allows to solve various production planning problems in a flexible and fast way by means of modules that the user can configure and join to create their own implementations; Each module is implemented in cuda and runs directly on the GPU with the help of the numba and cupy libraries, which guarantees a parallel execution and much shorter waiting times than if the execution were done on the CPU.

Installation
------------------------

Requirements

* GPU NVIDIA con soporte para cuda 10 +
* typing
* numpy
* cupy
* numba 
* python 3.5 +

Google Colab


In the case of Google Colab, the cuda environment is already configured, so it is enough to activate a session with gpu support and execute the following commands to install the library and be able to view the gantt charts associated with the results.

.. code-block::

  pip install gepapy --no-deps 
  pip install chart_studio

Personal Computer with NVIDIA GPU

It is recommended to use docker technology and use a preconfigured image by Nvidia so that you do not have to manually install all the cuda libraries and do not have to resolve possible incompatibilities one by one in the installation process, which usually becomes quite tedious. Since docker is cross-platform so it works on a wide range of operating systems and it is enough to refer to the following nvidia repository https://github.com/NVIDIA/nvidia-docker and follow the instructions, then the image can be modified to install python and their respective libraries, as well as a jupyter server if you want to use it from a notebook https://docs.docker.com/. Another alternative is to install an environment with ANCONDA that supports cuda https://www.anaconda.com/. After configuring the environment with support for cuda, just run the following commands to install the library automatically from the pypi repositories.

.. code-block::

  # Si ya se encuentran instaladas las librerias, typing, numpy, cupy,  y numba y configurado el ambiente para cuda
  pip install gepapy --no-deps 


  # Si no se encuentran instalada alguna de las librerias, typing, numpy, cupy,  y numba
  pip install gepapy 



  # si se quieren visualizar graficos de Gantt
  pip install chart_studio
  
Examples
------------------------

Single Machine

The ** single machine ** problem is a scheduling case in which there is only one machine and a number of operations greater than one that must be performed on it, so the objective is to optimize the order of execution of did many operations based on priority weights assigned to each machine and expected delivery times.


The following code can be divided into 4 parts:


> * ** Import of libraries. **
> * ** Import or definition of the problem data. **
> * ** Instantiation of the SingleMachine class object. **
> * ** Definition of the loop that will be repeated over each epoch. **


** Import of libraries. **

.. code:: python

  import time 
  from gen_scheduling_cuda.single_machine import Single_Machine
  import cupy as cp
  import numpy as np
  import matplotlib.pyplot as plt

The time library is only to measure the execution time, so it is not strictly necessary to use it, the gen_scheduling_cuda library contains the different classes that allow us to create objects for the different types of problems that are supported in the library, for example in this case SingleMachine , FlowShop and JobShop. It should be said that Single Machine and FlowShop are specific cases of JobShop, therefore the JobShop type objects can also solve this type of more specific problems, but the user must make the respective modifications in the input data manually, so for ease of use You can create SingleMachine and FlowShop objects and thus not have to worry about the adequacy of the problem. In general, the JobShop type objects implemented in the library can be used to solve any problem that is encoded as sequences of repeated numbers, for example [1, 2, 3, 4, 3, 1, 1, 2, 3, 2 , 4, 4]

The cupy library is an extension of numpy for nvidia GPUs and it is necessary to import it since it allows the handling of all kinds of arrays that are stored in the GPU memory, in turn it is also necessary to import numpy to use these arrays when they are require passing to the CPU memory, either to view them, to save them or to use them in another library that does not have GPU support.

Finally we import the matplotlib library to be able to make graphs and visualize the optimization curve.

** Import or definition of the problem data. **

.. code:: python

  T_ = cp.array([10,10,13,4,9,4,8,15,7,1,9,3,15,9,11,6,5,14,18,3],dtype=cp.float32)
  d_ = cp.array([50,38,49,12,20,105,73,45,6,64,15,6,92,43,78,21,15,50,150,99],dtype=cp.float32)
  w_ = cp.array([10,5,1,5,10,1,5,10,5,1,5,10,10,5,1,10,5,5,1,5],dtype=cp.float32)

For this specific ** Single Machine ** problem, the following data must be defined to solve the problem; T_, d_, w_. Each position in the vector represents an operation, so T_ [0] -> 10 says that the zero operation time is 10, then d_ [0] -> 50, says that the expected delivery time for the zero operation is 50 and W_ [0] -> 10, says that the delivery priority for the zero operation is 1/10, so then the problem conditions are defined.


** SingleMachine class object instantiation. **

.. code:: python

  p = Single_Machine(n_samples=100000,
                     n_jobs=20,
                     processing_time=T_,
                     due_date=d_,
                     weights=w_,
                     percent_cross=0.8,
                     percent_mutation=0.8,
                     percent_migration=0.1,
                     fitness_type="E_LTw")


The instantiation of an object is the way to obtain a set of tools to solve the problem associated with said object, once it is created and initialized with the parameters of the problem, then the different compatible operations can be applied, in terms of the Some initialization parameters can be modified afterwards at any moment of the execution and others cannot, among which they can be changed are, for example, the percentages of crossing, mutation and migration, probation size, fitness and a few more. Actually, in this example, not all the possible parameters of a SingleMachine type problem are presented, since they were not necessary for this case, but these will be exposed in the complete documentation of the library.



** Definition of the loop that will be repeated over each epoch. **

.. code:: python

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


This loop can be built according to the wishes of the user and the order that he considers pertinent, within the loop at any time he can decide to change any of the object's parameters or even if he has sufficient expertise to modify the population at some point with another code tool or library, tie everything and then continue with the training. For the specific example, a crossing was defined to be made in each epoch, then the fitness is calculated, then the population is rearranged according to the fitness, then a mutation, then again the fitness and the rearrangement and finally a migration and a redenomination and so on. 100 generations or epochs.

.. code:: python
  plt.plot(fitness)

  
  
