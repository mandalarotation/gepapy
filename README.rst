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

.. code:: sh
  pip install gepapy --no-deps 
  pip install chart_studio


