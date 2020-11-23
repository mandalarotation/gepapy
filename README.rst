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
  
