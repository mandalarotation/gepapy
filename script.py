import time 
from gsc.single_machine import Single_Machine
from gsc.job_shop import Job_Shop

p = Job_Shop()

print(p.get_population())

p.exec_crossA0001()

print(p.get_population())

p.exec_mutationA0001()

print(p.get_population())

p.exec_migrationA0001()

print(p.get_population())

