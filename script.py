import time 
from gsc.single_machine import Single_Machine
from gsc.job_shop import Job_Shop


p = [10,10,13,4,9,4,8,15,7,1,9,3,15,9,11,6,5,14,18,3]
d = [50,38,49,12,20,105,73,45,6,64,15,6,92,43,78,21,15,50,150,99]
w = [10,5,1,5,10,1,5,10,5,1,5,10,10,5,1,10,5,5,1,5]

p = Single_Machine(n_samples=10,n_jobs=20,processing_time=p,due_date=d,weights=w)

print(p.get_population())

p.exec_crossA0001()

print(p.get_population())

p.exec_mutationA0001()

print(p.get_population())

p.exec_migrationA0001()

print(p.get_population())

p.exec_fitnessSM0001()

print(p.get_fitness())

p.exec_sortA0001()

print(p.get_fitness())

