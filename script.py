from gsc.single_machine import Single_Machine


p = Single_Machine()

print(p.get_population())

p.exec_crossA0001()

print(p.get_population())