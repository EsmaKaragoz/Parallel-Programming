import random as rd
n = 50000000
f = open('constants.txt', 'w')
f.write('%d\n' %n)
for i in range(n):
    f.write('%f\n' %rd.uniform(1, 10))
f.close()