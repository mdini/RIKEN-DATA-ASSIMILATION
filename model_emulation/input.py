import numpy as np
from random import random,seed


tmax, n = 100, 10000
# tmax, n = 100, 1000


for i in range(700000, 1000000):
        f=np.load("data/{}.npy".format(i))
        m,s=f.mean(axis=0),f.std(axis=0)
        print(i,m,s)
        fr=np.random.normal(m,s,(n,3))
        fr[:,0].sort(),fr[:,1].sort(),fr[:,2].sort()
        np.save('random/{}.npy'.format(i),fr)
