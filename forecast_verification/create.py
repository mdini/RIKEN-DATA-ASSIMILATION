
# coding: utf-8

import numpy as np 


noise_dim = (64, 64, 1)
prange = range(20000,100000)

for i in prange:
    print(i,end='-')
    x= np.random.rand(*noise_dim)
    np.save('RANDOM/{}.npy'.format(i),x)
   
   

        
        
        
        


