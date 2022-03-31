import numpy as np 
import os, glob, math, re
import pyqg
import datetime
import sys
import math 

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
np.set_printoptions(threshold=np.inf)



hour=60*60; day=24*hour; month= 30*day; year =360*day
dt=60
tsnapint=6*hour 
tmax_hr=year*5

          
       

def process(r):
          qh=[] 
          mh = pyqg.QGModel(nx=32,  tmax=tmax_hr, twrite=1000, tavestart=5*year)
          sig = 1.e-7
          qi = sig*np.vstack([np.random.randn(mh.nx,mh.ny)[np.newaxis,],
                  	        np.random.randn(mh.nx,mh.ny)[np.newaxis,]])
          np.save('input/{}.npy'.format(r),qi[0])
          mh.set_q(qi)
          mh.run()
          np.save('output/{}.npy'.format(r),mh.q[0] + mh.Qy[0]*mh.y)
          return 

chunk=int(sys.argv[1])
limit=chunk+90000
for r in range(chunk,limit):
      print(r)
      process(r)
       
       
