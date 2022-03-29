import numpy as np 
import os, glob, math, re
import pyqg
import datetime
from multiprocessing import Pool
#import tensorflow as tf
from noise import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import math 

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
np.set_printoptions(threshold=np.inf)



hour=60*60; day=24*hour; month= 30*day; year =360*day
dt=60
tsnapint=6*hour
tsnapstart_hr=10*year 
tmax_hr=year*10+month*2
tsnapstart_pr=0
tmax_pr=month*2
tsnapstart_pt=0
tmax_pt=month*1
tsnapstart_lr=0
tmax_lr=month*1
noise=1.e-9
br_int=4

def upscale_q(big,factor,func=np.mean):
    (bz,by,bx)=big.shape
    (sz,sy,sx)=bz,int(bx/factor),int(by/factor)
    small = func(func(big.reshape([sz, sx, factor, sy, factor]),axis=4),axis=2)
    return small
    

    

          
          
       

def process(r):
          qh=[] 
          mh = pyqg.QGModel(nx=64,  tmax=tmax_hr, twrite=1000, tavestart=5*year)
          sig = 1.e-7
          qi = sig*np.vstack([np.random.randn(mh.nx,mh.ny)[np.newaxis,],
                  	        np.random.randn(mh.nx,mh.ny)[np.newaxis,]])
          mh.set_q(qi)
          for i,_ in enumerate(mh.run_with_snapshots(tsnapstart=tsnapstart_hr, tsnapint=tsnapint)):
              if not i: 
                 		qil=mh.q.copy()
              else:
              			qh.append(mh.q.copy())
          ######################################          
          mp = pyqg.QGModel(nx=64,  tmax=tmax_pr,twrite=1000,tavestart=5*year)          
          qip=qil+0.0     
          norm=3.e-6
          ep=5.e-7
          qip[0]=qip[0]+ep*create_red_noise(64, 64, 10)/10
          qip[1]=qip[1]+ep*create_red_noise(64, 64, 10)/10
          mp.set_q(qip)       
          ####################################              
                  
          qp,errs=[],[]
          for i,_ in enumerate(mp.run_with_snapshots(tsnapstart=tsnapstart_pr, tsnapint=tsnapint)):
              qp.append(mp.q.copy())
              diff=qp[i]-qh[i]
              err=np.sqrt(np.mean(diff**2))
              if i:
                  errs.append(err)
              if (i==120): 
                  ql,qht,qpt=[],[],[]
                  print(i, err)
                  if err>norm:
                      mp.set_q(qh[i]+(diff/err)*norm)
                  if i:
                      qil=mp.q.copy()
                      ############
                      mpt = pyqg.QGModel(nx=64,  tmax=tmax_pt,twrite=1000,tavestart=5*year)
                      mpt.set_q(qil)
                      for j,_ in enumerate(mpt.run_with_snapshots(tsnapstart=tsnapstart_pt, tsnapint=tsnapint)):
              		        qpt.append(mpt.q.copy())
                      ############
                      ml = pyqg.QGModel(nx=32,  tmax=tmax_lr,twrite=1000,tavestart=5*year)
                      qir=upscale_q(qil,2,np.max)
                      ml.set_q(qir)
                      for j,_ in enumerate(ml.run_with_snapshots(tsnapstart=tsnapstart_lr, tsnapint=tsnapint)):
              		        ql.append(ml.q.copy())
                      ############
                      qht=qh[i:i+120]
                      ############
                      for x in ['HR','PHR','PLR']:   
                          os.makedirs('PV/{}/{}'.format(r,x), exist_ok=True)
                      ############
                      print(len(qht),len(qpt),len(ql))
                      ############
                      
                      for j in range(int(month/tsnapint)):
                          np.save('PV/{}/{}/{}.npy'.format(r,'HR',j),qht[j][0] + mh.Qy[0]*mh.y)
                          np.save('PV/{}/{}/{}.npy'.format(r,'PHR',j),qpt[j][0] + mp.Qy[0]*mp.y)
                          np.save('PV/{}/{}/{}.npy'.format(r,'PLR',j),ql[j][0] + ml.Qy[0]*ml.y)
                      
          ############  
          return 

chunk=int(sys.argv[1])
limit=chunk+5000
for r in range(chunk,limit):
      print(r)
      process(r)
       
       
