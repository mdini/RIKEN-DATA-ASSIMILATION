#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import os, glob, math, re
import pyqg
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys


# In[2]:


def upscale_q(big,factor,func=np.mean):
    (bz,by,bx)=big.shape
    (sz,sy,sx)=bz,int(bx/factor),int(by/factor)
    small = func(func(big.reshape([sz, sx, factor, sy, factor]),axis=4),axis=2)
    return small  


# In[3]:


def plot(xh,yh,qhref,qh,f):
    mx= np.nanmax(qhref); mn= np.nanmin(qhref)
    #mx=max(mx,mn)
    #mn=-mx
    qh=np.clip(qh,mn,mx)
    levels=np.arange(mn,mx+(mx-mn)/12,(mx-mn)/12)
    plt.figure(figsize=(6,5)) 
    ax=plt.subplot(111)
    ax.set_xticks([]) 
    ax.set_yticks([])
    plt.contourf(xh, yh, qh, cmap='RdBu_r', levels=levels )
    plt.colorbar()
    plt.show()
    plt.savefig(f, transparent=True)
    plt.close()


# In[78]:


hour=60*60; day=24*hour; month= 30*day; year =360*day
dt=60
tsnapint=6*hour


# In[79]:


sig = 1.e-5
qi=np.zeros((2,64,64))
(x,y,a,b)=tuple(np.random.randint(1,5,size=4))
(e,v)=tuple(np.random.randint(-10,10,size=2))
qi[0,32-y:32+y,32-x:32+x]=sig*e
qi[0,28-a:28+a,28-b:28+b]=sig*v
tsnapstart_m=0
tmax_m=int(year*.5/(a+b+x+y))


# In[80]:


mm = pyqg.QGModel(nx=64,  tmax=tmax_m, twrite=1000, tavestart=5*year)
mm.set_q(qi)
mm.run()


# In[81]:


cut=np.percentile(np.fabs(mm.q[0]),90)
q=np.where(np.fabs(mm.q[0])>cut,mm.q[0].copy(),np.nan)
print(tmax_m/day,e,v)
plot(mm.x,mm.y,q, mm.q[0], 'test.png')
plot(mm.x,mm.y,q, q, 'test.png')


# In[82]:


q=np.where(np.fabs(mm.q[0])>cut,mm.q[0],0)
y,x=np.nonzero(q)
y0,y1,x0,x1=min(y),max(y),min(x),max(x)


# In[83]:


y0,y1,x0,x1


# In[84]:


tmax_hr=year*5


# In[85]:


mh = pyqg.QGModel(nx=64,  tmax=tmax_hr, twrite=1000, tavestart=5*year)
sig = 1.e-7
qi = sig*np.vstack([np.random.randn(mh.nx,mh.ny)[np.newaxis,],
                np.random.randn(mh.nx,mh.ny)[np.newaxis,]])
mh = pyqg.QGModel(nx=64,  tmax=tmax_hr, twrite=1000, tavestart=5*year)
mh.set_q(qi)
mh.run()
qh=mh.q[0] + mh.Qy[0]*mh.y
plot(mh.x,mh.y,qh, qh, 'test.png')


# In[86]:


dy,dx=y1-y0,x1-x0
qda=qh.copy()
ya=np.random.randint(64-dy)
xa=np.random.randint(64-dx)
qda[ya:ya+dy,xa:xa+dx]+=q[y0:y1,x0:x1]
plot(mh.x,mh.y,qh, qda, 'test.png')
qdb=qh.copy()
yb=np.random.randint(64-dy)
xb=np.random.randint(64-dx)
qdb[yb:yb+dy,xb:xb+dx]+=q[y0:y1,x0:x1]
plot(mh.x,mh.y,qh, qdb, 'test.png')


# In[87]:


displace=qda-qdb
plot(mh.x,mh.y,q, displace, 'test.png')


# In[ ]:





# In[ ]:





# In[ ]:




