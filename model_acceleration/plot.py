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

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
np.set_printoptions(threshold=np.inf)

hour=60*60; day=24*hour; month= 30*day; year =360*day
dt=60
tsnapint=6*hour
tsnapstart_hr=10*year 
tmax_hr=year*10+year*2
tsnapstart_lr=0
tmax_lr=year*1


mh = pyqg.QGModel(nx=64,  tmax=tmax_hr, twrite=1000, tavestart=5*year)