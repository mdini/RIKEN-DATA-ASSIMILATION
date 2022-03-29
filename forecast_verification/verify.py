# coding: utf-8


import os,glob,re
import numpy as np
import tensorflow as tf
from numpy.random import randint,choice
from metrics import *
from augment import * 
from multiprocessing import Pool
import itertools
import sys
from  scipy.special import logit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

config = tf.ConfigProto()
sess = tf.Session(config=config)


folder="../data3d"
dim=(320,320,1)
epochs=5
drop=0.1

layers={}
rgl=tf.keras.regularizers.l1(10e-10)


inp,inps=[],[]
for i in range(2):
        inp.append(tf.keras.layers.Input(shape=dim))
        x=tf.keras.layers.Conv2D(32, (3,3), activation='relu', activity_regularizer=rgl)(inp[i])
        x=tf.keras.layers.MaxPooling2D((2,2), padding='valid')(x)  
        inps.append(x)
        
layers[0]=tf.keras.layers.add(inps)


layers[1]=tf.keras.layers.Conv2D(32, (3,3), activation='relu', activity_regularizer=rgl)(layers[0])
layers[1.1]=tf.keras.layers.Dropout(drop)(layers[1])
layers[1.9]=tf.keras.layers.MaxPooling2D((4,4))(layers[1.1])


layers[2]=tf.keras.layers.Conv2D(32, (3,3), activation='relu', activity_regularizer=rgl)(layers[1.9])
layers[2.1]=tf.keras.layers.Dropout(drop)(layers[2])
layers[2.9]=tf.keras.layers.MaxPooling2D((4,4))(layers[2.1])



layers[3]=tf.keras.layers.Conv2D(32, (3,3), activation='relu', activity_regularizer=rgl)(layers[2.9])
layers[3.1]=tf.keras.layers.Dropout(drop)(layers[3])
layers[3.9]=tf.keras.layers.MaxPooling2D((4,4))(layers[3.1])


layers[3.95]=tf.keras.layers.Flatten()(layers[3.9])


layers[4]=tf.keras.layers.Dense(1, activation='sigmoid', activity_regularizer=rgl)(layers[3.95])


model = tf.keras.models.Model(inp, layers[4])
model.summary()

bce = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae', metrics=['accuracy'])

checkpoint_path = "verif.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


#model.load_weights(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,verbose=1)






def process(i):
	try:
		a=np.load("{}/{}.npy".format(folder,i))
		a=np.mean(a,axis=0)
		return a
	except:
		print(i) 
   
def prepare(data):
    data=[np.where(im<0, 0, im) for im in data]
    data=[a[:-1,:-1] for a in data]
    data=[a.reshape(dim) for a in data]
    return data 
    
    
def mse(A,B):
    return ((A - B)**2).mean()


indices=np.load('indices.npy')

def train():
    while True:
          print('start')
          prange=indices
          data=[process(x) for x in prange]
          data=prepare(data) 
          print('2222222') # noise
          noise=[np.random.normal(loc=0,scale=2,size=dim)  for x in prange]
          data2=[data[i]+noise[i] for i in range(len(prange))]
          print('3333333') # displacement  
          data3=[np.roll(m, np.random.randint(100), axis=np.random.randint(2)) for m in data]
          print("44444444") # different 
          data4=np.random.permutation(data)
          print("55555555") # scale
          scales=[np.random.choice(np.arange(0.25, 2.01, 0.25)) for m in data ]
          data5=[clipped_zoom(m[:,:,0], scales[i]) for i,m in enumerate(data) ]
          data5=[a.reshape(dim) for a in data5]
          print(data5[0].shape)
          print("66666666")
          data_a=[a for a in data for _ in range(4)] 
          data_b=[l[i] for i in range(len(data)) for l in [data2,data3,data4,data5]]
          data_o=[x for i in range(len(data)) for x in range(4)]
          data_a=np.array(data_a)
          data_b=np.array(data_b)
          data_o=np.array(data_o)
          print("##################################################################")
          history = model.fit([data_a,data_b], data_o, epochs=epochs, batch_size=1 ,validation_split=0.1, callbacks=[cp_callback])
        
        
def eval():
    prange=indices[:2000:1000]
    prange=[5665,31259]
    print(prange)
    data=[process(x) for x in prange]
    data=prepare(data) 
    in1=data
    in2=[data[0] for i in range(2)]
    in1=np.array(in1)
    in2=np.array(in2)
    res=model.predict([in1,in2],verbose=1)
    res=[logit(x) for x in res]    
    res=[.05*x+.5 for x in res]
    res=[x-res[0] for x in res]
    res=[min(x,1) for x in res]
    for i,r in enumerate(res):
        m=mse(in1[i][:,:,0],in2[i][:,:,0])
        print(i,r,m)
    return res 
    
    
train()
    

