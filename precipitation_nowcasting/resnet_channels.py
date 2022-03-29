
# coding: utf-8


import os,glob,re
import numpy as np
import tensorflow as tf
from numpy.random import randint,choice
from metrics import *
from multiprocessing import Pool
import itertools
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

config = tf.ConfigProto()
sess = tf.Session(config=config)


folder="../data3d"
dim=(56,320,320)
steps=5
forecast=120
pace=int(forecast/(steps*2))
ndim=(56,320,320,1)
out_dim=320*320*56
epochs=5
drop=0.05

def process(i):
	try:
		a=np.load("{}/{}.npy".format(folder,i))
		return a
	except:
		print(i) 

bce = tf.keras.losses.binary_crossentropy
msle=tf.keras.losses.mean_squared_logarithmic_error
rgl=tf.keras.regularizers.l1(10e-10)

# Encoder
inp,inps=[],[]
for i in range(steps):
        inp.append(tf.keras.layers.Input(shape=ndim))
        x=tf.keras.layers.Conv3D(16, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(inp[i])
        #x=tf.keras.layers.MaxPooling3D((2,2,2), padding='valid')(x) # steps >5 
        inps.append(x)


res1,res2=[],[]
layers={}

layers[1]=tf.keras.layers.add(inps)
#layers[2]=tf.keras.layers.Conv3D(16, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[1])
layers[3]=tf.keras.layers.Conv3D(32, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[1])
res1.append(layers[3])
layers[4]=tf.keras.layers.MaxPooling3D((2,2,2), padding='valid')(layers[3])
layers[5]=tf.keras.layers.Dropout(drop)(layers[4])
#layers[6]=tf.keras.layers.Conv3D(64, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[5])
layers[7]=tf.keras.layers.Conv3D(64, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[5])
res2.append(layers[7])
layers[8]=tf.keras.layers.MaxPooling3D((2,2,2), padding='valid')(layers[7])
#layers[9]=tf.keras.layers.Conv3D(16, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[8])

code=tf.keras.layers.Conv3D(1, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[8])
encoder = tf.keras.models.Model(inp, code)
encoder.summary()
# Decoder
code2=tf.keras.layers.UpSampling3D((2,2,2))(code)

layers[10]=tf.keras.layers.Conv3D(64, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(code2)
#layers[11]=tf.keras.layers.Conv3D(64, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[10])
layers[12]=tf.keras.layers.add([res2[0], layers[10]])
layers[13]=tf.keras.layers.UpSampling3D((2,2,2))(layers[12])
#layers[14]=tf.keras.layers.Conv3D(32, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[13])
layers[15]=tf.keras.layers.Conv3D(32, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[13])
layers[16] = tf.keras.layers.add([res1[0], layers[15]])
layers[17]= tf.keras.layers.Conv3D(1, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[16])
#for steps > 5
#layers[17.5]=tf.keras.layers.UpSampling3D((2,2,2))(layers[17])
#layers[17]= tf.keras.layers.Conv3D(1, (3,3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[17.5])

decoded=layers[17]

autoencoder = tf.keras.models.Model(inp, decoded)
autoencoder.summary()


autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=[TP,FN,FP])



checkpoint_path = "training/cp_{}.ckpt".format(steps)
checkpoint_dir = os.path.dirname(checkpoint_path)




try:
        autoencoder.load_weights(checkpoint_path)
        print("find weights")
except:
        print('No weights file')


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


#####################################################################################################



def prepare(i):
        data=process(i) 
        data=np.where(data<0, 0, data) 
        data=data[:-1,:-1,:-1] 
        data=data.reshape(ndim) 
        #data=np.array(data)#*100000
        return data

indices=np.load('indices.npy')


def prepare1(prange,k,pace):
        data=[[process(x) for x in range(i,i+pace)] for i in prange]
        data=[np.mean(np.array(d), axis = 0) for d in data]
        data=np.array([np.where(im<0, 0, im) for im in data])
        data=np.array([a[:-1,:-1,:-1] for a in data])
        data=np.array([a.reshape(dim) for a in data])
        data=[data[i:i+k] for i in range(0,len(data),k)]
        data=[x.reshape((56,320,320,k)) for x in data]
        data=np.array(data)#10000
        return data

for e in range(1000):
        print(e)
        print("##################################################")
        for forecast in [120,240,360,480,600,720]:#[50,10,100,20,480,80,200,720,40,120]:
            pace=int(forecast/(steps*2))
            prange=np.random.choice(indices,10)
            print(steps,forecast,prange)
            din=[]
            try:
                for i in range(steps):
                      rin=[y+i*pace  for y in prange]
                      din.append(prepare1(rin,1,pace))
                rout=[x+forecast  for x in prange]
                dout=prepare1(rout,1,pace)
                print("##################################################")
                history=autoencoder.fit(din,dout,epochs=epochs,validation_split=0.02, batch_size=1, callbacks=[cp_callback])
            except:
                print('range issue')
          
