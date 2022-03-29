# coding: utf-8
import os,glob,re
import numpy as np 
import tensorflow as tf
from numpy.random import randint, choice
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


#########################################

folder="PV"
in_dim=(32,32,1)
out_dim=(64,64,1)
epochs=50
drop=0.1
layers={}
rgl=tf.keras.regularizers.l1(10e-10)

layers[0]=tf.keras.layers.Input(shape=in_dim)

layers[1]=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
layers[1.1]=tf.keras.layers.Dropout(drop)(layers[1])
layers[1.9]=tf.keras.layers.UpSampling2D((2,2))(layers[1.1])

layers[2]=tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
layers[2.1]=tf.keras.layers.Dropout(drop)(layers[2])
layers[2.3]=tf.keras.layers.UpSampling2D((2,2))(layers[2.1])
layers[2.5]=tf.keras.layers.add([layers[1.9],layers[2.3]])
layers[2.7]=tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', activity_regularizer=rgl)(layers[2.5])
layers[2.9]=tf.keras.layers.Dropout(drop)(layers[2.7])

layers[3]=tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
layers[3.1]=tf.keras.layers.Dropout(drop)(layers[3])
layers[3.3]=tf.keras.layers.UpSampling2D((2,2))(layers[3.1])
layers[3.5]=tf.keras.layers.add([layers[2.9],layers[3.3]])
layers[3.7]=tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', activity_regularizer=rgl)(layers[3.5])
layers[3.9]=tf.keras.layers.Dropout(drop)(layers[3.7])


layers[4]=tf.keras.layers.Conv2D(32, (11, 11), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
layers[4.1]=tf.keras.layers.Dropout(drop)(layers[4])
layers[4.3]=tf.keras.layers.UpSampling2D((2,2))(layers[4.1])
layers[4.5]=tf.keras.layers.add([layers[3.9],layers[4.3]])
layers[4.7]=tf.keras.layers.Conv2D(32, (11, 11), activation='relu', padding='same', activity_regularizer=rgl)(layers[4.5])
layers[4.9]=tf.keras.layers.Dropout(drop)(layers[4.7])


layers[5]=tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[4.9])

model_hpsm = tf.keras.models.Model(layers[0], layers[5])


model_hpsm.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mse')

checkpoint_path = "low.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_hpsm.load_weights(checkpoint_path)

##########################################
##########################################

layers={}
in_dim=(64,64,1)

inp,inps=[],[]
for i in range(2):
        inp.append(tf.keras.layers.Input(shape=in_dim))
        x=tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', activity_regularizer=rgl)(inp[i])
        inps.append(x)
        

layers[0]=tf.keras.layers.add(inps)
layers[1]=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
layers[2]=tf.keras.layers.Dropout(drop)(layers[1])
layers[3]=tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[2])

model = tf.keras.models.Model(inp, layers[3])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mse')

checkpoint_path = "combine.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)




model.load_weights(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        


############################################################
############################################################
def process(r, i):
	#try:
		#a=np.load("{}/{}/HR/{}.npy".format(folder,r,i))
		b=np.load("{}/{}/PHR/{}.npy".format(folder,r,i))
		c=np.load("{}/{}/PLR/{}.npy".format(folder,r,i))
		d=np.load("{}/{}/PHR/{}.npy".format(folder,r,0))
		return (b,c,d)#(a,b,c,d)


def ponder(x):
        return np.exp(-.01*x)/(1+np.exp(-.01*x))
        

for e in range(1000):
        prange=range(100000)#np.random.choice(100000,10000)
        times=[np.random.randint(100) for r in prange]
        print(times[:5])
        data=[process(r,times[i]) for i,r in enumerate(prange)]
        phr=np.array([a[0] for a in data]) 
        in_dim=(64,64,1)
        phr=np.array([np.reshape(a,in_dim)*100000 for a in phr])
        
        plr=np.array([a[1] for a in data])
        in_dim=(32,32,1)
        plr=np.array([np.reshape(a,in_dim)*100000 for a in plr])
        plr=model_hpsm.predict(plr)
        
        phr0=np.array([a[2] for a in data]) 
        times=[ponder(x) for x in times]
        print(times[:5])
        in_dim=(64,64,1)
        phr0=np.array([np.reshape(a,in_dim)*100000*times[i] for i,a in enumerate(phr0)])
        
        in1=plr
        in2=phr0
        out=phr
        history = model.fit([in1,in2], out, epochs=epochs, batch_size=1 ,validation_split=0.05, callbacks=[cp_callback])
        
        
        
        
        
  
