
# coding: utf-8


import os,glob,re
import numpy as np 
import tensorflow as tf
from numpy.random import randint, choice
from multiprocessing import Pool
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import horovod.tensorflow as hvd
hvd_valid = True

hvd.init()


folder="PV"
in_dim=(64,64,1)
out_dim=(64,64,1)
epochs=200
drop=0.1


def process(r, i):
	try:
		a=np.load("{}/{}/PHR/{}.npy".format(folder,r,i+1))
		b=np.load("{}/{}/PHR/{}.npy".format(folder,r,i))
		return (a,b)
	except:
		print(i) 
   
   

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()   
   
 
with strategy.scope():      
      layers={}
      rgl=tf.keras.regularizers.l1(10e-10)
      
      layers[0]=tf.keras.layers.Input(shape=in_dim)
      
      layers[1]=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
      layers[1.1]=tf.keras.layers.Dropout(drop)(layers[1])
      layers[1.9]=tf.keras.layers.UpSampling2D((1,1))(layers[1.1])
      
      layers[2]=tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
      layers[2.1]=tf.keras.layers.Dropout(drop)(layers[2])
      layers[2.3]=tf.keras.layers.UpSampling2D((1,1))(layers[2.1])
      layers[2.5]=tf.keras.layers.add([layers[1.9],layers[2.3]])
      layers[2.7]=tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', activity_regularizer=rgl)(layers[2.5])
      layers[2.9]=tf.keras.layers.Dropout(drop)(layers[2.7])
      
      layers[3]=tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
      layers[3.1]=tf.keras.layers.Dropout(drop)(layers[3])
      layers[3.3]=tf.keras.layers.UpSampling2D((1,1))(layers[3.1])
      layers[3.5]=tf.keras.layers.add([layers[2.9],layers[3.3]])
      layers[3.7]=tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', activity_regularizer=rgl)(layers[3.5])
      layers[3.9]=tf.keras.layers.Dropout(drop)(layers[3.7])
      
      
      layers[4]=tf.keras.layers.Conv2D(32, (11, 11), activation='relu', padding='same', activity_regularizer=rgl)(layers[0])
      layers[4.1]=tf.keras.layers.Dropout(drop)(layers[4])
      layers[4.3]=tf.keras.layers.UpSampling2D((1,1))(layers[4.1])
      layers[4.5]=tf.keras.layers.add([layers[3.9],layers[4.3]])
      layers[4.7]=tf.keras.layers.Conv2D(32, (11, 11), activation='relu', padding='same', activity_regularizer=rgl)(layers[4.5])
      layers[4.9]=tf.keras.layers.Dropout(drop)(layers[4.7])
      
      
      layers[5]=tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[4.9])
      
      
      model = tf.keras.models.Model(layers[0], layers[5])
      
      
      model.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mse')
      
      checkpoint_path = "high.ckpt"
      checkpoint_dir = os.path.dirname(checkpoint_path)
      
      
      
      
      model.load_weights(checkpoint_path)
      
      
      cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                       save_weights_only=True,
                                                       verbose=1)
      
      
      
      
      for e in range(1000):
        prange=range(10000,20000)
        for i in range(12):
          data=[process(r,i) for r in prange]
          train_hr=np.array([a[0] for a in data])
          train_lr=np.array([a[1] for a in data])
          train_lr=[np.reshape(a,in_dim)*100000 for a in train_lr]
          train_hr=[np.reshape(a,out_dim)*100000 for a in train_hr]
          train_hr=np.array(train_hr)
          train_lr=np.array(train_lr)
          print("##################################################################",i)
          history = model.fit(train_lr, train_hr, epochs=epochs, batch_size=500, validation_split=0.02, callbacks=[cp_callback])


