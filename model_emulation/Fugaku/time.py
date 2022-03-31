
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

in_dim=(3,10)
out_dim=(3,)
epochs=2000
drop=0.1



strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()   
   
 
with strategy.scope():   
      layers={}
      
      layers[0]=tf.keras.layers.Input(shape=in_dim)
      
      layers[1]=tf.keras.layers.Dense(32)(layers[0])
      layers[2]=tf.keras.layers.Dense(3)(layers[1])
      
      model = tf.keras.models.Model(layers[0], layers[2])
      model.summary()

      model.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mse')
      
      checkpoint_path = "time.ckpt"
      checkpoint_dir = os.path.dirname(checkpoint_path)
      
      
      
      
      #model.load_weights(checkpoint_path)
      
      
      cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                       save_weights_only=True,
                                                       verbose=1)
      
      
      
      
      n=10000
      data=np.load('63.npy')[:n]
      print(data)
      for e in range(10000):
          print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
          ins=data[:-10]
          ins=np.array([data[i:i+10] for i in range(n-10)])
          outs=data[10:]
          while True:
              print("##################################################################")
              history = model.fit(ins,outs, epochs=epochs, validation_split=0.02, callbacks=[cp_callback])
