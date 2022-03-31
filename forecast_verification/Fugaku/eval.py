
# coding: utf-8


import os,glob,re
import numpy as np 
import tensorflow as tf
from numpy.random import randint, choice
from multiprocessing import Pool
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
#import horovod.tensorflow as hvd
#hvd_valid = True

#hvd.init()






folder="./"
epochs = 50
noise_dim = (32, 32, 1)
prange = range(0,5)
out_dim=(32,32,1)
drop=.1

#strategy = tf.contrib.distribute.MirroredStrategy(["cpu:0","gpu:0", "gpu:1"])


if True:
      
    layers={}
    rgl=tf.keras.regularizers.l1(10e-10)
    layers[0]=tf.keras.layers.Input(shape=noise_dim)
    i=0
    for x in range(5):
        layers[i+1]=tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', activity_regularizer=rgl)(layers[i])
        i=i+1
        layers[i+1]=tf.keras.layers.Dropout(drop)(layers[i])
        i=i+1
        if i>1:
            layers[i+1]=tf.keras.layers.Add()([layers[i],layers[i-2]])
            i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1

    generator = tf.keras.models.Model(layers[0], layers[i])

    generator.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae')


    generator.summary()

    checkpoint_path_gen = "generator.ckpt"
    checkpoint_dir_gen = os.path.dirname(checkpoint_path_gen)


    generator.load_weights(checkpoint_path_gen)

    cp_callback_gen = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_gen,
                                                 save_weights_only=True,
                                                 verbose=1)



#####################################################################
    layers2={}

    layers2[0]=tf.keras.layers.Input(shape=out_dim)
    i=0

    for x in range(3):
        layers2[i+1]=tf.keras.layers.Conv2D(32, (5, 5), activation='relu',  activity_regularizer=rgl)(layers2[i])
        i=i+1
        layers2[i+1]=tf.keras.layers.Dropout(drop)(layers2[i])
        i=i+1
    layers2[i+1]=tf.keras.layers.MaxPool2D((2,2))(layers2[i])
    i=i+1
    layers2[i+1]=tf.keras.layers.Conv2D(1, (3, 3), activation='relu', activity_regularizer=rgl)(layers2[i])
    i=i+1
    
    layers2[i+1]=tf.keras.layers.Flatten()(layers2[i])
    i=i+1
    layers2[i+1]=tf.keras.layers.Dense(16)(layers2[i])
    i=i+1
    layers2[i+1]=tf.keras.layers.Dense(1)(layers2[i])
    i=i+1

    discriminator = tf.keras.models.Model(layers2[0],layers2[i])

    discriminator.compile(optimizer=tf.keras.optimizers.Adamax(), loss='binary_crossentropy', metrics=['accuracy'])


    discriminator.summary()

    checkpoint_path_dis = "discriminator.ckpt"
    checkpoint_dir_dis = os.path.dirname(checkpoint_path_dis)


    discriminator.load_weights(checkpoint_path_dis)

    cp_callback_dis = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_dis,
                                                 save_weights_only=True,
                                                 verbose=1)
                                                       
                                                       
#######################################################################    
in_data=np.array([np.load("input/{}.npy".format(i))*100000 for i in prange])
print(in_data.shape)
artificial= generator.predict(in_data)                           
print(artificial)
np.save('output.npy',artificial)
        
        
        
        


