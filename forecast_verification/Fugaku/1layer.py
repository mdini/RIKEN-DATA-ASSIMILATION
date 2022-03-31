
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
prange = np.random.choice(10000,5000)
out_dim=(32,32,1)
drop=.1

#strategy = tf.contrib.distribute.MirroredStrategy(["cpu:0","gpu:0", "gpu:1"])


if True:
      
    layers={}
    rgl=tf.keras.regularizers.l1(10e-10)
    layers[0]=tf.keras.layers.Input(shape=noise_dim)
    i=0
    layers[i+1]=tf.keras.layers.Flatten()(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Dense(2048)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Dense(1024)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Reshape(out_dim)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    generator = tf.keras.models.Model(layers[0], layers[i])

    generator.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae')


    generator.summary()

    checkpoint_path_gen = "generatorL1.ckpt"
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


    #discriminator.summary()

    checkpoint_path_dis = "discriminator.ckpt"
    checkpoint_dir_dis = os.path.dirname(checkpoint_path_dis)


    discriminator.load_weights(checkpoint_path_dis)

    cp_callback_dis = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_dis,
                                                 save_weights_only=True,
                                                 verbose=1)
                                                       
                                                       
#######################################################################    



for i in range(100000000000000):
    prange = np.random.choice(10000,100)
    out_data=np.array([np.load("output/{}.npy".format(i))*100000 for i in prange])
    #in_data=[np.random.normal(loc=np.mean(hr), scale=np.std(hr), size=(32,32)) for hr in out_data]
    #in_data=np.array([ np.reshape(np.sort(y.flatten()), (32,32))  for y in in_data ])
    in_data=np.array([ np.reshape(np.sort(y.flatten()), (32,32))  for y in out_data ])
    in_data=np.array([np.reshape(a,noise_dim) for a in in_data])
    out_data=np.array([np.reshape(a,out_dim) for a in out_data])
    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh',i)
    history_gen = generator.fit(in_data, out_data,  epochs=5000, validation_split=0.02, callbacks=[cp_callback_gen])
    artificial= generator.predict(in_data)
    print(artificial.shape,out_data.shape)
    np.save('out.npy',artificial)
    p = np.random.permutation(len(prange))
    images=np.concatenate((artificial, out_data), axis=0)[p]
    classes=np.concatenate((np.ones(len(prange)),np.zeros(len(prange))) ,axis=0)[p]
    history_dis = discriminator.fit(images, classes,  epochs=1, validation_split=0.02, callbacks=[cp_callback_dis])
                                                 

        
        
        
        


