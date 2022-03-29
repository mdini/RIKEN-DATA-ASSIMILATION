
# coding: utf-8


import os,glob,re
import numpy as np 
import tensorflow as tf






folder="./"
epochs = 50
dim = (10000, 2, 1)
drop=.1

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():

    layers={}
    rgl=tf.keras.regularizers.l1(10e-10)
    layers[0]=tf.keras.layers.Input(shape=dim)
    i=0
    layers[i+1]=tf.keras.layers.Flatten()(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Dense(20000)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Dense(20000)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Flatten()(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Reshape(dim)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(32, (5, 2), padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(32, (11, 2), padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(1, (3, 2), padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    
    
    generator = tf.keras.models.Model(layers[0], layers[i])

    generator.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae')


    generator.summary()

    checkpoint_path_gen = "generatorL.ckpt"
    checkpoint_dir_gen = os.path.dirname(checkpoint_path_gen)


    generator.load_weights(checkpoint_path_gen)

    cp_callback_gen = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_gen,
                                                 save_weights_only=True,
                                                 save_freq='epoch',
                                                 period=50,
                                                 verbose=1)




    print('Hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
#####################################################################
    layers2={}

    layers2[0]=tf.keras.layers.Input(shape=dim)
    i=0

    for x in range(3):
        layers2[i+1]=tf.keras.layers.Conv2D(32, (5, 2), activation='relu',  padding='same', activity_regularizer=rgl)(layers2[i])
        i=i+1
        layers2[i+1]=tf.keras.layers.Dropout(drop)(layers2[i])
        i=i+1
    layers2[i+1]=tf.keras.layers.MaxPool2D((2,1))(layers2[i])
    i=i+1
    layers2[i+1]=tf.keras.layers.Conv2D(1, (3, 2), activation='relu',  padding='same', activity_regularizer=rgl)(layers2[i])
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

    checkpoint_path_dis = "discriminatorL.ckpt"
    checkpoint_dir_dis = os.path.dirname(checkpoint_path_dis)


    #discriminator.load_weights(checkpoint_path_dis)

    cp_callback_dis = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_dis,
                                                 save_weights_only=True,
                                                 verbose=1)
                                                       
                                                       
#######################################################################    



    for i in range(10000):
        prange = range(10)#np.random.choice(range(1000000),10)
        in_data=np.array([np.load("random/{}.npy".format(i))[:,:2]*100000 for i in prange])
        out_data=np.array([np.load("data/{}.npy".format(i))[:,:2]*100000 for i in prange])
        in_data=np.array([ np.reshape(y, dim)  for y in in_data ])
        out_data=np.array([ np.reshape(y, dim)  for y in out_data ])
        print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh',in_data.shape)
        history_gen = generator.fit(in_data, out_data,  epochs=100,  callbacks=[cp_callback_gen])
        artificial= generator.predict(in_data)
        print(artificial.shape,out_data.shape)
        np.save('outL.npy',artificial[:5])
        p = np.random.permutation(len(prange))
        images=np.concatenate((artificial, out_data), axis=0)[p]
        classes=np.concatenate((np.ones(len(prange)),np.zeros(len(prange))) ,axis=0)[p]
        history_dis = discriminator.fit(images, classes,  epochs=1, validation_split=0.02, callbacks=[cp_callback_dis])
                                                     
    
            
        
        
        


