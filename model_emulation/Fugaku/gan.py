
# coding: utf-8


import os,glob,re
import numpy as np 
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(32)
tf.config.threading.set_intra_op_parallelism_threads(32)

print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))






folder="./"
epochs = 50
dim = (10000, 3, 1)
drop=.1

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope
    
    layers={}
    rgl=tf.keras.regularizers.l1(10e-10)
    layers[0]=tf.keras.layers.Input(shape=dim)
    i=0
    layers[i+1]=tf.keras.layers.Reshape((dim[2],dim[1],dim[0]))(layers[i])
    #i=i+1
    layers[i+1]=tf.keras.layers.Dense(2500)(layers[1])
    i=i+1
    layers[i+1]=tf.keras.layers.UpSampling1D(size=2)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Dense(2500)(layers[1])
    i=i+1
    layers[i+1]=tf.keras.layers.UpSampling1D(size=2)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Concatenate()([layers[i],layers[i-2]])
    i=i+1
    #layers[i+1]=tf.keras.layers.Flatten()(layers[i])
    #i=i+1
    #layers[i+1]=tf.keras.layers.Reshape(dim)(layers[i])
    #i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(128, (5, 3), padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(32, (11, 3), padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    layers[i+1]=tf.keras.layers.Conv2D(1, (3, 3), padding='same', activity_regularizer=rgl)(layers[i])
    i=i+1
    
    
    generator = tf.keras.models.Model(layers[0], layers[i])

    generator.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae')


    generator.summary()

    checkpoint_path_gen = "generator.ckpt"
    checkpoint_dir_gen = os.path.dirname(checkpoint_path_gen)


    #generator.load_weights(checkpoint_path_gen)

    cp_callback_gen = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_gen,
                                                 save_weights_only=True,
                                                 verbose=1)



#####################################################################
    layers2={}

    layers2[0]=tf.keras.layers.Input(shape=dim)
    i=0

    for x in range(3):
        layers2[i+1]=tf.keras.layers.Conv2D(32, (5, 3), activation='relu',  padding='same', activity_regularizer=rgl)(layers2[i])
        i=i+1
        layers2[i+1]=tf.keras.layers.Dropout(drop)(layers2[i])
        i=i+1
    layers2[i+1]=tf.keras.layers.MaxPool2D((2,1))(layers2[i])
    i=i+1
    layers2[i+1]=tf.keras.layers.Conv2D(1, (3, 3), activation='relu',  padding='same', activity_regularizer=rgl)(layers2[i])
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


    #discriminator.load_weights(checkpoint_path_dis)

    cp_callback_dis = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_dis,
                                                 save_weights_only=True,
                                                 verbose=1)
                                                       
                                                       
#######################################################################    



    for i in range(10000):
        prange = range(10)#np.random.choice(range(1000000),10)
        in_data=np.array([np.load("random/random/{}.npy".format(i))[:dim[0]]*100000 for i in prange])
        out_data=np.array([np.load("data/data/{}.npy".format(i))[:dim[0]]*100000 for i in prange])
        in_data=np.array([ np.reshape(y, dim)  for y in in_data ])
        out_data=np.array([ np.reshape(y, dim)  for y in out_data ])
        print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh',in_data.shape)
        history_gen = generator.fit(in_data, out_data,  epochs=100,  callbacks=[cp_callback_gen])
        artificial= generator.predict(in_data)
        print(artificial.shape,out_data.shape)
        np.save('out.npy',artificial[:5])
        p = np.random.permutation(len(prange))
        images=np.concatenate((artificial, out_data), axis=0)[p]
        classes=np.concatenate((np.ones(len(prange)),np.zeros(len(prange))) ,axis=0)[p]
        history_dis = discriminator.fit(images, classes,  epochs=1, validation_split=0.02, callbacks=[cp_callback_dis])
                                                     
    
            
        
        
        


