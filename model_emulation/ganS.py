
# coding: utf-8


import os,glob,re
import numpy as np 
import tensorflow as tf






folder="./"
epochs = 50
dim = (10000, 3, 1)
drop=.1

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope
  rgl=tf.keras.regularizers.l1(10e-10)
  generator = tf.keras.Sequential([
            tf.keras.layers.Input(shape=dim),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(30000),
            tf.keras.layers.Dense(30000),
            tf.keras.layers.Reshape(dim),
            tf.keras.layers.Conv2D(32, (5, 3), padding='same', activity_regularizer=rgl),
            tf.keras.layers.Conv2D(32, (11, 3), padding='same', activity_regularizer=rgl),
            tf.keras.layers.Conv2D(1, (3, 3), padding='same', activity_regularizer=rgl)])
  generator.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae')


  generator.summary()

  checkpoint_path_gen = "generatorS.ckpt"
  checkpoint_dir_gen = os.path.dirname(checkpoint_path_gen)


  #generator.load_weights(checkpoint_path_gen)

  cp_callback_gen = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_gen,
                                               save_weights_only=True,
                                               save_freq='epoch',
                                               period=50,
                                               verbose=1)



#####################################################################


  discriminator = tf.keras.Sequential([
          tf.keras.layers.Input(shape=dim),
          tf.keras.layers.Conv2D(32, (5, 3), activation='relu',  padding='same', activity_regularizer=rgl),
          tf.keras.layers.Dropout(drop),
          tf.keras.layers.MaxPool2D((2,1)),
          tf.keras.layers.Conv2D(1, (3, 3), activation='relu',  padding='same', activity_regularizer=rgl),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(16),
          tf.keras.layers.Dense(1)])

  discriminator.compile(optimizer=tf.keras.optimizers.Adamax(), loss='binary_crossentropy', metrics=['accuracy'])


  discriminator.summary()

  checkpoint_path_dis = "discriminatorS.ckpt"
  checkpoint_dir_dis = os.path.dirname(checkpoint_path_dis)


  #discriminator.load_weights(checkpoint_path_dis)

  cp_callback_dis = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_dis,
                                               save_weights_only=True,
                                               verbose=1)
                                                     
                                                       
#######################################################################    



for i in range(10000):
    prange = range(10)#np.random.choice(range(1000000),10)
    in_data=np.array([np.load("random/{}.npy".format(i))*100000 for i in prange])
    out_data=np.array([np.load("data/{}.npy".format(i))*100000 for i in prange])
    in_data=np.array([ np.reshape(y, dim)  for y in in_data ])
    out_data=np.array([ np.reshape(y, dim)  for y in out_data ])
    print(in_data.shape,out_data.shape)
    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh',in_data.shape)
    history_gen = generator.fit(in_data, out_data,  epochs=100,  callbacks=[cp_callback_gen])
    artificial= generator.predict(in_data)
    print(artificial.shape,out_data.shape)
    np.save('out.npy',artificial[:5])
    p = np.random.permutation(len(prange))
    images=np.concatenate((artificial, out_data), axis=0)[p]
    classes=np.concatenate((np.ones(len(prange)),np.zeros(len(prange))) ,axis=0)[p]
    history_dis = discriminator.fit(images, classes,  epochs=1, validation_split=0.02, callbacks=[cp_callback_dis])
                                                     
    
            
        
        
        


