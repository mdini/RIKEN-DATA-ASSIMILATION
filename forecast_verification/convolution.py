#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# In[28]:


layers={}
rgl=tf.keras.regularizers.l1(10e-10)
dim=(64,64,1)
epochs=5
drop=0.1
k=16
fk=7
l=5

# In[29]:



class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.1):
            print("\n\n\nReached 0.01 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True


trainingStopCallback = haltCallback()






layers[0]=tf.keras.layers.Input(shape=dim)

for i in range(l):
    layers[0.1+i]=tf.keras.layers.Conv2D(k, (fk,fk), activation='relu', padding='same', activity_regularizer=rgl)(layers[0+i])
    layers[0.5+i]=tf.keras.layers.MaxPool2D((2,2), padding='same')(layers[0.1+i])
    layers[1+i]=tf.keras.layers.Dropout(drop)(layers[0.5+i])

for i in range(l,2*l):
    layers[0.1+i]=tf.keras.layers.Conv2D(k, (fk,fk), activation='relu', padding='same', activity_regularizer=rgl)(layers[0+i])
    layers[0.5+i]=tf.keras.layers.UpSampling2D((2,2))(layers[0.1+i])
    layers[1+i]=tf.keras.layers.Dropout(drop)(layers[0.5+i])


layers[-2]=tf.keras.layers.add([layers[2*l],layers[0.1]])

layers[-1]=tf.keras.layers.Conv2D(1, (3,3), activation='relu', padding='same', activity_regularizer=rgl)(layers[-2])


# In[30]:


model = tf.keras.models.Model(layers[0], layers[-1])
model.summary()


# In[31]:


model.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae')


# In[32]:


ind=np.array([np.load("sample1/1.npy").reshape(dim)*100000 ])


# In[33]:


outd=np.array([np.load("sample2/1.npy").reshape(dim)*100000 ])


# In[38]:

history = model.fit(ind, outd, epochs=10000, callbacks=[trainingStopCallback])


# In[39]:


for e,layer in enumerate(model.layers):
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    else:
        print(layer.name)
        filters, biases = layer.get_weights()
        # normalize filter values to 0-1 so we can visualize them
        #f_min, f_max = filters.min(), filters.max()
        #filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        print(filters.shape)
        k1,k2=filters.shape[2],filters.shape[3]
        n_filters = k1*k2 
        plt.figure(figsize=(k*2,k*2)  if e<len(model.layers)-1 else (5,k) )
        for i in range(k2):
            for j in range(k1):
                # get the filter
                f = filters[:, :, j, i]
                # plot each channel separately
                # specify subplot and turn of axis
                ax = plt.subplot(k1,k2,i*k1+j+1)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(f, cmap='RdBu_r',vmin=filters.min(),vmax=filters.max())
        plt.savefig("features/filter_{}".format(e))
        # show the figure
        plt.show()
        p_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer.output)
        feature_maps = p_model.predict(ind[:1])
        
        plt.figure(figsize=(k*3,5)  if e<len(model.layers)-1 else (5,5))
        for i in range(k2):
                m=feature_maps[0,:,:,i]
                ax = plt.subplot(1,k2,i+1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(m, cmap='RdBu_r')
        plt.savefig("features/features_{}".format(e))




