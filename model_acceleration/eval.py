import os,glob,re
import numpy as np 
import tensorflow as tf
from numpy.random import randint, choice
from scipy import interpolate
import scipy.stats as stats
import datetime
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#import cv2
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def mean_squared_error(A,B):
    mse = (np.square(A - B)).mean(axis=None)
    return mse 
    

def mean_absolute_error(A,B):
    mae = (np.absolute(A - B)).mean(axis=None)
    return mae 
    


#########################################

in_dim=(32,32,1)
out_dim=(64,64,1)
epochs=5
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


in_dim=(64,64,1)
out_dim=(64,64,1)
epochs=5
drop=0.1


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


model_ddsm = tf.keras.models.Model(layers[0], layers[5])


model_ddsm.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mse')

checkpoint_path = "high.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_ddsm.load_weights(checkpoint_path)


##########################################


layers={}

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

############################################################

folder="EV"



def process(r, i):
	#try:
		a=np.load("{}/{}/HR/{}.npy".format(folder,r,i))
		b=np.load("{}/{}/PHR/{}.npy".format(folder,r,i))
		c=np.load("{}/{}/PLR/{}.npy".format(folder,r,i))
		d=np.load("{}/{}/PHR/{}.npy".format(folder,r,i-1))
		e=np.load("{}/{}/PHR/{}.npy".format(folder,r,0))
		return (a,b,c,d,e)
	#except:
	#	print("{}/{}/HR/{}.npy".format(folder,r,i))
   



   
def ACC(FC,OBS,F,O):
	top = np.mean((FC-F)*(OBS-O))
	bottom = np.sqrt(np.mean((FC-F)**2)*np.mean((OBS-O)**2))
	ACC = top/bottom
	return ACC
 
 
def ponder(x):
        if x <100:
              return np.exp(-.01*x)/(1+np.exp(-.01*x))
        else:
              return np.exp(-1)/(1+np.exp(-1))
              
              

def fft(image):
        npix = image.shape[0]
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2

        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, npix//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                             statistic = "mean",
                                             bins = kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        return Abins


 
 
   
breed=120
prange=range(breed,5*1440,breed)
mae_ref,mae_hpsm,mae_ddsm,mae_inter,mae_combine=[],[],[],[],[]
mse_ref,mse_hpsm,mse_ddsm,mse_inter,mse_combine=[],[],[],[],[]
acc_ref,acc_hpsm,acc_ddsm,acc_inter,acc_combine=[],[],[],[],[]
fft_0,fft_ref,fft_hpsm,fft_ddsm,fft_inter,fft_combine=[],[],[],[],[],[]
for i in range(1,480,4):
      print('####################################',i)
      data=[process(r,i) for r in prange]
      hr=np.array([a[0] for a in data])
      np.save("SCREENSHOTS/HR/{}.npy".format(i),hr) ########
      fft_hr=np.mean(np.array([fft(hr[i]) for i in range(len(data))]),axis=0)
      fft_0.append(fft_hr)
      phr=np.array([a[1] for a in data])
      np.save("SCREENSHOTS/PHR/{}.npy".format(i),phr) ########
      ################ Reference 
      fft_phr=np.mean(np.array([fft(phr[i]) for i in range(len(data))]),axis=0)
      mae_phr=np.mean(np.array([mean_absolute_error(hr[i],phr[i]) for i in range(len(data))]))
      mse_phr=np.mean(np.array([mean_squared_error(hr[i],phr[i]) for i in range(len(data))]))
      O=np.mean(hr,axis=0)
      F=np.mean(phr,axis=0)
      acc_phr=np.mean(np.array([ACC(phr[i],hr[i],F,O) for i in range(len(data))]))
      print( '{}: {}: {}, {}'.format(str(i*.25), 'phr', mae_phr, acc_phr) )
      mae_ref.append(mae_phr)
      mse_ref.append(mse_phr)
      acc_ref.append(acc_phr)
      fft_ref.append(fft_phr)
      ################ hpsm 
      plr=np.array([a[2] for a in data])
      in_dim=(32,32,1)
      plr=np.array([np.reshape(a,in_dim)*100000 for a in plr])
      t = datetime.datetime.now() ### time 
      plra=model_hpsm.predict(plr) 
      t = (datetime.datetime.now() -t)/120 ### time 
      plr=np.array([a[:,:,0]/100000 for a in plra])
      np.save("SCREENSHOTS/HPSM/{}.npy".format(i),plr) #########
      fft_plr=np.mean(np.array([fft(plr[i]) for i in range(len(data))]),axis=0)
      mae_plr=np.mean(np.array([mean_absolute_error(hr[i],plr[i]) for i in range(len(data))]))
      mse_plr=np.mean(np.array([mean_squared_error(hr[i],plr[i]) for i in range(len(data))]))
      F=np.mean(plr,axis=0)
      acc_plr=np.mean(np.array([ACC(plr[i],hr[i],F,O) for i in range(len(data))]))
      print( '{}: {}, {} {} {}'.format(str(i*.25), 'plr', mae_plr, acc_plr, t) )
      mae_hpsm.append(mae_plr)
      acc_hpsm.append(acc_plr)
      mse_hpsm.append(mse_plr) 
      fft_hpsm.append(fft_plr)
      ################ li 
      li=np.array([a[2] for a in data])
      #li=np.array([cv2.resize(a, dsize=(64,64)) for a in li])
      t = datetime.datetime.now() ### time 
      li=np.array([interpolate.interp2d(np.arange(0,64,2), np.arange(0,64,2), a, kind='cubic') for a in li])
      t = (datetime.datetime.now() -t)/120 ### time 
      li=np.array([a(np.arange(0,64,1), np.arange(0,64,1)) for a in li])
      np.save("SCREENSHOTS/LI/{}.npy".format(i),li) #########
      fft_li=np.mean(np.array([fft(li[i]) for i in range(len(data))]),axis=0)
      mae_li=np.mean(np.array([mean_absolute_error(hr[i],li[i]) for i in range(len(data))]))
      mse_li=np.mean(np.array([mean_squared_error(hr[i],li[i]) for i in range(len(data))]))
      F=np.mean(li,axis=0)
      acc_li=np.mean(np.array([ACC(li[i],hr[i],F,O) for i in range(len(data))]))
      print( '{}: {}, {} {} {}'.format(str(i*.25), 'li', mae_li, acc_li, t) )
      mae_inter.append(mae_li)
      mse_inter.append(mse_li)
      acc_inter.append(acc_li)
      fft_inter.append(fft_li)
      ################ ddsm
      if i==1:
          phr_1=np.array([a[3] for a in data]) 
      in_dim=(64,64,1)
      phr_1=np.array([np.reshape(a,in_dim)*100000 for a in phr_1])
      t = datetime.datetime.now() ### time 
      phr_1=model_ddsm.predict(phr_1)
      t = (datetime.datetime.now() -t)/120 ### time
      phr_1=np.array([a[:,:,0]/100000 for a in phr_1])
      mn,mx=hr.min(),hr.max()
      phr_1=np.clip(phr_1,mn,mx)
      np.save("SCREENSHOTS/DDSM/{}.npy".format(i), phr_1) #########
      fft_phr_1=np.mean(np.array([fft(phr_1[i]) for i in range(len(data))]),axis=0)
      mae_phr_1=np.mean(np.array([mean_absolute_error(hr[i],phr_1[i]) for i in range(len(data))]))
      mse_phr_1=np.mean(np.array([mean_squared_error(hr[i],phr_1[i]) for i in range(len(data))]))
      F=np.mean(phr_1,axis=0)
      acc_phr_1=np.mean(np.array([ACC(phr_1[i],hr[i],F,O) for i in range(len(data))]))
      print( '{}: {}, {} {} {}'.format(str(i*.25), 'phr_1', mae_phr_1, acc_phr_1, t) )
      mae_ddsm.append(mae_phr_1)
      acc_ddsm.append(acc_phr_1)
      mse_ddsm.append(mse_phr_1)
      fft_ddsm.append(fft_phr_1)
      ################ combine
      phr0=np.array([a[4] for a in data])*ponder(i)
      in_dim=(64,64,1)
      phr0=np.array([np.reshape(a,in_dim)*100000 for a in phr0])
      t = datetime.datetime.now() ### time 
      cmb=model.predict([plra,phr0])
      t = (datetime.datetime.now() -t)/120 ### time
      cmb=np.array([a[:,:,0]/100000 for a in cmb])
      np.save("SCREENSHOTS/HPSM-HR0/{}.npy".format(i), cmb) #########
      fft_cmb=np.mean(np.array([fft(cmb[i]) for i in range(len(data))]),axis=0)
      mae_cmb=np.mean(np.array([mean_absolute_error(hr[i],cmb[i]) for i in range(len(data))]))
      mse_cmb=np.mean(np.array([mean_squared_error(hr[i],cmb[i]) for i in range(len(data))]))
      F=np.mean(cmb,axis=0)
      acc_cmb=np.mean(np.array([ACC(cmb[i],hr[i],F,O) for i in range(len(data))]))
      print( '{}: {}, {} {} {}'.format(str(i*.25), 'cmb', mae_cmb, acc_cmb, t) )
      mae_combine.append(mae_cmb)
      acc_combine.append(acc_cmb)
      mse_combine.append(mse_cmb)
      fft_combine.append(fft_cmb)
      
np.save('skills/mae/ref.npy',np.array(mae_ref))
np.save('skills/mae/hpsm.npy',np.array(mae_hpsm))
np.save('skills/mae/ddsm.npy',np.array(mae_ddsm))
np.save('skills/mae/li.npy',np.array(mae_inter))
np.save('skills/mae/cmb.npy',np.array(mae_combine))

np.save('skills/acc/ref.npy',np.array(acc_ref))
np.save('skills/acc/hpsm.npy',np.array(acc_hpsm))
np.save('skills/acc/ddsm.npy',np.array(acc_ddsm))
np.save('skills/acc/li.npy',np.array(acc_inter))
np.save('skills/acc/cmb.npy',np.array(acc_combine))

np.save('skills/mse/ref.npy',np.array(mse_ref))
np.save('skills/mse/hpsm.npy',np.array(mse_hpsm))
np.save('skills/mse/ddsm.npy',np.array(mse_ddsm))
np.save('skills/mse/li.npy',np.array(mse_inter))
np.save('skills/mse/cmb.npy',np.array(mse_combine))

np.save('skills/fft/0.npy',np.array(fft_0))
np.save('skills/fft/ref.npy',np.array(fft_ref))
np.save('skills/fft/hpsm.npy',np.array(fft_hpsm))
np.save('skills/fft/ddsm.npy',np.array(fft_ddsm))
np.save('skills/fft/li.npy',np.array(fft_inter))
np.save('skills/fft/cmb.npy',np.array(fft_combine))
