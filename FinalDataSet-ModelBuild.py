# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 02:16:32 2021

@author: aravi
"""

import pandas as pd
import os
# os.environ["KERAS_BACKEND"] = "tensorflow.keras.backend"
import numpy as np
import h5py, array
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow. keras.layers import Dense
from tensorflow.keras.layers import Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv1D, SeparableConv1D
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import random
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt
import datetime
import tensorflow.keras.backend as kb
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

#%%
output_directory = r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\models'
output_model_name = 'best_model_cnn_cce2018_p2.h5'

#%%

inputdataset = r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\model-series-m1.h5'

cropfinalplots = r"C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\FinalDataset-Yield.xlsx"

# final/selected crop plots
pdf = pd.read_excel(cropfinalplots)

crop_list = pdf['crop_name'].unique().tolist()

h5f = h5py.File(inputdataset,'r')
X = h5f['X'][:]
y=h5f['Y'][:]

h5f.close()

X.shape
y.shape

#array_sum = np.sum(X)
#array_has_nan = np.isnan(array_sum)

#np.isnan(np.min(X[:,0,:]))

#np.isnan(np.min(X))

for i in range(X.shape[0]):
    if np.isnan(np.min(X[i,:,:])):
        print(i)

#%%

yieldnorm=[95,7,9,27,15,35,19,20,15]

#yieldnorm=[107.5, 8.5, 9.5, 28.5, 20.0, 18.5, 37.5, 22.0, 20.0, 15.0]

#yieldnorm=[120,10,10,30,25,20,40,25,20,15]

yielddict={}

for index in range(6,15):
    yielddict[index]=yieldnorm[index-6]






#%%

## j takes X columns- 3 to 12
for i in range(X.shape[0]):
    for j in range(6,15):
        if X[i,:,j][0]==1:
            y[i]=y[i]/yielddict[j]
            
#%%

series=[1, 10.5, 1.4, 180, 47, 30 ]




#%%

X[:,:,0]= X[:,:,0]/1
X[:,:,1]= X[:,:,1]/10.8
X[:,:,2]= X[:,:,2]/1.4
X[:,:,3]= X[:,:,3]/180
X[:,:,4]= X[:,:,4]/47
X[:,:,5]= X[:,:,5]/30

cropvalue=[]
temp=pd.DataFrame(X[:,0,6:15])

temp=np.array(temp)

for i in range(temp.shape[0]):
    cropvalue.append(np.argmax(temp[i]))


#%%


 
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=cropvalue)

cropvalue=[]
temp=pd.DataFrame(Xtest[:,0,6:15])
temp=np.array(temp)
for i in range(temp.shape[0]):
    cropvalue.append(np.argmax(temp[i]))

Xtest,Xvalid, ytest, yvalid = train_test_split(Xtest, ytest, test_size=0.3, random_state=42, stratify=cropvalue)


temp=pd.DataFrame(Xtrain[:,0,6:15])
cropvalue=[]
temp=np.array(temp)

for i in range(temp.shape[0]):
    cropvalue.append(np.argmax(temp[i]))

cropvalue=pd.DataFrame(cropvalue)

print(cropvalue.value_counts())

# =============================================================================
# for i in range(11):
#     print(cropvalue.count(i))
# =============================================================================
    

temp=pd.DataFrame(Xtest[:,0,6:15])
cropvalue=[]
temp=np.array(temp)

for i in range(temp.shape[0]):
    cropvalue.append(np.argmax(temp[i]))
    
cropvalue=pd.DataFrame(cropvalue)

print(cropvalue.value_counts())
    
temp=pd.DataFrame(Xvalid[:,0,6:15])

temp=np.array(temp)
cropvalue=[]
for i in range(temp.shape[0]):
    cropvalue.append(np.argmax(temp[i]))
    
cropvalue=pd.DataFrame(cropvalue)

print(cropvalue.value_counts())





#%%



Xnewtrain=np.zeros((1507,215,17))

ynewtrain=np.zeros((1507,1))

Xnewtrain[0:137,:,:]= Xtest
ynewtrain[0:137]=ytest
count=137
county=137
#%%

def jitter(x,num_list, sigma=0.03):
    for i in range(len(x)):
        if i in num_list:
            x[i]= x[i] +  np.random.normal(loc=0., scale=sigma)
    return x


#%%

for ytemp in ytest:
    ynewtrain[county:county+10,:]=ytemp
    county=county+10

for Xtemp in Xtest:
    duration=np.count_nonzero(Xtemp[:,0])
    ndvi_jitter =[]
    vhvv_jitter =[]
    b7b6_jitter =[]
    num_list=[]
    
    for i in range(20,48,3):
        num_list.append(random.sample(range(0,duration),i))
        
    ndvi_season= Xtemp[0:duration,0]
    vhvv_season= Xtemp[0:duration,1]
    b7b6_season= Xtemp[0:duration,2]
    ndvi_season=pd.Series(ndvi_season)
    vhvv_season=pd.Series(vhvv_season)
    b7b6_season=pd.Series(b7b6_season)
    
    for i in range(0,10):
        ndvitemp=ndvi_season.copy()
        vhvvtemp=vhvv_season.copy()
        b7b6temp=b7b6_season.copy()
        ndvi_jitter.append(jitter(ndvi_season,num_list[i],sigma=0.01))
        vhvv_jitter.append(jitter(vhvv_season,num_list[i],sigma=0.01))
        b7b6_jitter.append(jitter(b7b6_season,num_list[i],sigma=0.01))
        ndvi_season=ndvitemp
        vhvv_season=vhvvtemp
        b7b6_season=b7b6temp
        
        
    


    for i in range(0,10):
        dates = vhvv_jitter[i].index
        vhvv_jitter[i] = savgol_filter(vhvv_jitter[i], window_length = 47, polyorder = 1)
        vhvv_jitter[i] = pd.Series(vhvv_jitter[i], dates)
        
        dates = ndvi_jitter[i].index
        ndvi_jitter[i] = savgol_filter(ndvi_jitter[i], window_length = 51, polyorder = 1)
        ndvi_jitter[i] = pd.Series(ndvi_jitter[i], dates)
        
        dates = b7b6_jitter[i].index
        b7b6_jitter[i] = savgol_filter(b7b6_jitter[i], window_length = 51, polyorder = 1)
        b7b6_jitter[i] = pd.Series(b7b6_jitter[i], dates)
        

# =============================================================================
#     for i in range(0,5):
#         ndvi_final_jitter[i][0:duration] = ndvi_jitter[i]
#         vhvv_final_jitter[i][0:duration] = vhvv_jitter[i]
#         b7b6_final_jitter[i][0:duration] = b7b6_jitter[i]
# =============================================================================
        
    for i in range(0,10):
        Xnewtrain[count+i,0:duration,0]=ndvi_jitter[i]
        Xnewtrain[count+i,0:duration,1]=vhvv_jitter[i]
        Xnewtrain[count+i,0:duration,2]=b7b6_jitter[i]
        
        Xnewtrain[count+i,:,3:17]=Xtemp[:,3:17]
        
    
    count=count+10
        
    
idxs = [i for i in range(Xnewtrain.shape[0])]
random.shuffle(idxs)                            

Xnewtrain=Xnewtrain[idxs,:,:]
ynewtrain=ynewtrain[idxs,:]



#%%

#define optimizer
opt = Adam(lr = 0.001)
n_steps = Xnewtrain.shape[1]
n_features = Xnewtrain.shape[2]
model = Sequential()
model.add(Conv1D(64,5, input_shape = (n_steps,n_features,), activation = 'relu'))
#model.add(MaxPooling1D(2))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Conv1D(128,5, activation  = 'relu'))
#model.add(MaxPooling1D(2))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(Conv1D(64,3, activation  = 'relu'))
#model.add(MaxPooling1D(2))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation  = 'relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(16, activation  = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=opt, loss='mse', metrics = ['mae'])



#%%
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=10, min_lr=0.000001, verbose =1)
file_path = output_directory + '/' + output_model_name
model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_mae',save_best_only=True, verbose = 1)
early_stopping = EarlyStopping(monitor = 'val_mae',min_delta = 0.001, patience  = 15, verbose =1  )
callbacks = [reduce_lr, model_checkpoint, early_stopping]
#%%

model.fit(Xnewtrain,ynewtrain, epochs = 500, batch_size = 64, validation_data = (Xvalid, yvalid),  verbose = 1, shuffle = True, callbacks = callbacks)

#%%

model = load_model(file_path)



model.evaluate(Xtrain,ytrain)
#%%

pred=model.predict(Xtrain)

from scipy.stats import pearsonr

predflat=pred.flatten()
ytestflat=ytrain.flatten()

corr, _ = pearsonr(predflat, ytestflat)
print('Pearsons correlation: %.3f' % corr)

#%%



import matplotlib.pyplot as plt
plt.plot(ytestflat,'g*', predflat, 'ro')
plt.show()

#%%
