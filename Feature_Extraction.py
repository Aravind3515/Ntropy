# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 03:30:15 2021

@author: aravi
"""

#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor




#%%

dataset=r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\datasetupdate.h5'

data = pd.read_hdf(dataset)



#%%

X=data[[i for i in list(data.columns) if i != 'Yield']]

y=data[[i for i in list(data.columns) if i == 'Yield']]


labels = np.array(y['Yield']/119)

feature_list = list(X.columns)

features=np.array(X)
#%%

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#%%

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


#%%
################################   Random Forest Regressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)

rf.fit(train_features, train_labels)



#%%

rf.feature_importances_

#%%

plt.barh(feature_list, rf.feature_importances_)

#%%

a=np.array(feature_list)
sorted_idx = rf.feature_importances_.argsort()
plt.barh(a[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

plt.xaxis.set_major_locator(plt.MaxNLocator(10))
#%%

featureimportances=rf.feature_importances_

featurelist=feature_list

df=pd.DataFrame(columns=featurelist)

df.loc[0]=featureimportances

df.to_csv('Feature_Values.csv')

#%%

pred=rf.predict(test_features)

#pred_list = [item for sublist in pred for item in sublist]
df2 = pd.DataFrame()
df2['actuals'] = test_labels.astype('float32')/100.
df2['preds'] = pred
r = df2['actuals'].corr(df2['preds'])

#%%

############################################ XGBoost
import xgboost
from xgboost import XGBClassifier,XGBRegressor


#%%


#%%



#%%

gb=xgboost.XGBRegressor(eval_metric = 'rmse',n_estimators = 2000,learning_rate = 0.01,seed=162,random_state = 162,colsample_bytree=0.65)

gb.fit(train_features,train_labels)
#%%

plt.bar(feature_list, gb.feature_importances_)

#%%

featureimport=gb.feature_importances_

featurelist=feature_list

af=pd.DataFrame(columns=featurelist)

af.loc[0]=featureimport

af.to_csv('Feature_Values(XGBoost).csv')

#%%

pred=gb.predict(test_features)
af2 = pd.DataFrame()
af2['actuals'] = test_labels.astype('float32')/100.
af2['preds'] = pred
ar = af2['actuals'].corr(af2['preds'])

#%%


############################################ CNN Model


#%%

n_features=1
train_features = train_features.reshape((train_features.shape[0], train_features.shape[1], n_features))

n_timesteps, n_features, n_outputs = train_features.shape[1], train_features.shape[2], train_features.shape[1]
#%%

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
#create model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(train_features,train_labels,epochs=10)

#%%

test_features = test_features.reshape((test_features.shape[0], test_features.shape[1], n_features))

pred=model.predict(test_features)

#%%

