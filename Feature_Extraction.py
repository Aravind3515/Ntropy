# -*- coding: utf-8 -*-
"""
@author: aravi
"""

#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


dataset=r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\datasetupdate.h5'

data = pd.read_hdf(dataset)



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


############################################ XGBoost
import xgboost
from xgboost import XGBClassifier,XGBRegressor

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

