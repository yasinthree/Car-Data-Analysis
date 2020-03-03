# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:33:15 2020

@author: hp
"""
import pandas as pd 
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
df = pd.read_csv('https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv')
df.info()
df.isnull().any()
df.dtypes
df['horsepower'] = pd.to_numeric(df['horsepower'],errors ='coerce')
print(df[pd.to_numeric(df['horsepower'],errors = 'coerce').isnull()])
df = df.dropna()
df['model_year'].unique()

df = df.drop(['name', 'origin', 'model_year'],axis = 1)

df.columns

df['acceleration'].unique()
y = df['mpg']
cylinders = pd.get_dummies(df['cylinders'],drop_first = True)

df = pd.concat([df,cylinders],axis = 1)

df = df.drop(['mpg', 'cylinders','displacement'], axis = 1)

x = df.copy()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=1 )

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2_score(y_test,y_pred)





















