# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization
from termcolor import colored as cl # text customization
import warnings                                 
warnings.filterwarnings('ignore')

# Loading the dataset 
df = pd.read_csv("C:\\Users\\Donatus\\Documents\\HousePricePrediction\\House_Data.csv")
df.set_index('Id', inplace = True) # We set the Id col as the index column

# Ckeing the nulls and the dtype at once
df = df.dropna(axis = 0)

# Droping the GrLiArea column
df.drop('GrLivArea', axis = 1, inplace = True)

# Identify the important features from the correlation above
imprt_features = df.drop(columns='SalePrice')
y_var = df['SalePrice'].values #Finding the value of Y

# Modeling 
from sklearn.model_selection import train_test_split # data split
from sklearn.linear_model import LinearRegression # OLS algorithm

from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Standard scalling our input_columns towards modelling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
x_scaled = MinMaxScaler().fit_transform(imprt_features)

# Spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_var, test_size = 0.2, random_state = 0)

# OLS
ols = LinearRegression()
ols.fit(X_train, y_train)
ols_yhat = ols.predict(X_test)

# Explained Variance Score

print(cl('EXPLAINED VARIANCE SCORE:', attrs = ['bold']))
print(cl('Explained Variance Score of OLS model is {}'.format(evs(y_test, ols_yhat))))

# R-squared
print(cl('R-SQUARED:', attrs = ['bold']))
print(cl('R-Squared of OLS model is {}'.format(r2(y_test, ols_yhat))))

# Save the model
import pickle 
filename = 'hsmodel.pkl'
pickle.dump(ols, open(filename, 'wb'))