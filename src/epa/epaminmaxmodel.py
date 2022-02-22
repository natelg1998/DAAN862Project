"""
Module that uses MinMaxScaler to scale our data and build a model
epaminmaxmodel.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import scipy.stats as stats
from linearassumptions import calculate_residuals, linear_assumption,normal_errors_assumption,multicollinearity_assumption, autocorrelation_assumption
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from epatransformtools import calc_outliers
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

"""Tried this for building a model with MinMaxScaler. Did not work out as well as yeo-johnson"""
os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete1.pkl")

#Scale data using MinMax - we have negative values so MinMax will work instead of StandardScaler
#Also create a Pipeline to use yeo-johnson to normailize the distribution
scaler = MinMaxScaler(feature_range=(-1, 1))
pt = PowerTransformer(method = "yeo-johnson")
pipeline = Pipeline(steps =[('s', scaler), ('p', pt)])
epacomplete = pd.DataFrame(pipeline.fit_transform(epacomplete), index = epacomplete.index, columns = epacomplete.columns)
calc_outliers(epacomplete)

#Split features from target
X = epacomplete.loc[:, epacomplete.columns != 'comb08']
y = epacomplete['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)


mlr_minmax = LinearRegression()
mlr_minmax.fit(X_train, y_train)
y_pred = mlr_minmax.predict(X_test)

mae_minmax = metrics.mean_absolute_error(y_test, y_pred)
mse_minmax = metrics.mean_squared_error(y_test, y_pred)
rmse_minmax = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_minmax.score(X_train,y_train)*100))
print("Mean Absolute Error:", mae_minmax)
print("Mean Square Error:", mse_minmax)
print("Root Mean Square Error:", rmse_minmax)

#Use OLS to give a summary report
x = sm.add_constant(X_train)
result = sm.OLS(y_train, x).fit()
print(result.summary())
#
# # #Test linearity assumption
# linear_assumption(mlr_minmax, X_train, y_train)
#
#
# #Test for normaility of residuals
# normal_errors_assumption(mlr_minmax, X_train, y_train)
#
# #Test for multicollinearity
# multicollinearity_assumption(mlr_minmax, X_train, y_train, X.columns)
#
# #Test for autocorrelation
# autocorrelation_assumption(mlr_minmax, X_train, y_train)

"""LassoCV """
folds = KFold(n_splits = 10, shuffle = True, random_state = 5758)
alpha_values = [0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]


lasso_mod = LassoCV(alphas = alpha_values, cv = folds)
sfm = SelectFromModel(lasso_mod)
sfm.fit(X_train, y_train)
feature_sel = sfm.get_support()
feature_names = X_train.columns[feature_sel]
print(feature_names)

#Select the following variables from Lasso
X = epacomplete[['charge240', 'city08', 'city08U', 'cityCD', 'cityE', 'co2TailpipeGpm',
       'fuelCost08', 'highway08', 'hpv', 'pv4', 'range', 'rangeCity', 'UCity',
       'UCityA', 'UHighwayA']]

y = epacomplete['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)


mlr_minmax = LinearRegression()
mlr_minmax.fit(X_train, y_train)
y_pred = mlr_minmax.predict(X_test)

mae_minmax = metrics.mean_absolute_error(y_test, y_pred)
mse_minmax = metrics.mean_squared_error(y_test, y_pred)
rmse_minmax = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_minmax.score(X_train,y_train)*100))
print("Mean Absolute Error:", mae_minmax)
print("Mean Square Error:", mse_minmax)
print("Root Mean Square Error:", rmse_minmax)

#Use OLS to give a summary report
x = sm.add_constant(X_train)
result = sm.OLS(y_train, x).fit()
print(result.summary())
#
# # #Test linearity assumption
# linear_assumption(mlr_minmax, X_train, y_train)
#
#
# #Test for normaility of residuals
# normal_errors_assumption(mlr_minmax, X_train, y_train)
#
# #Test for multicollinearity
# multicollinearity_assumption(mlr_minmax, X_train, y_train, X.columns)
#
# #Test for autocorrelation
# autocorrelation_assumption(mlr_minmax, X_train, y_train)

#Drop variables by VIF
X = epacomplete[['charge240', 'city08U', 'cityCD',
       'fuelCost08', 'hpv', 'pv4', 'rangeCity',
        'UHighwayA']]

y = epacomplete['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)


mlr_minmax = LinearRegression()
mlr_minmax.fit(X_train, y_train)
y_pred = mlr_minmax.predict(X_test)

mae_minmax = metrics.mean_absolute_error(y_test, y_pred)
mse_minmax = metrics.mean_squared_error(y_test, y_pred)
rmse_minmax = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_minmax.score(X_train,y_train)*100))
print("Mean Absolute Error:", mae_minmax)
print("Mean Square Error:", mse_minmax)
print("Root Mean Square Error:", rmse_minmax)

#Use OLS to give a summary report
x = sm.add_constant(X_train)
result = sm.OLS(y_train, x).fit()
print(result.summary())
#
# # #Test linearity assumption
linear_assumption(mlr_minmax, X_train, y_train)


#Test for normaility of residuals
normal_errors_assumption(mlr_minmax, X_train, y_train)

#Test for multicollinearity
multicollinearity_assumption(mlr_minmax, X_train, y_train, X.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_minmax, X_train, y_train)
