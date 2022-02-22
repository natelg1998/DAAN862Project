"""
Module for us building models with our miceforest imputed data
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import scipy.stats as stats
from scipy.stats import yeojohnson
from linearassumptions import calculate_residuals, linear_assumption,normal_errors_assumption,multicollinearity_assumption, autocorrelation_assumption
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from epatransformtools import calc_missing_values,  calc_outliers
from sklearn.pipeline import Pipeline

os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

epamfimpute = pd.read_pickle(f"{data_dir}\\epadatamfimputed.pkl")

#Let's compare outliers in both models
print("Iterative Imputed Outliers")
calc_outliers(epamfimpute)

print("Miceforest Imputed Outliers")
calc_outliers(epamfimpute)


scaler = MinMaxScaler(feature_range=(-1, 1))
pt = PowerTransformer(method = "yeo-johnson")
pipeline = Pipeline(steps =[('s', scaler), ('p', pt)])

#Model with MFImpute data
epaiterdf = pd.DataFrame(pipeline.fit_transform(epamfimpute), index = epamfimpute.index, columns = epamfimpute.columns)

#Split features from target
X = epamfimpute.loc[:, epamfimpute.columns != 'comb08']
y = epamfimpute['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)

mlr_iterimp = LinearRegression()
mlr_iterimp.fit(X_train, y_train)
y_pred = mlr_iterimp.predict(X_test)

mae_minmax = metrics.mean_absolute_error(y_test, y_pred)
mse_minmax = metrics.mean_squared_error(y_test, y_pred)
rmse_minmax = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_iterimp.score(X_train,y_train)*100))
print("Mean Absolute Error:", mae_minmax)
print("Mean Square Error:", mse_minmax)
print("Root Mean Square Error:", rmse_minmax)

#Use OLS to give a summary report
x = sm.add_constant(X_train)
result = sm.OLS(y_train, x).fit()
print(result.summary())
#
# #Test linearity assumption
# linear_assumption(mlr_iterimp, X_train, y_train)
#
# #Test for normality of residuals
# normal_errors_assumption(mlr_iterimp, X_train, y_train)
#
# #Test for multicollinearity
# multicollinearity_assumption(mlr_iterimp, X_train, y_train, X.columns)
#
# #Test for autocorrelation
# autocorrelation_assumption(mlr_iterimp, X_train, y_train)

"""LassoCV """
folds = KFold(n_splits = 10, shuffle = True, random_state = 5758)
alpha_values = [0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]


lasso_mod = LassoCV(alphas = alpha_values, cv = folds)
sfm = SelectFromModel(lasso_mod)
sfm.fit(X_train, y_train)
feature_sel = sfm.get_support()
feature_names = X_train.columns[feature_sel]
print(feature_names)

#Based on Lasso Regression, below are the features selected
X = epamfimpute[['barrels08', 'city08', 'city08U', 'co2', 'co2TailpipeGpm', 'comb08U',
       'fuelCost08', 'highway08', 'highway08U', 'lv4', 'pv4', 'UCity',
       'UHighway', 'youSaveSpend']]

y = epamfimpute['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)

mlr_iterimp = LinearRegression()
mlr_iterimp.fit(X_train, y_train)
y_pred = mlr_iterimp.predict(X_test)

mae_minmax = metrics.mean_absolute_error(y_test, y_pred)
mse_minmax = metrics.mean_squared_error(y_test, y_pred)
rmse_minmax = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_iterimp.score(X_train,y_train)*100))
print("Mean Absolute Error:", mae_minmax)
print("Mean Square Error:", mse_minmax)
print("Root Mean Square Error:", rmse_minmax)

#Use OLS to give a summary report
x = sm.add_constant(X_train)
result = sm.OLS(y_train, x).fit()
print(result.summary())
#
# # #Test linearity assumption
# linear_assumption(mlr_iterimp, X_train, y_train)
#
# #Test for normality of residuals
# normal_errors_assumption(mlr_iterimp, X_train, y_train)
#
# #Test for multicollinearity
# multicollinearity_assumption(mlr_iterimp, X_train, y_train, X.columns)
#
# #Test for autocorrelation
# autocorrelation_assumption(mlr_iterimp, X_train, y_train)

#Drop features based on VIF values
X = epamfimpute[[ 'city08U', 'co2',
         'lv4',  'UCity',
        'youSaveSpend']]

y = epamfimpute['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)

mlr_iterimp = LinearRegression()
mlr_iterimp.fit(X_train, y_train)
y_pred = mlr_iterimp.predict(X_test)

mae_minmax = metrics.mean_absolute_error(y_test, y_pred)
mse_minmax = metrics.mean_squared_error(y_test, y_pred)
rmse_minmax = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_iterimp.score(X_train,y_train)*100))
print("Mean Absolute Error:", mae_minmax)
print("Mean Square Error:", mse_minmax)
print("Root Mean Square Error:", rmse_minmax)

#Use OLS to give a summary report
x = sm.add_constant(X_train)
result = sm.OLS(y_train, x).fit()
print(result.summary())
#
# # #Test linearity assumption
linear_assumption(mlr_iterimp, X_train, y_train)

#Test for normality of residuals
normal_errors_assumption(mlr_iterimp, X_train, y_train)

#Test for multicollinearity
multicollinearity_assumption(mlr_iterimp, X_train, y_train, X.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_iterimp, X_train, y_train)