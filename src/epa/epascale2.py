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
from sklearn.preprocessing import PowerTransformer
from sklearn import metrics
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import scipy.stats as stats
from scipy.stats import yeojohnson
from linearassumptions import calculate_residuals, linear_assumption,normal_errors_assumption,multicollinearity_assumption, autocorrelation_assumption
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete1.pkl")

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(epacomplete)
epascaled = pd.DataFrame(scaler.fit_transform(epacomplete),
                         columns = epacomplete.columns, index = epacomplete.index)
def calc_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    print("Outliers Count")
    print(((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).sum())

# X = epascaled.loc[:, epacomplete.columns != "comb08"]
# y = epascaled.loc[:,"comb08"]

# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler.fit(X,y)
# X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
# print(X.head())
# y = pd.DataFrame(scaler.fit_transform(y), columns = y.columns)
# print(y.head())


# X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, random_state=345)
# #
# mlr_minmax = LinearRegression()
# mlr_minmax.fit(X_train, y_train)
# y_pred = mlr_minmax.predict(X_test)
#
# mae_minmax = metrics.mean_absolute_error(y_test, y_pred)
# mse_minmax = metrics.mean_squared_error(y_test, y_pred)
# rmse_minmax = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#
# print("R-squared : {:.2f}".format(mlr_minmax.score(X_train,y_train)*100))
# print("Mean Absolute Error:", mae_minmax)
# print("Mean Square Error:", mse_minmax)
# print("Root Mean Square Error:", rmse_minmax)
#
# #Use OLS to give a summary report
# x = sm.add_constant(X_train)
# result = sm.OLS(y_train, x).fit()
# print(result.summary())
#
# # #Test linearity assumption
# linear_assumption(mlr_minmax, X_train, y_train)
#
#
# #Test for normaility of residuals
# normal_errors_assumption(mlr_minmax, X_train, y_train)
#
# #Test for multicollinearity
# multicollinearity_assumption(mlr_minmax, X_train, y_train, X_train.columns)
#
# #Test for autocorrelation
# autocorrelation_assumption(mlr_minmax, X_train, y_train)
#
# folds = KFold(n_splits = 10, shuffle = True, random_state = 5758)
# alpha_values = [0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]
#
#
# lasso_mod = LassoCV(alphas = alpha_values, cv = folds)
# sfm = SelectFromModel(lasso_mod)
# sfm.fit(X_train, y_train)
# feature_sel = sfm.get_support()
# feature_names = X_train.columns[feature_sel]
# print(feature_names)


"""
Based on Lasso, here is our new model
"""
X = epascaled[['city08', 'co2TailpipeGpm', 'highway08', 'highway08U']]
y = epascaled["comb08"]
print(X.head())

# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler.fit(X,y)
# X = pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
# y = scaler.fit_transform(y)
#
#
#
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, random_state=345)

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

# #Test linearity assumption
linear_assumption(mlr_minmax, X_train, y_train)


#Test for normaility of residuals
normal_errors_assumption(mlr_minmax, X_train, y_train)

#Test for multicollinearity
multicollinearity_assumption(mlr_minmax, X_train, y_train, X.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_minmax, X_train, y_train)