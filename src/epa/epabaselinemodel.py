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

X = epacomplete.loc[:, epacomplete.columns != "comb08"]
y = epacomplete["comb08"]

mlr_base = LinearRegression()
mlr_base.fit(X, y)
y_pred = mlr_base.predict(X)

mae_minmax = metrics.mean_absolute_error(y, y_pred)
mse_minmax = metrics.mean_squared_error(y, y_pred)
rmse_minmax = np.sqrt(metrics.mean_squared_error(y, y_pred))

print("R-squared : {:.2f}".format(mlr_base.score(X,y)*100))
print("Mean Absolute Error:", mae_minmax)
print("Mean Square Error:", mse_minmax)
print("Root Mean Square Error:", rmse_minmax)

#Use OLS to give a summary report
x = sm.add_constant(X)
result = sm.OLS(y, x).fit()
print(result.summary())

# #Test linearity assumption
linear_assumption(mlr_base, X, y)

#Test for normality of residuals
normal_errors_assumption(mlr_base, X, y)

#Test for multicollinearity
multicollinearity_assumption(mlr_base, X, y, X.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_base, X, y)