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

X = epascaled.loc[:, epascaled.columns != 'comb08']
y = epascaled['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)
mlr_minmax = LinearRegression()
mlr_minmax.fit(X_train, y_train)
y_pred = mlr_minmax.predict(X_test)

mae_base = metrics.mean_absolute_error(y_test, y_pred)
mse_base = metrics.mean_squared_error(y_test, y_pred)
rmse_base = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_minmax.score(X_train,y_train)*100))
print("Mean Absolute Error:", mae_base)
print("Mean Square Error:", mse_base)
print("Root Mean Square Error:", rmse_base)

#Use OLS to give a summary report
x = sm.add_constant(X_train)
result_base = sm.OLS(y_train, x).fit()
print(result_base.summary())

# #Test linearity assumption
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

"Use GridSearchCV to find the optimal number of features. Use RFE (Recursive Feature Elimination)"
folds = KFold(n_splits = 10, shuffle = True, random_state = 5758)
hyper_params = [{'n_features_to_select': list(range(len(epascaled.columns)))}]
rfe = RFE(mlr_minmax)

grid_cv = GridSearchCV(estimator = rfe,
                       param_grid = hyper_params,
                       scoring = 'r2',
                       cv = folds,
                       verbose = 1,
                       return_train_score = True)


grid_cv.fit(X_train, y_train)

cv_results = pd.DataFrame(grid_cv.cv_results_)
print(cv_results)

num_features = 35
lmfinal = LinearRegression()
lmfinal.fit(X_train, y_train)

rfe = RFE(lmfinal, n_features_to_select=num_features)
rfe = rfe.fit(X_train, y_train)
print(X_train.shape)
for i in range(X_train.shape[1]):
    print('Column: %s, Selected %s, Rank: %.3f' % (X_train.columns.values[i], rfe.support_[i], rfe.ranking_[i]))
# print(rfe.support_)
# print(rfe.ranking_)

X2 = epascaled.loc[:, ~epascaled.columns.isin(['co2', 'co2A', 'co2TailpipeAGpm', 'co2TailpipeGpm', 'fuelCostA08', 'hlv',
                                           'hpv', 'lv2', 'lv4', 'pv2', 'pv4', 'rangeCityA', 'UHighwayA', 'comb08'])]
print(X2.columns)
y2 = epascaled['comb08']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.7, random_state = 245)
mlr_final = LinearRegression()
mlr_final.fit(X_train2, y_train2)

y_pred2 = mlr_final.predict(X_test2)
r2 =  metrics.r2_score(y_test2, y_pred2)
print(r2)
print(mlr_final.coef_)
print(mlr_final.n_features_in_)
print(mlr_final.feature_names_in_)

# #Test linearity assumption
linear_assumption(mlr_final, X_train2, y_train2)


#Test for normaility of residuals
normal_errors_assumption(mlr_final, X_train2, y_train2)

#Test for multicollinearity
multicollinearity_assumption(mlr_final, X_train2, y_train2, X_train2.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_final, X_train2, y_train2)