import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from linearassumptions import calculate_residuals, linear_assumption,normal_errors_assumption,multicollinearity_assumption, autocorrelation_assumption
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()



epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete.pkl")
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
print(epacomplete.head())
print(epacomplete.dtypes)

#Check for nulls
print("Null Values List")
print(epacomplete.isna().sum())




"""Build a baseline model """
X = epacomplete.loc[:, epacomplete.columns != 'comb08']
y = epacomplete['comb08']

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)
mlr_base = LinearRegression()
mlr_base.fit(X, y)
y_pred = mlr_base.predict(X)

mae_base = metrics.mean_absolute_error(y, y_pred)
mse_base = metrics.mean_squared_error(y, y_pred)
rmse_base = np.sqrt(metrics.mean_squared_error(y, y_pred))

print("R-squared : {:.2f}".format(mlr_base.score(X,y)*100))
print("Mean Absolute Error:", mae_base)
print("Mean Square Error:", mse_base)
print("Root Mean Square Error:", rmse_base)

#Use OLS to give a summary report
x = sm.add_constant(X)
result_base = sm.OLS(y, x).fit()
print(result_base.summary())

#Test linearity assumption
linear_assumption(mlr_base, X, y)


#Test for normaility of residuals
normal_errors_assumption(mlr_base, X, y)

#Test for multicollinearity
multicollinearity_assumption(mlr_base, X, y, X.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_base, X, y)


X1 = epacomplete.iloc[:,[15, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
corrmatrix1 = X1.corr().round(2)
sns.heatmap(corrmatrix1, annot = True)
plt.show()

X2 = epacomplete.iloc[:,[15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
corrmatrix2 = X2.corr().round(2)
sns.heatmap(corrmatrix2, annot = True)
plt.show()

X3 = epacomplete.iloc[:,[15, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
corrmatrix3 = X3.corr().round(2)
sns.heatmap(corrmatrix3, annot = True)
plt.show()

X4 = epacomplete.iloc[:,[15, 46,47,48,49]]
corrmatrix4 = X4.corr().round(2)
sns.heatmap(corrmatrix4, annot = True)
plt.show()


# Visualize distributions for outliers
group1 = epacomplete.iloc[:,0:10]
group1_melt = pd.melt(group1)
sns.boxplot(x= "variable", y = "value", data = group1_melt)
plt.show()

group2 = epacomplete.iloc[:,11:21]
group2_melt = pd.melt(group2)
sns.boxplot(x= "variable", y = "value", data = group2_melt)
plt.show()

group3 = epacomplete.iloc[:,22:32]
group3_melt = pd.melt(group3)
sns.boxplot(x= "variable", y = "value", data = group3_melt)
plt.show()

group4 = epacomplete.iloc[:,33:43]
group4_melt = pd.melt(group4)
sns.boxplot(x= "variable", y = "value", data = group4_melt)
plt.show()

group5 = epacomplete.iloc[:,44:49]
group5_melt = pd.melt(group5)
sns.boxplot(x= "variable", y = "value", data = group5_melt)
plt.show()

# #Drop charge120 as it is not providing any correlation
epacomplete.drop('charge120', axis = 1, inplace = True)
epacomplete.to_pickle(f"{data_dir}\\epacomplete1.pkl")

"Use GridSearchCV to find the optimal number of features. Use RFE (Recursive Feature Elimination)"
# folds = KFold(n_splits = 10, shuffle = True, random_state = 5758)
# hyper_params = [{'n_features_to_select': list(range(len(epacomplete.columns)))}]
# lm2 = LinearRegression()
# lm2.fit(X, y)
# rfe = RFE(lm2)
#
# grid_cv = GridSearchCV(estimator = rfe,
#                        param_grid = hyper_params,
#                        scoring = 'r2',
#                        cv = folds,
#                        verbose = 1,
#                        return_train_score = True)
#
#
# grid_cv.fit(X, y)
#
# cv_results = pd.DataFrame(grid_cv.cv_results_)
# print(cv_results)
