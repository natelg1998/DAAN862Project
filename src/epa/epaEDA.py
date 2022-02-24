"""
Module that performs our exploratory analysis
epaEDA.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from epatransformtools import calc_missing_values,  calc_outliers


os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()



epadata = pd.read_pickle(f"{data_dir}\\epadata.pkl")
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
print(epadata.head())
print(epadata.dtypes)


#Check for nulls
print("Null Values List")
print(epadata.isna().sum())
calc_missing_values(epadata)
print(epadata.describe())
print(epadata.info())


X1 = epadata.iloc[:,[15, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
corrmatrix1 = X1.corr().round(2)
sns.heatmap(corrmatrix1, annot = True)
plt.show()

X2 = epadata.iloc[:,[15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
corrmatrix2 = X2.corr().round(2)
sns.heatmap(corrmatrix2, annot = True)
plt.show()

X3 = epadata.iloc[:,[15, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
corrmatrix3 = X3.corr().round(2)
sns.heatmap(corrmatrix3, annot = True)
plt.show()

X4 = epadata.iloc[:,[15, 46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]]
corrmatrix4 = X4.corr().round(2)
sns.heatmap(corrmatrix4, annot = True)
plt.show()

X5 = epadata.iloc[:,[15, 61,62,63,64,65,66,67,68,69,70,71,72,73,74,75]]
corrmatrix5 = X5.corr().round(2)
sns.heatmap(corrmatrix5, annot = True)
plt.show()

X6 = epadata.iloc[:,[15, 76,77,78,79,80,81,82]]
corrmatrix6 = X6.corr().round(2)
sns.heatmap(corrmatrix6, annot = True)
plt.show()


# Visualize distributions for outliers
# group1 = epadata.iloc[:,0:10]
# group1_melt = pd.melt(group1)
# sns.boxplot(x= "variable", y = "value", data = group1_melt)
# plt.show()
#
# group2 = epadata.iloc[:,11:21]
# group2_melt = pd.melt(group2)
# sns.boxplot(x= "variable", y = "value", data = group2_melt)
# plt.show()
#
# group3 = epadata.iloc[:,22:32]
# group3_melt = pd.melt(group3)
# sns.boxplot(x= "variable", y = "value", data = group3_melt)
# plt.show()
#
# group4 = epadata.iloc[:,33:43]
# group4_melt = pd.melt(group4)
# sns.boxplot(x= "variable", y = "value", data = group4_melt)
# plt.show()
#
# group5 = epadata.iloc[:,44:54]
# group5_melt = pd.melt(group5)
# sns.boxplot(x= "variable", y = "value", data = group5_melt)
# plt.show()
#
# group6 = epadata.iloc[:,55:65]
# group6_melt = pd.melt(group6)
# sns.boxplot(x= "variable", y = "value", data = group6_melt)
# plt.show()
#
# group7 = epadata.iloc[:,66:76]
# group7_melt = pd.melt(group7)
# sns.boxplot(x= "variable", y = "value", data = group7_melt)
# plt.show()
#
# group7 = epadata.iloc[:,77:82]
# group7_melt = pd.melt(group7)
# sns.boxplot(x= "variable", y = "value", data = group7_melt)
# plt.show()

# #Drop charge120 as it is not providing any correlation
# epadata.drop('charge120', axis = 1, inplace = True)
# epadata.to_pickle(f"{data_dir}\\epadata1.pkl")

# """Build a baseline model """
# X = epadata.loc[:, epadata.columns != 'comb08']
# y = epadata['comb08']
#
# # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)
# mlr_base = LinearRegression()
# mlr_base.fit(X, y)
# y_pred = mlr_base.predict(X)
#
# mae_base = metrics.mean_absolute_error(y, y_pred)
# mse_base = metrics.mean_squared_error(y, y_pred)
# rmse_base = np.sqrt(metrics.mean_squared_error(y, y_pred))
#
# print("R-squared : {:.2f}".format(mlr_base.score(X,y)*100))
# print("Mean Absolute Error:", mae_base)
# print("Mean Square Error:", mse_base)
# print("Root Mean Square Error:", rmse_base)
#
# #Use OLS to give a summary report
# x = sm.add_constant(X)
# result_base = sm.OLS(y, x).fit()
# print(result_base.summary())
#
# #Test linearity assumption
# linear_assumption(mlr_base, X, y)
#
#
# #Test for normaility of residuals
# normal_errors_assumption(mlr_base, X, y)
#
# #Test for multicollinearity
# multicollinearity_assumption(mlr_base, X, y, X.columns)
#
# #Test for autocorrelation
# autocorrelation_assumption(mlr_base, X, y)
#
#
# X1 = epadata.iloc[:,[15, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
# corrmatrix1 = X1.corr().round(2)
# sns.heatmap(corrmatrix1, annot = True)
# plt.show()
#
# X2 = epadata.iloc[:,[15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
# corrmatrix2 = X2.corr().round(2)
# sns.heatmap(corrmatrix2, annot = True)
# plt.show()
#
# X3 = epadata.iloc[:,[15, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
# corrmatrix3 = X3.corr().round(2)
# sns.heatmap(corrmatrix3, annot = True)
# plt.show()
#
# X4 = epadata.iloc[:,[15, 46,47,48,49]]
# corrmatrix4 = X4.corr().round(2)
# sns.heatmap(corrmatrix4, annot = True)
# plt.show()
#
#
# # Visualize distributions for outliers
# group1 = epadata.iloc[:,0:10]
# group1_melt = pd.melt(group1)
# sns.boxplot(x= "variable", y = "value", data = group1_melt)
# plt.show()
#
# group2 = epadata.iloc[:,11:21]
# group2_melt = pd.melt(group2)
# sns.boxplot(x= "variable", y = "value", data = group2_melt)
# plt.show()
#
# group3 = epadata.iloc[:,22:32]
# group3_melt = pd.melt(group3)
# sns.boxplot(x= "variable", y = "value", data = group3_melt)
# plt.show()
#
# group4 = epadata.iloc[:,33:43]
# group4_melt = pd.melt(group4)
# sns.boxplot(x= "variable", y = "value", data = group4_melt)
# plt.show()
#
# group5 = epadata.iloc[:,44:49]
# group5_melt = pd.melt(group5)
# sns.boxplot(x= "variable", y = "value", data = group5_melt)
# plt.show()
#
# # #Drop charge120 as it is not providing any correlation
# epadata.drop('charge120', axis = 1, inplace = True)
# epadata.to_pickle(f"{data_dir}\\epadata1.pkl")