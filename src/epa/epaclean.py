"""
Module for cleaning our data
epaclean.py
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

#Drop any columns with 50% or more values. Also drop certain attributes that will not add value to a model
epadata = epadata.drop(["guzzler", "trans_dscr", "tCharger", "sCharger", "atvType", "fuelType2", "rangeA",
                        "evMotor", "mfrCode", "c240Dscr","charge240b", "c240bDscr", "startStop",
                        "createdOn", "modifiedOn", "engId", "feScore", "ghgScore", "ghgScoreA", "id",
                        "mpgData", "phevBlended", "year"], axis = 1)

#Drop all non-numeric values
epacomplete = epadata.copy()
epacomplete = epacomplete.apply(pd.to_numeric, errors = 'coerce').dropna(axis = 1)

"""
EDA again on newly formed data
"""

print(epacomplete.info())
print(epacomplete.isna().sum())
calc_missing_values(epacomplete)
calc_outliers(epacomplete)

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


