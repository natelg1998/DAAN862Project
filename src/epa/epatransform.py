"""
This module is for transforming our data for being utilized in a models

"""
import pandas as pd
import numpy as np
import miceforest as mf
import pickle
import os
from pathlib import Path
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import miceforest as mf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import scipy.stats as stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

epascaled = pd.read_pickle(f"{data_dir}\\epascaled.pkl")
print(epascaled.head())

def calc_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    print("Outliers Count")
    print(((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).sum())

epafeatures = epascaled.loc[:, epascaled.columns != 'comb08']
print(epafeatures.info())

calc_outliers(epafeatures)

Q1 = epafeatures.quantile(q=.25)
Q3 = epafeatures.quantile(q=.75)
IQR = epafeatures.apply(stats.iqr)
lower_bound = Q1 - 1.5 * IQR
print(lower_bound)
upper_bound = Q3 + 1.5 * IQR
#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3

epaclean1 = epafeatures.copy()
for name in epaclean1.columns:
    epaclean1.loc[(epaclean1[name] < lower_bound[name]) | (epaclean1[name] > upper_bound[name]), name] = np.nan
print(epaclean1.head(15))

#Now that we cleaned the data removing the outliers, we will use MICE to impute for the outliers
lr = LinearRegression()
imp = IterativeImputer(estimator = lr, missing_values=np.nan, max_iter = 75, imputation_order = "roman",
                       random_state= 5769)

epaimp = imp.fit_transform(epaclean1)
epaimputed = pd.DataFrame(epaimp, index = epaclean1.index, columns = epaclean1.columns)
print(epaimputed.head())
calc_outliers(epaimputed)

# kds = mf.ImputationKernel(
#     data = epaclean1,
#     datasets =1,
#     mean_match_candidates=7,
#     save_all_iterations=True,
#     random_state=5345
# )
# # #
# kds.mice(5, verbose = True)
# completed_data = kds.complete_data(0)
#
# calc_outliers(completed_data)
#
# optimal_parameters, losses = kds.tune_parameters(
#     dataset =0,
#     verbose = True,
#     optimization_steps=5
# )
# print(optimal_parameters)
# print("#" * 1000)
# kds.mice(1, variable_parameters=optimal_parameters)
# completedata2 = kds.complete_data(0)
# calc_outliers(completedata2)
# completedata2.to_pickle(f"{data_dir}\\epaimputed2.pkl")







