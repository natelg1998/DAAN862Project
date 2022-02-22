"""
Module to show imputation using mice and miceforest

"""

import pandas as pd
import numpy as np
import miceforest as mf
import pickle
import os
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
import statsmodels.api as sm
from sklearn import metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from linearassumptions import calculate_residuals, linear_assumption,normal_errors_assumption,multicollinearity_assumption, autocorrelation_assumption
from epatransformtools import calc_missing_values,  calc_outliers

os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete1.pkl")

Q1 = epacomplete.quantile(q=.25)
Q3 = epacomplete.quantile(q=.75)
IQR = epacomplete.apply(stats.iqr)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

epaclean = epacomplete.copy()
epaclean = pd.DataFrame(epaclean)
for name in epaclean.columns:
    epaclean.loc[(epaclean[name] < lower_bound[name]) | (epaclean[name] > upper_bound[name]), name] = np.nan
print(epaclean.head(15))

#Now that we cleaned the data removing the outliers, we will use MICE to impute for the outliers
#first with IterativeImputer
# lr = LinearRegression()
# imp = IterativeImputer(estimator = lr, missing_values=np.nan, max_iter = 200, imputation_order = "roman",
#                        random_state= 5769)
#
# epaimp = imp.fit_transform(epaclean)
# epaimputed = pd.DataFrame(epaimp, index = epaclean.index, columns = epaclean.columns)
# print(epaimputed.head())
# calc_outliers(epaimputed)
# epaimputed.to_pickle(f"{data_dir}\\epadataiterimputed.pkl")
#
# #Second with miceforest - DO NOT RUN THIS AGAIN THIS WILL TAKE ALMOST 3 HOURS TO COMPLETE
# kds = mf.ImputationKernel(
#     data = epaclean,
#     datasets =1,
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
# completedata = kds.complete_data(0)
# calc_outliers(completedata)
# completedata.to_pickle(f"{data_dir}\\epadatamfimputed.pkl")