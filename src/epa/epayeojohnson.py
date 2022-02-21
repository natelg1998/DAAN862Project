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
def calc_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    print("Outliers Count")
    print(((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).sum())

# epafeatures = epacomplete.loc[:, epacomplete.columns != 'comb08']
# y = epacomplete['comb08']
# print(epafeatures.info())
# print(epafeatures.head())
# calc_outliers(epafeatures)

pt = PowerTransformer(method = "yeo-johnson")
epatransformed = pt.fit_transform(epacomplete)

epatransformdf = pd.DataFrame(epatransformed, columns = epacomplete.columns)
print(epatransformdf.head(20))
calc_outliers(epatransformdf)

X = epatransformdf.loc[:, epatransformdf.columns != 'comb08']
y = epatransformdf['comb08']
#
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)
mlr_yj = LinearRegression()
mlr_yj.fit(X_train, y_train)
y_pred = mlr_yj.predict(X_test)

mae_base = metrics.mean_absolute_error(y_test, y_pred)
mse_base = metrics.mean_squared_error(y_test, y_pred)
rmse_base = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_yj.score(X,y)*100))
print("Mean Absolute Error:", mae_base)
print("Mean Square Error:", mse_base)
print("Root Mean Square Error:", rmse_base)

#Use OLS to give a summary report
x = sm.add_constant(X)
result_base = sm.OLS(y, x).fit()
print(result_base.summary())

#Test linearity assumption
linear_assumption(mlr_yj, X, y)


#Test for normaility of residuals
normal_errors_assumption(mlr_yj, X, y)

#Test for multicollinearity
multicollinearity_assumption(mlr_yj, X, y, X.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_yj, X, y)

