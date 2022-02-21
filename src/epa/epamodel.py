import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from linearassumptions import calculate_residuals, linear_assumption,normal_errors_assumption,multicollinearity_assumption, autocorrelation_assumption

os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

epaimpute = pd.read_pickle(f"{data_dir}\\epaimputed.pkl")
epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete1.pkl")
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

#Have to put y back in
epaimpute.insert(loc = 15, column ="comb08", value = epacomplete["comb08"])
epafull = epaimpute.copy()
#Check for nulls
print("Null Values List")
print(epafull.isna().sum())
print(epafull.head())

def calc_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    print("Outliers Count")
    print(((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).sum())

def calc_missing_values(series):
    """
    Calculate percentage of NAs in dataframe
    :param series:
    :return:
    """
    num = series.isnull().sum()
    length = len(series)
    print(round(num/length, 2))
    
calc_outliers(epafull)
#
X = epafull.loc[:, epafull.columns != 'comb08']
y = epafull['comb08']
#
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 245)
mlr_base = LinearRegression()
mlr_base.fit(X_train, y_train)
y_pred = mlr_base.predict(X_test)

mae_base = metrics.mean_absolute_error(y_test, y_pred)
mse_base = metrics.mean_squared_error(y_test, y_pred)
rmse_base = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("R-squared : {:.2f}".format(mlr_base.score(X,y)*100))
print("Mean Absolute Error:", mae_base)
print("Mean Square Error:", mse_base)
print("Root Mean Square Error:", rmse_base)

#Use OLS to give a summary report
x = sm.add_constant(X)
result_base = sm.OLS(y, x).fit()
print(result_base.summary())
#
#Test linearity assumption
linear_assumption(mlr_base, X, y)


#Test for normaility of residuals
normal_errors_assumption(mlr_base, X, y)

#Test for multicollinearity
multicollinearity_assumption(mlr_base, X, y, X.columns)

#Test for autocorrelation
autocorrelation_assumption(mlr_base, X, y)

# X1 = epacomplete.iloc[:,[15, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
# corrmatrix1 = X1.corr().round(2)
# sns.heatmap(corrmatrix1, annot = True)
# plt.show()
#
# X2 = epacomplete.iloc[:,[15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
# corrmatrix2 = X2.corr().round(2)
# sns.heatmap(corrmatrix2, annot = True)
# plt.show()
#
# X3 = epacomplete.iloc[:,[15, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
# corrmatrix3 = X3.corr().round(2)
# sns.heatmap(corrmatrix3, annot = True)
# plt.show()
#
# X4 = epacomplete.iloc[:,[15, 46,47,48,49]]
# corrmatrix4 = X4.corr().round(2)
# sns.heatmap(corrmatrix4, annot = True)
# plt.show()