import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete.pkl")
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

X = epacomplete[:, epacomplete.columns != 'comb08']
y = epacomplete['comb08']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =5328)

