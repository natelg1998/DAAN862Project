import pandas as pd
import miceforest as mf
import pickle
import os
from pathlib import Path
from urllib.request import urlretrieve

#Define the data directory
os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()


#Load our data. Update your file path as necessary
URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
epadata = pd.read_csv(URL)

#Save data locally
urlretrieve(URL, f"{data_dir}\\vehicles.csv")
#
# "Pandas config settings for display"
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

#EDA

print(epadata.head())
print(epadata.describe())
print(epadata.info())
print(epadata.isna().sum())


def calc_missing_values(series):
    """
    Calculate percentage of NAs in dataframe
    :param series:
    :return:
    """
    num = series.isnull().sum()
    length = len(series)
    print(round(num/length, 2))

calc_missing_values(epadata)

#Drop any columns with 50% or more values. Also drop certain attributes that will not add value to a model
epadata = epadata.drop(["guzzler", "trans_dscr", "tCharger", "sCharger", "atvType", "fuelType2", "rangeA",
                        "evMotor", "mfrCode", "c240Dscr","charge240b", "c240bDscr", "startStop",
                        "createdOn", "modifiedOn", "engId", "feScore", "ghgScore", "ghgScoreA", "id",
                        "mpgData", "phevBlended", "year"], axis = 1)

#Drop all non-numeric values
epanumeric = epadata.copy()
epanumeric = epanumeric.apply(pd.to_numeric, errors = 'coerce').dropna(axis = 1)
print(epanumeric.info())
print(epanumeric.dtypes)
print(epanumeric.isna().sum())
calc_missing_values(epanumeric)

epacomplete = pd.DataFrame(epanumeric)
epacomplete.to_pickle(f"{data_dir}\\epacomplete.pkl")








