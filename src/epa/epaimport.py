"""
Module that extracts and pickles data
epaimport.py
"""

import pandas as pd
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

#Verification
print(epadata.head())


epadata.to_pickle(f"{data_dir}\\epadata.pkl")