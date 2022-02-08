import os
from pathlib import Path
import pandas as pd
# print(os.getcwd())

os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete.pkl")
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
print(epacomplete.head())
#Check for nulls
print(epacomplete.isna().sum())

