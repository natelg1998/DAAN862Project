import os
from pathlib import Path
import pandas as pd
# print(os.getcwd())

os.chdir('..\\..')
print(os.getcwd())
data_dir = Path('.' + '\\data').resolve()

epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete.pkl")
pd.options.display.width= None
pd.options.display.max_columns= None
print(epacomplete.head())

