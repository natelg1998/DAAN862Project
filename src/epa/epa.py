import pandas as pd
import miceforest as mf
import pickle
import os
from pathlib import Path

#Load our data. Update your file path as necessary
epadata = pd.read_csv("C:\\Users\\ntlg4\\PycharmProjects\\DAAN862Project\\data\\vehicles.csv")

"Pandas config settings for display"
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

#Drop any columns with 50% or more values
epadata = epadata.drop(["guzzler", "trans_dscr", "tCharger", "sCharger", "atvType", "fuelType2", "rangeA",
                        "evMotor", "mfrCode", "c240Dscr","charge240b", "c240bDscr", "startStop"], axis = 1)

print(epadata.info())
print(epadata.isna().sum())

#For mice forest, we need to change all objects to either categoric or numeric type
#all our values are category
epadata['drive'] = epadata['drive'].astype('category')
epadata['eng_dscr'] = epadata['eng_dscr'].astype('category')
epadata['fuelType'] = epadata['fuelType'].astype('category')
epadata['fuelType1'] = epadata['fuelType1'].astype('category')
epadata['make'] = epadata['make'].astype('category')
epadata['model'] = epadata['model'].astype('category')
#Values were Y and N so made it bool
epadata['mpgData'] = epadata['mpgData'].astype('bool')
epadata['trany'] = epadata['trany'].astype('category')
epadata['VClass'] = epadata['VClass'].astype('category')
epadata['createdOn'] = epadata['createdOn'].astype('category')
epadata['modifiedOn'] = epadata['modifiedOn'].astype('category')

print(epadata.info())

# epadatadf1 = epadata
# epadatadf1.drop(['createdOn', 'modifiedOn'])
# #Let's handle the rest of the missing values
kds = mf.ImputationKernel(
    data = epadata,
    datasets =1,
    save_all_iterations=True,
    random_state=5345
)
#
kds.mice(3)
#
completed_data = kds.complete_data(0)
# print(completed_data)
# print(type(completed_data))

epacomplete = pd.DataFrame(completed_data)
os.chdir('..\\..')
data_folder = Path('data')
epacomplete.to_pickle(f"{data_folder}\\epacomplete.pkl")
