import pandas as pd
import miceforest as mf

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

#Let's handle the rest of the missing values
kds = mf.ImputationKernel(
    data = epadata,
    save_all_iterations=True,
    random_state=5345
)

kds.mice(5)

completed_data = kds.complete_data()
print(completed_data)
