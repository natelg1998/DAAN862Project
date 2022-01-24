import pandas as pd

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
