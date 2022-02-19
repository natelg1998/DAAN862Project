import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# print(os.getcwd())

os.chdir('..\\..')
data_dir = Path('.' + '\\data').resolve()

epacomplete = pd.read_pickle(f"{data_dir}\\epacomplete.pkl")
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
print(epacomplete.head())
print(epacomplete.dtypes)
#Check for nulls
print(epacomplete.isna().sum())


"""
Normalize our variables 
"""

scaler = MinMaxScaler()
scaler.fit(epacomplete)
epascaled = pd.DataFrame(scaler.fit_transform(epacomplete),
                         columns = epacomplete.columns, index = epacomplete.index)
print(epascaled.head())
#
# #Index 15 is our dependent variable comb08
#
# X1 = epascaled.iloc[:,[15, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
# corrmatrix1 = X1.corr().round(2)
# sns.heatmap(corrmatrix1, annot = True)
# plt.show()
#
# X2 = epascaled.iloc[:,[15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
# corrmatrix2 = X2.corr().round(2)
# sns.heatmap(corrmatrix2, annot = True)
# plt.show()
#
# X3 = epascaled.iloc[:,[15, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
# corrmatrix3 = X3.corr().round(2)
# sns.heatmap(corrmatrix3, annot = True)
# plt.show()
#
# X4 = epascaled.iloc[:,[15, 46,47,48,49]]
# corrmatrix4 = X4.corr().round(2)
# sns.heatmap(corrmatrix4, annot = True)
# plt.show()

#melt the data to create boxplots
# epascaled_melt = pd.melt(epascaled)
# print(epascaled_melt.head())
#

#Visualize distributions for outliers
# group1 = epascaled.iloc[:,0:10]
# group1_melt = pd.melt(group1)
# sns.boxplot(x= "variable", y = "value", data = group1_melt)
# plt.show()
#
# group2 = epascaled.iloc[:,11:21]
# group2_melt = pd.melt(group2)
# sns.boxplot(x= "variable", y = "value", data = group2_melt)
# plt.show()
#
# group3 = epascaled.iloc[:,22:32]
# group3_melt = pd.melt(group3)
# sns.boxplot(x= "variable", y = "value", data = group3_melt)
# plt.show()
#
# group4 = epascaled.iloc[:,33:43]
# group4_melt = pd.melt(group4)
# sns.boxplot(x= "variable", y = "value", data = group4_melt)
# plt.show()
#
# group5 = epascaled.iloc[:,44:49]
# group5_melt = pd.melt(group5)
# sns.boxplot(x= "variable", y = "value", data = group5_melt)
# plt.show()

epascaled.drop('charge120', axis = 1, inplace = True)
epascaled.to_pickle(f"{data_dir}\\epascaled.pkl")
# sns.boxplot(x= "variable", y = "value", data = epascaled_melt)
# plt.show()


#Split dataframe into splits to see correlations
# epa1 = epacomplete.loc[:,[y.name, 'barrels08', 'barrelsA08', 'charge120', 'charge240', 'city08', 'city08U', 'cityA08',
#                           'cityA08U', 'cityCD', 'cityE', 'cityUF', 'co2']]
# epaheatmap1 = sns.heatmap(epa1.corr(), annot = True)
# plt.show()
#
# epa2 = epacomplete.loc[:,[y.name, 'co2A', 'co2TailpipeAGpm', 'co2TailpipeGpm', 'comb08U', 'combA08', 'combA08U', 'combE',
#                           'combinedCD', 'cylinders', 'displ', 'fuelCost08', 'fuelCostA08']]
# epaheatmap2 = sns.heatmap(epa2.corr(), annot = True)
# plt.show()
#
# epa3 = epacomplete.loc[:,[y.name, 'highway08', 'highway08U', 'highwayA08', 'highwayA08U', 'highwayCD', 'highwayE', 'highwayUF',
#                           'hlv', 'hpv', 'lv2', 'lv4', 'pv2']]
# epaheatmap3 = sns.heatmap(epa3.corr(), annot = True)
# plt.show()
#
# epa4 = epacomplete.loc[:,[y.name, 'pv4', 'range', 'rangeCity', 'rangeCityA', 'rangeHwy', 'rangeHwyA', 'UCity',
#                           'UCityA', 'UHighway', 'UHighwayA', 'year', 'youSaveSpend', 'phevCity', 'phevHwy', 'phevComb']]
# epaheatmap4 = sns.heatmap(epa4.corr(), annot = True)
# plt.show()


# fig, ax = plt.subplots(3,3)
#
# ax[0,0].plot(epacomplete['year'], epacomplete['barrels08'])
# ax[0,0].set_title("Annual Petroleum consumption in barrels for FuelType1")
# ax[0,0].set_xlabel("Year")
# ax[0,0].set_ylabel("Annual Barrels Consumption")
#
# ax[0,1].plot(epacomplete['year'], epacomplete['barrelsA08'])
# ax[0,1].set_title("Annual Petroleum consumption in barrels for FuelType2")
# ax[0,1].set_xlabel("Year")
# ax[0,1].set_ylabel("Annual Barrels Consumption")
#
# ax[0,2].plot(epacomplete['year'], epacomplete['city08'])
# ax[0,2].set_title("City Mpg for FuelType1")
# ax[0,2].set_xlabel("Year")
# ax[0,2].set_ylabel("City MPG")
#
# ax[1,0].plot(epacomplete['year'], epacomplete['city08U'])
# ax[1,0].set_title("Unrounded City Mpg for FuelType1")
# ax[1,0].set_xlabel("Year")
# ax[1,0].set_ylabel("Unrounded City MPG")
#
# ax[1,0].plot(epacomplete['year'], epacomplete['cityA08'])
# ax[1,0].set_title("City Mpg for FuelType2")
# ax[1,0].set_xlabel("Year")
# ax[1,0].set_ylabel("City MPG")
#
# ax[1,1].plot(epacomplete['year'], epacomplete['cityA08U'])
# ax[1,1].set_title("Unrounded City Mpg for FuelType2")
# ax[1,1].set_xlabel("Year")
# ax[1,1].set_ylabel("Unrounded City MPG")
#
# ax[1,2].plot(epacomplete['year'], epacomplete['cityCD'])
# ax[1,2].set_title("City gasoline consumption (gallons/100 miles) in charge depleting mode")
# ax[1,2].set_xlabel("Year")
# ax[1,2].set_ylabel("Gasoline Consumption")
#
# ax[2,0].plot(epacomplete['year'], epacomplete['cityE'])
# ax[2,0].set_title("City electricity consumption in kw-hrs/100 miles")
# ax[2,0].set_xlabel("Year")
# ax[2,0].set_ylabel("Electricity Consumption")
#
# ax[2,1].plot(epacomplete['year'], epacomplete['cityUF'])
# ax[2,1].set_title("EPA city utility factor (share of electricity) for PHEV")
# ax[2,1].set_xlabel("Year")
# ax[2,1].set_ylabel("City Utility Factor")
#
# ax[2,2].plot(epacomplete['year'], epacomplete['co2'])
# ax[2,2].set_title("Tailpipe CO2 in grams/mile for FuelType1")
# ax[2,2].set_xlabel("Year")
# ax[2,2].set_ylabel("Tailpipe CO2 (g/mile)")
#
# plt.show()

# plt.plot(epacomplete['year'], epacomplete['barrels08'])
# plt.show()

# sns.lineplot(x ="year", y = "barrels08", data = epacomplete)
# plt.title('Annual Petroleum consumption in barrels for type2 fuel')
# plt.ylabel('Annual Barrels consumption')
# plt.show()


