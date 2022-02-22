"""
Module for some handy transform functions
epatransformtools.py
"""

def calc_missing_values(df):
    """
    Calculate percentage of NAs in dataframe
    :param df:
    :return:
    """
    num = df.isnull().sum()
    length = len(df)
    print("Percent NA")
    print(round(num/length, 2))

def calc_outliers(df):
    """
    Calculate the IQR to detect outliers.
    :param df
    :return: list of outliers counted per column
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    print("Outliers Count")
    print(((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).sum())