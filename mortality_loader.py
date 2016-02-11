"""
mordality_loader
~~~~~~~~~~~~~~~~
load mortality dataset.
"""
import numpy as np
import pandas as pd

def loadCsv(file):
    return pd.read_csv(file, sep= ",")

def standardize(my_df):
    for c in my_df.columns:
        mean = np.mean(my_df[c])
        std = np.std(my_df[c])
        my_df[c] = np.round([(x-mean)/std for x in my_df[c]], decimals=4)
    return my_df

def formatize(my_df):
    training = [x for x in np.array(my_df).tolist()]
    inputs = []
    results = []
    for x in training:
        inputs.append(x[1:])
        results.append(x[0])
    #results = np.array(my_df[my_df.columns[0]])
    data = zip(inputs, results)
    return data

def doItTogether():
    my_df = loadCsv("mortalityt.csv")
    my_df = standardize(my_df)
    mortality = formatize(my_df)
    return mortality





    #my_standardized_data =



#print my_data
