"""
census_loader
~~~~~~~~~~~~
load census income dataset
formatize the dataset
"""

import numpy as np
import pandas as pd

def loadCsv():
    colName = ["age", "workclass", "fnlwgt", "education", "education-num",
              "marital-status", "occupation", "relationship", "race",
              "sex", "capital-gain", "capital-loss", "hours-per-week",
              "native-country", "agrossincome"]
    training_data = pd.read_csv("data/adult.data", names = colName,
                                header = None, na_values="?", keep_default_na = False, skipinitialspace=True)
    test_data = pd.read_csv("data/adult.test", names = colName,
                                header = None, na_values="?", keep_default_na = False, skipinitialspace=True)
    training_data.loc[training_data["agrossincome"]=="<=50K", "agrossincome"]= 0
    training_data.loc[training_data["agrossincome"]==">50K", "agrossincome"]= 1

    test_data.loc[test_data["agrossincome"]=="<=50K.", "agrossincome"]= 0
    test_data.loc[test_data["agrossincome"]==">50K.", "agrossincome"]= 1

    return training_data.dropna(), test_data.dropna()

def categorize(training_data, test_data):
    training_data_len = len(training_data)
    test_data_len = len(test_data)
    data = pd.concat([training_data, test_data])
    data["workclass"] = data["workclass"].astype("category")
    data["education"] = data["education"].astype("category")
    data["marital-status"] = data["marital-status"].astype("category")
    data["occupation"] = data["occupation"].astype("category")
    data["relationship"] = data["relationship"].astype("category")
    data["race"] = data["race"].astype("category")
    data["sex"] = data["sex"].astype("category")
    data["native-country"] = data["native-country"].astype("category")

    return data[:training_data_len], data[training_data_len:training_data_len+test_data_len]

def toDummy(data, columnNames):
    for columnName in columnNames:
        values = data[columnName].cat.categories.get_values()

        for value in values:
            data[value] = 0
            data.loc[data[columnName]==value, value] = 1
        data.drop(columnName, 1, inplace=True)
    #print len(data.columns)
    agrossincome = data["agrossincome"]
    data.drop("agrossincome",1, inplace=True)
    #print len(data.columns)
    data["agrossincome"] = agrossincome
    print len(data.columns)
    return data

def formatize(data):
    inputs = [np.reshape(x[:-1], (len(x[:-1]), 1)) for x in np.array(data).tolist()]
    results = [x[-1] for x in np.array(data).tolist()]
    data = zip(inputs, results)
    return data

def load_wapper():
    training_data, test_data = loadCsv()
    categoricalColumnNames = ["workclass","education","marital-status","occupation",
                   "relationship","race","sex","native-country"]
    cotinousColumnNames = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    #normalization
    training_data = standardize(training_data,cotinousColumnNames)
    test_data = standardize(test_data,cotinousColumnNames)

    #categorize
    training_data,test_data  = categorize(training_data, test_data)

    #convert categorical valirbales to dummy variables
    training_data = toDummy(training_data,categoricalColumnNames)
    test_data = toDummy(test_data,categoricalColumnNames)

    training_data = formatize(training_data)
    #test_inputs = [np.reshape(x[:-1], (len(x[:-1]), 1)) for x in np.array(test_data).tolist()]
    #test_results = [x[-1] for x in np.array(test_data).tolist()]

    test_data = formatize(test_data)
    #test_data = zip(test_inputs, test_results)
    return training_data, test_data

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...1) into a corresponding desired output from the neural
    network."""
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def standardize(data, columnNames):
    for columnName in columnNames:
        mean = np.mean(data[columnName])
        std = np.std(data[columnName])
        data[columnName] = [(x-mean)/std for x in data[columnName]]
    return data


#print len(training_data), len(test_data), training_data.head(10)