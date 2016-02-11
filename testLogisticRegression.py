import census_loader
import pandas as pd
import numpy as np
from sklearn import cross_validation,linear_model,metrics
import random

def loadToDf():
    training_data, test_data = census_loader.loadCsv()
    categoricalColumnNames = ["workclass","education","marital-status","occupation",
                   "relationship","race","sex","native-country"]
    cotinousColumnNames = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    #normalization
    training_data = census_loader.standardize(training_data,cotinousColumnNames)
    test_data = census_loader.standardize(test_data,cotinousColumnNames)

    #categorize
    training_data,test_data  = census_loader.categorize(training_data, test_data)

    #convert categorical valirbales to dummy variables
    training_data = census_loader.toDummy(training_data,categoricalColumnNames)
    test_data = census_loader.toDummy(test_data,categoricalColumnNames)

    return training_data, test_data

def cvLR(training_data):
    #training_data =  training_data
    cv = cross_validation.KFold(len(training_data), n_folds=10)
    for trainCV, testCV in cv:
        validation_training = training_data.iloc[trainCV]
        validation_test = training_data.iloc[testCV]

        lr = linear_model.LogisticRegression()
        lr.fit(validation_training[np.arange(103)], validation_training.loc[:,"agrossincome"])

        train_preds = lr.predict(validation_training[np.arange(103)])
        train_y = list(validation_training.loc[:,"agrossincome"])

        results = lr.predict(validation_test[np.arange(103)])
        # preds = [result[1] for result in results]
        ys = list(validation_test.loc[:,"agrossincome"])

        train_fpr, train_tpr, train_thresholds = metrics.roc_curve(np.array(train_preds), np.array(train_y), pos_label=1)

        fpr, tpr, thresholds = metrics.roc_curve(np.array(results), np.array(ys), pos_label=1)
        print metrics.auc(train_fpr, train_tpr),metrics.auc(fpr, tpr)
        # print sum(int(t==y and t==0) for (t, y) in zip(targets, ys)), sum(int(t==y and t==1) for (t, y) in zip(targets, ys)), len(targets)

training_data, test_data = loadToDf()


#cvLR(training_data)

# lr = linear_model.LogisticRegression()
# lr.fit(training_data[np.arange(103)], training_data.loc[:,"agrossincome"])
#
# train_preds = lr.predict(training_data[np.arange(103)])
# train_y = list(training_data.loc[:,"agrossincome"])
#
# preds = lr.predict(test_data[np.arange(103)])
# ys = list(test_data.loc[:,"agrossincome"])
#
# train_fpr, train_tpr, train_thresholds = metrics.roc_curve(np.array(train_preds), np.array(train_y), pos_label=1)
# fpr, tpr, thresholds = metrics.roc_curve(np.array(preds), np.array(ys), pos_label=1)
#
# print metrics.auc(train_fpr, train_tpr), metrics.auc(fpr, tpr)

