import neural_network
import census_loader
import numpy as np
import pandas as pd
from sklearn import cross_validation

training_data, test_data = census_loader.load_wapper()

def crossValidation(training_data):
    cv = cross_validation.KFold(len(training_data), n_folds=10, shuffle=True)
    for trainCV, testCV in cv:
        validation_training = map(training_data.__getitem__, trainCV)
        validation_test = map(training_data.__getitem__, testCV)
        y = [result[1] for result in validation_test]
        #zeros = sum(y)
        #ones = len(validation_test) - zeros
        #print "validation_test baseline: {0}/{1}".format(sum(y), len(y))

        net = neural_network.Network([104,1])
        net.SGD(training_data, epochs=4, mini_batch_size=10, eta=0.1, test_data=validation_test)


for i in xrange(10):
    targets = [target[1] for target in test_data]
    # zeros = np.sum(targets)
    # print "test baseline: {0}/{1}".format(sum(targets), len(targets))
    # net = neural_network.Network([104,10, 1])
    # net.SGD(training_data, epochs=4, mini_batch_size=10, eta=0.5, test_data=test_data)
    crossValidation(training_data)

# for i in xrange(10):
#     net = neural_network.Network([104, 10, 1])
#     net.SGD(training_data, epochs=4, mini_batch_size=10, eta=0.1, test_data=test_data)


#print "training_data: {0}\n test_data: {1}".format(training_data[1:5], test_data[1:5])

#print training_data[0]

