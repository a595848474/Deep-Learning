import mortality_loader
import SGD
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cross_validation

mortality = mortality_loader.doItTogether()

cv = cross_validation.KFold(len(mortality), n_folds=10)
errors_cv = []
for i in xrange(100):
    error_cv = []
    for trainCV, testCV in cv:
        training_data = map(mortality.__getitem__, trainCV)
        test_data = map(mortality.__getitem__, testCV)
        errors_epoch, errors_batch, ws, bs = SGD.SGD(training_data=training_data, epochs=4, batch_size=53,
                                                     stepsize=0.08, init_w=None,init_b=None,
                                                     test_data=test_data, regular=0.99, decay=1)
        error_cv.append(errors_epoch[-1])
    errors_cv.append(np.mean(error_cv))

print np.nanmin(errors_cv)


# slices = [mortality[i::10] for i in xrange(10)]
#
# for j in xrange(len(slices)):
#     errors_epoch, errors_batch, ws, bs = SGD.SGD(mortality, 4, 1, 0.05, init_w=None,init_b=None, test_data=slices[j], regular=None, decay=0.98)
