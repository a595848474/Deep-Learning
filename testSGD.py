import mortality_loader
import SGD
import numpy as np
from matplotlib import pyplot as plt


mortality = mortality_loader.doItTogether()

def run():
    intial_vars = []
    end_vars = []
    errors_epoches = []
    errors_batches = []

    for i in xrange(100):
        errors_epoch, errors_batch, ws, bs = SGD.SGD(mortality, 4, 10, 0.4, init_w=None,init_b=None, test_data=None, regular=None, decay=0.98)

        temp = list(ws[0])
        temp.append(bs[0])
        intial_vars.append(temp)

        temp = list(ws[-1])
        temp.append(bs[-1])
        end_vars.append(temp)

        errors_epoches.append(errors_epoch)
        errors_batches.append(errors_batch)

    return intial_vars, end_vars, errors_epoches, errors_batches

intial_vars, end_vars, errors_epoches, errors_batches = run()

results = [errors[-1] for errors in errors_epoches]
mean = np.nanmean(results)
min = np.nanmin(results)
max = np.nanmax(results)
minIndex = results.index(min)

print 'mean: {0} \nmax: {1} \nmin: {2} \nmin Index: {3} \ninitial variables: {4}' \
      '\nend variables: {5} \nerrors_epoches: {6}'.format(mean,max,min,minIndex,intial_vars[minIndex], end_vars[minIndex], errors_epoches[minIndex])


#print ws[0], ws[-1], errors_epoches
plt.plot(errors_epoches[minIndex])
#plt.plot(errors_epoches)
plt.show()
#plt.plot(errors_square)
#plt.ylabel('squared error')
#plt.show()

#errors_epoch, errors_batch, ws, bs = SGD.SGD(mortality, 1, 3, 0.004, init_w=None,init_b=None, decay=1)
