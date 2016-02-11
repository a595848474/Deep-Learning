#
# training_data: training dataset
# epochs: how many times to run it again
# mini_batch_size: size of batch, which is smaller than the number of observations
# eta: stepsize
# decay: decay
# test_data: test dataset
#
import numpy as np
#import sklearn
def SGD(training_data, epochs, batch_size, stepsize,init_w, init_b, test_data, regular, decay = 1):
    w = 2*np.round(np.random.random_sample(4), decimals=2)-1 # initialize weights
    b = 2*np.round(np.random.random_sample(), decimals=2)-1 # initialize b
    if init_w:
        w = np.array(init_w)
    if init_b:
        b = np.array(init_b)
    l = len(training_data)

    data = training_data
    for j in xrange(epochs-1):
        data = np.concatenate((data, training_data))
    n = len(data)
    #stepsizes = [stepsize]
    #for indexStepsize in xrange(n-1):
     #   stepsize = stepsize*decay
     #   stepsizes.append(stepsize)

    batchs = [data[k:k+batch_size] for k in xrange(0, n, batch_size)]
   # steps = [stepsizes[k:k+batch_size] for k in xrange(0, n, batch_size)]
    errors_square = []
    mean_batch_errors_square = []
    #errors = []
    ws = [w]
    bs = [b]
    for batch in batchs:
        x = [a[0] for a in batch]
        y = [a[1]for a in batch]
        target = np.dot(x, np.reshape(w, (4, 1))) + b

        error = np.reshape(target, (1, len(batch))) - np.array(y)
        error_square = np.square(error)
        #errors = np.concatenate((errors, error))
        errors_square = np.concatenate((errors_square,error_square[0])) # [0]get the 1-dimensional array. lol...
        mean_batch_errors_square.append(np.mean(error))
        #w = np.subtract(w, np.multiply(x[-1], stepsize*error[-1]))
        if regular:
            w = w*regular - np.mean(x, axis=0)*stepsize*np.mean(error)
        else:
            w = w - np.mean(x, axis=0)*stepsize*np.mean(error)
        stepsize = stepsize*decay
        ws.append(w)
        #b = b - stepsize*error[-1]
        b = b - stepsize*np.mean(error)
        bs.append(b)
        #stepsize = stepsize*decay
    errors_square_epochs = [np.mean(errors_square[m:m+l]) for m in xrange(0, n, l)]
    if test_data:
        errors_square_epochs.append(evaluate(test_data,w,b))
    return errors_square_epochs,mean_batch_errors_square,ws,bs

def evaluate(test_data,w,b):
    x = [a[0] for a in test_data]
    y = [a[1] for a in test_data]
    target = np.dot(x, np.reshape(w, (4, 1))) + b
    error_square = np.square(np.reshape(target, (1, len(test_data))) - np.array(y))
    return np.mean(error_square)




