import numpy as np
x = np.array([[-1,0,0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]]) # input
t = np.array([0, 1, 1, 1]) # targets
w = np.array([-0.05, -0.02, 0.02]) # weights

# perceptron function
# x: input
# w: weight
#
def g(x, w):
    g = np.dot(x, w)
    return np.where(g>0, 1, 0)
    
def compare(y, t):
    for i in range(y.size):
        if y[i]!=t[i]:
            return False
    return True

def update(x, t, w):
    y = g(x, w)
    if compare(y, t)==True:
        print w
        return w
    else:
        print w
        for i in range(y.size):
            if y[i]!=t[i]:
                for j in range(w.size):
                    w[j] += - 0.1*(y[i]-t[i])*x[i][j]
        print "updating w"
        return update(x,t,w)

update(x,t,w)



    



        
            
