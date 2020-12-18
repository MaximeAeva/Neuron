import cupy as cp
import numpy as np

def genRecogBase(m, shape):
    """
    Generate recognition database
    
    Arguments:
    m -- number of examples
    shape -- shape

    Returns:
    X -- images
    Y -- labels
    """
    X = 0.5*cp.random.rand(m, shape[0], shape[1], shape[2])
    Y = cp.zeros((3, m))
    for i in range(m):
        typeM = np.random.randint(0, 3)
        if(typeM == 0):
            X[i, :, :, :] += cp.zeros(shape)
            Y[0, i] += 1
        if(typeM == 1):
            X[i, :, :, :] += cp.ones(shape)
            Y[1, i] += 1
        if(typeM == 2):
            X[i, :, :, np.random.randint(0, shape[2])] += cp.eye(shape[0], shape[1])
            Y[2, i] += 1
    return cp.minimum(X, 1), Y
        