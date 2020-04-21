import numpy as np

def functions(direction, f, Z, dA = None):
    if direction == 'forward':
        if f == 'relu':
            return relu(Z)
        elif f == 'sigmoid':
            return sigmoid(Z)
        elif f == 'tanh':
            return tanh(Z)
        elif f == 'bentid':
            return bentid(Z)
        elif f == 'identity':
            return identity(Z)
    else : 
        if f == 'relu':
            return relu_backward(dA, Z)
        elif f == 'sigmoid':
            return sigmoid_backward(dA, Z)
        elif f == 'tanh':
            return tanh_backward(dA, Z)
        elif f == 'bentid':
            return bentid_backward(dA, Z)
        elif f == 'identity':
            return identity_backward(dA, Z)


def identity(Z):
    return Z, Z

def identity_backward(dA, Z):
    return dA

def bentid(Z):
    return ((np.sqrt((Z**2)+1)-1)/2) + Z, Z

def bentid_backward(dA, Z):
    return dA * ((Z/(2*np.sqrt((Z**2)+1)))+1)

def tanh(Z):
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z)), Z

def tanh_backward(dA, Z):
    return dA * (1 - ((np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z)))**2)

def sigmoid(Z):
    return 1/(1+np.exp(-Z)), Z

def relu(Z):
    return np.maximum(0,Z), Z


def relu_backward(dA, Z):  
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, Z):

    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

