import cupy as cp

def activation(direction, function, z, dA = None):
    if direction == 'forward':
        if function == 'relu':
            return relu(z)
        elif function == 'sigmoid':
            return sigmoid(z)
        elif function == 'tanh':
            return tanh(z)
        elif function == 'bentid':
            return bentid(z)
        elif function == 'identity':
            return identity(z)
    else : 
        if function == 'relu':
            return relu_backward(dA, z)
        elif function == 'sigmoid':
            return sigmoid_backward(dA, z)
        elif function == 'tanh':
            return tanh_backward(dA, z)
        elif function == 'bentid':
            return bentid_backward(dA, z)
        elif function == 'identity':
            return identity_backward(dA, z)


def identity(z):
    return z

def identity_backward(dA, z):
    return dA

def bentid(z):
    return ((cp.sqrt((z**2)+1)-1)/2) + z

def bentid_backward(dA, z):
    return dA * ((z/(2*cp.sqrt((z**2)+1)))+1)

def tanh(z):
    return (cp.exp(z)-cp.exp(-z))/(cp.exp(z)+cp.exp(-z))

def tanh_backward(dA, z):
    return dA * (1 - ((cp.exp(z)-cp.exp(-z))/(cp.exp(z)+cp.exp(-z)))**2)

def sigmoid(z):
    return 1/(1+cp.exp(-z))

def sigmoid_backward(dA, z):
    return dA * (1/(1+cp.exp(-z))) * (1-(1/(1+cp.exp(-z))))

def relu(z):
    return cp.maximum(0,z)

def relu_backward(dA, z):  
    dz = cp.array(dA, copy=True)  
    dz = cp.maximum(0, dz)
    return dz

