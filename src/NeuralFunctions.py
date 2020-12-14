import numpy as np
import cupy as cp
import math
from NeuralActivation import activation
from timeit import default_timer as timer
import dask.array as da
import psutil as cpuInfo
import GPUtil as gpuInfo

print("----------Hardware config information----------")
print("CPU :")
print("          Physical cores:", cpuInfo.cpu_count(logical=False))
print("          Total cores:", cpuInfo.cpu_count(logical=True))
gpus = gpuInfo.getGPUs()
print("GPU :")
for gpu in gpus:
    print("          "+gpu.name)
    
'''
print("----------Speed test----------")
start = timer()
rs = da.random.RandomState(RandomState=np.random.RandomState) 
x = rs.normal(10, 1, size=(50000, 50000), chunks=(1000, 1000))
(x + 1)[::2, ::2].sum().compute()
print("MultiThreading CPU : %f s" %(timer()-start))
start = timer()
rs = da.random.RandomState(RandomState=cp.random.RandomState)
x = rs.normal(10, 1, size=(50000, 50000), chunks=(1000, 1000))
(x + 1)[::2, ::2].sum().compute()
print("MultiThreading GPU : %f s" %(timer()-start))
'''

def initialize_parameters_he(in_dim, out_dim):
    '''
    Initialize parameters for a function layer

    Parameters
    ----------
    in_dim : int
        Size of the input (vector must be v(in_dim, examples)).
    out_dim : int
        Size of the output (vector must fit the next layer).

    Returns
    -------
    W : cp.array(out_dim, in_dim)
        Weight matrix computing Z = W*A.
    gamma : cp.array(out_dim, 1)
        Weight vector computing z = gamma*zhat + beta.
    beta : cp.array(out_dim, 1)
        Bias vector computing z = gamma*zhat + beta..
    mu : empty array
        Mean for batch norm.
    sigma : empty array
        Standard deviation for batch norm.

    '''
    W = cp.random.randn(out_dim, in_dim)*cp.sqrt(2/in_dim)
    gamma = cp.ones((out_dim, 1))
    beta = cp.zeros((out_dim, 1))
    mu = cp.zeros((out_dim, 1))
    sigma = cp.zeros((out_dim, 1))
    
    return W, gamma, beta, mu, sigma

def initialize_parameters_he_conv(f, depth, filter_number):
    '''
    Initialize filter and bias for convolution

    Parameters
    ----------
    f : int
        Filter size.
    depth : int
        Filter depth.
    filter_number : int
        Number of filter in the layer

    Returns
    -------
    Filter : cp.array(f, f, depth, number_filter)
        Filter for convolution.
    bias : cp.array(1, 1, 1, number_filter)
        Filter bias
    '''
    Filter = cp.random.randn(f, f, depth, filter_number)*cp.sqrt(2/(f**2))
    bias = cp.zeros((1, 1, 1, filter_number))
    
    return Filter, bias

def initialize_adam(W, beta, gamma) :
    '''
    Initialize Adam optimizer variables for a function layer

    Parameters
    ----------
    W : cp.array(out_dim, in_dim)
        Weight matrix.
    beta : cp.array(out_dim, 1)
        Bias matrix.
    gamma : cp.array(out_dim, 1)
        Weight matrix.

    Returns
    -------
    vdW : cp.array(W)
        Moving average of dW.
    vdgamma : cp.array(gamma)
        Moving average of dgamma.
    vdbeta : cp.array(beta)
        Moving average of dbeta.
    sdW :  cp.array(W)
        Moving average of squared dW.
    sdgamma : cp.array(gamma)
        Moving average of squared dgamma.
    sdbeta : cp.array(beta)
        Moving average of squared dbeta.

    '''
    vdW = cp.zeros((W.shape[0], W.shape[1]))
    vdbeta = cp.zeros((beta.shape[0], beta.shape[1]))
    vdgamma = cp.zeros((gamma.shape[0], gamma.shape[1]))
    sdW = cp.zeros((W.shape[0], W.shape[1]))
    sdbeta = cp.zeros((beta.shape[0], beta.shape[1]))
    sdgamma = cp.zeros((gamma.shape[0], gamma.shape[1]))

    
    return vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta

def initialize_adam_conv(Filter, bias) :
    '''
    Initialize Adam optimizer variables for a function layer

    Parameters
    ----------
    Filter : cp.array(f, f, depth, filter_number)
        filter vector of volume
    bias : cp.array(1, 1, 1, filter_number)
        Bias matrix.

    Returns
    -------
    vdFilter : cp.array(Filter)
        Moving average of dFilter.
    vdbias : cp.array(bias)
        Moving average of dbias.
    sdFilter :  cp.array(Filter)
        Moving average of squared dFilter.
    sdbias : cp.array(bias)
        Moving average of squared dbias.

    '''
    vdFilter = cp.zeros((Filter.shape[0], Filter.shape[1], 
                         Filter.shape[2], Filter.shape[3]))
    vdbias = cp.zeros((bias.shape[0], bias.shape[1], 
                         bias.shape[2], bias.shape[3]))
    sdFilter = cp.zeros((Filter.shape[0], Filter.shape[1], 
                         Filter.shape[2], Filter.shape[3]))
    sdbias = cp.zeros((bias.shape[0], bias.shape[1], 
                         bias.shape[2], bias.shape[3]))

    
    return vdFilter, vdbias, sdFilter, sdbias

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    '''
    Splite training data into multiple mini_batches

    Parameters
    ----------
    X : cp.array(examples, features_in)
        Input dataset.
    Y : cp.array(features_out, examples)
        Output target.
    mini_batch_size : int (a power of 2), optional
        Size of a batch. The default is 64.
    seed : int
        Caring about random function behaviour
    Returns
    -------
    mini_batches : cp.list of tuple (X, Y)
        A list of batches with the given size.

    '''
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : m*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : m*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def forward_function(A_previous, W, mu, sigma, gamma, beta, function, dropout = 1):
    '''
    A forward function step

    Parameters
    ----------
    A_previous : cp.array(in_dim, examples)
        Result of the previous layer.
    W : cp.array(out_dim, in_dim)
        Weight matrix.
    mu : cp.array(out_dim, number of epochs)
        Gather Z mean over epochs.
    sigma : cp.array(out_dim, number of epochs)
        Gather Z std over epochs.
    gamma : cp.array(out_dim, 1)
        Weight matrix.
    beta : cp.array(out_dim, 1)
        Bias matrix.
    function : string
        The desired activation function.
    dropout : float (in (0,1)), optional
        Percentage of disabled neurons. The default is 1.

    Returns
    -------
    A : cp.array(out_dim, examples)
        Output layer result.
    z : cp.array(out_dim, examples)
        Activation function input.
    zhat : cp.array(out_dim, examples)
        Normalized Z.
    Z : cp.array(out_dim, examples)
        Result after applying W weights.
    mu : cp.array(out_dim, number of epochs)
        Updated mu.
    sigma : cp.array(out_dim, number of epochs)
        Updated sigma.
    D : cp.array(out_dim, 1)
        Mask matrix for dropout (filled with ones if dropout = 1).

    '''
    
    m = A_previous.shape[1]
    eps = 1e-8
    
    Z = cp.dot(W, A_previous)
    if mu.any():
        mu = cp.concatenate((mu, cp.mean(Z, axis = 1, keepdims = True)), axis = 1)
        sigma = cp.concatenate((sigma, cp.std(Z, axis = 1, keepdims = True)), axis = 1)
    else :
        mu = cp.mean(Z, axis = 1, keepdims = True)
        sigma = cp.std(Z, axis = 1, keepdims = True)
    
    zhat = ((Z-cp.mean(mu, axis = 1, keepdims = True))
            /(((m/(m-1))*cp.mean(sigma, axis = 1, keepdims = True))+eps))
    z = (gamma*zhat)+beta
    A = activation('forward', function, z)
    D = cp.random.rand(A.shape[0], A.shape[1])                                        
    D = (D <= dropout).astype(int) 
    A = A*D                                       
    A = A/dropout 
    
            
    return A, z, zhat, Z, mu, sigma, D

def forward_conv(A_previous, Filter, Bias, pad, stride):
    '''
    A forward convolution step.
    Calcul output shape : ((x-f+2*pad)/stride)+1

    Parameters
    ----------
    A_previous : cp.array(examples, height, width, depth)
        Input images from the previous layer.
    Filter : cp.array(f, f, depth, number of filter)
        Filter to convolve with the input image.
    Bias : cp.array(1, 1, 1, number of filter)
        Bias for each filter.
    pad : int
        Padding edge width.
    stride : int
        Stride number.

    Returns
    -------
    Z : cp.array(examples, ((h-f+2*pad)/stride)+1, ((w-f+2*pad)/stride)+1), number of filter)
        Output layer image.

    '''
    (m, n_H_prev, n_W_prev, n_C_prev) = A_previous.shape

    (f, f, n_C_prev, n_C) = Filter.shape
    
    n_H = int(((n_H_prev-f+2*pad)/stride)+1)
    n_W = int(((n_W_prev-f+2*pad)/stride)+1)
    
    

    Z = cp.zeros([m, n_H, n_W, n_C])
    
    A_prev_pad = cp.pad(A_previous, ((0,0), (pad,pad), (pad,pad), (0,0),), mode='constant', constant_values = (0,0))

    i0 = cp.repeat(cp.arange(f), f)
    i1 = stride * cp.repeat(cp.arange(n_W), n_H)
    j0 = cp.tile(cp.arange(f), f)
    j1 = stride * cp.tile(cp.arange(n_H), n_W)
    i = cp.reshape(i0, (-1, 1))+cp.reshape(i1, (1, -1))
    j = cp.reshape(j0, (-1, 1))+cp.reshape(j1, (1, -1))
    k = cp.reshape(cp.repeat(cp.arange(n_C_prev), f**2), (-1, 1))
    Ztest = cp.squeeze(A_prev_pad[:, i, j, :])
    weights = cp.reshape(Filter, (f**2, n_C_prev, n_C))
    conV = cp.tensordot(weights, Ztest, ((0, 1), (1, 3)))
    Z = cp.reshape(cp.transpose(conV, (1, 2, 0)), (m, n_H, n_W, n_C))
    '''
    for i in range(m):               
        a_prev_pad = A_prev_pad[i, :, :, :]             
        for h in range(n_H):     
            vert_start = h*stride
            vert_end = h*stride+f
            
            for w in range(n_W):       
                horiz_start = w*stride
                horiz_end = w*stride+f
                
                a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                for c in range(n_C):  
                    Z[i, h, w, c] = cp.squeeze(cp.sum(a_slice_prev*Filter[:, :, :, c])+Bias[:, :, :, c])
    '''               
    return Z

def forward_pool(A_previous, stride, f, mode = "max"):
    '''
    A forward pool step
    Calcul output shape : (1 + (x - f) / stride)

    Parameters
    ----------
    A_previous : cp.array(examples, height, width, depth)
        Input images from the previous layer.
    stride : int
        Stride number.
    f : int
        Square filter dimension.
    mode : string, optional
        Pool mode 'mean' or 'max'. The default is "max".

    Returns
    -------
    A : cp.array(examples, 1 + (height - f) / stride, 1 + (width - f) / stride, depth)
        Output layer image.

    '''
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_previous.shape
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = cp.zeros((m, n_H, n_W, n_C))       

    i0 = cp.repeat(cp.arange(f), f)
    i1 = stride * cp.repeat(cp.arange(n_W), n_H)
    j0 = cp.tile(cp.arange(f), f)
    j1 = stride * cp.tile(cp.arange(n_H), n_W)
    i = cp.reshape(i0, (-1, 1))+cp.reshape(i1, (1, -1))
    j = cp.reshape(j0, (-1, 1))+cp.reshape(j1, (1, -1))
    R = cp.squeeze(A_previous[:, i, j, :])
    if mode == "max":
        pl = cp.max(R, 1)
    elif mode == "mean":
        pl = cp.mean(R, 1)
    A = cp.reshape(pl, (m, n_H, n_W, n_C))
    '''
    for i in range(m):                       
        for h in range(n_H):                   
            vert_start = h*stride
            vert_end = h*stride+f
            
            for w in range(n_W):        
                horiz_start = w*stride
                horiz_end = w*stride+f
                
                for c in range (n_C):
                    a_prev_slice = A_previous[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = cp.max(a_prev_slice)
                    elif mode == "mean":
                        A[i, h, w, c] = cp.mean(a_prev_slice)
    '''
    return A

def backward_function(dA_previous, A_previous, D, Z, z, zhat, gamma, beta, W, mu, sigma, function, dropout):
    '''
    A backward function step

    Parameters
    ----------
    dA_previous : cp.array(out_dim, examples)
        Cost derivates from A of the l+1 layer.
    A_previous : cp.array(in_dim, examples)
        Output forward result from the l-1 layer.
    D : cp.array(out_dim, examples)
        Dropout mask of the output from layer l.
    Z : cp.array(out_dim, examples)
        Linear result of layer l.
    z : cp.array(out_dim, examples)
        Activation input.
    zhat : cp.array(out_dim, examples)
        Normalized linear result.
    gamma : cp.array(out_dim, 1)
        Weight matrix.
    beta : cp.array(out_dim, 1)
        Bias matrix.
    W : cp.array(out_dim, in_dim)
        Weight matrix.
    mu : cp.array(out_dim, number of epochs)
        Mean collection.
    sigma : cp.array(out_dim, number of epochs)
        Standard deviation collection.
    function : string
        Activation function choice.
    dropout : float (in (0, 1))
        Dropout factor.
    last : bool
        Is it the last backward step. Default = False

    Returns
    -------
    dA : cp.array(in_dim, examples)
        Output layer cost derivation.
    dW : cp.array(out_dim, in_dim)
        Weight matrix derivation.
    dgamma : cp.array(out_dim, 1)
        Weight matrix derivation.
    dbeta : cp.array(out_dim, 1)
        Bias matrix derivation.

    '''
    
    m = dA_previous.shape[1]
    eps = 1e-8 
    
    mu = cp.mean(mu, axis = 1, keepdims = True)
    sigma = (m/(m-1))*cp.mean(sigma, axis = 1, keepdims = True)
    
    dz = activation('backward', function, z, dA_previous)
    dbeta = cp.sum(dz, axis=1, keepdims = True)
    dgamma = cp.sum(zhat*dz, axis=1, keepdims = True)
    dzhat = dz*gamma
    dsigma = (dbeta*(Z-mu))*(-gamma/(2*((sigma**3)+eps)))
    dmu = (dbeta*(gamma/(sigma+eps)))+(dsigma*(1/m)*cp.sum((-2)*(Z-mu), axis=1, keepdims = True))
    dZ = (dzhat/(sigma+eps))+(dsigma*(1/m)*(2*(Z-mu)))+((1/m)*dmu)
    dW = 1./m * cp.dot(Z, A_previous.T)
    dA = cp.dot(W.T, dZ)
    dA = dA*D
    dA = dA/dropout
    
    
    return dA, dW, dgamma, dbeta

def backward_conv(dZ, A_previous, Filter, Bias, pad, stride):
    '''
    A backward convolution step

    Parameters
    ----------
    dZ : cp.array(examples, ((h-f+2*pad)/stride)+1, ((w-f+2*pad)/stride)+1), number of filter)
        Cost derivative from the l+1 layer.
    A_previous : cp.array(examples, height, width, depth)
        Output image from the l-1 layer.
    Filter : cp.array(f, f, depth, number of filter)
        Convolutionnal filter.
    Bias : cp.array(1, 1, 1, number of filter)
        Bias respective to each filter.
    pad : int
        Padding parameter.
    stride : int
        Stride parameter.

    Returns
    -------
    dA : cp.array(examples, height, width, depth)
        Cost derivative from the current layer.
    dFilter : cp.array(f, f, depth, number of filter)
        Cost derivative from filter.
    dBias : cp.array(1, 1, 1, number of filter)
        Cost derivative from Bias.

    '''

    (m, n_H_prev, n_W_prev, n_C_prev) = A_previous.shape
    
    (f, f, n_C_prev, n_C) = Filter.shape
    
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA = cp.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dFilter = cp.zeros((f, f, n_C_prev, n_C))
    dBias = cp.zeros((1, 1, 1, n_C))
    dBias = cp.sum(dZ, axis=(0, 1, 2))
    
    A_prev_pad = cp.pad(A_previous, ((0,0), (pad,pad), (pad,pad), (0,0),), mode='constant', constant_values = (0,0))
    dA_prev_pad = cp.pad(dA, ((0,0), (pad,pad), (pad,pad), (0,0),), mode='constant', constant_values = (0,0))
    i0 = cp.repeat(cp.arange(f), f)
    i1 = stride * cp.repeat(cp.arange(n_W), n_H)
    j0 = cp.tile(cp.arange(f), f)
    j1 = stride * cp.tile(cp.arange(n_H), n_W)
    i = cp.reshape(i0, (-1, 1))+cp.reshape(i1, (1, -1))
    j = cp.reshape(j0, (-1, 1))+cp.reshape(j1, (1, -1))
    k = cp.reshape(cp.repeat(cp.arange(n_C_prev), f**2), (-1, 1))
    Ztest = cp.squeeze(A_prev_pad[:, i, j, :])
    dZtest = cp.reshape(cp.squeeze(dZ), (m, -1, n_C))
    dFiltertest = cp.tensordot(dZtest, cp.transpose(Ztest, (1, 0, 2, 3)), ((0, 1), (1, 2)))
    dFilter = cp.reshape(cp.transpose(dFiltertest, (1, 2, 0)), (f, f, n_C_prev, n_C))
    weights = cp.reshape(Filter, (f**2, n_C_prev, n_C))
    i0 = cp.tile(i0, n_C_prev)
    j0 = cp.tile(cp.arange(f), f * n_C_prev)
    i = cp.reshape(i0, (-1, 1)) + cp.reshape(i1, (1, -1))
    j = cp.reshape(j0, (-1, 1)) + cp.reshape(j1, (1, -1))
    Ztest = cp.squeeze(A_prev_pad[:, i, j, k])
    dAt = cp.tensordot(weights, dZtest.T, (2, 0))
    padded = cp.transpose(cp.zeros(A_prev_pad.shape), (0, 3, 1, 2))
    dAt_rshp = cp.transpose(cp.reshape(dAt, (n_C_prev*(f**2), -1, m)), (2, 0, 1))
    cp.scatter_add(padded, (slice(None), k, i, j), dAt_rshp)
    dA = cp.transpose(padded[:, :, pad:dA_prev_pad.shape[1]-pad, pad:dA_prev_pad.shape[2]-pad], (0, 2, 3, 1))
    '''
    for i in range(m):                     
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]
        
        for h in range(n_H):
            vert_start = h*stride
            vert_end = h*stride + f
            for w in range(n_W):
                horiz_start = w*stride
                horiz_end = w*stride + f
                a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                for c in range(n_C):
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += Filter[:,:,:,c] * dZ[i, h, w, c]
                    dFilter[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    dBias[:,:,:,c] += dZ[i, h, w, c]
        dA[i, :, :, :] = da_prev_pad[pad:da_prev_pad.shape[0]-pad, pad:da_prev_pad.shape[1]-pad, :]
    '''
    return dA, dFilter, dBias

def backward_pool(dA, A_previous, stride, f, mode = "max"):
    '''
    A backward pool step

    Parameters
    ----------
    dA : cp.array(examples, 1 + (height - f) / stride, 1 + (width - f) / stride, depth)
        Cost derivative from l+1 layer.
    A_previous : cp.array(examples, height, width, depth)
        Output image from the l-1 layer.
    stride : int
        Stride parameter.
    f : int
        Square filter dimension.
    mode : string, optional
        Filter type. The default is "max".

    Returns
    -------
    dA_prev : cp.array(examples, height, width, depth)
        Cost derivative from the current layer.

    '''
    
    m, n_H_prev, n_W_prev, n_C_prev = A_previous.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = cp.zeros(A_previous.shape)
    
    for i in range(m):               
        a_prev = A_previous[i, :, :, :]
        
        for h in range(n_H):
            vert_start = h*stride
            vert_end = h*stride + f                 
            for w in range(n_W):
                horiz_start = w*stride
                horiz_end = w*stride + f
                for c in range(n_C):           
                    
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = (a_prev_slice == cp.max(a_prev_slice))
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i,h,w,c]
                        
                    elif mode == "mean":

                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += dA[i,h,w,c] * cp.ones((f, f))/f**2
                        
    return dA_prev

def cost(AL, Y, mode = 'SEL'):
    '''
    

    Parameters
    ----------
    AL : cp.array(out_dim, examples)
        Final layer output.
    Y : cp.array(out_dim, examples)
        Expected output.
    mode : string, optional
        Type of cost computation. The default is 'SEL'.

    Returns
    -------
    cost : cp.array(out_dim, 1)
        Cost output (cost per features).
    dAL : cp.array(out_dim, examples)
        Cost derivates function.

    '''
    
    m = Y.shape[1]
    
    if mode == 'XC':
        cost = -(1/m)*cp.sum((Y*cp.log(AL)+((1-Y)*cp.log(1-AL))), axis = 1)
        dAL = - (cp.divide(Y, AL) - cp.divide(1 - Y, 1 -AL))
    elif mode == 'SEL':
        cost = (1/(2*m))*cp.sum((AL - Y)**2, axis = 1)
        dAL = AL - Y
        
    cost = cp.squeeze(cost)
    
    return cost, dAL

def update_parameters_with_adam(W, gamma, beta, dW, dgamma, dbeta,
                                vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta,
                                t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    '''
    

    Parameters
    ----------
    W : cp.array(out_dim, in_dim)
        Weight matrix.
    gamma : cp.array(out_dim, 1)
        Weight matrix.
    beta : cp.array(out_dim, 1)
        Bias matrix.
    dW : cp.array(out_dim, in_dim)
        Weight matrix derivative.
    dgamma : cp.array(out_dim, 1)
        Weight matrix derivative.
    dbeta : cp.array(out_dim, 1)
        Bias matrix derivative.
    vdW : cp.array(out_dim, in_dim)
        Derivative moving average weight matrix.
    vdgamma : cp.array(out_dim, 1)
        Derivative moving average weight matrix.
    vdbeta : cp.array(out_dim, 1)
        Derivative moving average bias matrix.
    sdW : cp.array(out_dim, in_dim)
        Squared derivative moving average weight matrix.
    sdgamma : cp.array(out_dim, 1)
        Squared derivative moving average weight matrix.
    sdbeta : cp.array(out_dim, 1)
        Squared derivative moving average bias matrix.
    t : int
        Power Adam evolution.
    learning_rate : float, optional
        Length of each step. The default is 0.01.
    beta1 : float, optional
        Moving average parameter. The default is 0.9.
    beta2 : float, optional
        Moving average parameter. The default is 0.999.
    epsilon : float, optional
        Ensure non zeros division. The default is 1e-8.

    Returns
    -------
    W : cp.array(out_dim, in_dim)
        Updated weight matrix.
    gamma : cp.array(out_dim, 1)
        Updated weight matrix.
    beta : cp.array(out_dim, 1)
        Updated bias matrix.
    vdW : cp.array(out_dim, in_dim)
        Updated derivative moving average weight matrix.
    vdgamma : cp.array(out_dim, 1)
        Updated derivative moving average weight matrix.
    vdbeta : cp.array(out_dim, 1)
        Updated derivative moving average bias matrix.
    sdW : cp.array(out_dim, in_dim)
        Updated squared derivative moving average weight matrix.
    sdgamma : cp.array(out_dim, 1)
        Updated squared derivative moving average weight matrix.
    sdbeta : cp.array(out_dim, 1)
        Updated squared derivative moving average bias matrix.

    '''  

    vdW = beta1*vdW+(1-beta1)*dW
    vdbeta = beta1*vdbeta+(1-beta1)*dbeta
    vdgamma = beta1*vdgamma+(1-beta1)*dgamma

    v_cdW= vdW/(1-pow(beta1, t))
    v_cdbeta = vdbeta/(1-pow(beta1, t))
    v_cdgamma = vdgamma/(1-pow(beta1, t))

    sdW = beta2*sdW+(1-beta2)*pow(dW, 2)
    sdbeta = beta2*sdbeta+(1-beta2)*pow(dbeta, 2)
    sdgamma = beta2*sdgamma+(1-beta2)*pow(dgamma, 2)

    s_cdW = sdW/(1-pow(beta2, t))
    s_cdbeta = sdbeta/(1-pow(beta2, t))
    s_cdgamma = sdgamma/(1-pow(beta2, t))

    W = W-learning_rate*(v_cdW/(cp.sqrt(s_cdW)+epsilon))
    beta = beta-learning_rate*(v_cdbeta/(cp.sqrt(s_cdbeta)+epsilon))
    gamma = gamma-learning_rate*(v_cdgamma/(cp.sqrt(s_cdgamma)+epsilon))


    return W, gamma, beta, vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta

def update_parameters_with_adam_conv(Filter, bias, dFilter, dbias,
                                vdFilter, vdbias, sdFilter, sdbias,
                                t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    '''
    

    Parameters
    ----------
    Filter : cp.array(f, f, depth, filter_number)
        Filters
    bias : cp.array(1, 1, 1, filter_number)
        Bias matrix.
    dFilter : cp.array(f, f, depth, filter_number)
        Filters matrix derivative.
    dbias : cp.array(1, 1, 1, filter_number)
        Bias matrix derivative.
    vdFilter : cp.array(f, f, depth, filter_number)
        Derivative moving average Filters matrix.
    vdbias : cp.array(1, 1, 1, filter_number)
        Derivative moving average bias matrix.
    sdFilter : cp.array(f, f, depth, filter_number)
        Squared derivative moving average Filters matrix.
    sdbias : cp.array(1, 1, 1, filter_number)
        Squared derivative moving average bias matrix.
    t : int
        Power Adam evolution.
    learning_rate : float, optional
        Length of each step. The default is 0.01.
    beta1 : float, optional
        Moving average parameter. The default is 0.9.
    beta2 : float, optional
        Moving average parameter. The default is 0.999.
    epsilon : float, optional
        Ensure non zeros division. The default is 1e-8.

    Returns
    -------
    Filter : cp.array(f, f, depth, filter_number)
        Updated filter matrix.
    bias : cp.array(1, 1, 1, filter_number)
        Updated bias matrix.
    vdFilter : cp.array(f, f, depth, filter_number)
        Updated derivative moving average filter matrix.
    vdbias : cp.array(1, 1, 1, filter_number)
        Updated derivative moving average bias matrix.
    sdFilter : cp.array(f, f, depth, filter_number)
        Updated squared derivative moving average filter matrix.
    sdbias : cp.array(1, 1, 1, filter_number)
        Updated squared derivative moving average bias matrix.

    '''  

    vdFilter = beta1*vdFilter+(1-beta1)*dFilter
    vdbias = beta1*vdbias+(1-beta1)*dbias

    v_cdFilter= vdFilter/(1-pow(beta1, t))
    v_cdbias = vdbias/(1-pow(beta1, t))

    sdFilter = beta2*sdFilter+(1-beta2)*pow(dFilter, 2)
    sdbias = beta2*sdbias+(1-beta2)*pow(dbias, 2)

    s_cdFilter = sdFilter/(1-pow(beta2, t))
    s_cdbias = sdbias/(1-pow(beta2, t))

    Filter = Filter-learning_rate*(v_cdFilter/(cp.sqrt(s_cdFilter)+epsilon))
    bias = bias-learning_rate*(v_cdbias/(cp.sqrt(s_cdbias)+epsilon))


    return Filter, bias, vdFilter, vdbias, sdFilter, sdbias

AlexNet = (('input', (224, 224, 3)),
           ('conv', (8, 3, 96, 0, 4)),('pool', (3, 2), 'max'), 
           ('conv', (5, 96, 256, 2, 0)), ('pool', (3, 2), 'max'), 
           ('conv', (3, 256, 384, 1, 0)), ('conv', (3, 384, 384, 1, 0)), 
           ('conv', (3, 384, 256, 1, 0)), ('pool', (3, 2)), 
           ('flatten', 9216), 
           ('dense', 4096, 'relu'), ('dense', 4096, 'relu'),
           ('dense', 1000, 'relu'), 
           ('dense', 10, 'sigmoid'))

LeNet = (('input', (28, 28, 3)), 
         ('conv', (5, 3, 6, 2, 1)), ('pool', (2, 2), 'max'),
         ('conv', (5, 6, 16, 0, 1)), ('pool', (2, 2), 'max'), 
         ('flatten', 400), 
         ('dense', 120, 'relu'), ('dense', 84, 'relu'),
         ('dense', 10, 'sigmoid'))

def train_CNN(X, Y, layers, learning_rate = 0.7, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, 
          keep_prob = 0.9, print_cost = True, pourcentageStop = 0.1, cost_mode = 'SEL', log = True):
    """
    Modelize the designed CNN.
    
    Arguments:
    X -- input data, of shape (height, width, depth, number of examples)
    Y -- shape (number of possible results, number of examples)
    layers -- python list, fill with layers parameters
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    seed = 0
    L = len(layers)-1             # number of layers in the neural networks
    costs = 0                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update   
    m = X.shape[0]                   # number of training examples
    
    # Initialize parameters
    parameters = {}
    adam = {}
    for l in range(1, len(layers)):
        if(layers[l][0] == 'conv'):
            W, beta = initialize_parameters_he_conv(layers[l][1][0], 
                                                         layers[l][1][1], 
                                                         layers[l][1][2])
            vdW, vdbeta, sdW, sdbeta = initialize_adam_conv(W, beta)
        if(layers[l][0] == 'dense'):
            W, gamma, beta, mu, sigma = initialize_parameters_he(layers[l-1][1], 
                                                                 layers[l][1])
            vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta = initialize_adam(W, 
                                                                         beta, 
                                                                         gamma)
            parameters["gamma"+str(l)] = gamma
            parameters["mu"+str(l)] = mu
            parameters["sigma"+str(l)] = sigma
            adam["vdgamma"+str(l)] = vdgamma
            adam["sdgamma"+str(l)] = sdgamma
            
            
        parameters["W"+str(l)] = W
        parameters["beta"+str(l)] = beta
        adam["vdW"+str(l)] = vdW
        adam["vdbeta"+str(l)] = vdbeta
        adam["sdW"+str(l)] = sdW
        adam["sdbeta"+str(l)] = sdbeta
        

    x = []
    fpa_cache = []
    deg = 4
    
    # Optimization loop
    for i in range(num_epochs):
        
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        # Select a minibatch
        for minibatch in minibatches:
            cache = {}
            tim = [0, 0, 0, 0, 0, 0]
            (minibatch_X, minibatch_Y) = minibatch
            cache["A0"] = minibatch_X
            
            # Forward propagation
            print("----------Forward prop----------")
            for l in range(1, len(layers)):
                if(layers[l][0] == 'conv'):
                    start = timer()
                    cache["A"+str(l)] = forward_conv(cache["A"+str(l-1)], 
                                                     parameters["W"+str(l)], 
                                     parameters["beta"+str(l)], 
                                     layers[l][1][3],  layers[l][1][4])
                    tim[0] = tim[0] + timer()-start 
                if(layers[l][0] == 'pool'):
                    start = timer()
                    cache["A"+str(l)] = forward_pool(cache["A"+str(l-1)], 
                                     layers[l][1][1], layers[l][1][0],
                                     layers[l][2])
                    tim[1] = tim[1] + timer()-start
                if(layers[l][0] == 'dense'):
                    start = timer()
                    cache["A"+str(l)], cache["z"+str(l)], cache["zhat"+str(l)], cache["Z"+str(l)], parameters["mu"+str(l)], parameters["sigma"+str(l)], cache["D"+str(l)] = forward_function(cache["A"+str(l-1)], 
                        parameters["W"+str(l)], parameters["mu"+str(l)],
                        parameters["sigma"+str(l)], parameters["gamma"+str(l)], 
                        parameters["beta"+str(l)], layers[l][2], keep_prob)
                    tim[2] = tim[2] + timer()-start
                if(layers[l][0] == 'flatten'):
                    cache["A"+str(l)] = cp.reshape(cache["A"+str(l-1)], 
                                   (cache["A"+str(l-1)].shape[1]*
                                    cache["A"+str(l-1)].shape[2]*
                                    cache["A"+str(l-1)].shape[3],
                                    cache["A"+str(l-1)].shape[0]))
                    cache["D"+str(l)] = cp.ones(cache["A"+str(l)].shape)
                print("A%d: %s" %(l, str(cache["A"+str(l)].shape)))
                
            
            # Compute cost and add to the cost total
            print("----------Cost computation----------")
            costs, cache["dA"+str(L+1)] = cost(cache["A"+str(L)],
                                                minibatch_Y, cost_mode)
            
            cost_total += cp.sum(abs(costs))
            print("Cost: %f" %(cost_total))
            
            

            # Backward propagation
            print("----------Bacward prop----------")
            for l in reversed(range(1, len(layers))):
                print("dA%d: %s" %(l+1, str(cache["dA"+str(l+1)].shape)))
                if(layers[l][0] == 'conv'):
                    start = timer()
                    cache["dA"+str(l)], cache["dW"+str(l)], cache["dbeta"+str(l)] = backward_conv(cache["dA"+str(l+1)],
                                cache["A"+str(l-1)], parameters["W"+str(l)], 
                                parameters["beta"+str(l)], 
                                layers[l][1][3],  layers[l][1][4])
                    tim[3] = tim[3] + timer()-start
                if(layers[l][0] == 'pool'):
                    start = timer()
                    cache["dA"+str(l)] = backward_pool(cache["dA"+str(l+1)], 
                                     cache["A"+str(l-1)], layers[l][1][1], 
                                     layers[l][1][0], layers[l][2])
                    tim[4] = tim[4] + timer()-start
                if(layers[l][0] == 'dense'):
                    start = timer()
                    cache["dA"+str(l)], cache["dW"+str(l)], cache["dgamma"+str(l)], cache["dbeta"+str(l)] = backward_function(cache["dA"+str(l+1)], 
                        cache["A"+str(l-1)], cache["D"+str(l-1)], cache["Z"+str(l)],
                        cache["z"+str(l)], cache["zhat"+str(l)], 
                        parameters["gamma"+str(l)], parameters["beta"+str(l)],
                        parameters["W"+str(l)], parameters["mu"+str(l)], 
                        parameters["sigma"+str(l)], layers[l][2], keep_prob)
                    tim[5] = tim[5] + timer()-start
                if(layers[l][0] == 'flatten'):
                    cache["dA"+str(l)] = cp.reshape(cache["dA"+str(l+1)], 
                                   cache["A"+str(l-1)].shape)
                
            print("----------Time spent----------")
            print("forward conv: %f" %(tim[0]))
            print("forward pool: %f" %(tim[1]))
            print("forward dense: %f" %(tim[2]))
            print("backward conv: %f" %(tim[3]))
            print("backward pool: %f" %(tim[4]))
            print("backward dense: %f" %(tim[5]))
            
            # Update parameters
            t = t + 1 # Adam counter
            print("----------Optimisation----------")
            for l in range(1, len(layers)):
                print("layer: %d" %(l))
                if(layers[l][0] == 'conv'):
                    parameters["W"+str(l)], parameters["beta"+str(l)], adam["vdW"+str(l)], adam["vdbeta"+str(l)], adam["sdW"+str(l)], adam["sdbeta"+str(l)] = update_parameters_with_adam_conv(parameters["W"+str(l)], 
                                 parameters["beta"+str(l)], cache["dW"+str(l)], 
                                 cache["dbeta"+str(l)], adam["vdW"+str(l)], 
                                 adam["vdbeta"+str(l)], adam["sdW"+str(l)], 
                                 adam["sdbeta"+str(l)], t, learning_rate, 
                                 beta1, beta2, epsilon)
                if(layers[l][0] == 'dense'):
                    parameters["W"+str(l)], parameters["gamma"+str(l)], parameters["beta"+str(l)], adam["vdW"+str(l)], adam["vdgamma"+str(l)], adam["vdbeta"+str(l)], adam["sdW"+str(l)], adam["sdgamma"+str(l)], adam["sdbeta"+str(l)] = update_parameters_with_adam(parameters["W"+str(l)],
                                parameters["gamma"+str(l)], parameters["beta"+str(l)],
                                cache["dW"+str(l)], cache["dgamma"+str(l)], 
                                cache["dbeta"+str(l)], adam["vdW"+str(l)],
                                adam["vdgamma"+str(l)], adam["vdbeta"+str(l)],
                                adam["sdW"+str(l)], adam["sdgamma"+str(l)],
                                adam["sdbeta"+str(l)], t, learning_rate, 
                                beta1, beta2, epsilon)
          
        # Break if cost derivative trend is flat
        if log:
            cost_avg = np.log10(1+(cost_total / m))
        else:
            cost_avg = cost_total / m
        
        '''
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
            x.append(i/100)
        if len(x)>15:
            A, ls, Val = leastSquare(x, costs, deg)
            fpa, y = tangente(A, len(x)-10, Val, deg)
            fpa_cache.append(abs(fpa))
            if pourcentageStop*np.asarray(fpa_cache).max()>fpa_cache[len(fpa_cache)-1]:
                break
                
    # plot the cost
    plt.figure
    plt.plot(costs)
    plt.plot(ls)
    plt.plot(y)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    plt.close
'''
    return parameters
   
X = cp.random.rand(1000, 28, 28, 3)
Y = cp.random.rand(10, 1000)

train_CNN(X, Y, LeNet)
