import numpy as np
import cupy as cp
import math
import time 
from NeuralActivation import activation

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

def random_mini_batches(X, Y, mini_batch_size = 64):
    '''
    Splite training data into multiple mini_batches

    Parameters
    ----------
    X : cp.array(features_in, examples)
        Input dataset.
    Y : cp.array(features_out, examples)
        Output target.
    mini_batch_size : int (a power of 2), optional
        Size of a batch. The default is 64.

    Returns
    -------
    mini_batches : cp.list of tuple (X, Y)
        A list of batches which the given size.

    '''
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(cp.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
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
                    elif mode == "average":
                        A[i, h, w, c] = cp.mean(a_prev_slice)
    
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
    dZ = (dzhat/sigma)+(dsigma*(1/m)*(2*(Z-mu)))+((1/m)*dmu)
    dW = 1./m * cp.dot(Z, A_previous.T)
    dA = cp.dot(W.T, dZ)
    dA= dA*D
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

    A_prev_pad = cp.pad(A_previous, ((0,0), (pad,pad), (pad,pad), (0,0),), mode='constant', constant_values = (0,0))
    dA_prev_pad = cp.pad(dA, ((0,0), (pad,pad), (pad,pad), (0,0),), mode='constant', constant_values = (0,0))
    
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

        dA[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
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
                        
                    elif mode == "average":

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


for i in range(1):
    
    Y = cp.random.rand(50, 128)
    
    layers = ((1000, 'in'), (100, 'relu'), (150, 'relu'), (70, 'relu'), (50, 'relu'),(20, 'relu'),(50, 'sigmoid'))
    parameters = {}
    adam = {}
    forward = {}
    backward = {}
    for l in range(1, len(layers)):
        W, gamma, beta, mu, sigma = initialize_parameters_he(layers[l-1][0], layers[l][0])
        vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta = initialize_adam(W, beta, gamma)
        parameters["W"+str(l)] = W
        parameters["gamma"+str(l)] = gamma
        parameters["beta"+str(l)] = beta
        parameters["mu"+str(l)] = mu
        parameters["sigma"+str(l)] = sigma
        adam["vdW"+str(l)] = vdW
        adam["vdgamma"+str(l)] = vdgamma
        adam["vdbeta"+str(l)] = vdbeta
        adam["sdW"+str(l)] = sdW
        adam["sdgamma"+str(l)] = sdgamma
        adam["sdbeta"+str(l)] = sdbeta
        
    forward["A0"] = cp.ones((1000, 128))
    forward["D0"] = cp.ones((1000, 128))
    
    for l in range(1, len(layers)):
        F = cp.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])[:, :,  cp.newaxis,  cp.newaxis]
        B = 10*cp.ones((1, 1, 1, 1))
        H = forward["A0"][cp.newaxis, :, :, cp.newaxis]
        Q = forward_conv(H, F, B, 3, 3)   
        A, z, zhat, Z, mu, sigma, D = forward_function(forward["A"+str(l-1)], 
                                                       parameters["W"+str(l)], parameters["mu"+str(l)], 
                                                       parameters["sigma"+str(l)], parameters["gamma"+str(l)], 
                                                       parameters["beta"+str(l)], layers[l][1], dropout = 0.8)
        forward["A"+str(l)] = A
        forward["z"+str(l)] = z
        forward["zhat"+str(l)] = zhat
        forward["Z"+str(l)] = Z
        forward["D"+str(l)] = D
        parameters["mu"+str(l)] = mu
        parameters["sigma"+str(l)] = sigma
        
        
    cost, dAL = cost(forward["A"+str(len(layers)-1)], Y, mode = 'SEL')
    backward["dA"+str(len(layers)-1)] = dAL
    
    for l in reversed(range(1, len(layers))):     
        dA, dW, dgamma, dbeta = backward_function(backward["dA"+str(l)],
                                                                  forward["A"+str(l-1)],
                                                                  forward["D"+str(l-1)], forward["Z"+str(l)],
                                                                  forward["z"+str(l)], forward["zhat"+str(l)],
                                                                  parameters["gamma"+str(l)], parameters["beta"+str(l)],
                                                                  parameters["W"+str(l)], parameters["mu"+str(l)], parameters["sigma"+str(l)],
                                                                  layers[l][1], 0.8)
        W, gamma, beta, vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta = update_parameters_with_adam(parameters["W"+str(l)], parameters["gamma"+str(l)], parameters["beta"+str(l)],
                                                                                                 dW, dgamma, dbeta,
                                adam["vdW"+str(l)], adam["vdgamma"+str(l)], adam["vdbeta"+str(l)],
                                adam["sdW"+str(l)], adam["sdgamma"+str(l)], adam["sdbeta"+str(l)],
                                1, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8)
        
        backward["dA"+str(l-1)] = dA