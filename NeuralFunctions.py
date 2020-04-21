import numpy as np
import cupy as cp
import math
import time 
from NeuralActivation import activation

def initialize_parameters_he(in_dim, out_dim):

    W = cp.random.randn(out_dim, in_dim)*cp.sqrt(2/in_dim)
    gamma = cp.ones((out_dim, 1))
    beta = cp.zeros((out_dim, 1))
    mu = cp.zeros((out_dim, 1))
    sigma = cp.zeros((out_dim, 1))
    
    return W, gamma, beta, mu, sigma

def initialize_adam(W, beta, gamma) :

    vdW = cp.zeros((W.shape[0], W.shape[1]))
    vdbeta = cp.zeros((beta.shape[0], beta.shape[1]))
    vdgamma = cp.zeros((gamma.shape[0], gamma.shape[1]))
    sdW = cp.zeros((W.shape[0], W.shape[1]))
    sdbeta = cp.zeros((beta.shape[0], beta.shape[1]))
    sdgamma = cp.zeros((gamma.shape[0], gamma.shape[1]))

    
    return vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta

def random_mini_batches(X, Y, mini_batch_size = 64, train = 0.7):
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

def forward_function(A_previous, W, mu, sigma, gamma, beta, function, dropout = False):

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

def backward_propagation_with_dropout(dA_previous, A_previous, D, Z, z, zhat, gamma, beta, W, mu, sigma, function, dropout):

    m = dA_previous.shape[1]
    eps = 1e-8 
    
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

def compute_cost(AL, Y, mode = 'SEL'):
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
    

    vdW = beta1*vdW+(1-beta1)*dW
    vdbeta = beta1*vdbeta+(1-beta1)*dbeta
    vdgamma = beta1*vdgamma+(1-beta1)*dgamma

    v_cdW= vdW/(1-pow(beta1, 2))
    v_cdbeta = vdbeta/(1-pow(beta1, 2))
    v_cdgamma = vdgamma/(1-pow(beta1, 2))

    sdW = beta2*sdW+(1-beta2)*pow(dW, 2)
    sdbeta = beta2*sdbeta+(1-beta2)*pow(dbeta, 2)
    sdgamma = beta2*sdgamma+(1-beta2)*pow(dgamma, 2)

    s_cdW = sdW/(1-pow(beta2, 2))
    s_cdbeta = sdbeta/(1-pow(beta2, 2))
    s_cdgamma = sdgamma/(1-pow(beta2, 2))

    W = W-learning_rate*(v_cdW/(cp.sqrt(s_cdW)+epsilon))
    beta = beta-learning_rate*(v_cdbeta/(cp.sqrt(s_cdbeta)+epsilon))
    gamma = gamma-learning_rate*(v_cdgamma/(cp.sqrt(s_cdgamma)+epsilon))


    return W, gamma, beta, vdW, vdgamma, vdbeta, sdW, sdgamma, sdbeta


for i in range(10000):
    
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
        
        
    cost, dAL = compute_cost(forward["A"+str(len(layers)-1)], Y, mode = 'SEL')
    backward["dA"+str(len(layers)-1)] = dAL
    
    for l in reversed(range(1, len(layers))):                                          
        dA, dW, dgamma, dbeta = backward_propagation_with_dropout(backward["dA"+str(l)],
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