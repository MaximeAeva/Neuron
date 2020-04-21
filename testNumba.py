import numpy as np
import time 
import cupy as cp

dim = 2000

def f (A, W, gamma, beta):
    Z = np.dot(W, A)
    m = np.mean(Z, axis = 1, keepdims = True)
    s = np.std(Z, axis = 1, keepdims = True)
    zhat = (Z-m)/s
    z = gamma*zhat + beta
    a = 1/(1+np.exp(-z))
    d = np.random.rand(a.shape[0], a.shape[1])
    d = (d<0.8).astype(int)
    a = a*d
    a = a/0.8
    return a

def g (A, W, gamma, beta):
    Z = cp.dot(W, A)
    m = cp.mean(Z, axis = 1, keepdims = True)
    s = cp.std(Z, axis = 1, keepdims = True)
    zhat = (Z-m)/s
    z = gamma*zhat + beta
    a = 1/(1+cp.exp(-z))
    d = cp.random.rand(a.shape[0], a.shape[1])
    d = (d<0.8).astype(int)
    a = a*d
    a = a/0.8
    return a

py = np.array([])
nu = np.array([])
cu = np.array([])

for i in range (100):


    starti = time.time()
    A = np.random.rand(dim, 800)
    W = np.random.rand(dim, dim)
    gamma = np.random.rand(dim, 1)
    beta = np.random.rand(dim, 1)
    k = f (A, W, gamma, beta)
    endi = time.time()
    py = np.append(py, endi - starti)
    

    startj = time.time()
    Ac = cp.random.rand(dim, 800)
    Wc = cp.random.rand(dim, dim)
    gammac = cp.random.rand(dim, 1)
    betac = cp.random.rand(dim, 1)
    l = g (Ac, Wc, gammac, betac)
    endj = time.time()
    cu = np.append(cu, endj - startj)
    
print ("numpy running time -> mean: %f, m:%f, M:%f" %(py.mean(), py.min(), py.max()))
print ("cupy running time -> mean: %f, m:%f, M:%f" %(cu.mean(), cu.min(), cu.max()))
print('improvement ration: %f' %(py.mean()/(np.min(nu.mean(), cu.mean()))))