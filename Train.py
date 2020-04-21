import numpy as np
import math
import matplotlib.pyplot as plt
from Functions import *
from Scrapper import *
from Display import *


p = 1
X, Y, X_test, Y_test = LoadDataSet ("OceaniqueAquitain", param, [25, 2000], 0.8)

mX, sX = normnreduc(X, len(param))
mY, sY = normnreduc(Y, len(param))
X = (((X-mX)/(sX**p)))
Y = ((Y-mY)/(sY**p))
X_test = (((X_test-mX)/(sX**p)))
Y_test = Y_test

q = 100*(np.ones((5, 1)))
attempt = 0
cost_vector = []

while (abs(q[0])>2):
    attempt = attempt + 1
    print ("tentative = " +str(attempt))
    kprob = np.random.rand()*0.1+0.9
    lrate = 10**(-(np.random.rand()*1+3))
    
    tab = []
    for i in range(5):
        tab.append((int(np.random.rand()*15+5), 'identity'))
    
    
    layers_dims = [(X.shape[0], 'input'), (int(np.random.rand()*7+3), 'tanh')]
    layers_dims = layers_dims + tab
    layers_dims.append((Y.shape[0], 'identity'))
    
    
    print("Current test : %f, %f, %s" %(lrate, kprob, layers_dims))
    
    
    parms = model(X, Y, layers_dims, learning_rate = lrate, mini_batch_size = 64, beta = 0.9,
              beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000,
              keep_prob = 0.98, print_cost = True, pourcentageStop = 0.01,
              cost_mode = 'SEL', log = True)
    
    a, caches = L_model_forward(X_test, parms, layers_dims)
    
    a = ((a)*(sY**p))+mY
    
    q = a - Y_test
    
    q = np.std(q, axis = 1)
    
    if q.shape != (5, 1):
        q = np.asarray(q)[:, np.newaxis]
    
    print(q.T)

    cost_vector.append(np.sum(abs(q)))
    
    
    

print("Found !")


loadMyNet(parms, layers_dims)

