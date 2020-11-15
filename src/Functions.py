import numpy as np
import math
import matplotlib.pyplot as plt
from ActivationFunctions import *



plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0, train = 0.7):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : m*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : m*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l][0], layers_dims[l-1][0])*np.sqrt(2/layers_dims[l-1][0])
        parameters['gamma' + str(l)] = np.ones((layers_dims[l][0], 1))
        parameters['beta' + str(l)] = np.zeros((layers_dims[l][0], 1))
        parameters['m' + str(l)] = []
        parameters['s' + str(l)] = []
        
        ### END CODE HERE ###
        
    return parameters

def linear_forward(A, W):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A)
    ### END CODE HERE ###
    
    cache = (A, W)
    
    return Z, cache

def linear_activation_forward(A_prev, W, gamma, beta, m, s, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    Z, linear_cache = linear_forward(A_prev, W)
    z = gamma*((Z-m)/s) + beta
    A, activation_cache = functions('forward', activation, z)

    
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, layers_dims):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 5                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L+1):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, 
         parameters["W"+str(l)], parameters["gamma"+str(l)], parameters["beta"+str(l)],
         parameters["m"+str(l)], parameters["s"+str(l)],layers_dims[l][1])
        caches.append(cache)

            
    return A, caches


def forward_propagation_with_dropout(X, parameters, layers_dims, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (vect, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    m = X.shape[1]
    paramliste = {}
    paramliste["A0"] = X
    L = len(parameters) // 5          # number of layers in the neural network
    eps = 1e-8
    alpha = 0.9
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L+1):
        paramliste["Z"+str(l)] = np.dot(parameters["W"+str(l)], paramliste["A"+str(l-1)])
        if parameters["m"+str(l)] == []:
            parameters["m"+str(l)] = np.mean(paramliste["Z"+str(l)], axis = 1, keepdims = True)
            parameters["s"+str(l)] = np.std(paramliste["Z"+str(l)], axis = 1, keepdims = True)
        else:
            parameters["m"+str(l)] = (alpha*parameters["m"+str(l)]) + ((1-alpha)*np.mean(paramliste["Z"+str(l)], axis = 1, keepdims = True))
            parameters["s"+str(l)] = (alpha*parameters["s"+str(l)]) + ((1-alpha)*(np.std(paramliste["Z"+str(l)], axis = 1, keepdims = True)))
        paramliste["zhat"+str(l)] = ((paramliste["Z"+str(l)]-parameters["m"+str(l)])/(parameters["s"+str(l)]+eps))
        paramliste["z"+str(l)] = (parameters["gamma"+str(l)]*paramliste["zhat"+str(l)])+parameters["beta"+str(l)]
        paramliste["A"+str(l)], cache = functions('forward', layers_dims[l][1], np.asarray(paramliste["z"+str(l)]))
        if l != L:
            paramliste["D"+str(l)] = np.random.rand(np.asarray(paramliste["A"+str(l)]).shape[0],np.asarray(paramliste["A"+str(l)]).shape[1])                                         # Step 1: initialize matrix D1 = np.random.rand(..., ...)
            paramliste["D"+str(l)] = (paramliste["D"+str(l)] < keep_prob).astype(int) # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
            paramliste["A"+str(l)] = paramliste["A"+str(l)]*paramliste["D"+str(l)]                                        # Step 3: shut down some neurons of A1
            paramliste["A"+str(l)] = paramliste["A"+str(l)]/keep_prob 
    
            
    return paramliste, L


def backward_propagation_with_dropout(X, Y, parms, parameters, L, dAL, layers_dims, keep_prob = 0.5):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    gradients = {}
    eps = 1e-8 
    gradients["dA"+str(L)] = dAL
    
    for l in reversed(range(1, L+1)):
        gradients["dz"+str(l)] = functions('backward', layers_dims[l][1],
                                   parms["z"+str(l)], gradients["dA"+str(l)])
        gradients["dbeta"+str(l)] = np.sum(gradients["dz"+str(l)], axis=1, keepdims = True)
        gradients["dgamma"+str(l)] = np.sum(parms["zhat"+str(l)]*gradients["dz"+str(l)], axis=1, keepdims = True)
        gradients["dzhat"+str(l)] = gradients["dz"+str(l)]*parameters["gamma"+str(l)]
        gradients["ds"+str(l)] = (gradients["dbeta"+str(l)]*(parms["Z"+str(l)]-parameters["m"+str(l)]))*(-parameters["gamma"+str(l)]/(2*((parameters["s"+str(l)]**3)+eps)))
        gradients["dm"+str(l)] = (gradients["dbeta"+str(l)]*(parameters["gamma"+str(l)]/(parameters["s"+str(l)]+eps)))+(gradients["ds"+str(l)]*(1/m)*np.sum((-2)*(parms["Z"+str(l)]-parameters["m"+str(l)]), axis=1, keepdims = True))
        gradients["dZ"+str(l)] = (gradients["dzhat"+str(l)]/parameters["s"+str(l)])+(gradients["ds"+str(l)]*(1/m)*(2*(parms["Z"+str(l)]-parameters["m"+str(l)])))+((1/m)*gradients["dm"+str(l)])
        gradients["dW"+str(l)] = 1./m * np.dot(gradients["dZ"+str(l)], parms["A"+str(l-1)].T)
        if l > 1 :
            gradients["dA"+str(l-1)] = np.dot(parameters["W"+str(l)].T, gradients["dZ"+str(l)])
            gradients["dA"+str(l-1)]=gradients["dA"+str(l-1)]*parms["D"+str(l-1)]
            gradients["dA"+str(l-1)] = gradients["dA"+str(l-1)]/keep_prob
    
    
    return gradients



def compute_cost(AL, Y, mode = 'SEL'):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    
    if mode == 'XC':
        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
        cost = -(1/m)*np.sum((Y*np.log(AL)+((1-Y)*np.log(1-AL))), axis = 1)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 -AL))
        ### END CODE HERE ###
    elif mode == 'SEL':
        cost = (1/(2*m))*np.sum((AL - Y)**2, axis = 1)
        dAL = AL - Y
        
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost, dAL


def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 5 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
        v["dbeta" + str(l+1)] = np.zeros((parameters["beta" + str(l+1)].shape[0],parameters["beta" + str(l+1)].shape[1]))
        v["dgamma" + str(l+1)] = np.zeros((parameters["gamma" + str(l+1)].shape[0],parameters["gamma" + str(l+1)].shape[1]))
        s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
        s["dbeta" + str(l+1)] = np.zeros((parameters["beta" + str(l+1)].shape[0],parameters["beta" + str(l+1)].shape[1]))
        s["dgamma" + str(l+1)] = np.zeros((parameters["gamma" + str(l+1)].shape[0],parameters["gamma" + str(l+1)].shape[1]))
    ### END CODE HERE ###
    
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 5                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)]+(1-beta1)*grads['dW' + str(l+1)]
        v["dbeta" + str(l+1)] = beta1*v["dbeta" + str(l+1)]+(1-beta1)*grads['dbeta' + str(l+1)]
        v["dgamma" + str(l+1)] = beta1*v["dgamma" + str(l+1)]+(1-beta1)*grads['dgamma' + str(l+1)]
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-pow(beta1, t))
        v_corrected["dbeta" + str(l+1)] = v["dbeta" + str(l+1)]/(1-pow(beta1, t))
        v_corrected["dgamma" + str(l+1)] = v["dgamma" + str(l+1)]/(1-pow(beta1, t))
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)]+(1-beta2)*pow(grads['dW' + str(l+1)], 2)
        s["dbeta" + str(l+1)] = beta2*s["dbeta" + str(l+1)]+(1-beta2)*pow(grads['dbeta' + str(l+1)], 2)
        s["dgamma" + str(l+1)] = beta2*s["dgamma" + str(l+1)]+(1-beta2)*pow(grads['dgamma' + str(l+1)], 2)
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-pow(beta2, t))
        s_corrected["dbeta" + str(l+1)] = s["dbeta" + str(l+1)]/(1-pow(beta2, t))
        s_corrected["dgamma" + str(l+1)] = s["dgamma" + str(l+1)]/(1-pow(beta2, t))
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters['W' + str(l+1)]-learning_rate*(v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+epsilon))
        parameters["beta" + str(l+1)] = parameters['beta' + str(l+1)]-learning_rate*(v_corrected["dbeta" + str(l+1)]/(np.sqrt(s_corrected["dbeta" + str(l+1)])+epsilon))
        parameters["gamma" + str(l+1)] = parameters['gamma' + str(l+1)]-learning_rate*(v_corrected["dgamma" + str(l+1)]/(np.sqrt(s_corrected["dgamma" + str(l+1)])+epsilon))

        ### END CODE HERE ###

    return parameters, v, s


def model(X, Y, layers_dims, learning_rate = 0.7, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, 
          keep_prob = 0.5, print_cost = True, pourcentageStop = 0.1, cost_mode = 'SEL', log = True):
    """
    L-layer neural network model.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
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
    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update   
    m = X.shape[1]                   # number of training examples
    
    # Initialize parameters
    parameters = initialize_parameters_he(layers_dims)

    v, s = initialize_adam(parameters)
    x = []
    fpa_cache = []
    deg = 4
    # Optimization loop
    for i in range(num_epochs):
        
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            parms, L = forward_propagation_with_dropout(minibatch_X, parameters, layers_dims, keep_prob)

            # Compute cost and add to the cost total
            cost, dAL = compute_cost(np.asarray(parms["A"+str(L)]), minibatch_Y, cost_mode)
            
            cost_total += np.sum(abs(cost))

            # Backward propagation
            grads = backward_propagation_with_dropout(minibatch_X, minibatch_Y, parms, parameters, L, dAL, layers_dims, keep_prob)

            # Update parameters
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                              t, learning_rate, beta1, beta2,  epsilon)
            
        if log:
            cost_avg = np.log10(1+(cost_total / m))
        else:
            cost_avg = cost_total / m
        
        
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

    return parameters


def normnreduc(M, nbparam):
    k = int(M.shape[0]/nbparam)
    m = []
    s = []
    
    for i in range(nbparam):
        for z in range(k):
            m.append(M[i*k:(i+1)*k, :].mean())
            s.append(M[i*k:(i+1)*k, :].std())
        
    return np.asarray(m)[:, np.newaxis], np.asarray(s)[:, np.newaxis]

def leastSquare(x, Y, deg):
    '''Least Squares'''
    if type(x) != 'numpy.ndarray':
        x = np.asarray(x)[:, np.newaxis]
    if type(Y) != 'numpy.ndarray':
        Y = np.asarray(Y)[:, np.newaxis]
    if x.shape[1]>x.shape[0]:
        x = x.T
    if Y.shape[1]>Y.shape[0]:
        Y = Y.T
        
    X = []
    for i in range(deg+1):
        X.append(x**i)
    
    X = np.squeeze(np.asarray(X)).T
    
    A = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    
    return A, np.dot(X, A), X

def tangente(A, abscisse, X, deg, step = 0.01):
    '''Tangente in abscisse'''
    xpts = []
    for i in range(deg+1):
        xpts.append(np.asarray([[abscisse],[abscisse+step]])**i)
    xpts = np.squeeze(np.asarray(xpts)).T
    
    val = np.dot(xpts, A)
    fpa = float((val[1]-val[0])/step)
    fa = float(np.dot(np.asarray(xpts[0, :]), A))
    y = fpa*(X[:, 1][:, np.newaxis]-abscisse)+fa
    
    return fpa, y

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0),), mode='constant', constant_values = (0,0))
    ### END CODE HERE ###
    
    return X_pad
    
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- W0dow - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev*W
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z+float(b)
    ### END CODE HERE ###

    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    n_H = int(((n_H_prev-f+2*pad)/stride)+1)
    n_W = int(((n_W_prev-f+2*pad)/stride)+1)
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros([m, n_H, n_W, n_C])
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :, :, :]               # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h*stride
            vert_end = h*stride+f
            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                horiz_start = w*stride
                horiz_end = w*stride+f
                
                for c in range(n_C):   # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                                        
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache