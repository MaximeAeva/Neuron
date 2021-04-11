import numpy as np
import cupy as cp
import math
import Display
import matplotlib.pyplot as plt
from NeuralActivation import activation
from NeuralFunctions import *
from timeit import default_timer as timer
import dask.array as da
import psutil as cpuInfo
import GPUtil as gpuInfo
import DataCreator
import Utils

def unitTest_forward_conv():
    A_prev = cp.random.rand(1, 5, 5, 1)
    filt = cp.random.rand(3, 3, 1, 1)
    bias = cp.zeros((1, 1, 1, 1))
    functions = ('relu', 'sigmoid', 'tanh', 'bentid', 'softmax', 'identity')
    for ff in functions:
        print(" ")
        print("##### Checking for "+ff+" #####")
        Z = forward_conv(A_prev, filt, bias, 0, 1, ff)
        if(Z.shape != (1, 3, 3, 1)):
            print("current shape : "+str(Z.shape))
            print("expected: "+str((1, 3, 3, 1)))
            return Z
        
        Z = forward_conv(A_prev, filt, bias, 1, 1, ff)
        if(Z.shape != (1, 5, 5, 1)):
            print("current shape : "+str(Z.shape))
            print("expected: "+str((1, 5, 5, 1)))
            return Z
        
        Z = forward_conv(A_prev, filt, bias, 0, 2, ff, True)
        if(Z.shape == (1, 2, 2, 1)):
            print("shape : OK")
        else:
            print("current shape : "+str(Z.shape))
            print("expected: "+str((1, 2, 2, 1)))
            return Z
        
        K = cp.asnumpy(Z)
        U = K[0, :, :, 0]
    return U

print(unitTest_forward_conv())