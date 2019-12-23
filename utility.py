import numpy as np
import math
import matplotlib.pyplot as plt
import collections
from scipy.io import loadmat

# Loads one batch from the CIFAR-10 dataset
def loadData(fileName):
    # TODO Normalize data?
    data = loadmat(fileName)

    
    X = np.array(data["data"])
    
    X = X.astype(float) # Converts the data to float64
    X -= np.mean(X, axis=1).reshape(X.shape[0], 1) # Subtracts the mean
    X /= np.std(X, axis=1).reshape(X.shape[0], 1) # Divides by the STD

    Y = np.array(data["labels"]) # Labels

    OneHotY = np.eye(10)[Y.reshape(-1)] # Converts to one-hot encoding
    
    return X.T, Y.T, OneHotY.T