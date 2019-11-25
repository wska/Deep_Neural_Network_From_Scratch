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
    
    return X,Y,OneHotY

# Softmax function
# Not tested. Might need axis = 1
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

'''
# Softmax gradient
def softmax_gradient(x): 
    n, _ = x.shape
    softmax_grad = np.array([])
    jacobian_m = np.diag(x)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = x[i] * (1-x[i])
            else: 
                jacobian_m[i][j] = -x[i]*x[j]
    return jacobian_m
'''

# Sigmoid activation function centered around origin
def sigmoid(x):
    return (2/(1+np.exp(-x)))-1

# Gradient of sigmoid centered around origin
def sigmoid_gradient(x):
    return ((1+sigmoid(x))*(1-sigmoid(x)))/2

# ReLu activation function
def relu(x):
    return np.maximum(0,x)

# Gradient of the ReLu activation function
def relu_gradient(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

# Computes the cost given output(WX-T) using mean square error
def meanSquareError(predictions, targets):
    return np.mean((predictions-targets)**2)

# Compute the cost using binary cross-entropy
def binaryCrossEntropy(predictions, targets):
    m = predictions.shape[1]
    cost = -1 / m * (np.dot(targets, np.log(predictions).T) + np.dot(1 - targets, np.log(1 - predictions).T))
    return np.squeeze(cost)


# Categorical Cross Entropy
def categoricalCrossEntropy(predictions, targets):
    N = predictions.shape[0]
    entropy = -np.sum(targets*np.log(predictions))/N
    return entropy


# Threshold probabilites at 0.5
def threshold(output):
         probs = np.copy(output)
         probs[probs > 0.5] = 1
         probs[probs <= 0.5] = 0
         return probs


# The cost function used in backprop
def computeCost(predictions, targets, layer, costFunction, regularizationLambda):
    
    if costFunction == "centropy":
        loss = categoricalCrossEntropy(predictions, targets)
    elif costFunction == "meanSquareError":
        loss = meanSquareError(predictions, targets)
    
    regularizationTerm = regularizationLambda*(layer["W"].sum()**2)
    loss = loss + regularizationTerm

    # Divide by the size of the batch?(columns of X in old code)

    return loss


# Computes the accuracy(classification rate) of the forward pass versus the true labels of the data
def computeAccuracy(Outputs, targets):
    
        Outputs = np.argmax(Outputs, axis=1).reshape((len(Outputs), 1))
        return ((Outputs == targets).sum() / Outputs.shape[0]) * 100



