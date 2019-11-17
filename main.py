import numpy as np
import math
import matplotlib.pyplot as plt
import collections
from scipy.io import loadmat
from utilityFunctions import *
from nn_structs import *


def main():
    # Loads the relevant data. 
    # [a-z]*Data has shape 10000x3072 (10000 samples, 3072 features)
    # [a-z]*Labels has shape 10000x10 (10000 samples, 1 label as a one-hot encoding)
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")
    testingData, testingLabels, encodedTestingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_2.mat")
    validationData, validationlabels, encodedValidationlabels = loadData("Datasets/cifar-10-batches-mat/data_batch_3.mat")


    #W,b = initializeLayer(features, 50, 0, 0.01)
    nn = initializeNetwork(nn_architecture_1, 0, 0.01, 99)
    print(nn["L1"]["W"].shape)
    print(nn["L1"]["b"].shape)
    #out = single_layer_forwardPass(trainingData, nn["L1"])
    x = np.array([[1, 2], [1,2]])
    x = softmax(x)
    print(x)
    print(softmax_gradient(x))
    
    #out, outputMemory = forwardPass(trainingData, nn)
    

    




# Initializes the weight matrix and bias for a single layer to that of a zero-mean, std drawn from the gaussian normal dist
def initializeLayer(features, layerSize, mean, std):
    W = np.random.normal(mean, std, (features, layerSize))
    b = np.random.normal(mean, std, (1, layerSize))

    return W,b


# Network that initializes a dictionary with weights and activation functions from a given network architecture
def initializeNetwork(NN_architecture, mean, std, seed=None):
        if seed != None:
                np.random.seed(seed)

        neural_network = collections.OrderedDict()

        for layerNumber, layer in enumerate(NN_architecture):
                layerName = "L" + str(layerNumber+1)
                layerInputSize = layer["input"]
                layerOutputSize = layer["output"]
                W,b = initializeLayer(layerInputSize, layerOutputSize, mean, std)
                neural_network[layerName] = {"W": W, "b": b, "activation": layer["activation"]}

        return neural_network

# Trains the network using mini-batch gradient descent
def trainNetwork():
    return


# Performs a complete forward-pass through the entire network
def forwardPass(data, nn):
        inputMemory = {}
        layerInput = data

        for _,layer in enumerate(nn):
                prevousInput = layerInput
                layerInput, unactivatedInput = single_layer_forwardPass(layerInput, nn[layer])

                inputMemory[layer] = (layerInput, unactivatedInput)

        return layerInput, inputMemory

# TODO BACKWARD PASS ALGORITHM
def backwardPass(predictions, targets, nn, memory):

        return

def updateWeights():
        return

# Executes the forward pass for a given input vector X and returns the predictions of the class Y
def single_layer_forwardPass(input, layer):
        wx = np.dot(input, layer["W"]) + layer["b"]
        if layer["activation"] == "relu":
                return relu(wx), wx
        elif layer["activation"] == "softmax":
                return softmax(wx), wx
        else:
                raise Exception("Activation function not recognized")
    

# Performs the backward pass
def single_layer_backwardPass(dA_curr, layer, Z_curr, A_prev):
        m = A_prev.shape[1]

        if layer["activation"] == "relu":
                dZ_curr = relu_backward(dA_curr, Z_curr)
        elif layer["activation"] == "softmax":
                #dZ_curr = softmax(dA_curr, Z_curr)
                return
        
        return      

# Performs batch normalization
def batchNormalization():
    return



if __name__ == "__main__":
    main()