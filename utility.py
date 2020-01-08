# William Skagerström
# Last updated, 2019-01-07

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from scipy.io import loadmat
import os
from layers import BatchNormalization
from decimal import Decimal
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# Loads one batch from the CIFAR-10 dataset
def loadData(fileName, images = None, dimensions=None, normalize=False):
    data = loadmat(fileName)
    X = np.array(data["data"])

    if normalize:
        X = normalize(X)
   

    Y = np.array(data["labels"]) # Labels

    OneHotY = np.eye(10)[Y.reshape(-1)] # Converts to one-hot encoding
    
    if images is None:
        images = 10000 # Size of the batch
    if dimensions is None:
        dimensions = 3072 # 32x32x3 CIFAR Image
    
    return X.T[0:dimensions, 0:images], Y.T[0:dimensions, 0:images], OneHotY.T[0:dimensions, 0:images]

def loadAllData(path, valsplit=0.0):
    batchNames = ["data_batch_1.mat", "data_batch_2.mat", "data_batch_3.mat", "data_batch_4.mat", "data_batch_5.mat"]
    trainData, _, trainLabels = loadData(path+batchNames[0])
    
    for name in batchNames[1:]:
        batchX, _, batchY = loadData(path+name)   
        trainData = np.concatenate((trainData, batchX), axis=1) 
        trainLabels = np.concatenate((trainLabels, batchY), axis=1) 
    
    testData, _, testLabels = loadData(path+"test_batch.mat")

    trainData, testData = normalize(trainData, testData)

    valData = trainData[:, int(trainData.shape[1]*(1-valsplit)):]
    valLabels = trainLabels[:, int(trainLabels.shape[1]*(1-valsplit)):]

    trainData = trainData[:, 0:int(trainData.shape[1]*(1-valsplit))]
    trainLabels = trainLabels[:, 0:int(trainLabels.shape[1]*(1-valsplit))]
    
    return trainData, trainLabels, valData, valLabels, testData, testLabels

    

def normalize(X, testData=None):
    X = X.astype(float) # Converts the data to float64
    trainMean = np.mean(X, axis=1).reshape(X.shape[0], 1)
    trainSTD = np.std(X, axis=1).reshape(X.shape[0], 1) 

    X -=  trainMean # Subtracts the mean
    X /= trainSTD # Divides by the STD

    if testData is not None:
        testData = testData.astype(float)
        testData -= trainMean
        testData /= trainSTD

    return X, testData

def plotAccuracy(history, path_plots, timestamp, title='Model accuracy over epochs', fileName = None):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    try:
        plt.plot(history.history['val_accuracy'])
        plt.legend(['Train', 'Validation'], loc='upper left')
    except KeyError:
        plt.legend(['Train'], loc='upper left')

    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if fileName is None:
        plt.savefig(os.path.join(path_plots, timestamp + '_acc.png'))
    else:
        plt.savefig(os.path.join(path_plots, fileName + '.png'))
    # plt.show(), fileName = None):
    plt.cla()
    

def plotLoss(history, path_plots,timestamp, title='Loss function over epochs', fileName = None):
    # summarize history for loss
    plt.plot(history.history['cost'])
    try:
        plt.plot(history.history['val_cost'])
        plt.legend(['Train', 'Validation'], loc='upper left')
    except KeyError:
        plt.legend(['Train'], loc='upper left')

    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if fileName is None:
        plt.savefig(os.path.join(path_plots, timestamp + '_cost.png'))
    else:
        plt.savefig(os.path.join(path_plots, fileName + '.png'))
    # plt.show()
    plt.cla()



def multiPlotAccuracy(historys, path_plots,timestamp, title='Accuracy over epochs'):
    legend = []
    for history in historys:
        plt.plot(history.history['accuracy'])
        try:
            plt.plot(history.history['val_accuracy'])
            legend.append(history.history["name"] + '(train)')
            legend.append(history.history["name"] + '(validation)')

        except KeyError:
            legend.append(history.history["name"] + '(train)')
            
    plt.legend(legend, loc='upper left')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(path_plots, timestamp + '_accuracy.png'))
    # plt.show()
    plt.cla()



def multiPlotLoss(historys, path_plots,timestamp, title='Loss function over epochs'):
    legend = []
    for history in historys:
        plt.plot(history.history['cost'])
        try:
            plt.plot(history.history['val_cost'])
            legend.append(history.history["name"] + '(train)')
            legend.append(history.history["name"] + '(validation)')

        except KeyError:
            legend.append(history.history["name"] + '(train)')

    plt.legend(legend, loc='upper left')
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(path_plots, timestamp + '_cost.png'))
    # plt.show()
    plt.cla()



def plotGrid(data, path_plots=None,timestamp=None):
    # summarize history for loss
    x,y,z = data[0], data[1], data[2]
    axisDimensions = data[3]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    x = x.reshape(axisDimensions)
    y = y.reshape(axisDimensions)
    z = z.reshape(axisDimensions)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contourf(x, y, z, 50, cmap='jet' )
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Lambda')
    ax.set_zlabel('Accuracy')

    plt.show()

def plotSavedGrid(path_plots="" ,timestamp=None):

    x = np.load(path_plots+"x.npy")
    y = np.load(path_plots+"y.npy")
    z = np.load(path_plots+"z.npy")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(x, y, z, 50, cmap='jet' )
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Lambda')
    ax.set_zlabel('Accuracy')

    plt.show()


#data = [np.arange(0,1,0.05), np.arange(0,1,0.05), np.arange(0,1,0.05)]
#plotGrid(data)

# Analytical computations of the gradients for verifying correctness (centered difference theorem)
# @WARNING Takes quite a while for large matrices
def compute_grads(h, W, inputs, targets, network):
    grad = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j]+=h
            predictions = network.predict(inputs)
            cost = network.computeCost(predictions, targets)
            W[i,j] -=2*h
            predictions = network.predict(inputs)
            negativecost = network.computeCost(predictions, targets)
            grad[i,j] = (cost-negativecost)/(2*h)
            W[i,j]+=h
    #print(grad)
    return grad


# Analytical computations of the gradients for verifying correctness (centered difference theorem)
# @WARNING Takes quite a while for large matrices
def compute_grads_w_BN(h, W, inputs, targets, network):
    grad = np.zeros(W.shape)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            
            W[i,j]+=h
            predictions = network.predict(inputs, True)
            cost = network.computeCost(predictions, targets)


            W[i,j] -=2*h
            predictions = network.predict(inputs, True)
            negativecost = network.computeCost(predictions, targets)
            grad[i,j] = (cost-negativecost)/(2*h)
            W[i,j]+=h
    return grad


# Calculates the relative error between the gradients from backpropagation and the analytical gradients
def grad_difference(grad, numerical_grad):
    diff = np.abs(grad - numerical_grad)
    eps = np.finfo('float').eps # Minimum value that is distinguishable from 0
    #eps = np.finfo('float').tiny # Minimum value that is distinguishable from 0
    relative_error = diff/np.maximum(eps, (np.abs(grad) + np.abs(numerical_grad)))
    print("Mean relative error  |  Max relative error: ")
    print("& %.5E & %.5E \n" % (Decimal(np.mean(relative_error)), Decimal(np.max(relative_error))) )
    return relative_error

