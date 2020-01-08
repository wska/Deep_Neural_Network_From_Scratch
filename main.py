# William Skagerström
# Last updated, 2019-01-07

from model import Model
from layers import Linear, Relu, Softmax, BatchNormalization
from optimizers import SGD
from utility import *
import numpy as np
from datetime import datetime


def main():

    trainingData, trainingLabels, \
    validationData, validationLabels, \
    testingData, testingLabels = loadAllData("Datasets/cifar-10-batches-mat/", valsplit=.10)

    #Settings 1
    #reg = 0.065
    #lr = 0.002
    
    #Settings 2
    #reg = 0.0021162
    #lr = 0.061474

    #Settings 3
    #reg = 0.0010781
    #lr = 0.069686

    #Settings 4
    #reg = 0.0049132
    #lr = 0.07112


    #Settings 5
    reg = 0.005
    lr = 0.007
    network = Model()

    network.addLayer(Linear(32*32*3, 50, regularization=reg, initializer="he"))
    network.addLayer(BatchNormalization(50, trainable=True))
    network.addLayer(Relu())

    network.addLayer(Linear(50, 30, regularization=reg, initializer="he"))
    network.addLayer(BatchNormalization(30, trainable=True))
    network.addLayer(Relu())

    network.addLayer(Linear(30,10, regularization=reg, initializer="he"))
    network.addLayer(Softmax())

    sgd = SGD(lr=lr, lr_decay=0.95, momentum=0.7, shuffle=True, lr_min=1e-5)  
 
    network.compile(sgd, "cce")
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')

    network.fit(trainingData, trainingLabels, epochs=30, batch_size=100, validationData=(validationData, validationLabels))

    
    plotAccuracy(network, "plots/", timestamp, title="3-layer network accuracy over epochs, eta:{}, lambda:{}".format(lr, reg))
    plotLoss(network, "plots/", timestamp, title="3-layer network loss over epochs, eta:{}, lambda:{}".format(lr, reg))
    
    loss, acc = network.evaluate(testingData, testingLabels)
    print("Test loss: {} , Test acc: {}".format(loss, acc) )
    

if __name__ == "__main__":
    main()