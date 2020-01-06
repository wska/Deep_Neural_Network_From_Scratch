from model import Model
from layers import Linear, Relu, Softmax, BatchNormalization
from optimizers import SGD
from utility import *
import numpy as np
from datetime import datetime


def regularizationSearch():

    trainingData, trainingLabels, \
    validationData, validationLabels, \
    testingData, testingLabels = loadAllData("Datasets/cifar-10-batches-mat/", valsplit=0.10)    

    bestLambda = 0.0
    bestValAcc = 0.0
    bestLoss = 0.0
    
    for lambdaValue in np.arange(0, 0.2, 0.005):

        network = Model()
        network.addLayer(Linear(32*32*3, 50, regularization=lambdaValue, initializer="he"))
        network.addLayer(BatchNormalization(50, trainable=True))
        network.addLayer(Relu())

        network.addLayer(Linear(50, 30, regularization=lambdaValue, initializer="he"))
        network.addLayer(BatchNormalization(30, trainable=True))
        network.addLayer(Relu())

        network.addLayer(Linear(30,10, regularization=lambdaValue, initializer="he"))
        network.addLayer(Softmax())

        sgd = SGD(lr=0.01, lr_decay=0.95, momentum=0.7, shuffle=True, lr_min=1e-5)  
    
        network.compile(sgd, "cce")
        
        timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')

        network.fit(trainingData, trainingLabels, epochs=20, validationData=(validationData, validationLabels), batch_size=64)

        
        #plotAccuracy(network, "plots/", timestamp)
        #plotLoss(network, "plots/", timestamp)
        
        print("Lambda:{}".format(lambdaValue))
        loss, acc = network.evaluate(validationData, validationLabels)
        print("Val loss: {} , Val acc: {}".format(loss, acc) )
        print("\n\n")
        
        if acc > bestValAcc:
            bestLambda = lambdaValue
            bestValAcc = acc
            bestLoss = loss
    
    return bestLambda, bestValAcc, bestLoss


def paramSearch():

    trainingData, trainingLabels, \
    validationData, validationLabels, \
    testingData, testingLabels = loadAllData("Datasets/cifar-10-batches-mat/", valsplit=0.10)    

    bestLambda = 0.0
    bestLR = 0.0
    bestValAcc = 0.0
    bestLoss = 0.0

    data = [[],[],[]]
    
    for lambdaValue in np.arange(0, 0.1, 0.01):
        for lr in np.arange(0.01, 0.1, 0.01):

            print("Lambda:{}".format(lambdaValue))
            print("LR:{}".format(lr))

            network = Model()
            network.addLayer(Linear(32*32*3, 50, regularization=lambdaValue, initializer="he"))
            network.addLayer(BatchNormalization(50, trainable=True))
            network.addLayer(Relu())

            network.addLayer(Linear(50, 30, regularization=lambdaValue, initializer="he"))
            network.addLayer(BatchNormalization(30, trainable=True))
            network.addLayer(Relu())

            network.addLayer(Linear(30,10, regularization=lambdaValue, initializer="he"))
            network.addLayer(Softmax())

            sgd = SGD(lr=lr, lr_decay=0.95, momentum=0.7, shuffle=True, lr_min=1e-5)  
        
            network.compile(sgd, "cce")
            
            timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')

            network.fit(trainingData, trainingLabels, epochs=1, validationData=(validationData, validationLabels), batch_size=100, verbose=False)

            
            #plotAccuracy(network, "plots/", timestamp)
            #plotLoss(network, "plots/", timestamp)
            
            loss, acc = network.evaluate(validationData, validationLabels)
            print("Val loss: {} , Val acc: {}".format(loss, acc) )
            print("\n\n")
            
            data[0].append(lr)
            data[1].append(lambdaValue)
            data[2].append(acc)
            
            if acc > bestValAcc:
                bestLambda = lambdaValue
                bestLR = lr
                bestValAcc = acc
                bestLoss = loss
    
    return bestLambda, bestLR,  bestValAcc, bestLoss, data

def main():
    #regularizationSearch()
    bestLambda, bestLR,  bestValAcc, bestLoss , data = paramSearch()
    print("Best Parameters = Lambda:{}, LR:{}, Acc:{}, Loss:{}".format(bestLambda, bestLR, bestValAcc, bestLoss))
    plotGrid(data)

if __name__ == "__main__":
    main()