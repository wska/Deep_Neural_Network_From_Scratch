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


def paramSearch(method="range"):

    trainingData, trainingLabels, \
    validationData, validationLabels, \
    testingData, testingLabels = loadAllData("Datasets/cifar-10-batches-mat/", valsplit=0.20)    



    bestLambda = 0.0
    bestLR = 0.0
    bestValAcc = 0.0
    bestLoss = 0.0
    bestModel = None

    data = [[],[],[]]
    
    if method == "range":
        lambdaValues = np.arange(0, 0.05, 0.001)
        lrValues = np.arange(0.04, 0.08, 0.005)

    elif method == "sampling":
        lrValues = np.random.uniform(0.06, 0.07, 15)
        lambdaValues = np.random.uniform(0.001, 0.005, 15)
        
    data.append((lrValues.shape[0], lambdaValues.shape[0])) # Append axis dimensions for 3D plotting



    for lambdaValue in lambdaValues:
        for lr in lrValues:


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

            network.fit(trainingData, trainingLabels, epochs=20, validationData=(validationData, validationLabels), batch_size=100, verbose=False)

            
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
                bestModel = network
    


    loss, acc = bestModel.evaluate(testingData, testingLabels)
    print("Test loss: {} , Test acc: {}".format(loss, acc) )
    print("\n\n")

    return bestLambda, bestLR,  bestValAcc, bestLoss, data



def main():
    #regularizationSearch()
    #bestLambda, bestLR,  bestValAcc, bestLoss , data = paramSearch(method="sampling")
    #print("Best Parameters = Lambda:{}, LR:{}, Acc:{}, Loss:{}".format(bestLambda, bestLR, bestValAcc, bestLoss))
    #plotGrid(data)
    plotSavedGrid("plots/fineSearchSpace/")
if __name__ == "__main__":
    main()