from model import Model
from layers import Linear, Relu, Softmax, BatchNormalization
from optimizers import SGD
from utility import *
import numpy as np
from datetime import datetime


def main():

    '''
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")
    #trainingData1, trainingLabels1, encodedTrainingLabels1 = loadData("Datasets/cifar-10-batches-mat/data_batch_2.mat")
    #trainingData2, trainingLabels2, encodedTrainingLabels2 = loadData("Datasets/cifar-10-batches-mat/data_batch_3.mat")
    #trainingData3, trainingLabels3, encodedTrainingLabels3 = loadData("Datasets/cifar-10-batches-mat/data_batch_4.mat")

    validationData, validationlabels, encodedValidationlabels = loadData("Datasets/cifar-10-batches-mat/data_batch_5.mat")
    testingData, testingLabels, encodedTestingLabels = loadData("Datasets/cifar-10-batches-mat/test_batch.mat")
    subset = 10000

    trainingData = trainingData[:, 0:subset]
    trainingLabels = trainingLabels[:, 0:subset]
    encodedTrainingLabels = encodedTrainingLabels[:, 0:subset]
    '''

    trainingData, trainingLabels, \
    validationData, validationLabels, \
    testingData, testingLabels = loadAllData("Datasets/cifar-10-batches-mat/", valsplit=0.10)

    print(trainingData.shape)

    


    
    network = Model()

    reg = 0.005
    network.addLayer(Linear(32*32*3, 50, regularization=reg, initializer="he"))
    #network.addLayer(BatchNormalization(50, trainable=True))
    network.addLayer(Relu())

    network.addLayer(Linear(50, 30, regularization=reg, initializer="he"))
    #network.addLayer(BatchNormalization(30, trainable=True))
    network.addLayer(Relu())

    network.addLayer(Linear(30,10, regularization=reg, initializer="he"))
    network.addLayer(Softmax())

    sgd = SGD(lr=0.01, lr_decay=0.95, momentum=0.7, shuffle=True, lr_min=1e-5)  
 
    network.compile(sgd, "cce")
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')

    network.fit(trainingData, trainingLabels, epochs=20, validationData=(validationData, validationLabels), batch_size=64)

    
    plotAccuracy(network, "plots/", timestamp)
    plotLoss(network, "plots/", timestamp)
    
    loss, acc = network.evaluate(testingData, testingLabels)
    print("Test loss: {} , Test acc: {}".format(loss, acc) )
    

if __name__ == "__main__":
    main()