from model import Model
from layers import Linear, Relu, Softmax
from optimizers import SGD
from utility import loadData, plotAccuracy, plotLoss
import numpy as np
from datetime import datetime


def main():

    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")
    testingData, testingLabels, encodedTestingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_2.mat")
    validationData, validationlabels, encodedValidationlabels = loadData("Datasets/cifar-10-batches-mat/data_batch_3.mat")


    network = Model()
    network.addLayer(Linear(32*32*3, 10, regularization=0.3))
    #network.addLayer(Relu())
    #network.addLayer(Linear(30, 20))
    #network.addLayer(Linear(20,10))
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.9, shuffle=True)
 
    network.compile(sgd, "cce")
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')
    network.fit(trainingData, encodedTrainingLabels, epochs=40, validationData=(validationData, encodedValidationlabels), batch_size=64)

    print(network.history["epochs"])
    print(network.history["accuracy"])
    print(network.history["cost"])
    print(network.history["val_accuracy"])
    print(network.history["val_cost"])
    
    plotAccuracy(network, "plots/", timestamp)
    plotLoss(network, "plots/", timestamp)

    

if __name__ == "__main__":
    main()