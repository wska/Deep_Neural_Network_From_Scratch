from model import Model
from layers import Linear, Relu, Softmax
from optimizers import SGD
from utility import loadData
import numpy as np


def main():

    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")
    testingData, testingLabels, encodedTestingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_2.mat")
    validationData, validationlabels, encodedValidationlabels = loadData("Datasets/cifar-10-batches-mat/data_batch_3.mat")


    network = Model()
    network.addLayer(Linear(32*32*3, 10))
    #network.addLayer(Relu())
    #network.addLayer(Linear(30, 20))
    #network.addLayer(Linear(20,10))
    network.addLayer(Softmax())
   
    sgd = SGD(lr=0.01, lr_decay=0.99)
 
    network.compile(sgd, "cce")
    
    network.fit(trainingData, encodedTrainingLabels, epochs=10, validationData=(validationData, encodedValidationlabels))

    print(network.history["epochs"])
    print(network.history["accuracy"])
    print(network.history["cost"])
    print(network.history["val_accuracy"])
    print(network.history["val_cost"])
    
    

if __name__ == "__main__":
    main()