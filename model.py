# William Skagerström
# Last updated, 2019-01-07

import numpy as np
from layers import *
from optimizers import SGD
import copy

class Model():
    def __init__(self, name="Model"):
        self.layers = []
        self.name = name
        self.loss = None
        self.optimizer = None

    # Method for adding a layer
    def addLayer(self, layer):
        self.layers.append(layer)
    
    # Performs the forward pass and evaluates the network
    # Returns the loss value & metrics values
    def evaluate(self, inputs, targets, updateInternal=False):
        predictions = self.predict(inputs, updateInternal)
        cost = self.computeCost(predictions, targets)
        accuracy = self.computeAccuracy(predictions, targets)

        return cost, accuracy        

    
    # Performs a forward pass without training the network
    def predict(self, inputs, updateInternal=False):
        prediction = inputs

        for layer in self.layers:
            if type(layer) != BatchNormalization:
                prediction = layer.forward(prediction)
            else:
                prediction = layer.forward(prediction, updateInternal)
        
        return prediction

    # Propagates the targets(one hot encoding) back through the network
    def backpropagate(self, targets):
        grad = self.layers[-1].backward(targets)  
        for layer in self.layers[-2::-1]:
            grad = layer.backward(grad)

        return grad



    # Computes the cost
    def computeCost(self, predictions, targets):
        
        totaltCost = 0   

        ## Maybe dont need to use the probabilities. We have the predictions...
        if self.loss == "categorical_cross_entropy":
            assert self.layers[-1].type == "Softmax", "Loss is cross-entropy but last layer is not softmax"
            yhat = targets*np.log(self.layers[-1].probabilities)
            entropy = -np.sum(yhat)/targets.shape[1]
            totaltCost = totaltCost + entropy

            for layer in self.layers[0:-1]:
                totaltCost = totaltCost + layer.cost()

        # NOT TESTED YET
        elif self.loss == "binary_cross_entropy":
            m = predictions.shape[0]
            binaryEntropy = -1 / m * (np.dot(targets, np.log(predictions).T) + np.dot(1 - targets, np.log(1 - predictions).T))
            totaltCost = totaltCost + np.squeeze(binaryEntropy)

            for layer in self.layers[0:-1]:
                totaltCost = totaltCost + layer.cost()

        # NOT TESTED YET
        elif self.loss == "mse":
            totaltCost = totaltCost + np.mean((predictions-targets)**2)

            for layer in self.layers[0:-1]:
                totaltCost = totaltCost + layer.cost()

        elif self.loss == "None":
            for layer in self.layers:
                totaltCost = totaltCost + layer.cost()
        
        return totaltCost


    # Computes the accuracy of the predictions given the targets
    def computeAccuracy(self, predictions, targets):
        assert predictions.shape == targets.shape
        accuracy = np.sum(np.argmax(predictions, axis=0) == np.argmax(targets, axis=0)) / predictions.shape[1]
        return accuracy



    # Initializes the attributes for the optimizer and the loss function. 
    # Also adds a reference for the optimizer to the current model(for access to the forward and backward pass of the network)
    def compile(self, optimizer="SGD", loss="cce"):

        if type(optimizer) is str:
            if optimizer == "SGD":
                self.optimizer = SGD()
            else:
                raise NameError("Unrecognized optimizer")
        else:
            self.optimizer = copy.deepcopy(optimizer)
        
        # Adds reference for the optimizer to the model
        self.optimizer.model = self

        if loss == "cce" or loss == "categorical_cross_entropy":
            self.loss = "categorical_cross_entropy"
        else:
            raise NameError("Unrecognized loss function.")

        self.history = self.optimizer.history


    # Fits the model to the data using the optimizer and loss function specified during compile
    def fit(self, inputs, targets, epochs=1, validationData=None, batch_size=None, verbose=True):
        if self.loss is None or self.optimizer is None:
            raise ValueError("Model not compiled")
        
        
        self.optimizer.train(x_train=inputs, y_train=targets,\
              validationData=validationData,\
              epochs=epochs, batch_size=batch_size, verbose=verbose)


    def __str__(self):
        strrep = "Sequential Model: " + self.name +"\n"
        for i in range(len(self.layers)):
            strrep = strrep + "     Layer " + str(i) + ": Type:"  + " " + str(self.layers[i]) + "\n"
        return strrep
