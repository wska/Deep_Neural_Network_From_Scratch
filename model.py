import numpy as np
from layers import *

class Model():
    def __init__(self, name="Model", loss="categorical_cross_entropy"):
        self.layers = []
        self.name = name
        self.loss = loss

    # Method for adding a layer
    def addLayer(self, layer):
        self.layers.append(layer)
    
    # Performs the forward pass wand train the network
    # Returns the loss value & metrics values
    def evaluate(self, inputVector, targets,  train=True):
        predictions = self.predict(inputVector)
        cost = self.computeCost(predictions, targets)
        accuracy = self.computeAccuracy(predictions, targets)

        return cost, accuracy        

    
    # Performs a forward pass without training the network
    def predict(self, inputVector):
        prediction = inputVector
        for layer in self.layers:
            prediction = layer.forward(prediction)
        
        return prediction

    def backpropagate(self, targets):

        #assert self.layers[-1].type == "Softmax"
        grad = self.layers[-1].backward(targets)
        
        for layer in self.layers[1::-1]:
            grad = layer.backward(grad)

        return grad



    # Computes the cost
    def computeCost(self, predictions, targets):
        assert targets.shape == predictions.shape, "Predictions shape differs from target shape"

        totaltCost = 0

        if self.loss == "categorical_cross_entropy":
            assert self.layers[-1].type == "Softmax", "Loss is cross-entropy but last layer is not softmax"
            yhat = targets*np.log(self.layers[-1].probabilities)
            entropy = -np.sum(yhat)/targets.shape[1]
            totaltCost = totaltCost + entropy

            for layer in self.layers[0:-1]:
                totaltCost = totaltCost + layer.cost()

        # @TODO
        # elif self.loss == "binary_cross_entropy":
        # elif self.loss == "mse":
        # elif self.loss == "mape":
        
        elif self.loss == "None":
            for layer in self.layers:
                totaltCost = totaltCost + layer.cost()
        
        return totaltCost

    def computeAccuracy(self, predictions, targets):
        assert predictions.shape == targets.shape
        accuracy = np.sum(np.argmax(predictions, axis=0) == np.argmax(targets, axis=0)) / predictions.shape[1]
        return accuracy

    def __str__(self):
        strrep = ""
        for i in range(len(self.layers)):
            strrep = strrep + "Layer " + i + ": Type" + ""
        return strrep
