import numpy as np
from layers import *

class Model():
    def __init__(self, name="Model"):
        self.layers = []
        self.name = name

    # Method for adding a layer
    def addLayer(self, layer):
        self.layers.append(layer)
    
    # Performs the forward pass wand train the network
    def forwardPropagate(self, inputVector, train=True):

        prediction = inputVector
        for layer in self.layers:
            prediction = layer.forward(prediction)
        
        return prediction
    
    # Performs a forward pass without training the network
    def predict(self, inputVector):
        pass

    def backpropagate(self, targets):
        assert self.layers[-1].type == "Softmax"
        grad = self.layers[-1].backward(targets)
        
        for layer in self.layers[1::-1]:
            grad = layer.backward(grad)

    def computeCost(self, inputs, targets):
        predictions = self.forwardPropagate(inputs)
        assert targets.shape == predictions.shape

        totaltCost = 0
        for layer in self.layers:
            cost = cost + layer.cost()

    def computeAccuracy(self, predictions, targets):
        assert predictions.shape == targets.shape
        accuracy = np.sum(np.argmax(predictions, axis=0) == np.argmax(targets, axis=0)) / predictions.shape[1]
        return accuracy

    def __str__(self):
        strrep = ""
        for i in range(len(self.layers)):
            strrep = strrep + "Layer " + i + ": Type" + ""
        return strrep
