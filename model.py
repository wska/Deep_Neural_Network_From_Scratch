import numpy as np
from layers import *
from optimizers import SGD

class Model():
    def __init__(self, name="Model"):
        self.layers = []
        self.name = name
        self.loss = None
        self.optimizer = None

    # Method for adding a layer
    def addLayer(self, layer):
        self.layers.append(layer)
    
    # Performs the forward pass wand train the network
    # Returns the loss value & metrics values
    def evaluate(self, inputs, targets):
        predictions = self.predict(inputs)
        cost = self.computeCost(predictions, targets)
        accuracy = self.computeAccuracy(predictions, targets)

        return cost, accuracy        

    
    # Performs a forward pass without training the network
    def predict(self, inputs):
        prediction = inputs
        for layer in self.layers:
            prediction = layer.forward(prediction)
        
        return prediction


    def backpropagate(self, targets):
        #assert self.layers[-1].type == "Softmax"
        grad = self.layers[-1].backward(targets)  
        for layer in self.layers[-2::-1]:
            grad = layer.backward(grad)

        return grad



    # Computes the cost
    def computeCost(self, predictions, targets):
        assert targets.shape == predictions.shape, "Predictions shape differs from target shape"

        totaltCost = 0

        ## Maybe dont need to use the probabilities. We have the predictions...
        if self.loss == "categorical_cross_entropy":
            assert self.layers[-1].type == "Softmax", "Loss is cross-entropy but last layer is not softmax"
            yhat = targets*np.log(self.layers[-1].probabilities)
            entropy = -np.sum(yhat)/targets.shape[1]
            totaltCost = totaltCost + entropy

            for layer in self.layers[0:-1]:
                totaltCost = totaltCost + layer.cost()

        # @TODO
        # elif self.loss == "binary_cross_entropy":

        elif self.loss == "mse":
            totaltCost = totaltCost + np.mean((predictions-targets)**2)

            for layer in self.layers[0:-1]:
                totaltCost = totaltCost + layer.cost()

        # @TODO
        # elif self.loss == "mape":

        elif self.loss == "None":
            for layer in self.layers:
                totaltCost = totaltCost + layer.cost()
        
        return totaltCost



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
            self.optimizer = optimizer
        
        # Adds reference for the optimizer to the model
        self.optimizer.model = self

        if loss == "cce" or loss == "categorical_cross_entropy":
            self.loss = "categorical_cross_entropy"
        else:
            raise NameError("Unrecognized loss function.")

        self.history = self.optimizer.history


    # Fits the model to the data using the optimizer and loss function specified during compile
    def fit(self, inputs, targets, epochs, batch_size=None):
        if self.loss is None or self.optimizer is None:
            raise ValueError("Model not compiled")
        
        
        pass




    def __str__(self):
        strrep = "Sequential Model: " + self.name +"\n"
        for i in range(len(self.layers)):
            strrep = strrep + "Layer " + str(i) + ": Type:"  + " " + str(self.layers[i])
        return strrep
