import numpy as np
#from utilityFunctions import *

# Linear layer class.
class Linear():
    def __init__(self, inputDim, outputDim, name=None, regularization=0, initializer=None, std=0.01, mean=0):
        self.type = "Linear"
        self.name = self.type if name is None else name
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.regularization = regularization

        if initializer == None or initializer == "normal":
            # Initializes W and b with a gaussian normal of mean and std. 
            self.W = np.random.normal(mean, std, (outputDim, inputDim))
            self.b = np.random.normal(mean, std, (outputDim, 1))
        self.gradW = np.zeros(self.W.shape, dtype=float)
        self.gradb = np.zeros(self.b.shape, dtype=float)

        self.processedInput = None
        
    # Propagates the inputs through the layer
    def forward(self, inputs):
        assert inputs.T.shape == (self.inputDim, self.outputDim)
        self.processedInput = inputs
        return np.dot(self.W, inputs) + self.b
    

    def backward(self, grads):
        self.gradW = np.zeros(self.W.shape, dtype=float)
        self.gradb = np.zeros(self.b.shape, dtype=float)
        sizeOfMinibatch = self.processedInput.shape[1]

        # Computes the gradient for all inputs in the minibatch
        for i in range(sizeOfMinibatch):
            x = self.processedInput[:,i]
            g = grads[i, :]
            self.gradW += np.outer(g,x)
        
        # Divides by the size of the minibatch and adds the gradient for the regularization term (2*lambda*W)
        self.gradW = (self.gradW / sizeOfMinibatch) + 2*self.regularization*self.W
        # Set the bias gradient to be the mean of the gradients
        self.gradb = grads.T.mean(axis=1, keepdims=True)

        return np.dot(grads, self.W)

    def computeCost(self):
        return self.regularization * (self.W**2).sum()
    
    def __str__(self):
        return "{} Layer, {}->{}".format(self.type, self.inputDim, self.outputDim)

# Relu layer class.
class Relu():
    def __init__(self):
        self.type = "Relu"
        self.activatedOutputs = None
        self.processedInput = None
        
    # Propagates the inputs through the layer
    def forward(self, inputs):
        self.activatedOutputs = (inputs>0)
        inputs[inputs<0] = 0
        return inputs
    
    # Propagates the input backwards
    def backward(self, grads):
        return grads*self.activatedOutputs.T
    
    def __str__(self):
        return "{} Layer".format(self.type)

# Softmax layer class. Uses categorical crossentropy for loss.
class Softmax():
    def __init__(self):
        self.type = "Softmax"
        self.probabilities = None
    
    def forward(self, inputs):
        self.probabilities = np.exp(inputs) / np.sum(np.exp(inputs), axis = 0)
        return self.probabilities
    
    def backward(self, targets):
        assert targets.shape == self.probabilities.shape
        grads = -(targets-self.probabilities).T
        return grads
    
    # Calculates the categorical cross entropy between the softmax predictions and the true targets.
    def computeCost(self, targets):
        assert targets.shape == self.probabilities.shape
        yhat = targets*np.log(self.probabilities)
        entropy = -np.sum(yhat)/targets.shape[1]
        return entropy

    def __str__(self):
        return "{} Layer".format(self.type)


# @TODO
class BatchNormalization():
    def __init__(self, inputDim, outputDim, updateParameters=False, name=None, startingMean=None, startingVariance=None, alpha=0.99):
        self.type = "BatchNormalization"
        self.name = self.type if name is None else name
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.alpha = alpha

        self.mean = startingMean
        self.variance = startingVariance

        self.batchMean = np.zeros((inputDim, 1))
        self.batchVariance = np.zeros((inputDim, 1))

        
    # Propagates the inputs through the layer
    def forward(self, inputs):
        return

    def backward(self, grads):
       return

    def costFunction(self):
        return
    
    def __str__(self):
        return "{} Layer, {}->{}".format(self.type, self.inputDim, self.outputDim)