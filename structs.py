from abc import ABC, abstractmethod
import numpy as np
from utilityFunctions import *

class Linear():
    def __init__(self, input_dim, output_dim, name=None, activation=None, regularization=0, initializer=None):
        self.type = "Linear"
        self.name = self.type if name is None else name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.regularization = regularization
        if initializer == None:
            self.W = np.zeros((1,1)) # Add initializer here
            self.b = np.zeros((1,1))
        self.grad_W = np.zeros(self.W.shape, dtype=float)
        self.grad_b = np.zeros(self.b.shape, dtype=float)
        self.input = np.zeros((self.input_dim, 1))
        
        
        if activation.lower() == "relu":
            self.activation = relu
        elif activation.lower() == "sigmoid":
            self.activation == sigmoid
        elif activation.lower() == "softmax":
            self.activation == softmax
    
    def forwardPass(self, input):
        return
    
    def backwardPass(self, grads):
        pass
    
    def costFunction(self):
        pass
    
    def __str__(self):
        return "Layer {}, {}->{}".format(self.type, self.input_dim, self.output_dim)


a = Linear(3072, 50, activation="relu")
print(a.activation(-2))