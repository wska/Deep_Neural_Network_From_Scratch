import numpy as np


# Linear layer class.
class Linear():
    def __init__(self, inputDim, outputDim, name=None, regularization=0, initializer=None, std=0.01, mean=0, trainable=True):
        self.type = "Linear"
        self.name = self.type if name is None else name
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.regularization = regularization
        self.trainable = trainable

        if initializer == None or initializer == "normal":
            # Initializes W and b with a gaussian normal of mean and std. 
            self.W = np.random.normal(mean, std, (outputDim, inputDim))
            #self.b = np.random.normal(mean, std, (outputDim, 1))
            self.b = np.zeros((outputDim, 1))

        elif initializer.lower() == "he":
            var = 1 / (inputDim + outputDim)
            self.W = np.random.normal(0, np.sqrt(var), (outputDim, inputDim))
            #self.b = np.random.normal(mean, std, (outputDim, 1))
            self.b = np.zeros((outputDim, 1))

        elif initializer.lower() == "xavier":
            var = 2 / (inputDim + outputDim)
            self.W = np.random.normal(0, np.sqrt(var), (outputDim, inputDim))
            #self.b = np.random.normal(mean, std, (outputDim, 1))
            self.b = np.zeros((outputDim, 1))

        self.gradW = np.zeros(self.W.shape, dtype=float)
        self.previousGradW = np.zeros(self.W.shape, dtype=float)

        self.gradb = np.zeros(self.b.shape, dtype=float)
        self.previousGradb = np.zeros(self.b.shape, dtype=float)

        self.processedInput = None
        
    # Propagates the inputs through the layer
    def forward(self, inputs):

        # If single sample, expand dimensions from (X,) -> (X,1)
        if len(inputs.shape) == 1:
            inputs = np.resize(inputs, (inputs.shape[0], 1))

        self.processedInput = inputs
        return (np.dot(self.W, inputs) + self.b)
    

    def backward(self, grads):
        # Saves current gradients for use in momentum
        self.previousGradW = self.gradW
        self.previousGradb = self.gradb

        # Resets the current gradients and calculates the new ones for the current batch
        self.gradW = np.zeros(self.W.shape, dtype=float)
        self.gradb = np.zeros(self.b.shape, dtype=float)
        sizeOfMinibatch = self.processedInput.shape[1]
        
        self.gradW = np.dot(grads.T, self.processedInput.T)
        
        # Divides by the size of the minibatch and adds the gradient for the regularization term (2*lambda*W)
        self.gradW = (self.gradW / sizeOfMinibatch) + 2*self.regularization*self.W

        # Set the bias gradient to be the mean of the gradients
        self.gradb = grads.T.mean(axis=1, keepdims=True)
        #self.gradb = np.sum(grads, axis=0, keepdims=True)
        
        return np.dot(grads, self.W)

    def cost(self):
        return self.regularization * (self.W**2).sum()
    
    def __str__(self):
        return "{} Layer, {}->{}".format(self.type, self.inputDim, self.outputDim)





# Batch normalization layer class.
class BatchNormalization():
    def __init__(self, inputDim, name=None, mean=None, variance=None, gamma=None, beta=None, alpha=0.90, trainable=True):
        self.type = "BatchNormalization"
        self.name = self.type if name is None else name
        self.inputDim = inputDim
       
        self.alpha = alpha

        self.mu = mean
        self.var = variance

        self.trainable = trainable

        self.batchMean = None
        self.batchVar = None

        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = np.ones((inputDim, 1 ), dtype=float)
            
        if beta is not None:
            self.beta = beta
        else:
            self.beta = np.zeros((inputDim, 1 ), dtype=float)
            

        self.gradBeta = np.zeros((inputDim, 1 ))
        self.gradGamma = np.zeros((inputDim, 1 ))

        self.previousGradBeta = np.zeros((inputDim, 1 ))
        self.previousGradGamma = np.zeros((inputDim, 1 ))

    # @TODO TRY CHANGING IT TO USE BATCH VARIABLES FOR TRAINING INSTEAD

    # Propagates the inputs through the layer
    def forward(self, x, updateInternal):
        
        if updateInternal:
            
            self.n_x = x.shape[1]
            self.x_shape = x.shape

            self.x = x

            # Moving average mu and variance

            sampleMean = np.mean(x, axis=1, keepdims=True)
            self.batchMean = sampleMean

            sampleVar = np.var(x, axis=1, keepdims=True)
            self.batchVar = sampleVar

            if self.mu is None:
                self.mu = sampleMean
            else:
                self.mu = self.alpha*self.mu + (1-self.alpha)*sampleMean

            if self.var is None:
                self.var = sampleVar
            else:
                self.var = self.alpha*self.var + (1-self.alpha)*sampleVar
            

            # SOME USE THE BATCH VALUES INSTEAD FOR THE RUNNING MEAN/VAR (SELF.MU/SELF.VAR)

            self.x_norm = ((x - sampleMean)/np.sqrt(sampleVar + np.finfo(float).eps))
            out = self.gamma * self.x_norm + self.beta
        else:
            x_norm = ((x - self.mu)/np.sqrt(self.var + np.finfo(float).eps))
            out = self.gamma * x_norm + self.beta
       
        return out



    # Backward pass of BN
    def backward(self, grads):
        
        self.previousGradBeta = self.gradBeta
        self.previousGradGamma = self.gradGamma


        #x_mu = self.x - self.batchMean
        #var_inv = 1./np.sqrt(self.batchVar + np.finfo(float).eps)

        # Calculates the gradients for gamma and beta
        dbeta = np.dot(np.dot(1/self.x.shape[1], grads.T), np.ones((self.x.shape[1], 1)))
        dgamma = np.dot(np.dot(1/self.x.shape[1], (grads.T*self.x_norm)), np.ones((self.x.shape[1], 1)))

        #### Grads are fine. Must be below or in the forward pass.
        self.gradBeta = dbeta
        self.gradGamma = dgamma


        ########
        
        # Gradients through scale and shift
        grads = grads.T * np.dot(self.gamma, np.ones((self.x.shape[1], 1)).T)
        #grads = grads.T * self.gamma


        # BatchNormBackPass functionality ###################
        sigma1 = ((self.batchVar+np.finfo(float).eps)**(-0.5))
        sigma2 = ((self.batchVar+np.finfo(float).eps)**(-1.5))

        g1 = grads * np.dot(sigma1, np.ones((self.x.shape[1],1)).T)
        g2 = grads * np.dot(sigma2, np.ones((self.x.shape[1],1)).T)

        D = self.x - np.dot(self.batchMean, np.ones((self.x.shape[1],1)).T)

        c = np.dot(g2*D, np.ones((self.x.shape[1],1)))

        dx1 = g1 - np.dot((1/self.x.shape[1])*np.dot(g1, np.ones((self.x.shape[1],1))) , np.ones((self.x.shape[1],1)).T) \
        - (np.dot(1/self.x.shape[1], D) * np.dot(c, np.ones((self.x.shape[1],1)).T))

        return dx1.T


    def cost(self):
        return 0
    
    def __str__(self):
        return "{} Layer, {}".format(self.type, self.inputDim)



# Relu layer class.
class Relu():
    def __init__(self):
        self.type = "Relu"
        self.activatedOutputs = None
        self.processedInput = None
        
    # Propagates the inputs through the layer
    def forward(self, inputs):
        self.processedInput = inputs
        self.activatedOutputs = (inputs>0)
        inputs[inputs<0] = 0
        return inputs
    
    # Propagates the input backwards
    def backward(self, grads):
        return grads*self.activatedOutputs.T

    def cost(self):
        return 0
    
    def __str__(self):
        return "{} Layer".format(self.type)


# Sigmoid layer class.
class Sigmoid():
    def __init__(self):
        self.type = "Sigmoid"
        self.activatedOutputs = None
        self.processedInput = None
        
    # Propagates the inputs through the layer
    def forward(self, inputs):
        return (2/(1+np.exp(-inputs)))-1
    
    # Propagates the input backwards
    def backward(self, grads):
        x = self.processedInput
        return ((1+self.forward(x))*(1-self.forward(x)))/2

    def cost(self):
        return 0
    
    def __str__(self):
        return "{} Layer".format(self.type)


# Softmax layer class. Uses categorical crossentropy for loss.
class Softmax():
    def __init__(self):
        self.type = "Softmax"
        self.probabilities = None
    
    def forward(self, inputs):
        self.probabilities = (np.exp(inputs) / np.sum(np.exp(inputs), axis = 0))
        # If single sample, expand dimensions from (X,) -> (X,1)
        #if len(self.probabilities.shape) == 1:
        #    self.probabilities = np.resize(self.probabilities, (self.probabilities.shape[0], 1))

        return self.probabilities
    
    def backward(self, targets):
        assert targets.shape == self.probabilities.shape
        grads = -(targets-self.probabilities).T # PAPER DOES NOT HAVE TRANSPOSE HERE
        return grads
    

    def cost(self):
        return 0


    def __str__(self):
        return "{} Layer".format(self.type)






'''

# Batch normalization layer class.
class BatchNormalization():
    def __init__(self, inputDim, name=None, startingMean=None, startingVariance=None, alpha=0.99):
        self.type = "BatchNormalization"
        self.name = self.type if name is None else name
        self.inputDim = inputDim
       
        self.alpha = alpha

        self.mean = startingMean
        self.variance = startingVariance

        
    # Propagates the inputs through the layer
    def forward(self, inputs, updateInternal):
        # updateInternal Dictates if the BA layer is to use the internal mean and variance or update them during the processing of new data
        # Should be True for training(train set), false for prediction(validation, test)

        assert inputs.shape[0] == self.inputDim, "BN Mismatching input dimensions."

        if updateInternal:
            self.processedInput = inputs # Maybe inside update?
            self.batchMean = np.mean(inputs, axis=1, keepdims=True)
            self.batchVariance = np.var(inputs, axis=1, keepdims=True)
            self.batchVariance[self.batchVariance==0] = np.finfo(float).eps # Sets variances of 0 to minimum value distinct from 0 to avoid divide by 0 issues.

            if self.mean is None:
                self.mean = self.batchMean
            else:
                self.mean = self.alpha*self.mean+(1-self.alpha)*self.batchMean

            if self.variance is None:
                self.variance = self.batchVariance
            else:
                self.variance = self.alpha*self.variance+(1-self.alpha)*self.batchVariance

        else:
            self.batchMean = self.mean
            self.batchVariance = self.variance
        
        return (self.batchVariance**-0.5) * (inputs-self.batchMean)


    # Backward pass of BN
    def backward(self, grads):

        var = (self.batchVariance**-0.5)
        sizeOfMinibatch = self.processedInput.shape[1]
        inputs = (self.processedInput-self.batchMean) # Center inputs by subtracting the mean

        J_v = -(self.batchVariance**-1.5)*(grads.T * inputs).sum(axis=1, keepdims=True)
        J_my = var * grads.T.sum(axis=1, keepdims=True)

        grads = grads.T * var + J_v * inputs / sizeOfMinibatch + J_my / sizeOfMinibatch

        return grads.T


    def cost(self):
        return 0
    
    def __str__(self):
        return "{} Layer, {}".format(self.type, self.inputDim)

'''