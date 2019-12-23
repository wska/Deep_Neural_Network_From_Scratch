import numpy as np
from layers import *
import model


class SGD():
    def __init__(self, lr=0.001, lr_decay=1.0, momentum=0, shuffle=False, model=None):
        self.lr = lr
        self.previous_lr = lr
        self.lr_decay = lr_decay
        self.shuffle = shuffle
        self.model = None
        self.momentum = momentum
        self.history = {}

    def fit(self, x_train, y_train, validationData=None, epochs=1, batch_size=None):
        if batch_size is None:
            print(x_train.shape)
            batch_size = x_train.shape[0] # check if this is actually the number of samples

        for epoch in range(1, epochs+1):
            trainCost, trainAcc = self.fit_epoch(x_train, y_train, batch_size)
            self.append_history(trainCost, trainAcc, validationData, epoch)
            self.previous_lr = self.lr
            self.lr *= self.lr_decay 

    def fit_epoch(self, x_train, y_train, batch_size):
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples and labels does not match.")
        trainCost = []
        trainAccuracy = []

        if self.shuffle:
            trainingIndices = np.random.permutation(x_train.shape[0])
        else:
            trainingIndices = np.arange(x_train.shape[0])
        
        for batch in range(0, x_train.shape[0], batch_size):
            indices = trainingIndices[batch: batch+batch_size]
            cost, accuracy = self.model.evaluate(x_train[:, indices], y_train[:, indices])
            trainCost.append(cost)
            trainAccuracy.append(accuracy)

            self.model.backpropagate(y_train[:, indices])
            self.update_weights()
        
        return np.mean(trainCost), np.mean(trainAccuracy)


    # Performs the weight update for the neural network
    def update_weights(self):
        for layer in self.model.layers:
            # If layer is trainable
            try:
                if layer.trainable:

                    # Linear layer case
                    if type(layer) is Linear:

                        layer.W -= self.lr * layer.gradW \
                            + self.previous_lr * self.momentum * layer.previousGradW # Momentum term

                        layer.b -= self.lr * layer.gradb \
                            + self.previous_lr * self.momentum * layer.previousGradb # Momentum term
                            
            # Layers with no trainable attribute will be skipped
            except AttributeError:
                pass


    
    def append_history(self, trainCost, trainAcc, validationData, epoch):
        
        # EPOCHS
        if "epoch" in self.history.keys():
            self.history["epoch"].append(epoch)
        
        else:
            self.history["epoch"] = [epoch]

        # TRAINING ACCURACY
        if "accuracy" in self.history.keys():
            self.history["accuracy"].append(trainAcc)
        
        else:
            self.history["accuracy"] = [trainAcc]

        # TRAINING COST
        if "cost" in self.history.keys():
            self.history["cost"].append(trainCost)
        
        else:
            self.history["cost"] = [trainCost]

        if validationData is not None:
            valCost, valAcc = self.model.evaluate(validationData[0], validationData[1])

            # VALIDATION ACCURACY
            if "accuracy" in self.history.keys():
                self.history["val_accuracy"].append(valAcc)
            
            else:
                self.history["val_accuracy"] = [valAcc]

            # VALIDATION COST
            if "val_cost" in self.history.keys():
                self.history["val_cost"].append(valCost)
            
            else:
                self.history["val_cost"] = [valCost]


    def __str__(self):
        strrep = "SGD Optimizer\n  LR: "+str(self.lr)+"\n  LR-DECAY: "+str(self.lr_decay)+"\n  SHUFFLE ENABLED: "+str(self.shuffle)    
        return strrep