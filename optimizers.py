import numpy as np
from layers import *
import model


class SGD():
    def __init__(self, lr=0.001, lr_decay=1.0, momentum=0, shuffle=False, model=None, eta_max=None, lr_min=None):
        self.lr = lr
        self.previous_lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.shuffle = shuffle
        self.model = None
        self.momentum = momentum
        self.history = {}

    def train(self, x_train, y_train, validationData=None, epochs=1, batch_size=None, verbose=True):
        if batch_size is None:
            batch_size = x_train.shape[1] # check if this is actually the number of samples

        self.history["name"] = self.model.name
        
        for epoch in range(1, epochs+1):
            trainCost, trainAcc = self.train_epoch(x_train, y_train, batch_size)
            if validationData is not None:
                valCost, valAcc = self.model.evaluate(validationData[0], validationData[1], updateInternal=False)
            else:
                valCost, valAcc = None, None

            self.append_history(trainCost, trainAcc, valCost, valAcc, epoch)
            self.previous_lr = self.lr
            self.lr *= self.lr_decay 

            if self.lr_min is not None: # If enabled(lr_min not None): If current lr is below lr_min, set to lr_min
                if self.lr < self.lr_min:
                    self.lr = self.lr_min

            if verbose:
                if valAcc is not None:
                    print("Epoch {} - Loss:{:.6f}, Accuracy:{:.4f}, Val_loss:{:.6f}, Val_acc:{:.4f}".format(epoch, trainCost, trainAcc, valCost, valAcc))
                else:
                    print("Epoch {} - Loss:{:.6f}, Accuracy:{:.4f}".format(epoch, trainCost, trainAcc))

    def train_epoch(self, x_train, y_train, batch_size):
        assert x_train.shape[1] == y_train.shape[1], "Number of samples and labels does not match."

        trainCost = []
        trainAccuracy = []

        if self.shuffle:
            trainingIndices = np.random.permutation(x_train.shape[1])
        else:
            trainingIndices = np.arange(x_train.shape[1])
        
        for batch in range(0, x_train.shape[1], batch_size):
            indices = trainingIndices[batch: batch+batch_size]

            cost, accuracy = self.model.evaluate(x_train[:, indices], y_train[:, indices], updateInternal=True)
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
                        
                        # @TODO FIX TO SAVE PREVIOUS UPDATE VECTOR INSTEAD
                        layer.W -= (self.lr * layer.gradW \
                            + self.lr * self.momentum * layer.previousGradW) # Momentum term

                        layer.b -= (self.lr * layer.gradb \
                            + self.lr * self.momentum * layer.previousGradb) # Momentum term

                    elif type(layer) is BatchNormalization:
                        
                        layer.gamma -= (self.lr * layer.gradGamma \
                            + self.lr * self.momentum * layer.previousGradGamma)
                        layer.beta -= (self.lr * layer.gradBeta \
                            + self.lr * self.momentum * layer.previousGradBeta)


            # Layers with no trainable attribute will be skipped
            except AttributeError:
                pass


    
    def append_history(self, trainCost, trainAcc, valCost, valAcc, epoch):
        
        # EPOCHS
        if "epochs" in self.history.keys():
            self.history["epochs"].append(epoch)
        
        else:
            self.history["epochs"] = [epoch]

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

        if valAcc is not None:

            # VALIDATION ACCURACY
            if "val_accuracy" in self.history.keys():
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