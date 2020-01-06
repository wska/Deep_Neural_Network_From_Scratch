from model import Model
from layers import Linear, Relu, Softmax, BatchNormalization
from optimizers import SGD
from utility import *
import numpy as np
from datetime import datetime
from decimal import Decimal


def test1layergradients(samples=1, dimensions=3072):

    print("\n\nTesting 1-layer gradients (NO BN, NO REG) using a batch size of {}".format(samples))
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]
    trainingLabels = trainingLabels[0:dimensions, 0:samples]
    encodedTrainingLabels = encodedTrainingLabels[0:dimensions, 0:samples]
    

    network = Model()
    linear = Linear(dimensions, 10, regularization=0.00)
    network.addLayer(linear)
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.0, shuffle=True)
    network.compile(sgd, "cce")

    network.predict(trainingData)
    network.backpropagate(encodedTrainingLabels)
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')
    numerical_gradW = compute_grads(1e-6, linear.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb = compute_grads(1e-6, linear.b, trainingData, encodedTrainingLabels, network)

    print("W")
    relative_errorW = grad_difference(linear.gradW, numerical_gradW)
    print("b")
    relative_errorb = grad_difference(linear.gradb, numerical_gradb)

    return (relative_errorW, linear.gradW, numerical_gradW), (relative_errorb, linear.gradb, numerical_gradb)

   
def test2layergradients(samples=1, dimensions=3072):

    print("\n\nTesting 2-layer gradients (WITHOUT BN, NO REG) using a batch size of {}".format(samples))
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]
    trainingLabels = trainingLabels[0:dimensions, 0:samples]
    encodedTrainingLabels = encodedTrainingLabels[0:dimensions, 0:samples]
    
    

    network = Model()
    linear = Linear(dimensions, 50, regularization=0.00, initializer="he")
    network.addLayer(linear)
    network.addLayer(Relu())

    linear2 = Linear(50, 10, regularization=0.00, initializer="he")
    network.addLayer(linear2)
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.0, shuffle=True)
    network.compile(sgd, "cce")

    #network.fit(trainingData, encodedTrainingLabels, epochs=5, validationData=None, batch_size=samples)

    network.predict(trainingData)
    network.backpropagate(encodedTrainingLabels)

    
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')
    
    numerical_gradW1 = compute_grads(1e-5, linear.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb1 = compute_grads(1e-5, linear.b, trainingData, encodedTrainingLabels, network)

    numerical_gradW2 = compute_grads(1e-5, linear2.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb2 = compute_grads(1e-5, linear2.b, trainingData, encodedTrainingLabels, network)

    print("W1")
    relative_errorW = grad_difference(linear.gradW, numerical_gradW1)
    print("b1")
    relative_errorb = grad_difference(linear.gradb, numerical_gradb1)

    print("W2")
    relative_errorW2 = grad_difference(linear2.gradW, numerical_gradW2)
    print("b2")
    relative_errorb2 = grad_difference(linear2.gradb, numerical_gradb2)

    print("\n")



def test2layergradientsWBN(samples=1, dimensions=3072):

    print("\n\nTesting 2-layer gradients (WITH BN, NO REG) using a batch size of {}".format(samples))
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]
    trainingLabels = trainingLabels[0:dimensions, 0:samples]
    encodedTrainingLabels = encodedTrainingLabels[0:dimensions, 0:samples]
    
    

    network = Model()
    linear = Linear(dimensions, 50, regularization=0.00, initializer="xavier")
    network.addLayer(linear)

    bnlayer = BatchNormalization(50)
    network.addLayer(bnlayer)
    network.addLayer(Relu())

    linear2 = Linear(50, 10, regularization=0.00, initializer="xavier")
    network.addLayer(linear2)
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.0, shuffle=True)
    network.compile(sgd, "cce")

    #network.fit(trainingData, encodedTrainingLabels, epochs=200, validationData=None, batch_size=samples)

    network.predict(trainingData, updateInternal=True)
    network.backpropagate(encodedTrainingLabels)
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')

    
    
    numerical_gradW1 = compute_grads_w_BN(1e-4, linear.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb1 = compute_grads_w_BN(1e-4, linear.b, trainingData, encodedTrainingLabels, network)

    numerical_gradgamma = compute_grads_w_BN(1e-4, bnlayer.gamma, trainingData, encodedTrainingLabels, network)
    numerical_gradbeta = compute_grads_w_BN(1e-4, bnlayer.beta, trainingData, encodedTrainingLabels, network)

    numerical_gradW2 = compute_grads_w_BN(1e-4, linear2.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb2 = compute_grads_w_BN(1e-4, linear2.b, trainingData, encodedTrainingLabels, network)


    print("W1")
    relative_errorW = grad_difference(linear.gradW, numerical_gradW1)
    print("b1")
    relative_errorb = grad_difference(linear.gradb, numerical_gradb1)

    print("gamma1")
    relative_errorW = grad_difference(bnlayer.gradGamma, numerical_gradgamma)
    print("beta1")
    relative_errorb = grad_difference(bnlayer.gradBeta, numerical_gradbeta)

    print("W2")
    relative_errorW2 = grad_difference(linear2.gradW, numerical_gradW2)
    print("b2")
    relative_errorb2 = grad_difference(linear2.gradb, numerical_gradb2)

    print("\n")


def test3layergradients(samples=1, dimensions=3072):

    print("\n\nTesting 3-layer gradients using a batch size of {}".format(samples))
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]
    trainingLabels = trainingLabels[0:dimensions, 0:samples]
    encodedTrainingLabels = encodedTrainingLabels[0:dimensions, 0:samples]
    
    network = Model()

    linear = Linear(dimensions, 50, regularization=0.00, initializer="he")
    network.addLayer(linear)
    network.addLayer(Relu())

    linear2 = Linear(50, 30, regularization=0.00, initializer="he")
    network.addLayer(linear2)
    network.addLayer(Relu())

    linear3 = Linear(30, 10, regularization=0.00, initializer="he")
    network.addLayer(linear3)
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.0, shuffle=True)
    network.compile(sgd, "cce")

    network.predict(trainingData, updateInternal=True)
    network.backpropagate(encodedTrainingLabels)
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')
    
    numerical_gradW1 = compute_grads_w_BN(1e-4, linear.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb1 = compute_grads_w_BN(1e-4, linear.b, trainingData, encodedTrainingLabels, network)

    numerical_gradW2 = compute_grads_w_BN(1e-4, linear2.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb2 = compute_grads_w_BN(1e-4, linear2.b, trainingData, encodedTrainingLabels, network)

    numerical_gradW3 = compute_grads_w_BN(1e-4, linear3.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb3 = compute_grads_w_BN(1e-4, linear3.b, trainingData, encodedTrainingLabels, network)



    print("W1")
    relative_errorW = grad_difference(linear.gradW, numerical_gradW1)
    print("b1")
    relative_errorb = grad_difference(linear.gradb, numerical_gradb1)

    print("W2")
    relative_errorW2 = grad_difference(linear2.gradW, numerical_gradW2)
    print("b2")
    relative_errorb2 = grad_difference(linear2.gradb, numerical_gradb2)
    
    print("W3")
    relative_errorW3 = grad_difference(linear3.gradW, numerical_gradW3)
    print("b3")
    relative_errorb3 = grad_difference(linear3.gradb, numerical_gradb3)

    print("\n")


def test3layergradientsWBN(samples=1, dimensions=3072):

    print("\n\nTesting 3-layer gradients (WITH BN, NO REG) using a batch size of {}".format(samples))
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]
    trainingLabels = trainingLabels[0:dimensions, 0:samples]
    encodedTrainingLabels = encodedTrainingLabels[0:dimensions, 0:samples]
    
    

    network = Model()
    linear = Linear(dimensions, 50, regularization=0.00, initializer="he")
    network.addLayer(linear)

    bnlayer = BatchNormalization(50)
    network.addLayer(bnlayer)
    network.addLayer(Relu())

    linear2 = Linear(50, 30, regularization=0.00, initializer="he")
    network.addLayer(linear2)

    bnlayer2 = BatchNormalization(30)
    network.addLayer(bnlayer2)
    network.addLayer(Relu())

    linear3 = Linear(30, 10, regularization=0.00, initializer="he")
    network.addLayer(linear3)
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.0, shuffle=True)
    network.compile(sgd, "cce")

    #network.fit(trainingData, encodedTrainingLabels, epochs=200, validationData=None, batch_size=samples)

    network.predict(trainingData, updateInternal=True)
    network.backpropagate(encodedTrainingLabels)
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')

    
    
    numerical_gradW1 = compute_grads_w_BN(1e-4, linear.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb1 = compute_grads_w_BN(1e-4, linear.b, trainingData, encodedTrainingLabels, network)

    numerical_gradgamma1 = compute_grads_w_BN(1e-4, bnlayer.gamma, trainingData, encodedTrainingLabels, network)
    numerical_gradbeta1 = compute_grads_w_BN(1e-4, bnlayer.beta, trainingData, encodedTrainingLabels, network)

    numerical_gradW2 = compute_grads_w_BN(1e-4, linear2.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb2 = compute_grads_w_BN(1e-4, linear2.b, trainingData, encodedTrainingLabels, network)

    numerical_gradgamma2 = compute_grads_w_BN(1e-4, bnlayer2.gamma, trainingData, encodedTrainingLabels, network)
    numerical_gradbeta2 = compute_grads_w_BN(1e-4, bnlayer2.beta, trainingData, encodedTrainingLabels, network)

    numerical_gradW3 = compute_grads_w_BN(1e-4, linear3.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb3 = compute_grads_w_BN(1e-4, linear3.b, trainingData, encodedTrainingLabels, network)



    print("W1")
    relative_errorW = grad_difference(linear.gradW, numerical_gradW1)
    print("b1")
    relative_errorb = grad_difference(linear.gradb, numerical_gradb1)

    print("gamma1")
    relative_errorGamma1 = grad_difference(bnlayer.gradGamma, numerical_gradgamma1)
    print("beta1")
    relative_errorbeta1 = grad_difference(bnlayer.gradBeta, numerical_gradbeta1)

    print("W2")
    relative_errorW2 = grad_difference(linear2.gradW, numerical_gradW2)
    print("b2")
    relative_errorb2 = grad_difference(linear2.gradb, numerical_gradb2)

    print("gamma2")
    relative_errorGamma2 = grad_difference(bnlayer2.gradGamma, numerical_gradgamma2)
    print("beta2")
    relative_errorbeta2 = grad_difference(bnlayer2.gradBeta, numerical_gradbeta2)
    
    print("W3")
    relative_errorW3 = grad_difference(linear3.gradW, numerical_gradW3)
    print("b3")
    relative_errorb3 = grad_difference(linear3.gradb, numerical_gradb3)

    print("\n")





def test4layergradients(samples=1, dimensions=3072):

    print("\n\nTesting 4-layer gradients using a batch size of {}".format(samples))
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]
    trainingLabels = trainingLabels[0:dimensions, 0:samples]
    encodedTrainingLabels = encodedTrainingLabels[0:dimensions, 0:samples]
    
    network = Model()

    linear = Linear(dimensions, 50, regularization=0.00, initializer="he")
    network.addLayer(linear)
    network.addLayer(Relu())

    linear2 = Linear(50, 30, regularization=0.00, initializer="he")
    network.addLayer(linear2)
    network.addLayer(Relu())

    linear3 = Linear(30, 20, regularization=0.00, initializer="he")
    network.addLayer(linear3)
    network.addLayer(Relu())

    linear4 = Linear(20, 10, regularization=0.00, initializer="he")
    network.addLayer(linear4)
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.0, shuffle=True)
    network.compile(sgd, "cce")

    network.predict(trainingData, updateInternal=True)
    network.backpropagate(encodedTrainingLabels)
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')
    
    numerical_gradW1 = compute_grads_w_BN(1e-4, linear.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb1 = compute_grads_w_BN(1e-4, linear.b, trainingData, encodedTrainingLabels, network)

    numerical_gradW2 = compute_grads_w_BN(1e-4, linear2.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb2 = compute_grads_w_BN(1e-4, linear2.b, trainingData, encodedTrainingLabels, network)

    numerical_gradW3 = compute_grads_w_BN(1e-4, linear3.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb3 = compute_grads_w_BN(1e-4, linear3.b, trainingData, encodedTrainingLabels, network)

    numerical_gradW4 = compute_grads_w_BN(1e-4, linear4.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb4 = compute_grads_w_BN(1e-4, linear4.b, trainingData, encodedTrainingLabels, network)



    print("W1")
    relative_errorW = grad_difference(linear.gradW, numerical_gradW1)
    print("b1")
    relative_errorb = grad_difference(linear.gradb, numerical_gradb1)

    print("W2")
    relative_errorW2 = grad_difference(linear2.gradW, numerical_gradW2)
    print("b2")
    relative_errorb2 = grad_difference(linear2.gradb, numerical_gradb2)
    
    print("W3")
    relative_errorW3 = grad_difference(linear3.gradW, numerical_gradW3)
    print("b3")
    relative_errorb3 = grad_difference(linear3.gradb, numerical_gradb3)

    print("W4")
    relative_errorW4 = grad_difference(linear4.gradW, numerical_gradW4)
    print("b4")
    relative_errorb4 = grad_difference(linear4.gradb, numerical_gradb4)

    print("\n")


def test4layergradientsWBN(samples=1, dimensions=3072):

    print("\n\nTesting 4-layer gradients (WITH BN, NO REG) using a batch size of {}".format(samples))
    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]
    trainingLabels = trainingLabels[0:dimensions, 0:samples]
    encodedTrainingLabels = encodedTrainingLabels[0:dimensions, 0:samples]
    
    

    network = Model()
    linear = Linear(dimensions, 50, regularization=0.00, initializer="he")
    network.addLayer(linear)

    bnlayer = BatchNormalization(50)
    network.addLayer(bnlayer)
    network.addLayer(Relu())

    linear2 = Linear(50, 30, regularization=0.00, initializer="he")
    network.addLayer(linear2)

    bnlayer2 = BatchNormalization(30)
    network.addLayer(bnlayer2)
    network.addLayer(Relu())

    linear3 = Linear(30, 20, regularization=0.00, initializer="he")
    network.addLayer(linear3)

    bnlayer3 = BatchNormalization(20)
    network.addLayer(bnlayer3)
    network.addLayer(Relu())

    linear4 = Linear(20, 10, regularization=0.00, initializer="he")
    network.addLayer(linear4)
    network.addLayer(Softmax())

    sgd = SGD(lr=0.001, lr_decay=1.0, momentum=0.0, shuffle=True)
    network.compile(sgd, "cce")

    #network.fit(trainingData, encodedTrainingLabels, epochs=200, validationData=None, batch_size=samples)

    network.predict(trainingData, updateInternal=True)
    network.backpropagate(encodedTrainingLabels)
    
    timestamp = datetime.now().strftime('%Y-%b-%d--%H-%M-%S')

    
    
    numerical_gradW1 = compute_grads_w_BN(1e-4, linear.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb1 = compute_grads_w_BN(1e-4, linear.b, trainingData, encodedTrainingLabels, network)

    numerical_gradgamma1 = compute_grads_w_BN(1e-4, bnlayer.gamma, trainingData, encodedTrainingLabels, network)
    numerical_gradbeta1 = compute_grads_w_BN(1e-4, bnlayer.beta, trainingData, encodedTrainingLabels, network)

    numerical_gradW2 = compute_grads_w_BN(1e-4, linear2.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb2 = compute_grads_w_BN(1e-4, linear2.b, trainingData, encodedTrainingLabels, network)

    numerical_gradgamma2 = compute_grads_w_BN(1e-4, bnlayer2.gamma, trainingData, encodedTrainingLabels, network)
    numerical_gradbeta2 = compute_grads_w_BN(1e-4, bnlayer2.beta, trainingData, encodedTrainingLabels, network)

    numerical_gradW3 = compute_grads_w_BN(1e-4, linear3.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb3 = compute_grads_w_BN(1e-4, linear3.b, trainingData, encodedTrainingLabels, network)

    numerical_gradgamma3 = compute_grads_w_BN(1e-4, bnlayer3.gamma, trainingData, encodedTrainingLabels, network)
    numerical_gradbeta3 = compute_grads_w_BN(1e-4, bnlayer3.beta, trainingData, encodedTrainingLabels, network)

    numerical_gradW4 = compute_grads_w_BN(1e-4, linear4.W, trainingData, encodedTrainingLabels, network)
    numerical_gradb4 = compute_grads_w_BN(1e-4, linear4.b, trainingData, encodedTrainingLabels, network)



    print("W1")
    relative_errorW = grad_difference(linear.gradW, numerical_gradW1)
    print("b1")
    relative_errorb = grad_difference(linear.gradb, numerical_gradb1)

    print("gamma1")
    relative_errorGamma1 = grad_difference(bnlayer.gradGamma, numerical_gradgamma1)
    print("beta1")
    relative_errorbeta1 = grad_difference(bnlayer.gradBeta, numerical_gradbeta1)

    print("W2")
    relative_errorW2 = grad_difference(linear2.gradW, numerical_gradW2)
    print("b2")
    relative_errorb2 = grad_difference(linear2.gradb, numerical_gradb2)

    print("gamma2")
    relative_errorGamma2 = grad_difference(bnlayer2.gradGamma, numerical_gradgamma2)
    print("beta2")
    relative_errorbeta2 = grad_difference(bnlayer2.gradBeta, numerical_gradbeta2)
    
    print("W3")
    relative_errorW3 = grad_difference(linear3.gradW, numerical_gradW3)
    print("b3")
    relative_errorb3 = grad_difference(linear3.gradb, numerical_gradb3)

    print("gamma3")
    relative_errorGamma3 = grad_difference(bnlayer3.gradGamma, numerical_gradgamma3)
    print("beta3")
    relative_errorbeta3 = grad_difference(bnlayer3.gradBeta, numerical_gradbeta3)
    
    print("W4")
    relative_errorW4 = grad_difference(linear4.gradW, numerical_gradW4)
    print("b4")
    relative_errorb4 = grad_difference(linear4.gradb, numerical_gradb4)

    print("\n")

def bn_test():
    samples = 200
    dimensions = 100

    print("Performing BN Tests")
    trainingData, _, _ = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")
    testingData, _, _ = loadData("Datasets/cifar-10-batches-mat/data_batch_2.mat")
    validationData, _, _ = loadData("Datasets/cifar-10-batches-mat/data_batch_3.mat")

    
    trainingData = trainingData[0:dimensions, 0:samples]

    validationData = validationData[0:dimensions, 0:samples]

    testingData = testingData[0:dimensions, :]
    


    ### MEAN AND VAR TEST ###
    gamma = np.ones((dimensions, 1 ), dtype=float)
    beta = np.zeros((dimensions, 1 ), dtype=float)

    
    print("Mean and var before")
    print(np.mean(trainingData, axis=1))
    print(np.std(trainingData, axis=1))

    bn = BatchNormalization(100, gamma=gamma, beta=beta, trainable=True)
    data = bn.forward(trainingData, True)

    print("Mean and std after")
    print(np.mean(data, axis=1))
    print(np.std(data, axis=1))


    ########################


    ###### GAMMA AND BETA TEST #####
    #gamma = np.ones((dimensions, 1 ), dtype=float) + 5
    #beta = np.zeros((dimensions, 1 ), dtype=float) + 1
    gamma = np.array([i for i in range(0,100)]).reshape((dimensions, 1))
    beta = np.array([i for i in range(0,100)]).reshape((dimensions, 1))

    
    print("Mean and std before")
    print(np.mean(trainingData, axis=1))
    print(np.std(trainingData,  axis=1))

    bn = BatchNormalization(100, gamma=gamma, beta=beta, trainable=True)
    data = bn.forward(trainingData, True)

    print("Mean and std after")
    print(np.mean(data, axis=1))
    print(np.std(data, axis=1))

    #########################


    #TESTING TRAIN VS TEST NUMBERS(STD FOR TEST IS VERY HIGH)

    gamma = np.ones((dimensions, 1 ), dtype=float)
    beta = np.zeros((dimensions, 1 ), dtype=float)

    bn = BatchNormalization(100, gamma=gamma, beta=beta, trainable=True, alpha=0.90)
    for i in range(0,500):
        batch = np.random.randn(100,8)
        #batch = testingData[:, np.random.choice(testingData.shape[1], 8)]
        data = bn.forward(batch, True)
        
    data = bn.forward(np.random.randn(100,8), False)

    print("Mean and std after")
    print(np.mean(data, axis=1))
    print(np.std(data, axis=1))

    

def main():

    #bn_test()

    #test1layergradients(2,10)
    #test1layergradients(16,10)
    #test1layergradients(64,10)

    #test2layergradients(8, 10)
    #test2layergradients(16, 10)
    #test2layergradients(64, 10)

    pass


if __name__ == "__main__":
    main()