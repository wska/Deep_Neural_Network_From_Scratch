from model import Model
from layers import Linear, Relu, Softmax
from optimizers import SGD
from utility import loadData


def main():

    trainingData, trainingLabels, encodedTrainingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_1.mat")
    testingData, testingLabels, encodedTestingLabels = loadData("Datasets/cifar-10-batches-mat/data_batch_2.mat")
    validationData, validationlabels, encodedValidationlabels = loadData("Datasets/cifar-10-batches-mat/data_batch_3.mat")

    print(trainingData.shape)
    print(trainingLabels.shape)
    print(encodedTrainingLabels.shape)

    network = Model()
    network.addLayer(Linear(32*32*3,10))
    network.addLayer(Softmax())
   
    sgd = SGD(lr=0.01, lr_decay=0.99)
 
    network.compile(sgd, "cce")
    
    print(network.loss)
    print(network.optimizer)
    print(network)

    


if __name__ == "__main__":
    main()