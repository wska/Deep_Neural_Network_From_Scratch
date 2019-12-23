from model import Model
from layers import Linear, Relu, Softmax
from optimizers import SGD


def main():
    network = Model()
    network.addLayer(Linear(5,3))
    network.addLayer(Softmax())
   
    sgd = SGD(lr=0.01, lr_decay=0.99)
 
    network.compile(sgd, "cce")
    
    print(network.loss)
    print(network.optimizer)
    print(network)

    


if __name__ == "__main__":
    main()