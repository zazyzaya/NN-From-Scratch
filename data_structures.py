import numpy as np

from abc import ABC
from math import log
from random import sample

mm = np.matmul

'''
A few activation functions for nn layers to use
'''
def sigmoid(X, derivative=False):
    if not derivative:
        return 1/(1+np.exp(-X))
   
    return sigmoid(X)*(1-sigmoid(X))

def relu(X, derivative=False):
    if not derivative:
        X[X < 0] = 0
        return X
    
    X[X > 0] = 1
    X[X < 0] = 0

    return X

def smooth_relu(X, derivative=False):
    if not derivative:
        return np.log(1 + np.exp(X))

    return 1 / (1 + np.exp(-X))

'''
Basically equivilant to the Torch.nn.Linear class but built from scratch
'''
class Layer:
    def __init__(self, num_in, num_out, activation=sigmoid):
        self.weights = np.random.uniform(low=-1, high=1, size=(num_out, num_in))
        self.activation = activation

        self.z = None 
        self.derivative = None
        self.bias = np.random.uniform(low=-1, high=1,size=(num_out,1))

        self.train=True
        self.bias_update = []
        self.weight_update = []

    '''
    Returns sig(Wx + b)
    Stores the derivative and activations for use in back prop if training
    '''
    def forward(self, X):
        z = mm(self.weights, X) + self.bias
        a = self.activation(z)

        if self.train:
            self.derivative = self.activation(z, derivative=True)
            self.a = a

        return a
        
    '''
        Computes error for this layer based on previous layer (if needed), 
        then finds gradients for bias and weights

        returns weights^T X delta 
    '''
    def backward(self, a_prev, WTd_L=[], delta=[]):
        # Calculate this layers error if not provided
        # delta = \/C * z 
        # delta^L-1 = W^L Transposed X delta^L * z^L-1
        if len(WTd_L) != 0:
            delta = np.multiply(
                WTd_L,
                self.derivative
            )
        
        # If this is the last layer, delta is precalculated from the 
        # derivative of the loss function by the module containing
        # this layer
        else:
            assert len(delta) > 0

        #dC/db = delta
        self.bias_update.append(delta)
        
        # dC/dw_ij = aj * d^L_i
        # so dC/dw_i = delta x a.T
        weight_update =  mm(delta, a_prev.T)

        self.weight_update.append(weight_update)
        
        # Precompute this for the next step
        return mm(self.weights.T, delta)

    def update(self, lr):
        # Calculate the average to allow easy batching
        bu = sum(self.bias_update) / len(self.bias_update)
        wu = sum(self.weight_update) / len(self.weight_update)

        # Then update using grad descent 
        self.bias -= bu * lr 
        self.weights -= wu * lr 

        self.bias_update = []
        self.weight_update = []

'''
Abstract class to hold layers, and perform updates on the network 
'''
class Model(ABC):
    def __init__(self):
        self.layers = []

    '''
    Calls forward on all layers in the network
    Forward-propagation
    '''
    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)

        return X

    '''
    By default, use SSE
    If overridden, be sure to include the derivative wrt yprime 
    '''
    def loss(self, y, yprime, derivitive=False):
        if not derivitive:
            return sum((yprime - y)**2)
        else:
            return 2*(yprime - y)

    '''
    Back propagates error through each network layer. The real meat of the class
    A lot of the work is already done in the Layer class
    
    TODO improve batching 
    '''
    def backprop(self, X, y, lr):
        for i in range(y.shape[1]):
            # When slicing like this, np defaults to making them rows
            # so we have to transpose them into columns 
            yi = np.array([y[:, i]]).T
            x_in = np.array([X[:, i]]).T
            yprime = self.forward(x_in)

            d_L = np.multiply(
                self.loss(yi, yprime, derivitive=True),
                self.layers[-1].derivative
            )

            # Last layer we precalculate delta
            layer = 2
            WTd_L = self.layers[-1].backward(self.layers[-layer].a, delta=d_L)

            for layer in range(2, len(self.layers)):
                WTd_L = self.layers[-layer].backward(self.layers[-(layer+1)].a, WTd_L=WTd_L)
            
            # First layer we use the inputs as the activations
            self.layers[0].backward(x_in, WTd_L=WTd_L)

        for l in self.layers:
            l.update(lr)

    '''
    Sets all layers to train mode so they store derivatives
    '''
    def train(self):
        for l in self.layers:
            l.train = True

    '''
    Sets all layers to eval mode so they do not store any info
    '''
    def eval(self):
        for l in self.layers: 
            l.train = False

    '''
    Simple training loop. Does backprop on all samples. 
    '''
    def train_model(self, X, y, lr=0.001, epochs=800, verbose=1, lr_decay=1-1e-6, sgd=False):
        for epoch in range(epochs):
            self.train()
            
            if sgd:
                n = y.shape[1]
                batch = sample(range(n-1), 2+int(log(n)))

                self.backprop(X[:, batch], y[:, batch], lr)

            self.backprop(X, y, lr)

            self.eval()
            print("[%d] Loss: %0.4f" %
                (
                    epoch, 
                    self.loss(
                        y, 
                        self.forward(X)
                    ).sum() / y.shape[1]
                )
            )

            lr *= lr_decay

        print('y: ')
        print(y.T)
        print("\ny': ")
        print(self.forward(X).T)
            