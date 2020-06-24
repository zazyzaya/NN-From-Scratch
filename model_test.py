import numpy as np
from data_structures import Model, Layer, smooth_relu, sigmoid

'''
Simple proof of concept: train to solve the XOR problem.

Output after 1000 epochs:
y =
[[1 0]
 [1 0]
 [0 1]
 [0 1]]

y' = 
[[0.98768695 0.00950717]
 [0.9875749  0.00977973]
 [0.01046405 0.98948764]
 [0.01002394 0.99203538]]

 It works!
'''

class SimpleNN(Model):
    def __init__(self):
        super().__init__()

        self.layers = [
            Layer(2, 4, activation=smooth_relu),
            Layer(4, 2, activation=sigmoid)
        ]

X = np.array([
    [1,0], 
    [0,1],
    [0,0], 
    [1,1]
]).T

y = np.array([
    [1,0], 
    [1,0],
    [0,1], 
    [0,1]
]).T

nn = SimpleNN()
nn.train_model(X, y, epochs=1000, lr=0.5)