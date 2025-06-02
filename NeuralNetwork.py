# Starter Neural network
import math
import numpy

def sigmoid(x):
    return 1 / (1+numpy.exp(-x))

def weighted_sum(value, weight, bias):
    return numpy.dot(value, weight + bias)

w1 = 0
w2 = 1
bias = 4
weighted_sum = input1*w1 + input2*w2 + bias
print(sigmoid(weighted_sum))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedForward(self, inputs):
    # Weight inputs add bias then sigmoid
        total = numpy.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
weights = numpy.array([0,1])
bias = 4
n = Neuron(weights, bias)

x = numpy.array([2, 3])
print(n.feedForward(x))