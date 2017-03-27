import random
import numpy as np

from copy import deepcopy
from collections import OrderedDict

# basic class for fully-connected neural network
class Network(object):

    # topology: a list detailing the number of neurons in each layer
    # example: topology = [2,2,1] ==> 2 input neurons, one hidden layer
    # with 2 neurons, and then 1 output neuron (such is the case with the XOR problem)
    def __init__(self, topology):

        self.topology = topology
        self.nLayers = len(topology)
        self.weights = []
        self.biases = []

        # housekeeping step so that weights[l] and biases[l]
        # correspond to the weights and biases of layer l
        # the weights and biases of layer 0 (i.e. the input layer)
        # will never be used
        self.weights.append(np.ones((topology[0],1)))
        self.biases.append(np.ones((topology[0],1)))

        # every layer except the input layer will have weights and biases
        # associated with it
        # values randomly generated from normal distribution
        for l in range(1,len(topology)):
            n = topology[l]   # number of neurons in layer l
            m = topology[l-1]   # number of neurons in layer l-1
            self.weights.append(np.random.randn(n, m))
            self.biases.append(np.random.randn(n, 1))

        # real-time value of cost function whilst testing
        # used for reporting purposes
        self.cost = -1

    # tests the network using the given testing_set
    # reports the accuracy (# correct / # of testing instances)
    def test_XOR(self, testing_set):
        return self.evaluate_XOR(testing_set) / len(testing_set)

    # trains the network one epoch at a time
    # for each epoch, the weights/biases and the value of the cost function
    # are tracked
    # in this assignment, the mini_batch is the entire training set,
    # which is only four instances: [00,01,10,11] and their respective labels
    def train(self, training_data, nEpochs, learning_rate):
        n = len(training_data)

        epoch_weights = OrderedDict()
        epoch_biases = OrderedDict()
        epoch_costs = []
        for epoch in range(nEpochs):
            # print self.weights
            # print self.biases
            epoch_weights.update({epoch : deepcopy(self.weights)})
            epoch_biases.update({epoch : deepcopy(self.biases)})
            # for i in range(epoch+1):
                # print epoch_weights[i]
                # print epoch_biases[i]

            random.shuffle(training_data)

            self.cost = 0
            self.update_step(training_data, learning_rate)
            self.cost /= (2.0 * n)

            epoch_costs.append(self.cost)

            # print('Epoch %d finished training' % epoch)

        '''
        for e_b in epoch_biases:
            for layer_b in e_b:
                print layer_b
        '''

        return epoch_weights, epoch_biases, epoch_costs

    # performs one iteration: calculates average partials across
    # training set and uses them to update weights/biases according to
    # gradient descent update rule
    def update_step(self, training_data, learning_rate):
        n = len(training_data)
        # print self.topology

        # partials of cost w.r.t weights and biases for each layer
        delta_w = [np.zeros(layer_w.shape) for layer_w in self.weights]
        delta_b = [np.zeros(layer_b.shape) for layer_b in self.biases]
        for x, label in training_data:
            # partials of cost w.r.t weights and biases for each instance
            dw, db = self.backprop(x, label)
            for l in range(self.nLayers):
                # print 'layer %d' % l
                # print delta_w[l]
                # print dw[l]
                delta_w[l] += dw[l]
                # print delta_b[l]
                # print db[l]
                delta_b[l] += db[l]

        # gradient descent
        for l in range(1,self.nLayers):

            # print 'weights in layer %d' % l
            # print self.weights[l]
            # print 'biases in layer %d' % l
            # print self.biases[l]

            self.weights[l] -= learning_rate * delta_w[l] / float(n)
            self.biases[l] -= learning_rate * delta_b[l] / float(n)

    # x is the input to the network (no label)
    # calculates the weighted outputs (zs) and activations for each layer
    # activations[-1] = y' = output of entire network
    def feed_forward(self, x):
        activations = []
        # "output" from input layer is the input
        activations.append(x)

        zs = []
        # unused; this is weighted output from input layer
        zs.append(x)

        for l in range(1,self.nLayers):
            z = np.dot(self.weights[l], activations[l-1].reshape((-1,1))) + self.biases[l]
            zs.append(z)

            a = np.array([sigmoid(z_i) for z_i in z])
            activations.append(a)

        return zs, activations

    def backprop(self, x, label):
        zs, activations = self.feed_forward(x)

        # difference vector between calculated output and actual output (label)
        # also equivalent to the gradient of the cost function
        cost_gradient = activations[-1] - label

        self.cost += sum(cost_gradient * cost_gradient)

        delta_w = [np.zeros(layer_w.shape) for layer_w in self.weights]
        delta_b = [np.zeros(layer_b.shape) for layer_b in self.biases]

        # error in final layer
        error = cost_gradient * vectorized_sigmoid_prime(zs[-1])
        for l in reversed(range(1,self.nLayers)):
            delta_w[l] = np.dot(error.reshape((-1,1)), activations[l-1].reshape((-1,1)).transpose())
            delta_b[l] = error[:]

            # error in previous layer
            try:
                error = np.dot(self.weights[l].transpose(), error.reshape((-1,1))) * vectorized_sigmoid_prime(zs[l-1])
            except ValueError:
                 print l
                 print self.weights[l]
                 print error
                 print cost_gradient
                 print vectorized_sigmoid_prime(zs[-1])
                 print ''
                 print x
                 print activations[-1]
                 print label
                 exit(1)

        return delta_w, delta_b

    # returns the number of binary strings with correct XOR calculations
    def evaluate_XOR(self, testing_set):
        nCorrect = 0
        for x,label in testing_set:
            zs, activations = self.feed_forward(x)
            calc_label = round(activations[-1][0])

            nCorrect += calc_label == label

        return nCorrect

# takes in a vector z and outputs the sigmoid_prime function
# applied element-wise to z
def vectorized_sigmoid_prime(z):
    return np.array([sigmoid_prime(z_i) for z_i in z])

# sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# derivative of sigmoid function
def sigmoid_prime(z):
    sig_z = sigmoid(z)
    return sig_z * (1 - sig_z)
