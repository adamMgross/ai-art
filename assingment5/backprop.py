import numpy as np
import itertools
import functools
import operator

# length of binary strings
n = 2
samples = generate_samples(n)
LAYERS = 3
biases = [np.random.randn() for x in range(LAYERS)]
weights = [np.random.randn() for x in range(LAYERS)]

def generate_samples(length):
    """ Generates all binary vectors of the given length, then outputs a list of tuples
        where each element is (vector, (OR(vector), AND(vector), NAND(vector), XOR(vector)))
    """
    binary_strings = [''.join(seq) for seq in itertools.product('01', repeat=length)]
    inputs = [tuple([int(i) for i in s]) for s in binary_strings]
    return [(x, (any(x), all(x), not all(x), functools.reduce(operator.xor, x))) for x in inputs]



def hadamard(a, b):
    #TODO need to vectorize
    return a*b


def backprop(x):
    inputs, outputs = x
    zs = feed_forward(x)
    errors = []
    error = (sigmoid(zs[-1]) - outputs) - sigmoid_prime(zs[-1])
    errors.apppend(error)
    for z, i in enumerate(reversed(zs[:-1])):
        delta = hadamard(weights[-i -1] * errors[-i - 1], sigmoid_prime(zs[-i -1]))
        errors = [delta] + errors
        gradients = [{'dc_dw': sigmoid(zs[i-1] * errors[i]),
                      'dc_db': errors[i]}
                      for i in range(1, LAYERS + 1)]
    return gradients


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
