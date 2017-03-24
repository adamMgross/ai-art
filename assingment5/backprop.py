import numpy as np

LAYERS = 3
biases = [np.random.randn() for x in range(LAYERS)]
weights = [np.random.randn() for x in range(LAYERS)]


network_biases = 
    

def cost_derivative(output_activations, y):
    return (output_activations - y)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

