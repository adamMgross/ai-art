import numpy as np
import itertools
import functools
import operator
import os
#import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import network as net

# length of binary strings
n = 2

learning_rate = 3.0
nEpochs = 1000

def generate_samples(length):
    """ Generates all binary vectors of the given length, then outputs a list of tuples
        where each element is (vector, [OR(vector), AND(vector), NAND(vector), XOR(vector)])
    """
    binary_strings = [''.join(seq) for seq in itertools.product('01', repeat=length)]
    inputs = [np.array([int(i) for i in s]) for s in binary_strings]

    return [(x, np.array([int(any(x)), int(all(x)), int(not all(x)), functools.reduce(operator.xor, x)]).reshape((-1,1))) for x in inputs]

def main():
    training_set = generate_samples(n)

    # network with n inputs, one hidden layer with 2 units, one output unit (XOR of n input bits)
    XOR_topology = [n,n,1]
    XOR_training = [(x,np.array([y[3]])) for x,y in training_set]

    name = 'XOR_%d-%d-%d' % tuple(XOR_topology)
    # experiment(XOR_topology, XOR_training, name)

    for nUnits_1 in range(1,5):
        for nUnits_2 in range(5):
            if nUnits_2 == 0:
                topology = [n,nUnits_1,4]
                name = 'multi_%d-%d-%d' % tuple(topology)
            else:
                topology = [n,nUnits_1,nUnits_2,4]
                name = 'multi_%d-%d-%d-%d' % tuple(topology)

            experiment(topology,training_set,name)

def experiment(topology,training_set,name):
    network = net.Network(topology)
    results = network.train(training_set, nEpochs, learning_rate)

    create_plots(results,name)

def create_plots(results,name):
    folder = os.path.join(os.getcwd(),name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    epoch_weights, epoch_biases, epoch_costs = results
    epochs = range(nEpochs)

    ys = [epoch_costs]
    scatters = []

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    for y, c in zip(ys, colors):
        scatters.append(plt.scatter(epochs, y, color=c, s=10))

    plt.legend(tuple(scatters),
           ('Cost'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)

    plt.title('Cost vs. Epochs')
    #plt.show()
    plt.draw()
    fig = plt.gcf()
    fig.savefig(os.path.join(folder,'cost.png'))
    plt.clf()

    nLayers = len(epoch_weights[0])
    for l in range(1,nLayers):
        layer_folder = os.path.join(folder,'layer_%d' % l)
        if not os.path.exists(layer_folder):
            os.mkdir(layer_folder)

        nUnits = len(epoch_weights[0][l])
        for j in range(nUnits):
            # weights and biases for jth unit in layer l over all epochs
            ep_weights = [w[l][j] for w in epoch_weights.values()]
            ep_biases = [b[l][j] for b in epoch_biases.values()]

            nWeights = len(ep_weights[0])
            ys = [ep_biases]
            for k in range(nWeights):
                ep_ws = [w[k] for w in ep_weights]
                ys.append(ep_ws)

            scatters = []

            colors = cm.rainbow(np.linspace(0, 1, len(ys)))
            for y, c in zip(ys, colors):
                scatters.append(plt.scatter(epochs, y, color=c, s=10))

            names = ['Bias'] + ['Weight_%d' % k for k in range(nWeights)]
            names = tuple(names)

            plt.legend(tuple(scatters),
                   names,
                   scatterpoints=1,
                   loc='upper right',
                   ncol=3,
                   fontsize=8)

            plt.title('Layer %d, Unit %d: Bias/Weights vs. Epochs' % (l,j))
            #plt.show()
            plt.draw()
            fig = plt.gcf()
            fig.savefig(os.path.join(layer_folder,'unit_%d.png' % j))
            plt.clf()

if __name__ == '__main__':
    main()
