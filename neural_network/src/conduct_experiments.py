import mnist_loader
import network
import sys
from time import time
from copy import deepcopy
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

nEpochs = 1

# global variable that collects all the accuracies to be plotted
all_results = {}

"""
Syntax for important functions:
Network([num_inputs, num_hidden_layer, num_outputs])
net.SGD(examples. epochs, mini-batch size, learning rate, test_data)
This script runs training trials for the specified combinations of
parameter values in Assignment 4.
"""


# Check for provided output name
name = ''
try:
    name = sys.argv[1]
    if os.path.isfile('./record_{}.txt'.format(name)):
        print 'ERROR: File already exists'
        sys.exit()
except:
    print 'ERROR: Please provide name for test'
    sys.exit()

# Redirect "print" output to file
sys.stdout = open('../test_records/{}.txt'.format(name), 'w')


def log(variables, test_data_label):
    """ Prints to console the parameter information provided. Used for labeling
        test trials.
    """
    print '====================================='
    print 'CONDUCTING TRIAL'
    print '#HIDDEN_LAYERS={}'.format(variables['hidden_layers'])
    print '#UNITS_IN_HIDDEN_LAYER={}'.format(variables['hidden_layer_units'])
    print 'MINI_BATCH_SIZE={}'.format(variables['mini_batch_size'])
    print 'LEARNING_RATE={}'.format(variables['learning_rate'])
    print 'TEST_DATA={}'.format(test_data_label)


def trial(variables, training_data, test_data_label):
    """ Creates a network given the provided parameters, and runs Stochastic
        Gradient Descent on the network. Returns the elapsed time of the run.
    """
    log(variables, test_data_label)
    topology = [784]
    for i in range(variables['hidden_layers']):
        topology.append(variables['hidden_layer_units'])
    topology.append(10)
    net = network.Network(topology)
    t1 = time()
    accuracies = net.SGD(training_data,
                         nEpochs,
                         variables['mini_batch_size'],
                         variables['learning_rate'],
                         variables['test_data'])
    t2 = time()
    elapsed = t2-t1
    print 'ELAPSED TIME: {} seconds'.format(elapsed)
    print '====================================='
    variable_tuple = (variables['hidden_layers'],
                      variables['hidden_layer_units'],
                      variables['learning_rate'],
                      variables['mini_batch_size'])

    try:
        all_results[variable_tuple].update({test_data_label :   accuracies})
    except KeyError:
        all_results.update({variable_tuple  :   {test_data_label :   accuracies}})

    return elapsed

def create_plots(folder):
    epochs = range(nEpochs)
    for k,v in all_results.items():
        training = v['training']
        validation = v['validation']
        test = v['test']

        ys = [training, validation, test]

        scatters = []

        colors = cm.rainbow(np.linspace(0, 1, len(ys)))
        for y, c in zip(ys, colors):
            scatters.append(plt.scatter(epochs, y, color=c))

        plt.legend(tuple(scatters),
               ('Training', 'Validation', 'Testing'),
               scatterpoints=1,
               loc='lower right',
               ncol=3,
               fontsize=8)

        plt.title('%s Hypervariables: Accuracies' % str(k))
        #plt.show()
        plt.draw()
        fig = plt.gcf()
        fig.savefig(os.path.join(folder,'%s_plot.png' % str(k)))
        plt.clf()

def run(training_data, validation_data, test_data):
    """ 'Main' function. Displays hardware information, then conducts
        trials for each of the different parameters laid out in the *_trials
        variables (lists).
    """
    print ('CPUs: {},\n'
           'Memory: {} GB,\n'
           'Processor: {},\n'
           'Clock_Speed: {}\n').format(4, 16, 'Intel Xeon CPU E5-1620 v2', 'up to 3.7 GHz')

    # Parameter values to test
    hidden_layer_trials = [0, 1, 2]
    hidden_layer_units_trials = [10, 30, 50]
    learning_rate_trials = [0.01, 3, 30]
    mini_batch_size_trials = [1, 10, 100]

    # for testing the training set
    unvectorized_training_data = [(x,np.where(y==1)[0][0]) for x,y in training_data]
    test_data_trials = [('training', unvectorized_training_data),
                        ('validation', validation_data),
                        ('test', test_data)]

    #TODO training data as test data gives an error. Should be able to do
    # test_data_trials.append(('training', training_data))

    # Base parameters from which to change one at a time
    standard = {'hidden_layers': 1,
                'hidden_layer_units': 30,
                'learning_rate': 3.0,
                'mini_batch_size': 10,
                'test_data': 'dummy'}

    total_time = 0
    num_trials = 0

    for test_data_label, test_data in test_data_trials:

        variables = deepcopy(standard)
        variables['test_data'] = test_data

        total_time += trial(variables, training_data, test_data_label)
        num_trials += 1

        '''
        # Run trial for each value of hidden layers
        for hidden_layer_amt in hidden_layer_trials:
            variables['hidden_layers'] = hidden_layer_amt
            total_time += trial(variables, training_data, test_data_label)
            num_trials += 1

        variables['hidden_layers'] = standard['hidden_layers']

        # Run trial for each size of hidden layer
        for hidden_layer_units_amt in hidden_layer_units_trials:
            variables['hidden_layer_units'] = hidden_layer_units_amt
            total_time += trial(variables, training_data, test_data_label)
            num_trials += 1

        variables['hidden_layer_units'] = standard['hidden_layer_units']

        # Run trial for each learning rate value
        for learning_rate_amt in learning_rate_trials:
            variables['learning_rate'] = learning_rate_amt
            total_time += trial(variables, training_data, test_data_label)
            num_trials += 1

        variables['learning_rate'] = standard['learning_rate']

        # Run trial for each mini batch size
        for mini_batch_size_amt in mini_batch_size_trials:
            variables['mini_batch_size'] = mini_batch_size_amt
            total_time += trial(variables, training_data, test_data_label)
            num_trials += 1
        '''

    print '====================================='
    print 'TOTAL DURATION: {}'.format(total_time)
    print 'NUM_TRIALS: {}'.format(num_trials)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
run(training_data, validation_data, test_data)

prev_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
create_plots(os.path.join(prev_dir,'mnist_plots'))
