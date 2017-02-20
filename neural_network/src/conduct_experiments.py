import mnist_loader
import network
import sys
from time import time
import sys
from copy import deepcopy
import os

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


def log(variables):
    """ Prints to console the parameter information provided. Used for labeling
        test trials.
    """
    print '====================================='
    print 'CONDUCTING TRIAL'
    print '#HIDDEN_LAYERS={}'.format(variables['hidden_layers'])
    print '#UNITS_IN_HIDDEN_LAYER={}'.format(variables['hidden_layer_units'])
    print 'MINI_BATCH_SIZE={}'.format(variables['mini_batch_size'])
    print 'LEARNING_RATE={}'.format(variables['learning_rate'])

def trial(variables, training_data, test_data):
    """ Creates a network given the provided parameters, and runs Stochastic
        Gradient Descent on the network. Returns the elapsed time of the run.
    """
    log(variables)
    topology = [784]
    for i in range(variables['hidden_layers']):
        topology.append(variables['hidden_layer_units'])
    topology.append(10)
    net = network.Network(topology)
    t1 = time()
    net.SGD(training_data,
            30,
            variables['mini_batch_size'],
            variables['learning_rate'],
            test_data)
    t2 = time()
    elapsed = t2-t1
    print 'ELAPSED TIME: {} seconds'.format(elapsed)
    print '====================================='
    return elapsed

def run(training_data, test_data):
    """ 'Main' function. Displays hardware information, then conducts
        trials for each of the different parameters laid out in the *_trials
        variables (lists).
    """
    print ('CPUs: {},\n'
           'Memory: {}GB,\n'
           'Processor: {},\n'
           'Clock_Speed: {}\n').format(1, 0.5, 'Intel Xeon', 'up to 3.3 GHz')

    # Parameter values to test
    hidden_layer_trials = [0, 1, 2]
    hidden_layer_units_trials = [10, 30, 50]
    learning_rate_trials = [0.01, 3, 30]
    mini_batch_size_trials = [1, 10, 100]

    # Base parameters from which to change one at a time
    standard = {'hidden_layers': 1,
                'hidden_layer_units': 30,
                'learning_rate': 3.0,
                'mini_batch_size': 10}
    total_time = 0
    num_trials = 0

    # Run trial for each number of hidden layers
    for hidden_layer_amt in hidden_layer_trials:
        variables = deepcopy(standard)
        variables['hidden_layers'] = hidden_layer_amt
        total_time += trial(variables, training_data, test_data)
        num_trials += 1

    # Run trial for each size of hidden layer
    for hidden_layer_units_amt in hidden_layer_units_trials:
        variables = deepcopy(standard)
        variables['hidden_layer_units'] = hidden_layer_units_amt
        total_time += trial(variables, training_data, test_data)
        num_trials += 1

    # Run trial for each learning rate value
    for learning_rate_amt in learning_rate_trials:
        variables = deepcopy(standard)
        variables['learning_rate'] = learning_rate_amt
        total_time += trial(variables, training_data, test_data)
        num_trials += 1

    # Run trial for each mini batch size
    for mini_batch_size_amt in mini_batch_size_trials:
        variables = deepcopy(standard)
        variables['mini_batch_size'] = mini_batch_size_amt
        total_time += trial(variables, training_data, test_data)
        num_trials += 1

    print '====================================='
    print 'TOTAL DURATION: {}'.format(total_time)
    print 'NUM_TRIALS: {}'.format(num_trials)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
run(training_data, test_data)
