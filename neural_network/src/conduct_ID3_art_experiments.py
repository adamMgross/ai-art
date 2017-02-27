import network
import sys
from time import time
import sys
from copy import deepcopy
import os

import json
import random
import numpy as np

# proportion of total instances to be used for testing
test_ratio = 0.2

# proportion of training instances to be used for validation
cv_ratio = 1.0/3


"""
Syntax for important functions:
Network([num_inputs, num_hidden_layer, num_outputs])
net.SGD(examples. epochs, mini-batch size, learning rate, test_data)
This script runs training trials for the specified combinations of
parameter values in Assignment 4.
"""


def main():
    prev_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)

    # Check for provided output name
    name = ''
    record_file = ''
    try:
        name = sys.argv[1]
        record_file = os.path.join(prev_dir,'test_records','%s.txt' % name)
        if os.path.isfile(record_file):
            print 'ERROR: File already exists'
            sys.exit()
    except:
        print 'ERROR: Please provide name for test'
        #sys.exit()

    # Redirect "print" output to file
    #sys.stdout = open(record_file, 'w')
    git_root_dir = os.path.normpath(prev_dir + os.sep + os.pardir)
    #print git_root_dir

    training_data, validation_data, test_data = art_loader(
            os.path.join(git_root_dir,'id3','data.json'))

    run(training_data, validation_data, test_data)

def art_loader(json_file):
    return JSON_to_datasets(json_file)

def JSON_to_datasets(filename):
    data = []
    with open(filename, 'r') as f:
        data = json.load(f)

    data_size = len(data)
    test_size = int(data_size * test_ratio)
    test_indices = random.sample(range(data_size),test_size)

    labeled_data = []
    ordered_attrs = data[0].keys()[:]
    ordered_attrs.remove('genre')
    ordered_attrs.remove('url')
    for instance_label in data:
        instance = instance_label.copy()
        genre = instance.pop('genre')
        label = 2
        if genre == 'impressionist':
            label = 0
        elif genre == 'surrealist':
            label = 1

        instance.pop('url')
        instance_list = np.array([instance[k] for k in ordered_attrs])

        labeled_data.append((instance_list,label))

    training = []
    testing_set = []

    for i in range(data_size):
        if i in test_indices:
            testing_set.append(labeled_data[i])
        else:
            training.append(labeled_data[i])

    training_size = len(training)
    cv_size = int(training_size * cv_ratio)

    cv_indices = random.sample(range(training_size),cv_size)
    training_set = []
    validation_set = []

    for i in range(training_size):
        if i in cv_indices:
            validation_set.append(training[i])
        else:
            # change it so label is unit vector (for ease of network):
            #   - [1,0,0] : impressionist
            #   - [0,1,0] : surrealist
            #   - [0,0,1] : neither
            instance_list, label = training[i]
            if label == 0:
                training_set.append((instance_list,np.array([1,0,0])))
            elif label == 1:
                training_set.append((instance_list,np.array([0,1,0])))
            else:
                training_set.append((instance_list,np.array([0,0,1])))

    return training_set, validation_set, testing_set

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

    # number of attributes in input images
    topology = [10]
    for i in range(variables['hidden_layers']):
        topology.append(variables['hidden_layer_units'])

    # we have three possible classifications
    #   -impressionist: 0
    #   -surrealist: 1
    #   -neither: 2
    topology.append(3)
    net = network.Network(topology)
    t1 = time()
    net.SGD(training_data,
            30,
            variables['mini_batch_size'],
            variables['learning_rate'],
            variables['test_data'])
    t2 = time()
    elapsed = t2-t1
    print 'ELAPSED TIME: {} seconds'.format(elapsed)
    print '====================================='
    return elapsed

def run(training_data, validation_data, test_data):
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
    test_data_trials = [('training', training_data),
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

if __name__ == '__main__':
    main()
