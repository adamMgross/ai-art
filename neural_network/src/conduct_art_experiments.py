import network
import sys
from time import time
import sys
from copy import deepcopy
import os

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# scales images down to this size (making it square)
# so that network isn't gigantic
input_im_size = 30

nEpochs = 30

# global variable that collects all the accuracies to be plotted
all_results = {}

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
    sys.stdout = open(record_file, 'w')
    git_root_dir = os.path.normpath(prev_dir + os.sep + os.pardir)
    #print git_root_dir

    training_data, validation_data, test_data = art_loader(
            os.path.join(git_root_dir,'scraper','examples'))

    run(training_data, validation_data, test_data)

    create_plots(os.path.join(prev_dir,'art_plots'))

def art_dataset_loader(folder,vectorize_labels=False):
    files = os.listdir(folder)
    labeled_data = []
    for file in files:
        # creates grayscale image
        im = Image.open(os.path.join(folder,file)).convert('L')
        img = im.resize((input_im_size,input_im_size), Image.ANTIALIAS)

        pix = np.asarray(img)
        # flattens array to be 1-D
        pix = np.reshape(pix,(input_im_size**2,1))
        # print pix.shape

        label = file.split('.')[0]

        # surrealist corresponds to label=1
        if label == 'surrealist':
            if vectorize_labels:
                labeled_data.append((pix,np.array([0,1])))
            else:
                labeled_data.append((pix,1))
        else:
            if vectorize_labels:
                labeled_data.append((pix,np.array([1,0])))
            else:
                labeled_data.append((pix,0))

    return labeled_data

def art_loader(folder):
    training = art_dataset_loader(os.path.join(folder, "training"),True)
    validation = art_dataset_loader(os.path.join(folder, "validation"))
    test = art_dataset_loader(os.path.join(folder, "test"))

    return (training,validation,test)

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

    # number of pixels in imput images (which have been scaled down)
    topology = [input_im_size**2]
    for i in range(variables['hidden_layers']):
        topology.append(variables['hidden_layer_units'])

    # we have two possible classifications
    #   -impressionist: 0
    #   -surrealist: 1
    topology.append(2)
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

        # total_time += trial(variables, training_data, test_data_label)
        # num_trials += 1

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

    print '====================================='
    print 'TOTAL DURATION: {}'.format(total_time)
    print 'NUM_TRIALS: {}'.format(num_trials)

if __name__ == '__main__':
    main()
