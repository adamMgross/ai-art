import mnist_loader
import network
import sys
from time import time


sys.stdout = open('./record.txt', 'w')

#Network([num_inputs, num_hidden_layer, num_outputs])
#net.SGD(examples. epochs, mini-batch size, learning rate, test_data)

def log(hidden_layers, hidden_layer_units, mini_batch_size, learning_rate):
    print '====================================='
    print 'CONDUCTING TRIAL'
    print '#HIDDEN_LAYERS={}'.format(hidden_layers)
    print '#UNITS_IN_HIDDEN_LAUYER={}'.format(hidden_layer_units)
    print 'MINI_BATCH_SIZE={}'.format(mini_batch_size)
    print 'LEARNING_RATE={}'.format(learning_rate)

    
def trial(hidden_layers, hidden_layer_units, learning_rate, mini_batch_size):
    log(hidden_layers, hidden_layer_units, learning_rate, mini_batch_size)
        topology = [784]
        for i in range(hidden_layers):
            topology.append(hidden_layer_units)
        topology.append(10)
        net = network.Network(topology)
        t1 = time()
        net.SGD(training_data,
                30,
                mini_batch_size,
                learning_rate,
                test_data)
        t2 = time()
        elapsed = t2-t1
        print 'ELAPSED TIME: {} seconds'.format(elapsed)
        print '====================================='
        return elapsed
    
def run(training_data, test_data):
    hidden_layer_trials = [0, 1, 2]
    hidden_layer_units_trials = [10, 30, 50]
    learning_rate_trials = [0.01, 3, 30]
    mini_batch_size_trials = [1, 10, 100]
    standard = [1, 30, 3.0, 10]
    total_time = 0
    num_trials = 0
    
    for hidden_layer_amt in hidden_layer_trials:
        variables = standard[:]
        variables[0] = hidden_layer_amt
        total_time += trial(variables[0], variables[1], variables[2], variables[3])
        num_trials += 1
        
    for hidden_layer_units_amt in hidden_layer_units_trials:
        variables = standard[:]
        variables[1] = hidden_layer_units_amt
        total_time += trial(variables[0], variables[1], variables[2], variables[3])
        num_trials += 1
        
    for learning_rate_amt in learning_rate_trials:
        variables = standard[:]
        variables[2] = learning_rate_amt
        total_time += trial(variables[0], variables[1], variables[2], variables[3])
        num_trials += 1
        
    for mini_batch_size_amt in mini_batch_size_trials:
        variables = standard[:]
        variables[3] = mini_batch_size_amt
        total_time += trial(variables[0], variables[1], variables[2], variables[3])
        num_trials += 1
        
    print '====================================='
    print 'TOTAL DURATION: {}'.format(total_time)
    print 'NUM_TRIALS: {}'.format(num_trials)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
run(training_data, test_data)
