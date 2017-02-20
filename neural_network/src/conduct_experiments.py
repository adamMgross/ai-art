import mnist_loader
import network
import sys

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

def run(training_data, test_data):
    hidden_layer_trials = [0, 1, 2]
    hidden_layer_units_trials = [10, 30, 50]
    learning_rate_trials = [0.01, 3, 30]
    mini_batch_size_trials = [1, 10, 100]
    for hidden_layer_amt in hidden_layer_trials:
        for hidden_layer_units_amt in hidden_layer_units_trials:
            for learning_rate_amt in learning_rate_trials:
                for mini_batch_size_amt in mini_batch_size_trials:
                    log(hidden_layer_amt, hidden_layer_units_amt, mini_batch_size_amt, learning_rate_amt)
                    topology = [784]
                    for i in range(hidden_layer_amt):
                        topology.append(hidden_layer_units_amt)
                    topology.append(10)
                    net = network.Network(topology)
                    net.SGD(training_data,
                            30,
                            mini_batch_size_amt,
                            learning_rate_amt,
                            test_data)
    print '====================================='

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
run(training_data, test_data)
