import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import random, csv

from node import Tree

# proportion of total instances to be used for testing
test_ratio = 0.2

def main():
    tennis()
    #art()

def tennis():
    training, testing = JSON_to_datasets('play_tennis.json')
    label = 'Playtennis'

    folder = 'Tennis'

    directory = os.getcwd() + '\\%s' % folder
    if not os.path.exists(directory):
        os.makedirs(directory)

    solve_without_pruning(training, testing, label, folder)

def art():
    training, testing = JSON_to_datasets('data.json')
    label = 'genre'

    folder = 'Art'

    directory = os.getcwd() + '\\%s' % folder
    if not os.path.exists(directory):
        os.makedirs(directory)

    solve_without_pruning(training, testing, label, folder)
    for i in range(20):
        print i
        solve_with_pruning(training, testing, label, folder, i)

def solve_without_pruning(training, testing, label, folder):
    tree = Tree(training,label=label)
    # Iterations
    x = []

    training_errors = []
    testing_errors = []
    nNodes = []

    ys = [training_errors, testing_errors]

    # build tree iteratively
    ret_code = 0
    while ret_code == 0:
        res = tree.evaluate(testing)

        x.append(res.iteration)
        errors = res.errors

        training_errors.append(errors.training)
        testing_errors.append(errors.testing)
        nNodes.append(res.nNodes)

        ret_code = tree.expand_tree()

    tree.display(folder + '\\paths.txt')

    nNodes_normalized = normalize(np.array(nNodes))
    ys.append(nNodes_normalized)

    scatters = []

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    for y, c in zip(ys, colors):
        scatters.append(plt.scatter(x, y, color=c))

    plt.legend(tuple(scatters),
           ('Training', 'Testing', 'nNodes (normalized)'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)

    plt.title('%s Dataset: Errors and nNodes' % folder)
    #plt.show()
    plt.draw()
    fig = plt.gcf()
    fig.savefig('%s\\no_pruning_plot.png' % folder)
    plt.clf()

    column_labels = ['training_error', 'testing_error', 'nNodes']
    iterations = [column_labels] + [[tr,te,nodes] for tr,te,nodes in zip(training_errors,
            testing_errors, nNodes)]

    write2CSV(iterations,'%s\\no_pruning_data' % folder)

def solve_with_pruning(training, testing, label, folder, run):
    tree = Tree(training,label=label,perform_cv=True)

    directory = os.getcwd() + '\\%s\\run%s' % (folder, str(run))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterations
    x = []

    training_errors = []
    validation_errors = []
    testing_errors = []
    nNodes = []

    ys = [training_errors, validation_errors, testing_errors]

    # build tree iteratively
    ret_code = 0
    while ret_code == 0:
        res = tree.evaluate(testing)

        x.append(res.iteration)
        errors = res.errors

        training_errors.append(errors.training)
        validation_errors.append(errors.validation)
        testing_errors.append(errors.testing)
        nNodes.append(res.nNodes)

        ret_code = tree.expand_tree()

    tree.display(directory + '\\paths_before_pruning.txt')

    # prune tree iteratively
    ret_code = tree.prune()
    while ret_code == 0:
        res = tree.evaluate(testing)

        x.append(res.iteration)
        errors = res.errors

        training_errors.append(errors.training)
        validation_errors.append(errors.validation)
        testing_errors.append(errors.testing)
        nNodes.append(res.nNodes)

        ret_code = tree.prune()

    tree.display(directory + '\\paths_after_pruning.txt')

    nNodes_normalized = normalize(np.array(nNodes))
    ys.append(nNodes_normalized)

    scatters = []

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    for y, c in zip(ys, colors):
        scatters.append(plt.scatter(x, y, color=c))

    plt.legend(tuple(scatters),
           ('Training', 'Validation', 'Testing', 'nNodes (normalized)'),
           scatterpoints=1,
           loc='upper right',
           ncol=4,
           fontsize=8)

    plt.title('%s Dataset: Errors and nNodes' % folder)
    #plt.show()
    plt.draw()
    fig = plt.gcf()
    fig.savefig('%s\\with_pruning_plot.png' % directory)
    plt.clf()

    column_labels = ['training_error', 'validation_error', 'testing_error', 'nNodes']
    iterations = [column_labels] + [[tr,v,te,nodes] for tr,v,te,nodes in zip(training_errors,
            validation_errors, testing_errors, nNodes)]

    write2CSV(iterations,'%s\\with_pruning_data' % directory)

# used to normalize nNodes for display purposes
# borrowed from StackExchange
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def JSON_to_datasets(filename):
    data = []
    with open(filename, 'r') as f:
        data = json.load(f)

    data_size = len(data)
    test_size = int(data_size * test_ratio)
    test_indices = random.sample(range(data_size),test_size)

    training = []
    testing = []

    for i in range(data_size):
        if i in test_indices:
            testing.append(data[i])
        else:
            training.append(data[i])

    return training, testing
    # training = data[:11]
    # testing = data[11:]
    #
    # return training, testing

def write2CSV(iterations,fname):
    with open(fname + '.csv', 'wb') as file:
        wr = csv.writer(file,delimiter=',')
        wr.writerows(iterations)

if __name__ == '__main__':
    main()
