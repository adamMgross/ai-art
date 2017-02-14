from collections import OrderedDict, namedtuple
from Queue import Queue
import random
from math import log
from copy import copy
import operator

# container for all relevant info needed at each node
NodeInfo = namedtuple("NodeInfo","S attrs label a_vals path tree")

# container for all three possible errors upon evaluation
Errors = namedtuple("Errors","training validation testing")

# container for results from each iteration
Results = namedtuple("Results","errors nNodes iteration")

# proportion of training instances to be used for cross_validation
cv_ratio = 0.2

class Tree(object):
    """Represents decision tree task. Contains root node"""
    def __init__(self, training_set, label=None, perform_cv=False):
        self.validation = []
        self.perform_cv = perform_cv

        if perform_cv:
            training_size = len(training_set)
            cv_size = int(training_size * cv_ratio)

            cv_indices = random.sample(range(training_size),cv_size)
            self.training = []

            for i in range(training_size):
                if i in cv_indices:
                    self.validation.append(training_set[i])
                else:
                    self.training.append(training_set[i])
        else:
            self.training = training_set

        all_attributes = training_set[0].keys()
        try:
            all_attributes.remove('url')
        except ValueError:
            pass

        # determines dictionary of all possible values for each attribute
        self.attribute_values = {attr:set([]) for attr in all_attributes}
        for instance in self.training:
            for attr in all_attributes:
                # print attr
                # print instance[attr]
                # print self.attribute_values[attr]
                self.attribute_values[attr].add(instance[attr])

        # default: last attribute is label
        if label is None:
            self.label = all_attributes[-1]
            self.attrs = all_attributes[:-1]
        else:
            self.label = label
            self.attrs = all_attributes[:]
            self.attrs.remove(label)

        self.initial_info = NodeInfo(self.training, self.attrs,
                self.label, self.attribute_values, OrderedDict(), self)

        # number of non-leaf nodes
        # updated when a node is expanded (+1) or pruned (-1)
        self.nNodes = 0
        # also updated when a node is expanded (children added)
        self.expand_queue = Queue()

        # incremented whenever a node is expanded or pruned
        self.nIterations = 0

        # those to be considered when pruning
        # node is added when it is expanded
        # node is added when all of its children have been pruned
        # node is removed when one of its children is expanded
        self.pruning_candidates = []

        # extra housekeeping
        self.leaves = []

        self.root = Node(None,self.initial_info)
        self.expand_queue.put(self.root)

    # builds one iteration (i.e. builds one new node)
    # returns 0 if one node was successfully added to the tree
    # else returns -1 if no node can be added (no more attrs or all pure splits)
    def expand_tree(self):
        if self.expand_queue.empty():
            return -1

        node = self.expand_queue.get()
        ret_code = node.expand()
        while not self.expand_queue.empty() and ret_code == -1:
            node = self.expand_queue.get()
            ret_code = node.expand()

        return ret_code

    # prunes one iteration (i.e. converts one node into leaf node)
    # returns 0 if one node was successfully pruned from the tree
    # else returns -1 if no node should be pruned
    #   - this occurs when no pruning improves error on the validation set
    def prune(self):
        node_to_prune = None
        best_error = self.eval_set(self.validation)

        print('Validation error without pruning: %f\n' % best_error)

        # count = 0
        for node in self.pruning_candidates:
            # print node.display()
            node.prune()
            error = self.eval_set(self.validation)
            # print('Validation error with pruning: %f' % error)
            # count += 1
            node.unprune()

            # if error < best_error or abs(error - best_error) < 0.00001:
            if error < best_error:
                node_to_prune = node
                best_error = error

            else:
                self.pruning_candidates.remove(node)

        # should stop pruning
        if node_to_prune is None:
            return -1

        node_to_prune.prune()

        # print('pruning: %s' % node_to_prune.display())
        # self.display('test_paths_with_prune')

        self.nIterations += 1

        #raw_input('enter...')

        return 0

    def eval_set(self,set):
        error = 0
        for instance in set:
            if not self.root.eval(instance) == instance[self.label]:
                error += 1

        return error / float(len(set))

    def evaluate(self,test_set):
        training_error = self.eval_set(self.training)
        testing_error = self.eval_set(test_set)

        validation_error = 0
        if self.perform_cv:
            validation_error = self.eval_set(self.validation)

        errors = Errors(training_error, validation_error, testing_error)
        return Results(errors, self.nNodes, self.nIterations)

    # creates a list of all possible paths to leaf nodes in the tree
    # and the resulting label
    def display(self,fname=None):
        paths = []

        count = 0
        for leaf in self.leaves:
            string = 'Path %d: \nroot\n' % count
            for attr, value in leaf.info.path.items():
                string += ' --> (%s\t: %s)\n' % (str(attr), str(value))
            string += 'Label: %s\n\n' % str(leaf.vote)
            paths.append(string)
            count += 1

        if not fname is None:
            with open(fname, 'wb') as f:
                f.writelines(paths)

        return paths


class Node(object):
    """docstring for Node."""
    def __init__(self, parent, info):
        self.parent = parent
        self.info = info

        if self.parent is None:
            self.prev_attr = None
            self.prev_value = None
        else:
            self.prev_attr, self.prev_value = self.info.path.items()[-1]

        # self.vote to be instantiated when calculating the cur_entropy
        self.vote = None
        self.cur_entropy = self.entropy(self.info.S,determine_vote=True)

        # cur_entropy == 0
        # i.e. all instances have same label in this node
        self.isPure = abs(self.cur_entropy) < 0.00001

        # will change when self.expand() is called and
        # successfully expands, which occurs when both:
        #   not self.isPure
        #   number of remaining attrs is non-zero
        self.isLeaf = True

        # if this node gets expanded, self will be removed
        self.info.tree.leaves.append(self)

        # these will be instantiated when self.expand() is called
        self.attr = None
        self.children = None

        # used to determine whether or not this node
        # should be candidate (e.g. all its children are leaves)
        # instantiated when self.expand() is called
        self.children_are_leaves = None

        # print('Created node: %s' % self.display())

    def info_gain(self,attr):
        S_cardinality = len(self.info.S)
        subsets = {val:[] for val in self.info.a_vals[attr]}
        # print('S:')
        # print self.info.S
        # print('attr: %s' % attr)
        for instance in self.info.S:
            subsets[instance[attr]].append(instance)

        next_entropy = 0.0
        for value, subset in subsets.items():
            # print('value: %s' % value)
            # print('subset:')
            # print subset
            subset_cardinality = len(subset)
            next_entropy += subset_cardinality * self.entropy(subset)

        next_entropy /= S_cardinality

        return self.cur_entropy - next_entropy

    # calculates entropy
    # if determine_vote is True, then self.vote is updated to reflect plurality
    # of labels in set S of instances
    def entropy(self,set,determine_vote=False):
        cardinality = len(set)
        if cardinality == 0:
            return 0
        #print self.info.path
        label = self.info.label
        counts = {val:0 for val in self.info.a_vals[label]}
        for instance in set:
            counts[instance[label]] += 1

        if determine_vote:
            sorted_counts = sorted(counts.items(), key=operator.itemgetter(1))
            # sets self.vote to label value that is most representative
            # of instances in this node
            self.vote = sorted_counts[-1][0]

        ent = 0.0
        for value, count in counts.items():
            p_count = float(count) / cardinality
            if p_count != 0:
                ent += p_count * log(p_count,2)

        return -ent

    def eval(self,instance):
        # this is a leaf node
        if self.isLeaf:
            return self.vote

        val = instance[self.attr]
        try:
            return self.children[val].eval(instance)

        # val not seen as value for self.attr in any training instance
        # in this case we "guess" --- choose self.vote
        except KeyError:
            # print('unpreviously seen value %s for attribute %s' %
            #        (val, self.attr))
            return self.vote

    def expand(self):
        nAttrs = len(self.info.attrs)
        # print nAttrs
        # print self.isPure
        if nAttrs == 0 or self.isPure:
            # don't expand; maintain self.isLeaf
            return -1
        else:
            # definitely will expand now
            # thus no longer leaf node
            self.isLeaf = False
            self.info.tree.leaves.remove(self)
            self.info.tree.pruning_candidates.append(self)
            # informs our parent that we're no longer a leaf node
            if not self.parent is None:
                self.parent.children_are_leaves[self.prev_value] = False

            # as such we now try to remove parent as possible
            # pruning candidate
            try:
                self.info.tree.pruning_candidates.remove(self.parent)
            except ValueError:
                pass

            self.info.tree.nNodes += 1
            self.info.tree.nIterations += 1

            max_gain = -1
            gains = {a : 0 for a in self.info.attrs}
            for attr in self.info.attrs:
                gain = self.info_gain(attr)
                gains[attr] = gain
                if gain > max_gain:
                    self.attr = attr
                    max_gain = gain

            # for tup in gains.items():
            #     print tup

            subsets = {val:[] for val in self.info.a_vals[self.attr]}
            for instance in self.info.S:
                # print instance
                # print self.attr
                # print instance[self.attr]
                # print subsets
                subsets[instance[self.attr]].append(instance)

            self.children = {}
            self.children_are_leaves = {}
            for value, subset in subsets.items():
                child_attrs = copy(self.info.attrs)
                child_attrs.remove(self.attr)

                child_path = copy(self.info.path)
                child_path.update({self.attr    :   value})

                child_info = NodeInfo(subset, child_attrs, self.info.label,
                        self.info.a_vals, child_path, self.info.tree)

                self.children.update({value :   Node(self,child_info)})
                self.children_are_leaves.update({value :   True})

            # put the new children nodes on the queue to be expanded
            for value, node in self.children.items():
                self.info.tree.expand_queue.put(node)

            # print('\nExpanded node: %s\n' % self.display())

            # successfully expanded
            return 0

    def display(self):
        return '({prev_attr: prev_value}, attr) = ({%s: %s}, %s)' % (str(self.prev_attr), str(self.prev_value), str(self.attr))

    def prune(self):
        self.isLeaf = True
        self.info.tree.nNodes -= 1

        # have to now add node to list of leaves
        self.info.tree.leaves.append(self)

        # have to remove each child from list of leaves
        for child in self.children.values():
            try:
                self.info.tree.leaves.remove(child)
            except ValueError:
                pass

        if not self.parent is None:
            self.parent.children_are_leaves[self.prev_value] = True

            # if as a result of this pruning, all children of our
            # parent are now leaves, then our parent is once again
            # a pruning candidate
            if all(v for v in self.parent.children_are_leaves.values()):
                self.info.tree.pruning_candidates.append(self.parent)

    def unprune(self):
        self.isLeaf = False
        self.info.tree.nNodes += 1

        # no longer a leaf node
        try:
            self.info.tree.leaves.remove(self)
        except ValueError:
            pass

        # have to add each child from list of leaves
        for child in self.children.values():
            self.info.tree.leaves.append(child)

        if not self.parent is None:
            self.parent.children_are_leaves[self.prev_value] = False

            # now that we've unpruned this node, we know our parent is
            # once again no longer a pruning candidate
            try:
                self.info.tree.pruning_candidates.remove(self.parent)
            except ValueError:
                pass
