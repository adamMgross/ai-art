'''
modified from http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
and https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/12%20-%20Deep%20Q%20Network/dqn13.py
'''

from __future__ import print_function
import math
import random
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple
from itertools import count
from copy import deepcopy
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as var
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as T
import torchvision.models as M
import torchvision.datasets as datasets

import inception

image_folder = '../../imgs'
nEpochs = 100
learning_rate = 0.01
batch_size = 5

USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False
if USE_CUDA:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def create_plots(results,name):
    folder = os.path.join(os.getcwd(),name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    training_acc, testing_acc = results
    epochs = range(nEpochs)

    ys = [training_acc, testing_acc]
    scatters = []

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    for y, c in zip(ys, colors):
        scatters.append(plt.scatter(epochs, y, color=c, s=10))

    plt.legend(tuple(scatters),
           ('Training', 'Testing'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)

    plt.title('Accuracies vs. Epochs')
    #plt.show()
    plt.draw()
    fig = plt.gcf()
    fig.savefig(os.path.join(folder,'accs_no_aug_lr-001_bs-05.png'))
    plt.clf()

def train_test_split(dataset, test_size = 0.25, shuffle = False, random_seed = 0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and testing set.

    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    length = len(dataset)
    indices = list(range(1,length))

    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)

    if type(test_size) is float:
        split = math.floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]

# accepts path to root images directory
# returns two DataLoaders --- one for training and another for testing
def load_data(image_folder):

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    training_transforms = T.Compose([
            T.RandomSizedCrop(299),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
    ])

    testing_transforms = T.Compose([
            T.Scale(299),
            T.CenterCrop(299),
            T.ToTensor(),
            normalize,
    ])

    training_ds = datasets.ImageFolder(image_folder, testing_transforms)
    testing_ds = datasets.ImageFolder(image_folder, testing_transforms)

    # indices randomly selected from all instances in dataset to be used
    # for either training or testing
    train_indices, test_indices = train_test_split(training_ds, shuffle=True)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Both dataloader loads from the same dataset but with different indices
    train_loader = DataLoader(training_ds,
                      batch_size=20,
                      sampler=train_sampler,
                      num_workers=4,
                      pin_memory=True)

    test_loader = DataLoader(testing_ds,
                      batch_size=1,
                      sampler=test_sampler,
                      num_workers=1,
                      pin_memory=True)

    return train_loader, test_loader

def main():
    # model = M.vgg19(pretrained=True)
    # model = M.inception_v3(pretrained=True,aux_logits=False)

    # fuck this aux_logit bullshit; redefined inception.py to ignore this
    # transform_input identical to normalize Transform, which we already do
    model = inception.inception_v3(pretrained=True,transform_input=False)
    for param in model.parameters():
        # freeze all the layers
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(2048, 2) # assuming that the fc7 layer has 2048 neurons, otherwise change it
    #model.classifier._modules['6'] = nn.Linear(4096,2)

    # hack to get Inception_v3 to work (if aux_logits=True)
    #model.AuxLogits = M.inception.InceptionAux(768,2)

    if USE_CUDA:
        model = model.cuda()

    train_loader, test_loader = load_data(image_folder)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda() if USE_CUDA else nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.fc.parameters(), learning_rate,
                            momentum=0.9,
                            weight_decay=1e-4)

    train_start = time.time()

    train_accs = []
    test_accs = []
    for epoch in range(nEpochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_acc = train(train_loader, model, criterion, optimizer, epoch)
        test_acc = test(test_loader, model, criterion, epoch)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

    train_end = time.time()

    print('Total Training/Testing Time: %.3f' % (train_end - train_start))

    results = (train_accs, test_accs)
    create_plots(results,'Results')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()

        input_var = var(input)
        target_var = var(target)

        #print('okay0')
        # compute output
        output = model(input_var)
        #output = nn.Softmax()(output)
        #print('okay6')
        loss = criterion(output, target_var)
        #print('okay7')

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #print('okay8')
        loss.backward()
        #print('okay9')
        optimizer.step()
        #print('okay10')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [%d]\n'
          '\tTraining Set Performance:\n'
          '\t\tTime %.3f (avg: %.3f)\n'
          '\t\tData %.3f (avg: %.3f)\n'
          '\t\tLoss %.3f (avg: %.3f)\n'
          % (epoch,
          batch_time.val, batch_time.avg,
          data_time.val, data_time.avg,
          losses.val, losses.avg), end='')
    print('\n\t\tAccuracy {} (avg: {})\t'.format(top1.val[0],top1.avg[0]))

    return top1.avg[0]


def test(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    #losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        data_time.update(time.time() - end)

        if USE_CUDA:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        #output = nn.Softmax()(output)
        #loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        #print(prec1)
        #losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    '''
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    '''
    print('\n\tTest_Accuracy: {}'.format(top1.avg[0]))

    return top1.avg[0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #print(output)
    #print(target)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    #print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    #print(res)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
