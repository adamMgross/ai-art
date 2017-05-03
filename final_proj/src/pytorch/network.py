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
from collections import namedtuple
from itertools import count
from copy import deepcopy
import time

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
nEpochs = 1
learning_rate = 0.1
batch_size = 20

#USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
if USE_CUDA:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

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

    training_ds = datasets.ImageFolder(image_folder, training_transforms)
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
                      num_workers=4)

    test_loader = DataLoader(testing_ds,
                      batch_size=1,
                      sampler=test_sampler,
                      num_workers=1)

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

    for epoch in range(nEpochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        top1 = test(test_loader, model, criterion)
        print('Epoch: [{0}]\tAccuracy: {1}'.format(epoch,top1))

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
            target = target.cuda()

        input_var = var(input)
        target_var = var(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [%d][%d/%d]\t'
                  'Time %.3f (avg: %.3f)\t'
                  'Data %.3f (avg: %.3f)\t'
                  'Loss %.3f (avg: %.3f)\t'
                  'Prec@1 %.3f (avg: %.3f)\t'
                  % (epoch, i, len(train_loader),
                  batch_time.val, batch_time.avg,
                  data_time.val, data_time.avg,
                  losses.val, losses.avg,
                  top1.val, top1.avg))

def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        if USE_CUDA:
            target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        print(prec1)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % 10 == 0:
            print('Epoch: [%d][%d/%d]\t'
                  'Time %.3f (avg: %.3f)\t'
                  'Data %.3f (avg: %.3f)\t'
                  'Loss %.3f (avg: %.3f)\t'
                  'Prec@1 %.3f (avg: %.3f)\t'
                  % (epoch, i, len(train_loader),
                  batch_time.val, batch_time.avg,
                  data_time.val, data_time.avg,
                  losses.val, losses.avg,
                  top1.val, top1.avg))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    print(output)
    print(target)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
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
