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

image_folder = '../../imgs'
nEpochs = 10
learning_rate = 0.1
batch_size = 20

USE_CUDA = torch.cuda.is_available()
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
    length = dataset.real_length()
    indices = list(range(1,length))

    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)

    if type(test_size) is float:
        split = floor(test_size * length)
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
            T.RandomSizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
    ])

    testing_transforms = T.Compose([
            T.Scale(256),
            T.CenterCrop(224),
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
    model = M.vgg19(pretrained=True)
    for param in model.parameters():
        # freeze all the layers
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 2) # assuming that the fc7 layer has 512 neurons, otherwise change it
    #model.classifier._modules['6'] = nn.Linear(4096,2)

    model.cuda() if USE_CUDA else pass

    train_loader, test_loader = load_data(image_folder)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda() if USE_CUDA else nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                            momentum=0.9,
                            weight_decay=1e-4)

    for epoch in range(nEpochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        top1 = test(test_loader, model, criterion)
        print(Epoc)

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

        #target = target.cuda()
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
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

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

class AgentQLearn():
	def __init__(self, env, curiosity=1, learning_rate=0.01, discount_rate = 0.9):
		self.env = env
		self.curiosity = curiosity
		self.discount_rate = discount_rate

		# Initializes DQN and replay memory
		self.mem = ReplayMemory(capacity=10000)
		self.net = DQN(self.env.dims())
		if torch.cuda.is_available():
			self.net = self.net.cuda()

		# using Huber Loss so that we're not as sensitive to outliers
		self.criterion = nn.SmoothL1Loss()
		if torch.cuda.is_available():
			self.criterion = self.criterion.cuda()
		#self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1E-4)

	def learn(self, nEpochs=3000, mini_batch_size=50, nSteps=1000, test=True, nTestMazes=100, end_curiosity=0.1):
		avg_rewards = []
		success_rates = []

		start = time.time()
		for epoch in range(nEpochs):
			self.env = self.env.new_maze()
			state = self.env.observe()


			if (epoch + 100) % 150 == 0:
				self.curiosity = max(self.curiosity-0.1, end_curiosity)


			'''
			if epoch == 50:
				self.curiosity -= 0.1
			'''


			'''
			initial_state = state.copy()
			for flag in initial_state:
				print(flag)
			initial_q_values = self.net(to_var(initial_state, volatile=True))
			print(initial_q_values)
			'''

			restart = False
			for step in range(nSteps):
				# if we've reached the goal state in current maze,
				# restart with new random maze
				if restart:
					self.env = self.env.new_maze()
					state = self.env.observe()
					restart = False

				# get next action; either from Q (network output) or exploration
				action = self.policy(state)
				reward, square = self.env.act(action)
				next_state = self.env.observe()

				# state is terminal if trap or goal
				# note: for learning purposes, learning still happens on same maze
				# after trap is encountered
				isNotTerminal = not (self.env.isTrap(square) or self.env.isGoal(square))
				self.mem.push(state,action,next_state,isNotTerminal,reward)

				'''
				if self.env.isTrap(square) or self.env.isGoal(square):
					self.mem.push(state,action,None,reward)
				else:
					self.mem.push(state,action,next_state,reward)
				'''

				if len(self.mem) >= mini_batch_size:
					# Sample mini-batch transitions from memory
					batch = self.mem.sample(mini_batch_size)
					state_batch = np.array([trans.state for trans in batch])
					# TODO may have to change action format to be ndarray
					action_batch = np.array([trans.action for trans in batch])
					reward_batch = np.array([trans.reward for trans in batch])
					next_state_batch = np.array([trans.next_state for trans in batch])
					isNotTerminal_batch = np.array([int(trans.isNotTerminal) for trans in batch])

					# is 0 if state is terminal (trap or goal)
					non_terminal_mask = to_var(isNotTerminal_batch)

					'''
					print(next_state_batch)
					print(non_final_mask)
					sys.exit(0)
					'''

					'''
					# collects all next states that aren't terminal to be fed into model
					# volatile so that grad isn't calculated w.r.t. to this feed-forward
					non_terminal_next_states = to_var(np.array([s for s in next_state_batch
												if s is not None]),
												volatile=True)
					next_q_values = to_var(np.zeros((mini_batch_size,)))
					next_q_values[non_final_mask] = self.net(non_final_next_states).max(1)[0]
					notGoalFunc = lambda s: int(not self.env.isGoalState(s))
					next_NOTGoal_batch = np.array([notGoalFunc(s) for s in next_state_batch])
					'''

					# Forward + Backward + Optimize
					self.net.zero_grad()
					cur_input = to_var(state_batch, volatile=False)

					if torch.cuda.is_available():
						cur_input = cur_input.cuda()

					# feeds-forward state batch and collects the action outputs corresponding to the action batch
					q_values = self.net(cur_input).gather(1,to_var(action_batch).long().view(-1, 1))

					# Make volatile so that computational graph isn't affected
					# by this batch of inputs
					next_input = to_var(next_state_batch, volatile=True)
					if torch.cuda.is_available():
						next_input = next_input.cuda()

					next_max_q_values = self.net(next_input).max(1)[0].float()

					'''
					print(next_q_values.data)
					print(next_q_values.data.max(1))
					print(next_q_values.data.max(1)[0])
					'''

					# change volatile flag back to false so that weight gradients will be calculated
					next_max_q_values.volatile = False

					# only includes future q values if neither in goal state nor trap state
					target = to_var(reward_batch) + self.discount_rate * next_max_q_values * non_terminal_mask
					loss = self.criterion(q_values, target)

					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

				if self.env.isGoal(square):
					assert self.env.isGoalState(next_state)
					restart = True

				state = next_state

			print('\nFinished epoch %d....' % epoch)

			if test:
				nSuccesses, avg_reward = self.test(nTestMazes)
				print('Success rate: %d / %d\nAvg. Reward: %f'
					% (nSuccesses, nTestMazes, avg_reward))

				avg_rewards.append(avg_reward)
				success_rates.append(float(nSuccesses) / nTestMazes)

		end = time.time()

		print('\nLearning Time: %f' % (end - start))

		return avg_rewards, success_rates

		'''
		final_q_values = self.net(to_var(initial_state, volatile=True))
		print(final_q_values)
		'''
	def save_model(self,fname):
		torch.save(self.net.cpu().state_dict(),fname)

	def load_model(self,fname):
		state_dict = torch.load(fname)
		self.net.load_state_dict(state_dict)

	# state is state as given by env; not necessarily in suitable format for network input
	def policy(self,state,can_explore=True):
		if can_explore and self.explore():
			action = random.randrange(self.env.actions)
		else:
			# Q-values from running network on current state
			# Brackets around state give it batch dimension (of size 1)
			# q_values = self.net(batch_to_input([state]))
			v = to_var(state, volatile=True)
			if torch.cuda.is_available():
				v = v.cuda()

			q_values = self.net(v)
			# chooses best action
			action = q_values.data.max(1)[1][0,0]

		return int(action)

	def explore(self):
		return random.random() < self.curiosity

	def test(self, nMazes, debug=False):
		avg_reward = 0.0
		nSuccesses = 0
		max_nSteps = self.env.xbound * self.env.ybound
		for test in range(nMazes):
			env = self.env.new_maze()
			state = env.observe()
			cum_reward = 0.0

			if debug:
				print('Initial State:')
				env.print_state()

			for step in range(max_nSteps):

				action = self.policy(state,can_explore=False)
				reward, square = env.act(action)
				cum_reward += reward

				if debug:
					print('\n-------------------------------\n')
					print('Step %d:\nPrev Action: ' % step, end='')
					env.print_action(action)
					print()
					env.print_state()
					print('Cumulative reward: %f' % cum_reward)

					input('Press enter....')

				if env.isGoal(square):
					nSuccesses += 1
					break

				if env.isTrap(square):
					break

				state = env.observe()

			avg_reward += cum_reward

		return nSuccesses, avg_reward / nMazes




	# TODO can't enumerate over states
	'''
	def print_policy(self, start):
		self.env.set_position(start)
		s = self.env.observe()
		reward = 0
		print(["{0:10}".format(i) for i in self.env.actions_name])
		for s in range(self.env.states):
			self.env.print_state(s)
			action = self.Q[s].index(max(self.Q[s]))
			self.env.print_action(action)
			print('')
			print(["{0:10.2f}".format(i) for i in self.Q[s]])
	'''
# transforms numpy ndarray to Tensor
def to_tensor(ndarray):

	return torch.from_numpy(ndarray).float()

# transforms numpy ndarray to Variable for input into network
def to_var(ndarray, volatile=False):
	tensor = to_tensor(ndarray)
	if tensor.dim() == 3:
		tensor.unsqueeze_(0)

	v = var(tensor, volatile=volatile)
	if torch.cuda.is_available():
		v = v.cuda()

	return v

# TODO function that transforms given batch into format that aligns with
# input to network
def batch_to_input(batch):

	return to_tensor(np.array(batch), volatile=True)
