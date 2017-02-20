import os
import random
from subprocess import call
from time import sleep

surrealists = ['../imgs/surrealist/' + fname for fname in os.listdir('../imgs/surrealist/')]
impressionists = ['../imgs/impressionist/' + fname for fname in os.listdir('../imgs/impressionist/')]
fpaths = surrealists + impressionists

for fpath in fpaths:
    if '../' not in fpath:
        fpaths.remove(fpath)

random.shuffle(fpaths)
test_chop = len(fpaths)/5
validation_chop = len(fpaths)*4/15
training_chop = len(fpaths)*8/15
training = fpaths[:training_chop]
validation = fpaths[training_chop:training_chop + validation_chop]
test = fpaths[-test_chop:]

for training_example in training:
    sleep(0.01)
    genre = training_example.split('/')[-2]
    name = training_example.split('/')[-1]
    new_name = '{}.{}'.format(genre, name)
    print training_example
    call('cp {} ../examples/training/{}'.format(training_example, new_name).split(' '))

for validation_example in validation:
    sleep(0.01)
    genre = validation_example.split('/')[-2]
    name = validation_example.split('/')[-1]
    new_name = '{}.{}'.format(genre, name)
    call('cp {} ../examples/validation/{}'.format(validation_example, new_name).split(' '))

for test_example in test:
    sleep(0.01)
    genre = test_example.split('/')[-2]
    name = test_example.split('/')[-1]
    new_name = '{}.{}'.format(genre, name)
    call('cp {} ../examples/test/{}'.format(test_example, new_name).split(' '))
