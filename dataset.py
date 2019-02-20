import os 
import numpy as np 
from scipy.misc import imsave, imresize


def load_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels


def load_test_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels