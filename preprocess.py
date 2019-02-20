import numpy as np 
import pickle
from mnist import MNIST
import os
from scipy.misc import imread, imresize, imsave
import pickle
ROOT_FOLDER = './data'


def load_mnist_data(flag='training'):
    mndata = MNIST(os.path.join(ROOT_FOLDER, 'mnist'))
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images)
    images_array = np.concatenate(images_array, 0)
    return images_array.astype(np.uint8)


def load_fashion_data(flag='training'):
    mndata = MNIST(os.path.join(ROOT_FOLDER, 'fashion'))
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images)
    images_array = np.concatenate(images_array, 0)
    return images_array.astype(np.uint8)


def load_cifar10_data(flag='training'):
    if flag == 'training':
        data_files = ['data/cifar10/cifar-10-batches-py/data_batch_1', 'data/cifar10/cifar-10-batches-py/data_batch_2', 'data/cifar10/cifar-10-batches-py/data_batch_3', 'data/cifar10/cifar-10-batches-py/data_batch_4', 'data/cifar10/cifar-10-batches-py/data_batch_5']
    else:
        data_files = ['data/cifar10/cifar-10-batches-py/test_batch']
    x = []
    for filename in data_files:
        img_dict = unpickle(filename)
        img_data = img_dict[b'data']
        img_data = np.transpose(np.reshape(img_data, [-1, 3, 32, 32]), [0, 2, 3, 1])
        x.append(img_data)
    x = np.concatenate(x, 0)
    return x.astype(np.uint8)


def load_celeba_data(flag='training', side_length=None, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[45:173,25:153]
        if side_length is not None:
            img = imresize(img, [side_length, side_length])
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)


def load_celeba140_data(flag='training', side_length=None, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[39:179,19:159]
        if side_length is not None:
            img = imresize(img, [side_length, side_length])
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)


# Center crop 140x140 and resize to 64x64
# Consistent with the preporcess in WAE [1] paper
# [1] Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein auto-encoders. International Conference on Learning Representations, 2018.
def preprocess_celeba140():
    x_val = load_celeba140_data('val', 64)
    if not os.path.exists(os.path.join('data', 'celeba140')):
        os.mkdir(os.path.join('data', 'celeba140'))
    np.save(os.path.join('data', 'celeba140', 'val.npy'), x_val)
    x_test = load_celeba140_data('test', 64)
    np.save(os.path.join('data', 'celeba140', 'test.npy'), x_test)
    x_train = load_celeba140_data('training', 64)
    np.save(os.path.join('data', 'celeba140', 'train.npy'), x_train)

# Center crop 128x128 and resize to 64x64
def preprocess_celeba():
    x_val = load_celeba_data('val', 64)
    np.save(os.path.join('data', 'celeba', 'val.npy'), x_val)
    x_test = load_celeba_data('test', 64)
    np.save(os.path.join('data', 'celeba', 'test.npy'), x_test)
    x_train = load_celeba_data('training', 64)
    np.save(os.path.join('data', 'celeba', 'train.npy'), x_train)

def preprocess_mnist():
    x_train = load_mnist_data('training')
    x_train = np.reshape(x_train, [60000, 28, 28, 1])
    np.save(os.path.join('data', 'mnist', 'train.npy'), x_train)
    x_test = load_mnist_data('testing')
    x_test = np.reshape(x_test, [10000, 28, 28, 1])
    np.save(os.path.join('data', 'mnist', 'test.npy'), x_test)


def preporcess_cifar10():
    x_train = load_cifar10_data('training')
    np.save(os.path.join('data', 'cifar10', 'train.npy'), x_train)
    x_test = load_cifar10_data('testing')
    np.save(os.path.join('data', 'cifar10', 'test.npy'), x_test)


def preprocess_fashion():
    x_train = load_fashion_data('training')
    x_train = np.reshape(x_train, [60000, 28, 28, 1])
    np.save(os.path.join('data', 'fashion', 'train.npy'), x_train)
    x_test = load_fashion_data('testing')
    x_test = np.reshape(x_test, [10000, 28, 28, 1])
    np.save(os.path.join('data', 'fashion', 'test.npy'), x_test)


def preprocess_imagenet():
    for i in range(1, 11):
        train_file = 'train_data_batch_' + str(i)
        fid = open(os.path.join('data', 'imagenet', train_file), 'rb')
        x_batch = pickle.load(fid)
        fid.close()
        x_batch = x_batch['data']
        x_batch = np.reshape(x_batch, [np.shape(x_batch)[0], 3, 64, 64])
        x_batch = np.transpose(x_batch, [0, 2, 3, 1])
        np.save(os.path.join('data', 'imagenet', 'train' + str(i-1) + '.npy'), x_batch)
    fid = open(os.path.join('data', 'imagenet', 'val_data'), 'rb')
    x_batch = pickle.load(fid)
    fid.close()
    x_batch = x_batch['data']
    x_batch = np.reshape(x_batch, [np.shape(x_batch)[0], 3, 64, 64])
    x_batch = np.transpose(x_batch, [0, 2, 3, 1])
    np.save(os.path.join('data', 'imagenet', 'test.npy'), x_batch)
    


if __name__ == '__main__':
    preprocess_celeba()
    preprocess_celeba140()
#    preprocess_imagenet()
    preprocess_mnist()
    preprocess_fashion()
    preporcess_cifar10()
