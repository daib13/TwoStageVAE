from __future__ import absolute_import, division, print_function

import os.path, sys, tarfile
import numpy as np
from scipy import linalg
from six.moves import range, urllib
import tensorflow as tf
import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy.misc import imread
import urllib

import pathlib

import torch
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from dataset import load_test_dataset

import matplotlib.pyplot as plt

# =================================================================================
# tensorflow fid score

def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3
#-------------------------------------------------------------------------------


def get_activations_tf(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


# =================================================================================
# pytorch fid score

def get_activations_pt(images, model, batch_size=64, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def fid_score(codes_g, codes_r, eps=1e-6):
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d
    
    mn_g = codes_g.mean(axis=0)
    mn_r = codes_r.mean(axis=0)

    cov_g = np.cov(codes_g, rowvar=False)
    cov_r = np.cov(codes_r, rowvar=False)

    covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
    if not np.isfinite(covmean).all():
        cov_g[range(d), range(d)] += eps
        cov_r[range(d), range(d)] += eps 
        covmean = linalg.sqrtm(cov_g.dot(cov_r))

    score = np.sum((mn_g - mn_r) ** 2) + (np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
    return score 


def preprocess_fake_images(fake_images, norm=False):
    if np.shape(fake_images)[-1] == 1:
        fake_images = np.concatenate([fake_images, fake_images, fake_images], -1) 

    print('norm = ', norm)
    if norm:
        for j in range(np.shape(fake_images)[0]):
            fake_images[j] = (fake_images[j] - np.min(fake_images[j])) / (np.max(fake_images[j] - np.min(fake_images[j])))
    fake_images *= 255
    return fake_images[0:10000]


def preprocess_real_images(real_images):
    if np.shape(real_images)[-1] == 1:
        real_images = np.concatenate([real_images, real_images, real_images], -1) 
    real_images = real_images.astype(np.float32)
    return real_images


def check_or_download_inception():
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    model_file = 'classify_image_graph_def.pb'
    if not os.path.exists(model_file):
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb')
    return str(model_file)


def evaluate_fid_score(fake_images, dataset, root_folder, norm=True):
    real_images, _, _ = load_test_dataset(dataset, root_folder)
    np.random.shuffle(real_images)
    real_images = real_images[0:10000]
    real_images = preprocess_real_images(real_images)
    fake_images = preprocess_fake_images(fake_images, norm)

    inception_path = check_or_download_inception()

    create_inception_graph(inception_path)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print('calculating tf features...')
    real_out = get_activations_tf(real_images, sess)
    fake_out = get_activations_tf(fake_images, sess)
    fid_result = fid_score(real_out, fake_out)

    return fid_result