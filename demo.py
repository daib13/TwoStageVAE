import argparse 
import os 
from network.two_stage_vae_model import *
import numpy as np 
import tensorflow as tf 
import math 
import time 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from scipy.misc import imsave, imresize
import pickle
from dataset import load_dataset, load_test_dataset


def main():
    tf.reset_default_graph()
    # exp info
    exp_folder = os.path.join(args.output_path, args.dataset, args.exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model_path = os.path.join(exp_folder, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # dataset
    x, side_length, channels = load_dataset(args.dataset, args.root_folder)
    input_x = tf.placeholder(tf.uint8, [args.batch_size, side_length, side_length, channels], 'x')
    num_sample = np.shape(x)[0]
    print('Num Sample = {}.'.format(num_sample))

    # model
    if args.network_structure != 'Resnet':
        model = eval(args.network_structure)(input_x, args.latent_dim, args.second_depth, args.second_dim)
    else:
        model = Resnet(input_x, args.num_scale, args.block_per_scale, args.depth_per_block, args.kernel_size, args.base_dim, args.fc_dim, args.latent_dim, args.second_depth, args.second_dim)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(exp_folder, sess.graph)
    saver = tf.train.Saver()

    # train model
    iteration_per_epoch = num_sample // args.batch_size 
    if not args.val:
        # first stage
        for epoch in range(args.epochs):
            np.random.shuffle(x)
            lr = args.lr if args.lr_epochs <= 0 else args.lr * math.pow(args.lr_fac, math.floor(float(epoch) / float(args.lr_epochs)))
            epoch_loss = 0
            for j in range(iteration_per_epoch):
                image_batch = x[j*args.batch_size:(j+1)*args.batch_size]
                loss = model.step(1, image_batch, lr, sess, writer, args.write_iteration)
                epoch_loss += loss 
            epoch_loss /= iteration_per_epoch

            print('Date: {date}\t'
                  'Epoch: [Stage 1][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch, args.epochs, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
        saver.save(sess, os.path.join(model_path, 'stage1'))

        # second stage
        for epoch in range(args.epochs2):
            np.random.shuffle(x)
            lr = args.lr2 if args.lr_epochs2 <= 0 else args.lr2 * math.pow(args.lr_fac2, math.floor(float(epoch) / float(args.lr_epochs2)))
            epoch_loss = 0
            for j in range(iteration_per_epoch):
                image_batch = x[j*args.batch_size:(j+1)*args.batch_size]
                loss = model.step(2, image_batch, lr, sess, writer, args.write_iteration)
                epoch_loss += loss 
            epoch_loss /= iteration_per_epoch

            print('Date: {date}\t'
                  'Epoch: [Stage 2][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch, args.epochs2, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
        saver.save(sess, os.path.join(model_path, 'stage2'))
    else:
        saver.restore(sess, os.path.join(model_path, 'stage2'))

    # test dataset 
    x, side_length, channels = load_test_dataset(args.dataset, args.root_folder)
    np.random.shuffle(x)
    x = x[0:10000]

    # reconstruction and generation
    img_recons = model.reconstruct(sess, x)
    img_gens1 = model.generate(sess, 10000, 1)
    img_gens2 = model.generate(sess, 10000, 2)

    img_recons_sample = stich_imgs(img_recons)
    img_gens1_sample = stich_imgs(img_gens1)
    img_gens2_sample = stich_imgs(img_gens2)
    plt.imsave(os.path.join(exp_folder, 'recon_sample.jpg'), img_recons_sample)
    plt.imsave(os.path.join(exp_folder, 'gen1_sample.jpg'), img_gens1_sample)
    plt.imsave(os.path.join(exp_folder, 'gen2_sample.jpg'), img_gens2_sample)


def stich_imgs(x, img_per_row=10, img_per_col=10):
    x_shape = np.shape(x)
    assert(len(x_shape) == 4)
    output = np.zeros([img_per_row*x_shape[1], img_per_col*x_shape[2], x_shape[3]])
    idx = 0
    for r in range(img_per_row):
        start_row = r * x_shape[1]
        end_row = start_row + x_shape[1]
        for c in range(img_per_col):
            start_col = c * x_shape[2]
            end_col = start_col + x_shape[2]
            output[start_row:end_row, start_col:end_col] = x[idx]
            idx += 1
            if idx == x_shape[0]:
                break
        if idx == x_shape[0]:
            break
    if np.shape(output)[-1] == 1:
        output = np.reshape(output, np.shape(output)[0:2])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default='.')
    parser.add_argument('--output-path', type=str, default='./experiments')
    parser.add_argument('--exp-name', type=str, default='debug')

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--network-structure', type=str, default='Infogan')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--write-iteration', type=int, default=600)
    parser.add_argument('--latent-dim', type=int, default=64)

    parser.add_argument('--second-dim', type=int, default=1024)
    parser.add_argument('--second-depth', type=int, default=3)

    parser.add_argument('--num-scale', type=int, default=4)
    parser.add_argument('--block-per-scale', type=int, default=1)
    parser.add_argument('--depth-per-block', type=int, default=2)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--base-dim', type=int, default=16)
    parser.add_argument('--fc-dim', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-epochs', type=int, default=150)
    parser.add_argument('--lr-fac', type=float, default=0.5)

    parser.add_argument('--epochs2', type=int, default=800)
    parser.add_argument('--lr2', type=float, default=0.0001)
    parser.add_argument('--lr-epochs2', type=int, default=300)
    parser.add_argument('--lr-fac2', type=float, default=0.5)
    
    parser.add_argument('--val', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main()