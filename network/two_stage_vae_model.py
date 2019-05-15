import tensorflow as tf 
import math 
import numpy as np 
from tensorflow.python.training.moving_averages import assign_moving_average
from network.util import *



class TwoStageVaeModel(object):
    def __init__(self, x, latent_dim=64, second_depth=3, second_dim=1024, cross_entropy_loss=False):
        self.raw_x = x
        self.x = tf.cast(self.raw_x, tf.float32) / 255.0 
        self.batch_size = x.get_shape().as_list()[0]
        self.latent_dim = latent_dim
        self.second_dim = second_dim 
        self.second_depth = second_depth
        self.cross_entropy_loss = cross_entropy_loss

        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        self.__build_network()
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer()

    def __build_network(self):
        with tf.variable_scope('stage1'):
            self.build_encoder1()
            self.build_decoder1()
        with tf.variable_scope('stage2'):
            self.build_encoder2()
            self.build_decoder2()

    def __build_loss(self):
        HALF_LOG_TWO_PI = 0.91893

        self.kl_loss1 = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size)
        if not self.cross_entropy_loss:
            self.gen_loss1 = tf.reduce_sum(tf.square((self.x - self.x_hat) / self.gamma_x) / 2.0 + self.loggamma_x + HALF_LOG_TWO_PI) / float(self.batch_size)
        else:
            self.gen_loss1 = -tf.reduce_sum(self.x * tf.log(tf.maximum(self.x_hat, 1e-8)) + (1-self.x) * tf.log(tf.maximum(1-self.x_hat, 1e-8))) / float(self.batch_size)
        self.loss1 = self.kl_loss1 + self.gen_loss1 

        self.kl_loss2 = tf.reduce_sum(tf.square(self.mu_u) + tf.square(self.sd_u) - 2 * self.logsd_u - 1) / 2.0 / float(self.batch_size)
        self.gen_loss2 = tf.reduce_sum(tf.square((self.z - self.z_hat) / self.gamma_z) / 2.0 + self.loggamma_z + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss2 = self.kl_loss2 + self.gen_loss2 

    def __build_summary(self):
        with tf.name_scope('stage1_summary'):
            self.summary1 = []
            self.summary1.append(tf.summary.image('input', self.x))
            self.summary1.append(tf.summary.image('recon', self.x_hat))
            self.summary1.append(tf.summary.scalar('kl_loss', self.kl_loss1))
            self.summary1.append(tf.summary.scalar('gen_loss', self.gen_loss1))
            self.summary1.append(tf.summary.scalar('loss', self.loss1))
            self.summary1.append(tf.summary.scalar('gamma', self.gamma_x))
            self.summary1 = tf.summary.merge(self.summary1)

        with tf.name_scope('stage2_summary'):
            self.summary2 = []
            self.summary2.append(tf.summary.scalar('kl_loss', self.kl_loss2))
            self.summary2.append(tf.summary.scalar('gen_loss', self.gen_loss2))
            self.summary2.append(tf.summary.scalar('loss', self.loss2))
            self.summary2.append(tf.summary.scalar('gamma', self.gamma_z))
            self.summary2 = tf.summary.merge(self.summary2)

    def __build_optimizer(self):
        all_variables = tf.global_variables()
        variables1 = [var for var in all_variables if 'stage1' in var.name]
        variables2 = [var for var in all_variables if 'stage2' in var.name]
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
        self.opt1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss1, self.global_step, var_list=variables1)
        self.opt2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss2, self.global_step, var_list=variables2)
        
    def build_encoder2(self):
        with tf.variable_scope('encoder'):
            t = self.z 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.z, t], -1)
        
            self.mu_u = tf.layers.dense(t, self.latent_dim, name='mu_u')
            self.logsd_u = tf.layers.dense(t, self.latent_dim, name='logsd_u')
            self.sd_u = tf.exp(self.logsd_u)
            self.u = self.mu_u + self.sd_u * tf.random_normal([self.batch_size, self.latent_dim])
        
    def build_decoder2(self):
        with tf.variable_scope('decoder'):
            t = self.u 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.u, t], -1)

            self.z_hat = tf.layers.dense(t, self.latent_dim, name='z_hat')
            self.loggamma_z = tf.get_variable('loggamma_z', [], tf.float32, tf.zeros_initializer())
            self.gamma_z = tf.exp(self.loggamma_z)

    def extract_posterior(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        mu_z, sd_z = [], []
        for i in range(num_iter):
            mu_z_batch, sd_z_batch = sess.run([self.mu_z, self.sd_z], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            mu_z.append(mu_z_batch)
            sd_z.append(sd_z_batch)
        mu_z = np.concatenate(mu_z, 0)[0:num_sample]
        sd_z = np.concatenate(sd_z, 0)[0:num_sample]
        return mu_z, sd_z

    def step(self, stage, input_batch, lr, sess, writer=None, write_iteration=600):
        if stage == 1:
            loss, summary, _ = sess.run([self.loss1, self.summary1, self.opt1], feed_dict={self.raw_x: input_batch, self.lr: lr, self.is_training: True})
        elif stage == 2:
            loss, summary, _ = sess.run([self.loss2, self.summary2, self.opt2], feed_dict={self.z: input_batch, self.lr: lr, self.is_training: True})
        else:
            raise Exception('Wrong stage {}.'.format(stage))
        global_step = self.global_step.eval(sess)
        if global_step % write_iteration == 0 and writer is not None:
            writer.add_summary(summary, global_step)
        return loss 

    def reconstruct(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        recon_x = []
        for i in range(num_iter):
            recon_x_batch = sess.run(self.x_hat, feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            recon_x.append(recon_x_batch)
        recon_x = np.concatenate(recon_x, 0)[0:num_sample]
        return recon_x 

    def generate(self, sess, num_sample, stage=2):
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        gen_samples = []
        for i in range(num_iter):
            if stage == 2:
                # u ~ N(0, I)
                u = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
                # z ~ N(f_2(u), \gamma_z I)
                z, gamma_z = sess.run([self.z_hat, self.gamma_z], feed_dict={self.u: u, self.is_training: False})
                z = z + gamma_z * np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            else:
                z = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            # x = f_1(z)
            x = sess.run(self.x_hat, feed_dict={self.z: z, self.is_training: False})
            gen_samples.append(x)
        gen_samples = np.concatenate(gen_samples, 0)
        return gen_samples[0:num_sample]


class Infogan(TwoStageVaeModel):
    def __init__(self, x, latent_dim=64, second_depth=3, second_dim=1024, cross_entropy_loss=False):
        super(Infogan, self).__init__(x, latent_dim, second_depth, second_dim, cross_entropy_loss)

    def build_encoder1(self):
        with tf.variable_scope('encoder'):
            y = self.x 
            y = lrelu(conv2d(y, 64, 4, 4, 2, 2, name='conv1', use_sn=True))

            y = conv2d(y, 128, 4, 4, 2, 2, name='conv2', use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn2')
            y = lrelu(y)

            y = tf.reshape(y, [self.x.get_shape().as_list()[0], -1])
            y = linear(y, 1024, scope="fc3", use_sn=True)
            y = batch_norm(y, is_training=self.is_training, scope='bn3')
            y = lrelu(y)

            gaussian_params = linear(y, 2 * self.latent_dim, scope="en4", use_sn=True)
            self.mu_z = gaussian_params[:, :self.latent_dim]
            self.sd_z = 1e-6 + tf.nn.softplus(gaussian_params[:, self.latent_dim:])
            self.logsd_z = tf.log(self.sd_z)
            self.z = self.mu_z + tf.random_normal([self.batch_size, self.latent_dim]) * self.sd_z

    def build_decoder1(self):
        with tf.variable_scope('decoder'):
            y = self.z 
            final_side_length = self.x.get_shape().as_list()[1]
            data_depth = self.x.get_shape().as_list()[-1]

            y = tf.nn.relu(batch_norm(linear(y, 1024, 'fc1'), is_training=self.is_training, scope='bn1'))
            y = tf.nn.relu(batch_norm(linear(y, 128 * (final_side_length // 4) * (final_side_length // 4), scope='fc2'), is_training=self.is_training, scope='bn2'))
            y = tf.reshape(y, [self.batch_size, final_side_length // 4, final_side_length // 4, 128])
            y = tf.nn.relu(batch_norm(deconv2d(y, [self.batch_size, final_side_length // 2, final_side_length // 2, 64], 4, 4, 2, 2, name='conv3'), is_training=self.is_training, scope='bn3'))
            self.x_hat = tf.nn.sigmoid(deconv2d(y, [self.batch_size, final_side_length, final_side_length, data_depth], 4, 4, 2, 2, name='conv4'))

            self.loggamma_x = tf.get_variable('loggamma_x', [], tf.float32, tf.zeros_initializer())
            self.gamma_x = tf.exp(self.loggamma_x)


class Wae(TwoStageVaeModel):
    def __init__(self, x, latent_dim=64, second_depth=3, second_dim=1024, cross_entropy_loss=False):
        super(Wae, self).__init__(x, latent_dim, second_depth, second_dim, cross_entropy_loss)

    def build_encoder1(self):
        with tf.variable_scope('encoder'):
            y = self.x 

            y = tf.nn.relu(batch_norm(tf.layers.conv2d(y, 128, 5, 1, 'same'), self.is_training, 'bn1'))
            y = tf.nn.relu(batch_norm(tf.layers.conv2d(y, 256, 5, 2, 'same'), self.is_training, 'bn2'))
            y = tf.nn.relu(batch_norm(tf.layers.conv2d(y, 512, 5, 2, 'same'), self.is_training, 'bn3'))
            y = tf.nn.relu(batch_norm(tf.layers.conv2d(y, 1024, 5, 2, 'same'), self.is_training, 'bn4'))

            y = tf.layers.flatten(y)
            self.mu_z = tf.layers.dense(y, self.latent_dim)
            self.logsd_z = tf.layers.dense(y, self.latent_dim)
            self.sd_z = tf.exp(self.logsd_z)
            self.z = self.mu_z + tf.random_normal([self.batch_size, self.latent_dim]) * self.sd_z 

    def build_decoder1(self):
        with tf.variable_scope('decoder'):
            y = self.z 

            y = tf.nn.relu(tf.layers.dense(y, 8*8*1024))
            y = tf.reshape(y, [-1, 8, 8, 1024])

            y = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(y, 512, 5, 2, 'same'), self.is_training, 'bn1'))
            y = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(y, 256, 5, 2, 'same'), self.is_training, 'bn2'))
            y = tf.nn.relu(batch_norm(tf.layers.conv2d_transpose(y, 128, 5, 2, 'same'), self.is_training, 'bn3'))

            y = tf.layers.conv2d_transpose(y, 3, 5, 1, 'same')
            self.x_hat = tf.nn.sigmoid(y)

            self.loggamma_x = tf.get_variable('loggamma_x', [], tf.float32, tf.zeros_initializer())
            self.gamma_x = tf.exp(self.loggamma_x)


class Resnet(TwoStageVaeModel):
    def __init__(self, x, num_scale, block_per_scale=1, depth_per_block=2, kernel_size=3, base_dim=16, fc_dim=512, latent_dim=64, second_depth=3, second_dim=1024, cross_entropy_loss=False):
        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size 
        self.base_dim = base_dim 
        self.fc_dim = fc_dim
        super(Resnet, self).__init__(x, latent_dim, second_depth, second_dim, cross_entropy_loss)

    def build_encoder1(self):
        with tf.variable_scope('encoder'):
            dim = self.base_dim
            y = tf.layers.conv2d(self.x, dim, self.kernel_size, 1, 'same', name='conv0')
            for i in range(self.num_scale):
                y = scale_block(y, dim, self.is_training, 'scale'+str(i), self.block_per_scale, self.depth_per_block, self.kernel_size)

                if i != self.num_scale - 1:
                    dim *= 2
                    y = downsample(y, dim, self.kernel_size, 'downsample'+str(i))
            
            y = tf.reduce_mean(y, [1, 2])
            y = scale_fc_block(y, self.fc_dim, 'fc', 1, self.depth_per_block)

            self.mu_z = tf.layers.dense(y, self.latent_dim)
            self.logsd_z = tf.layers.dense(y, self.latent_dim)
            self.sd_z = tf.exp(self.logsd_z)
            self.z = self.mu_z + tf.random_normal([self.batch_size, self.latent_dim]) * self.sd_z 

    def build_decoder1(self):
        desired_scale = self.x.get_shape().as_list()[1]
        scales, dims = [], []
        current_scale, current_dim = 2, self.base_dim 
        while current_scale <= desired_scale:
            scales.append(current_scale)
            dims.append(current_dim)
            current_scale *= 2
            current_dim = min(current_dim*2, 1024)
        assert(scales[-1] == desired_scale)
        dims = list(reversed(dims))

        with tf.variable_scope('decoder'):
            y = self.z 
            data_depth = self.x.get_shape().as_list()[-1]

            fc_dim = 2 * 2 * dims[0]
            y = tf.layers.dense(y, fc_dim, name='fc0')
            y = tf.reshape(y, [-1, 2, 2, dims[0]])

            for i in range(len(scales)-1):
                y = upsample(y, dims[i+1], self.kernel_size, 'up'+str(i))
                y = scale_block(y, dims[i+1], self.is_training, 'scale'+str(i), self.block_per_scale, self.depth_per_block, self.kernel_size)
            
            y = tf.layers.conv2d(y, data_depth, self.kernel_size, 1, 'same')
            self.x_hat = tf.nn.sigmoid(y)

            self.loggamma_x = tf.get_variable('loggamma_x', [], tf.float32, tf.zeros_initializer())
            self.gamma_x = tf.exp(self.loggamma_x)