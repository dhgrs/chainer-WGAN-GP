# coding: UTF-8
import argparse
import os
import glob
import random

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert


class SubpixelConv(chainer.Chain):
    def __init__(self, out_channel):
        super(SubpixelConv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channel * 4, 1)

    def __call__(self, x):
        h = F.depth2space(self.conv(x), 2)
        return h


class BottleNeck(chainer.Chain):
    def __init__(self, in_channel, out_channel,
                 resample=None, norm=L.LayerNormalization):
        super(BottleNeck, self).__init__()
        self.resample = resample
        with self.init_scope():
            if resample == 'down':
                self.shortcut = L.Convolution2D(None, out_channel, 1,
                                                stride=2)
                self.conv1 = L.Convolution2D(None, max(1, in_channel // 2),
                                             3, pad=1)
                self.conv1b = L.Convolution2D(None, max(1, out_channel // 2),
                                              4, stride=2, pad=1)
                self.conv2 = L.Convolution2D(None, out_channel, 3, pad=1)
            elif resample == 'up':
                self.shortcut = SubpixelConv(out_channel)
                self.conv1 = L.Convolution2D(None, max(1, in_channel // 2),
                                             3, pad=1)
                self.conv1b = L.Deconvolution2D(None, max(1, out_channel // 2),
                                                4, stride=2, pad=1)
                self.conv2 = L.Convolution2D(None, out_channel, 3, pad=1)
            elif resample is None:
                if in_channel == out_channel:
                    self.shortcut = F.identity
                else:
                    self.shortcut = L.Convolution2D(None, out_channel, 1)
                self.conv1 = L.Convolution2D(None, max(1, in_channel // 2),
                                             3, pad=1)
                self.conv1b = L.Convolution2D(None, max(1, out_channel // 2),
                                              3, pad=1)
                self.conv2 = L.Convolution2D(None, out_channel, 3, pad=1)
            if norm == L.LayerNormalization:
                self.norm = norm(None)
            else:
                self.norm = norm(out_channel)

    def __call__(self, x):
        h1 = self.shortcut(x)
        h2 = F.relu(x)
        h2 = F.relu(self.conv1(h2))
        h2 = F.relu(self.conv1b(h2))
        h2 = self.conv2(h2)
        if isinstance(self.norm, L.LayerNormalization):
            shape = h2.shape
            h2 = h2.reshape((shape[0], -1))
            h2 = self.norm(h2)
            h2 = h2.reshape(shape)
        else:
            h2 = self.norm(h2)
        return h1 + 0.3 * h2


class Block(chainer.ChainList):
    def __init__(self, layers, *args, **kwargs):
        super(Block, self).__init__()
        for layer in range(layers):
            self.add_link(BottleNeck(*args, **kwargs))

    def __call__(self, x):
        for func in self.children():
            x = func(x)
        return x


class Generator(chainer.Chain):
    def __init__(self, noise=128, dim=64):
        super(Generator, self).__init__()
        self.dim = dim
        with self.init_scope():
            self.fc = L.Linear(None, dim * 8 * 4 * 4)
            self.res1 = Block(6, 8 * dim, 8 * dim,
                              norm=L.BatchNormalization)
            self.res1up = Block(1, 8 * dim, 4 * dim, resample='up',
                                norm=L.BatchNormalization)

            self.res2 = Block(6, 4 * dim, 4 * dim,
                              norm=L.BatchNormalization)
            self.res2up = Block(1, 4 * dim, 2 * dim, resample='up',
                                norm=L.BatchNormalization)

            self.res3 = Block(6, 2 * dim, 2 * dim,
                              norm=L.BatchNormalization)
            self.res3up = Block(1, 2 * dim, 1 * dim, resample='up',
                                norm=L.BatchNormalization)

            self.res4 = Block(6, 1 * dim, 1 * dim,
                              norm=L.BatchNormalization)
            self.res4up = Block(1, 1 * dim, dim // 2, resample='up',
                                norm=L.BatchNormalization)

            self.res5 = Block(5, dim // 2, dim // 2,
                              norm=L.BatchNormalization)

            self.conv = L.Deconvolution2D(None, 3, 3, pad=1)
        self.noise = noise

    def __call__(self, z):
        h = self.fc(z).reshape(-1, self.dim * 8, 4, 4)
        h = self.res1up(self.res1(h))
        h = self.res2up(self.res2(h))
        h = self.res3up(self.res3(h))
        h = self.res4up(self.res4(h))
        h = self.res5(h)
        y = F.tanh(self.conv(h) / 5)
        return y

    def make_z(self, n):
        z = self.xp.random.normal(size=(n, self.noise)).astype(self.xp.float32)
        return z


class Critic(chainer.Chain):
    def __init__(self, dim=64):
        super(Critic, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, max(1, dim // 2), 1)
            self.res1 = Block(5, max(1, dim // 2), max(1, dim // 2))
            self.res1down = Block(1, max(1, dim // 2), 1 * dim,
                                  resample='down')

            self.res2 = Block(6, 1 * dim, 1 * dim)
            self.res2down = Block(1, 1 * dim, 2 * dim, resample='down')

            self.res3 = Block(6, 2 * dim, 2 * dim)
            self.res3down = Block(1, 2 * dim, 4 * dim, resample='down')

            self.res4 = Block(6, 4 * dim, 4 * dim)
            self.res4down = Block(1, 4 * dim, 8 * dim, resample='down')

            self.res5 = Block(6, 8 * dim, 8 * dim)

            self.fc = L.Linear(None, 1)

    def __call__(self, x):
        h = self.conv(x)
        h = self.res1down(self.res1(h))
        h = self.res2down(self.res2(h))
        h = self.res3down(self.res3(h))
        h = self.res4down(self.res4(h))
        h = self.res5(h)
        y = self.fc(h) / 5
        return y


class WGANUpdater(training.StandardUpdater):
    def __init__(self, iterator, l, n_c, opt_g, opt_c, device):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.generator = opt_g.target
        self.critic = opt_c.target
        self.l = l
        self.n_c = n_c
        self._optimizers = {'generator': opt_g, 'critic': opt_c}
        self.device = device
        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self):
        # train critic
        for t in range(self.n_c):
            # read data
            batch = self._iterators['main'].next()
            x = self.converter(batch, self.device)
            x = F.resize_images(x, (64, 64))
            m = x.shape[0]
            xp = chainer.cuda.get_array_module(x)

            # generate
            z = self.generator.make_z(m)
            x_tilde = self.generator(z)

            # sampling along straight lines
            e = xp.random.uniform(0., 1., (m, 1, 1, 1))
            x_hat = e * x + (1 - e) * x_tilde

            # compute loss
            loss_gan = F.average(self.critic(x_tilde) - self.critic(x))
            grad, = chainer.grad([self.critic(x_hat)], [x_hat],
                                 enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))

            loss_grad = self.l * F.mean_squared_error(grad,
                                                      xp.ones_like(grad.data))
            loss_critic = loss_gan + loss_grad

            # update critic
            self.critic.cleargrads()
            loss_critic.backward()
            self._optimizers['critic'].update()

            # report
            chainer.reporter.report({
                'wasserstein distance': -loss_gan, 'loss/grad': loss_grad})

        # train generator
        # read data
        batch = self._iterators['main'].next()
        x = self.converter(batch, self.device)

        # generate and compute loss
        z = self.generator.make_z(m)
        loss_generator = F.average(-self.critic(self.generator(z)))

        # update generator
        self.generator.cleargrads()
        loss_generator.backward()
        self._optimizers['generator'].update()

        # report
        chainer.reporter.report({'loss/generator': loss_generator})


def main():
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=200000,
                        help='Number of iteration')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--directory', '-d', default='.',
                        help='root directory of CelebA Dataset')
    args = parser.parse_args()

    generator = Generator()
    critic = Critic()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()

    opt_g = chainer.optimizers.Adam(1e-4, beta1=0., beta2=0.9)
    opt_g.setup(generator)

    opt_c = chainer.optimizers.Adam(1e-4, beta1=0., beta2=0.9)
    opt_c.setup(critic)

    def preprocess(x):
        # crop 128x128 and flip
        top = random.randint(0, 218 - 128)
        left = random.randint(0, 178 - 128)
        bottom = top + 128
        right = left + 128
        # flip
        x = x[:, top:bottom, left:right]
        if random.randint(0, 1):
            x = x[:, :, ::-1]
        # to [-1, 1]
        x = x * (2 / 255) - 1
        return x

    train = chainer.datasets.TransformDataset(
        chainer.datasets.ImageDataset(glob.glob(
            os.path.join(args.directory, 'Img/img_align_celeba_png/*.png'))),
        preprocess)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = WGANUpdater(train_iter, 10, 5,
                          opt_g, opt_c, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'),
                               out=args.out)

    def out_generated_image(generator, H, W, rows, cols, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            # generate
            z = generator.make_z(rows * cols)
            with chainer.using_config('enable_backprop', False):
                # with chainer.using_config('train', False):
                x = generator(z)
            x = chainer.cuda.to_cpu(x.data)
            x = (x + 1) * (255 / 2)

            # convert to image
            x = np.asarray(np.clip(x, 0.0, 255.0), dtype=np.uint8)
            channels = x.shape[1]
            x = x.reshape((rows, cols, channels, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, channels))
            x = np.squeeze(x)

            # save
            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>6}.png'.format(trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x).save(preview_path)
        return make_image

    trainer.extend(extensions.dump_graph('wasserstein distance'))
    trainer.extend(extensions.snapshot(filename='snapshot'),
                   trigger=(500, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    trainer.extend(
        extensions.PlotReport(['wasserstein distance', 'loss/grad'],
                              'iteration', file_name='critic.png',
                              trigger=(10, 'iteration')))
    trainer.extend(
        extensions.PlotReport(
            ['loss/generator'], 'iteration', file_name='generator.png',
            trigger=(10, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'wasserstein distance', 'loss/grad',
         'loss/generator', 'elapsed_time']), trigger=(10, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_generated_image(generator, 64, 64, 4, 4, args.out),
                   trigger=(500, 'iteration'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        trainer.stop_trigger = chainer.training.trigger.get_trigger(
            (args.iteration, 'iteration'))
    trainer.run()


if __name__ == '__main__':
    main()
