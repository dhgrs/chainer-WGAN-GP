# coding: UTF-8
import argparse
import os

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


class Generator(chainer.Chain):
    def __init__(self, ):
        super(Generator, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 28 * 28)
            )

    def __call__(self, z):
        h = F.relu(self.fc1(z))
        y = F.reshape(F.sigmoid(self.fc2(h)), (-1, 1, 28, 28))
        return y

    def make_z(self, n):
        z = self.xp.random.normal(size=(n, 10)).astype(self.xp.float32)
        return z


class Critic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 1)
            )

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        y = self.fc2(h)
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
            m = x.shape[0]
            H, W = x.shape[2], x.shape[3]
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
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    generator = Generator()
    critic = Critic()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()

    opt_g = chainer.optimizers.Adam(1e-4, beta1=0.5, beta2=0.9)
    opt_g.setup(generator)

    opt_c = chainer.optimizers.Adam(1e-4, beta1=0.5, beta2=0.9)
    opt_c.setup(critic)

    train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = WGANUpdater(train_iter, 10, 5,
                          opt_g, opt_c, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    def out_generated_image(generator, H, W, rows, cols, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            n_images = rows * cols
            xp = generator.xp
            z = generator.make_z(rows * cols)
            with chainer.using_config('enable_backprop', False):
                with chainer.using_config('train', False):
                    x = generator(z)
            x = chainer.cuda.to_cpu(x.data)

            x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
            channels = x.shape[1]
            x = x.reshape((rows, cols, channels, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, channels))
            x = np.squeeze(x)

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>5}.png'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x).save(preview_path)
        return make_image

    trainer.extend(extensions.dump_graph('wasserstein distance'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport(['wasserstein distance', 'loss/grad'],
                              'epoch', file_name='critic.png'))
    trainer.extend(
        extensions.PlotReport(
            ['loss/generator'], 'epoch', file_name='generator.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'wasserstein distance', 'loss/grad',
         'loss/generator', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(out_generated_image(generator, 28, 28, 5, 5, args.out),
                   trigger=(1, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
