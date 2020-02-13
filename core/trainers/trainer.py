import tensorflow as tf
import copy
from tqdm import tqdm
import math

from util.general_functions import make_data_loader, get_model, get_loss, get_optimizer
from util.summary import TensorboardSummary

from constants import *


class Trainer(object):
    """Helper class to train neural networks."""

    def __init__(self, args):
        """
        Creates the model, dataloader, loss function, optimizer and tensorboard summary for training.

        Args:
            args (argparse.ArgumentParser): object that contains all the command line arguments.
        """
        self.args = args
        self.best_loss = math.inf
        self.summary = TensorboardSummary(args)
        self.model = get_model(args)

        if args.save_best_model:
            self.best_model = copy.deepcopy(self.model)

        if self.args.trainval:
            self.train_loader, self.test_loader = make_data_loader(args, TRAINVAL), make_data_loader(args, TEST)
        else:
            self.train_loader, self.test_loader = make_data_loader(args, TRAIN), make_data_loader(args, TEST)

        self.criterion = get_loss(args.loss_type)
        self.optimizer = get_optimizer(args)
        self.global_step = tf.train.get_or_create_global_step()

        if args.execute_graph:
            self.apply_gradients = tf.contrib.eager.defun(self.apply_gradients)
            self.model.call = tf.contrib.eager.defun(self.model.call)

    def apply_gradients(self, optimizer : tf.train.Optimizer, gradients, variables, global_step):
        optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    def run_epoch(self, epoch, split=TRAIN):
        """
        Trains the model for 1 epoch.

        Args:
          epoch (int): current training epoch
          split (string): current dataset split
        """
        total_loss = 0.0
        loader = self.train_loader if split == TRAIN else self.test_loader
        bar = tqdm(loader, total=loader.length)

        for i, sample in enumerate(bar):
            image, target = sample[0], sample[1]

            if split == TRAIN:
                with tf.GradientTape() as tape:
                    output = self.model(image, training=True)
                    loss = self.criterion(target, output)

                    l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables if 'bias' not in v.name]) * self.args.weight_decay
                    loss = tf.add(loss, l2_reg)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.apply_gradients(self.optimizer, gradients, self.model.trainable_variables, global_step=self.global_step)
            else:
                output = self.model(image)
                loss = self.criterion(target, output)

                l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables if 'bias' not in v.name]) * self.args.weight_decay
                loss = tf.add(loss, l2_reg)

            # Show 10 * 3 inference results each epoch
            if split != VISUALIZATION and i % (loader.length // 10) == 0:
                self.summary.visualize_image(image, target, output, split=split)
            elif split == VISUALIZATION:
                self.summary.visualize_image(image, target, output, split=split)

            total_loss += loss.numpy()
            bar.set_description(split + ' loss: %.3f' % (loss.numpy()))

        if split == TEST:
            if total_loss < self.best_loss:
                self.best_loss = total_loss

                if self.args.save_best_model:
                    self.best_model = copy.deepcopy(self.model)

        self.summary.add_scalar(split + '/total_loss_epoch', total_loss, epoch)
        print('\n=>Epoches %i, learning rate = %.4f, \previous best = %.4f' % (epoch, self.args.lr, self.best_loss))

    def save_network(self):
        self.summary.save_network(self.model)