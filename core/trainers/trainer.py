import tensorflow as tf
import copy
from tqdm import tqdm
import math
import time
import matplotlib.pyplot as plt
import os

from util.general_functions import make_data_loader, get_model, get_loss, get_optimizer, get_flat_images, tensor2im
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

        if self.args.inference:
            self.model = self.summary.load_network(self.model)
            self.inference_loader = make_data_loader(args, INFERENCE)
            self.test_loader = make_data_loader(args, TEST)
        elif self.args.trainval:
            self.train_loader, self.test_loader = make_data_loader(args, TRAINVAL), make_data_loader(args, TEST)
        else:
            self.train_loader, self.test_loader = make_data_loader(args, TRAIN), make_data_loader(args, TEST)

        if args.save_best_model:
            self.best_model = copy.deepcopy(self.model)

        if not self.args.inference:
            self.criterion = get_loss(args.loss_type)
            self.global_step = tf.train.get_or_create_global_step()
            self.optimizer = get_optimizer(args, self.global_step, self.train_loader.length)

        if args.execute_graph:
            self.apply_gradients = tf.contrib.eager.defun(self.apply_gradients)
            self.model.call = tf.contrib.eager.defun(self.model.call)

    def apply_gradients(self, gradients, variables, global_step):
        """
        Applies the gradients to the optimizer. This function exists so that tensorflow can define it as a graph function using tf.contrib.eager.defun

        Args:
            gradients:
            variables:
            global_step:
        """

        self.optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

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
                    if self.args.refine_network:
                        output, second_output = self.model(image, training=True)
                    else:
                        output = self.model(image, training=True)
                    loss = self.criterion(target, output)

                    if self.args.refine_network:
                        second_loss = self.criterion(target, second_output)
                        loss = tf.add(loss, second_loss)

                    l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables if 'bias' not in v.name]) * self.args.weight_decay
                    loss = tf.add(loss, l2_reg)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.apply_gradients(gradients, self.model.trainable_variables, global_step=self.global_step)
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
        print('\n=>Epoches %i, learning rate = %.6f, \previous best = %.4f' % (epoch, self.optimizer._lr_t.numpy(), self.best_loss))

    def inference(self):
        """
        Performs inference
        """

        loader = self.inference_loader
        bar = tqdm(loader, total=loader.length)
        times = []

        for i, sample in enumerate(bar):
            image = sample[0]
            target = None #sample[1] if len(sample) > 1 else None

            start = time.time()
            output = self.model(image)
            end = time.time()
            current_time = end - start

            output, target = get_flat_images(self.args.dataset, image, output, target)
            output = tf.stack([tensor2im(current) for current in output[:, : int(self.args.resize[0] / 2), : int(self.args.resize[0] / 2), :]])
            target = tf.stack([tensor2im(current) for current in target[:, : int(self.args.resize[0] / 2), : int(self.args.resize[0] / 2), :]]) if target is not None else None
            image = tf.image.resize(image, tf.convert_to_tensor([int(self.args.resize[0] / 2), int(self.args.resize[0] / 2)],dtype=tf.int32))

            if not os.path.exists(os.path.join(self.args.inference_dir, 'results')): os.makedirs(os.path.join(self.args.inference_dir, 'results'))
            plt.imsave(os.path.join(self.args.inference_dir, 'results', str(i)+'_og.jpg'), tf.squeeze(image).numpy())
            plt.imsave(os.path.join(self.args.inference_dir, 'results', str(i)+'_unwarped.jpg'), tf.squeeze(output).numpy())

            if target is not None:
                plt.imsave(os.path.join(self.args.inference_dir, 'results', str(i) + '_target.jpg'), tf.squeeze(target).numpy())

            times.append(current_time)

        print(sum(times) / len(times))

    def save_network(self):
        """Saves current model"""

        self.summary.save_network(self.model)