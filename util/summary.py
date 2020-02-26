import tensorflow as tf
from tensorflow.python.ops import array_ops
import os
import glob
import matplotlib.pyplot as plt

from constants import *
from util.general_functions import get_flat_images, tensor2im

class TensorboardSummary(object):
    """
    This class is used in order to save useful information for visualizing a model's progress through training
    """

    def __init__(self, args):
        """
        Initializes the tensorboard summary object.

        Args:
            args (argparse.ArgumentParser): Object that contains all the command line arguments
        """

        self.args = args
        self.experiment_dir = self.generate_directory(args)
        self.writer = tf.contrib.summary.create_file_writer(os.path.join(self.experiment_dir), flush_millis=10000)

        self.train_step = 0
        self.test_step = 0
        self.visualization_step = 0

    def generate_directory(self, args):
        """
        Generates the name of the folder where the training information will be saved

        Args:
            args (argparse.ArgumentParser): Object that contains all the command line arguments

        Returns:
            string: the name of the folder where the training information will be saved
        """

        checkname = 'debug' if args.debug else ''
        checkname += args.model
        checkname += '_sc' if args.separable_conv else ''
        checkname += '-refined' if args.refine_network else ''
        checkname += '-graph' if args.execute_graph else ''

        if 'deeplab' in args.model:
            checkname += '-os_' + str(args.output_stride)
            checkname += '-ls_1' if args.learned_upsampling else ''
            checkname += '-pt_1' if args.pretrained else ''
            checkname += '-aspp_0' if not args.use_aspp else ''

        if 'unet' in args.model:
            checkname += '-downs_' + str(args.num_downs) + '-ngf_' + str(args.ngf) + '-type_' + str(args.down_type)

        checkname += '-batch_' + str(args.batch_size)
        checkname += '-loss_' + args.loss_type
        checkname += '-sloss_' if args.second_loss else ''

        if args.clip > 0:
            checkname += '-clipping_' + str(args.clip)

        if args.resize:
            checkname += '-' + ','.join([str(x) for x in list(args.resize)])
        checkname += '-epochs_' + str(args.epochs)
        checkname += '-trainval' if args.trainval else ''

        current_dir = os.path.dirname(__file__)
        directory = os.path.join(current_dir, args.results_root, args.results_dir, args.dataset_dir, args.dataset, args.model, checkname)

        runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
        experiment_dir = os.path.join(directory, 'experiment_{}'.format(str(run_id)))

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        return experiment_dir

    def add_scalar(self, tag, value, step):
        """
        Adds a scalar value to tensorboard results file

        Args:
            tag (string): data identifier
            value (int): value to record
            step (int):  global step value to record
        """

        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(tag, value, step=step)

    def visualize_image(self, images, targets, outputs, split=TRAIN):
        """
        Saves visualization imagse to tensorboard file

        Args:
            images (tf.Tensor): input images
            outputs (tf.Tensor): output vector fields
            targets (tf.Tensor): target vector fields
            split (string):
        """

        step = self.get_step(split)

        outputs, targets = get_flat_images(self.args.dataset, images, outputs, targets)
        outputs = tf.stack([tensor2im(output) for output in outputs[:, : int(self.args.resize[0] /2), : int(self.args.resize[0] /2), :]])
        targets = tf.stack([tensor2im(target) for target in targets[:, : int(self.args.resize[0] /2), : int(self.args.resize[0] /2), :]])

        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image(split + '/ZZ Image', self.image_grid(images), step=step)
            tf.contrib.summary.image(split + '/Predicted label', self.image_grid(outputs), step=step)
            tf.contrib.summary.image(split + '/Groundtruth label', self.image_grid(targets), step=step)

    def get_step(self, split):
        """
        Returns the tensorboard visualization step of the current split

        Args:
            split (str): current training split

        Returns:
            int: split visualization step for tensorboard
        """

        if split == TRAIN:
            self.train_step += 1
            return self.train_step
        elif split == TEST:
            self.test_step += 1
            return self.test_step
        elif split == VISUALIZATION:
            self.visualization_step += 1
            return self.visualization_step

    def image_grid(self, images, num_channels=3):
        """Arrange a minibatch of images into a grid to form a single image.

        Args:
            images (tf.Tensor): minibatch of images
            num_channels (int): the number of channels in an image.
        Returns:
            tf.Tensor: single image in which the input images have been
            arranged into a grid.

        """

        num_cols = min(8, images.shape[0])
        image_shape = images.shape[1:3]
        grid_shape = (1, num_cols)

        height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
        reshape_size = tuple(grid_shape) + tuple(image_shape) + (num_channels,)
        images = array_ops.reshape(images, reshape_size)
        images = array_ops.transpose(images, [0, 1, 3, 2, 4])

        images = array_ops.reshape(images, [grid_shape[0], width, image_shape[0], num_channels])
        images = array_ops.transpose(images, [0, 2, 1, 3])
        images = array_ops.reshape(images, [1, height, width, num_channels])

        return images

    def save_network(self, model):
        """
        Save model to disk

        Args:
            model (tf.keras.Model): model to be saved
        """

        path = self.experiment_dir[self.experiment_dir.find(self.args.results_dir):].replace(self.args.results_dir, self.args.save_dir)
        if not os.path.isdir(path):
            os.makedirs(path)

        model.save_weights(path + '/network.hdf5')
        #root = tf.train.Checkpoint(model=model)
        #root.save(os.path.join(path, 'ckpt'))

    def load_network(self, model):
        """

        Args:
            model:

        Returns:

        """
        model.build((1, self.args.resize[0], self.args.resize[1], 3))
        path = self.args.pretrained_models_dir

        #root = tf.train.Checkpoint(model=model)
        #root.restore(tf.train.latest_checkpoint(path))
        model.load_weights(path + '.hdf5')
        return model

    def close_writer(self):
        """Closes the sumary writer"""
        self.writer.close()