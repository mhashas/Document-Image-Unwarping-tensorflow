import tensorflow as tf
import os
from scipy import io
import types
from constants import *

class InvertedDocunet(object):
    """Helper class for loading the inverted docunet dataset"""

    NUM_CLASSES = 2
    CLASSES = ["foreground", "background"]
    ROOT = '../../'
    DEFORMED = 'deformed_labels'
    DEFORMED_EXT = '.jpg'
    VECTOR_FIELD = 'inverted_vf'
    VECTOR_FIELD_EXT = '.mat'

    def __init__(self, args, split=TRAIN):
        """
        Initializes the dataset object

        Args:
            args (argparse.ArgumentParser): command line arguments
            split (str): current dataset split
        """

        super(InvertedDocunet, self).__init__()

        self.args = args
        self.split = split

    def _read_mat(self, label_path):
        """
        Reads mat vector field

        Args:
            label_path (str): disk label path

        Returns:
            ndarray: vector field in numpy format
        """

        label = io.loadmat(label_path.numpy())['inverted_vector_field']
        return label

    def preprocess(self, image_path, label_path):
        """
        Load and preprocess the input image and target vector field

        Args:
            image_path (str): disk image path
            label_path (str): disk label path

        Returns:
            tf.Tensor, tf.Tensor: image and label
        """

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = image / 255.0

        label = tf.py_function(self._read_mat, [label_path], tf.int16)
        label = tf.cast(label, tf.float32)

        return image, label

    def load_dataset(self):
        """
        Create the dataset loader object

        Returns:
            tf.data.Dataset: dataset object
        """

        current_dir = os.path.dirname(__file__)

        if self.split == TRAINVAL:
            raise NotImplementedError()

        images_path = os.path.join(current_dir, self.ROOT, self.args.dataset_dir, self.split, self.DEFORMED + '_' + 'x'.join(map(str, self.args.size)))
        labels_path = os.path.join(current_dir, self.ROOT, self.args.dataset_dir, self.split, self.VECTOR_FIELD + '_' + 'x'.join(map(str, self.args.size)))

        images_name = os.listdir(images_path)
        images_full_path = [os.path.join(images_path, image_name) for image_name in images_name if image_name.endswith(self.DEFORMED_EXT)]
        labels_full_path = [os.path.join(labels_path , image_name.replace(self.DEFORMED_EXT, self.VECTOR_FIELD_EXT)) for image_name in images_name if image_name.endswith(self.DEFORMED_EXT)]

        dataset = tf.data.Dataset.from_tensor_slices((images_full_path, labels_full_path))
        dataset = dataset.map(self.preprocess)

        if self.split == TRAIN:
            dataset = dataset.shuffle(buffer_size=len(images_path))

        dataset = dataset.batch(self.args.batch_size).prefetch(buffer_size=self.args.batch_size)
        dataset.length = int(len(images_full_path) / self.args.batch_size)

        return dataset