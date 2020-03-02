import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy import io

from core.trainers.trainer import Trainer
from dataloader.docunet_inverted import InvertedDocunet
from util.general_functions import apply_transformation_to_image
from parser_options import ParserOptions
from constants import *

def tensorflow_invert(invert=False):
    tf.enable_eager_execution()
    image = tf.io.read_file('../deformed_label.jpg')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.expand_dims(image, 0)

    label = io.loadmat('../fm.mat')['inverted_vector_field']
    label = tf.cast(label, tf.float32)
    label = tf.expand_dims(label, 0)

    flatten_label = apply_transformation_to_image(image, label)
    plt.imsave('../our_flatten_tensorflow.jpg', tf.squeeze(flatten_label).numpy())

def check_duplicates(source_folder_name, destination_folder_name):
    source_files = set(os.listdir(source_folder_name))
    destination_files = set(os.listdir(destination_folder_name))
    intersection = source_files.intersection(destination_files)
    intersection.remove('Thumbs.db')

    if len(intersection) == 0:
        print("OK")
    else:
        print("NOT OK")

def network_predict(pretrained_model=''):
    tf.enable_eager_execution(device_policy=tf.contrib.eager.DEVICE_PLACEMENT_SILENT)

    if not pretrained_model:
        raise NotImplementedError()

    args = ParserOptions().parse()
    args.cuda = False
    args.batch_size = 1
    args.inference = 1
    args.execute_graph = 1
    args.pretrained_models_dir = pretrained_model
    args.num_downs = 8
    args.resize, args.size = (256,256), (256,256)
    args.model = DEEPLAB_50
    args.inference_dir = 'prod_evaluation_dataset'
    args.debug = 0
    args.results_dir = 'results_inference'
    #args.refine_network = 1

    trainer = Trainer(args)
    trainer.inference()

if __name__ == "__main__":
    network_predict(pretrained_model='saved_models/deeplab_50')