import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from core.models.unet import UNet
from core.models.deeplabv3_plus import Deeplabv3_plus
from dataloader.docunet_inverted import InvertedDocunet
from util.docunet_loss import DocunetLoss
from constants import *

def make_data_loader(args, split=TRAIN):
    """
    Builds the model based on the provided arguments

    Args:
        args (ArgumentParser): input arguments
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float): scaling factor for normal, xavier and orthogonal.
    """

    if args.dataset == DOCUNET:
        raise NotImplementedError()
    elif args.dataset == DOCUNET_INVERTED:
        dataset = InvertedDocunet
    elif args.dataset == DOCUNET_IM2IM:
        raise NotImplementedError()
    else:
        raise NotImplementedError

    helper = dataset(args, split=split)
    return helper.load_dataset()

def get_model(args):
    """
    Builds the model based on the provided arguments and returns the initialized model

    Args:
        args (ArgumentParser): command line arguments

    Returns:
        tf.keras.Sequential: model
    """

    num_classes = get_num_classes(args.dataset)

    if UNET in args.model:
        model = UNet(args, num_classes)
    elif DEEPLAB in args.model:
        model = Deeplabv3_plus(args, num_classes)
    else:
        raise NotImplementedError()

    return model

def get_num_classes(dataset):
    """
    Builds the model based on the provided arguments and returns the initialized model

    Args:
        args (argparse): command line arguments

    Returns:
        int: dataset number of classes
    """

    if dataset == DOCUNET or dataset == DOCUNET_INVERTED:
        num_classes = 2
    elif dataset == DOCUNET_IM2IM:
        num_classes = 3
    else:
        raise NotImplementedError

    return num_classes

def get_loss(loss_type):
    """
    Builds and returns the loss function

    Args:
        args (str): command line arguments

    Returns:
        tf.keras.loss: loss function
    """

    if loss_type == DOCUNET_LOSS:
        loss = DocunetLoss()
    elif loss_type == L1_LOSS:
        loss = tf.keras.losses.MeanAbsoluteError()
    else:
        raise NotImplementedError

    return loss

def get_lr_scheduler(args, global_step, loader_length):
    """
    Builds and returns the lr scheduler

    Args:
        args (argparse): command line arguments
        global_step (tf.Tensor): the global step tensor
        loader_length (int): train loader length

    Returns:
        a function that computes the the decayed learning rate if in eager mode else the decayed learning rate

    """

    num_steps = int(args.epochs * loader_length)

    if args.lr_policy == LR_POLY_POLICY:
        lr_scheduler = tf.train.polynomial_decay(args.lr, global_step, num_steps, args.lr / 10)
    elif args.lr_policy == LR_NONE_POLICY:
        return args.lr
    else:
        raise NotImplementedError

    return lr_scheduler

def get_optimizer(args, global_step, loader_length):
    """
    Builds and returns the optimizer.

    Args:
        args (argparse): command line arguments
        global_step (tf.Tensor): the global step tensor
        loader length (int): train loader length

    Returns:
        tf.keras.optimizers
    """

    lr_scheduler = get_lr_scheduler(args, global_step, loader_length)

    if args.optim == SGD:
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_scheduler, momentum=args.momentum, use_nesterov=True)
    elif args.optim == ADAM:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_scheduler)
    elif args.optim == AMSGRAD:
        raise NotImplementedError
    else:
        raise NotImplementedError

    return optimizer

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Args:
        img (tf.Tensor): tensor of shape (B, H, W, C)
        x (tf.Tensor): flattened tensor of shape (B*H*W,)
        y (tf.Tensor): flattened tensor of shape (B*H*W,)

    Returns:
        tf.Tensor: shape (B, H, W, C)
    """

    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def scale_vector_field_tensor(vector_field):
    """
    Scales vector field to fit into the range [-1, 1]

    Args:
        vector_field (tf.Tensor): unormalized vector field

    Returns:
        tf.Tensor: normalized vector field
    """

    vector_field = tf.where(vector_field < 0, 3 * tf.shape(vector_field).numpy()[1] * tf.ones(tf.shape(vector_field).numpy(), dtype=vector_field.dtype), vector_field)
    vector_field = (vector_field / (tf.shape(vector_field).numpy()[1] / 2)) - 1

    return vector_field

def apply_transformation_to_image(img, vector_field):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Args:
        img (tf.Tensor): batch of images in (B, H, W, C) layout.
        vector_field (tf.Tensor): x, y which is the output of affine_grid_generator.

    Returns:
        out: interpolated images according to grids. Same size as grid.
    """

    vector_field = scale_vector_field_tensor(vector_field)
    x = vector_field[:, :, :, 0]
    y = vector_field[:, :, :, 1]

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def get_flat_images(dataset, images, outputs, targets):
    """
    Applies both predicted and target vector field to input image to flatten them

    Args:
        dataset (str): current dataset
        images (tf.Tensor): input images
        outputs (tf.Tensor): output vector fields
        targets (tf.Tensor): target vector fields

    Returns:
        tf.Tensor, tf.Tensor: flattened images
    """

    if dataset == DOCUNET:
        pass
    elif dataset == DOCUNET_INVERTED:
        outputs = apply_transformation_to_image(images, outputs)
        targets = apply_transformation_to_image(images, targets) if targets is not None else None
    else:
        pass

    return outputs, targets

def tensor2im(input_image):
    """
    Post processing tensor image

    Args:
        input_image (tf.Tensor): tensor input image

    Returns:
        tf.Tensor: tensor output image
    """

    if input_image.numpy().ndim == 3:
        input_image = (input_image - tf.reduce_min(input_image)) / (tf.reduce_max(input_image) - tf.reduce_min(input_image))
    input_image = (input_image + 1) / 2.0

    return input_image

def print_training_info(args):
    """
    Prints training parameters to output.

    Args:
        args (argparse): command line arguments
    """

    print("Built ", args.model)
    print("Dataset dir ", args.dataset_dir)
    print('Dataset ', args.dataset)
    print('Refine network', args.refine_network)
    print('Graph execution', args.execute_graph)

    if 'unet' in args.model:
        print('Ngf', args.ngf)
        print('Num downs', args.num_downs)
        print('Down type', args.down_type)

    if 'deeplab' in args.model:
        print('Output stride', args.output_stride)
        print('Learned upsampling', args.learned_upsampling)
        print('Pretrained', args.pretrained)
        print('Use aspp', args.use_aspp)

    print('Separable conv', args.separable_conv)
    print('Optimizer', args.optim)
    print('Learning rate', args.lr)
    print('Second loss', args.second_loss)

    if args.clip > 0:
        print('Gradient clip', args.clip)

    print('Resize', args.resize)
    print('Batch size', args.batch_size)
    print('Norm layer', args.norm_layer)
    print('Using cuda', args.cuda)
    print('Using ' + args.loss_type + ' loss')
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)





