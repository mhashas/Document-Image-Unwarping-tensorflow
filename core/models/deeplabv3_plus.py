import tensorflow as tf
import numpy as np

from core.models.resnet import ResNet50, ResNet101
from constants import *

class _ASPPModule(tf.keras.layers.Layer):
    """Create an ASPP Module"""

    def __init__(self, out_channels, kernel_size, dilation):
        """
        Initializes the ASPP Module

        Args:
            out_channels (int): convolution output channels
            kernel_size (int): convolution kernel size
            dilation (int): convolution dilation rate
        """
        super(_ASPPModule, self).__init__()

        self.atrous_conv = tf.keras.layers.Conv2D(out_channels, kernel_size=kernel_size, padding="same", dilation_rate=dilation, use_bias=False)
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        """
        Computes a forward pass through the ASPP Module

        Args:
            x (tf.Tensor): input
            training (bool): true if we are in training mode

        Returns:
            x (tf.Tensor): result of the forward pass
        """

        x = self.atrous_conv(x)
        x = self.norm(x, training=training)

        return self.relu(x)

class ASPP(tf.keras.layers.Layer):
    """Creates an ASPP layer consisting of multiple ASPP modules"""

    def __init__(self, output_stride, input_shape):
        """
        Initializes the ASPP layer

        Args:
            output_stride (int): determines the dilation rates of the ASPP modules
            input_shape (tuple): tensor input shape, used to determine kernel of avgpool
        """

        super(ASPP, self).__init__()

        avg_pool_kernel = tuple(np.array(input_shape) / 16)

        if output_stride == 16:
            dilations = [1, 6 , 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError()

        self.assp1 = _ASPPModule(256, 1, dilation=dilations[0])
        self.assp2 = _ASPPModule(256, 3, dilation=dilations[1])
        self.assp3 = _ASPPModule(256, 3, dilation=dilations[2])
        self.assp4 = _ASPPModule(256, 3, dilation=dilations[3])

        self.global_avg_pool = tf.keras.Sequential([tf.keras.layers.AvgPool2D(avg_pool_kernel),
                                                    tf.keras.layers.Conv2D(256, 1, use_bias=False),
                                                    tf.keras.layers.BatchNormalization(),
                                                    tf.keras.layers.ReLU()])

        self.upsampling = tf.keras.layers.UpSampling2D(avg_pool_kernel, interpolation='bilinear')


        self.conv = tf.keras.layers.Conv2D(256, 1, use_bias=False)
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=0.5)

    def call(self, x, training=False):
        """
        Computes a forward pass through the aspp layer

        Args:
            x (tf.Tensor): input
            training (bool): true if we are in training mode

        Returns:
            x (tf.Tensor): result of the forward pass
        """

        x1 = self.assp1(x, training=training)
        x2 = self.assp2(x, training=training)
        x3 = self.assp3(x, training=training)
        x4 = self.assp4(x, training=training)
        x5 = self.global_avg_pool(x, training=training)
        x5 = self.upsampling(x5)

        x = tf.concat([x1, x2, x3, x4, x5], axis=3)
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.relu(x)

        return self.dropout(x, training=training)

class Decoder(tf.keras.layers.Layer):
    """Creates a Decoder module for Deeplabv3_plus"""

    def __init__(self, num_classes):
        """
        Initializes the decoder module

        Args:
            num_classes (int): final number of classes
        """
        super(Decoder, self).__init__()

        self.conv = tf.keras.layers.Conv2D(48, 1, use_bias=False)
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.upsampling = tf.keras.layers.UpSampling2D((4,4), interpolation='bilinear')

        self.last_conv = tf.keras.Sequential([tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False),
                                              tf.keras.layers.BatchNormalization(),
                                              tf.keras.layers.ReLU(),
                                              tf.keras.layers.SpatialDropout2D(0.5),
                                              tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False),
                                              tf.keras.layers.BatchNormalization(),
                                              tf.keras.layers.ReLU(),
                                              tf.keras.layers.SpatialDropout2D(0.1),
                                              tf.keras.layers.Conv2D(num_classes, kernel_size=1)])

    def call(self, x, low_level_feat, training=False):
        """
        Computes a forward pass through the decoder

        Args:
            x (tf.Tensor): input
            low_level_feat (tf.Tensor): low level features
            training (bool): true if we are in training mode

        Returns:
            x (tf.Tensor): result of the forward pass
        """

        low_level_feat = self.conv(low_level_feat)
        low_level_feat = self.norm(low_level_feat, training=training)
        low_level_feat = self.relu(low_level_feat)

        x = self.upsampling(x)
        x = tf.concat([x, low_level_feat], axis=3)
        x = self.last_conv(x, training=training)

        return x

class Deeplabv3_plus(tf.keras.Model):
    """Creates a Deeplabv3_plus model"""

    def __init__(self, args, num_classes):
        """
        Initializes Deeplabv3_plus

        Args:
            args (argparse.ArgumentParser): command line arguments
            num_classes (int): specifies the output channels of the last convolution
        """
        super(Deeplabv3_plus, self).__init__()

        self.args = args

        if args.model == DEEPLAB:
            self.backbone = ResNet101(args.output_stride, args.init_type)
            if self.args.refine_network:
                self.refine_backbone = ResNet101(args.output_stride, args.init_type)
        elif args.model == DEEPLAB_50:
            self.backbone = ResNet50(args.output_stride, args.init_type)
            if self.args.refine_network:
                self.refine_backbone = ResNet50(args.output_stride, args.init_type)
        else:
            raise NotImplementedError()

        self.aspp = ASPP(args.output_stride, self.args.resize)
        self.decoder = Decoder(num_classes)
        self.upsampling = tf.keras.layers.UpSampling2D((4,4), interpolation='bilinear')

        if self.args.refine_network:
            self.refine_aspp = ASPP(args.output_stride, self.args.resize)
            self.refine_decoder = Decoder(num_classes)

    def call(self, x, training=False):
        """
        Computes a forward pass through Deeplab

        Args:
            x (tf.Tensor): input
            training (bool): true if we are in training mode

        Returns:
            x (tf.Tensor): result of the forward pass
        """
        input_image = x

        x, low_level_feat = self.backbone(x, training=training)
        x = self.aspp(x, training=training)
        x = self.decoder(x, low_level_feat, training=training)
        x = self.upsampling(x)

        if self.args.refine_network:
            refined_x = tf.concat((input_image, x), axis=3)
            refined_x, low_level_feat_refine = self.refine_backbone(refined_x, training=training)
            refined_x = self.refine_aspp(refined_x, training=training)
            refined_x = self.refine_decoder(refined_x, low_level_feat_refine, training=training)
            refined_x = self.upsampling(refined_x)
            return x, refined_x

        return x
