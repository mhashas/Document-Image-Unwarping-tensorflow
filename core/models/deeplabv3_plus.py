import tensorflow as tf

from constants import *

class _ASPPModule(tf.keras.layers.Layer):
    pass

class ASPP(tf.keras.layers.Layer):
    """Creates an ASPP layer"""

    def __init__(self, out_channels, kernel_size, dilation):
        super(ASPP, self).__init__()
        self.atrous_conv = tf.keras.layers.Conv2D(out_channels, kernel_size=kernel_size, dilation_rate=dilation, padding='same', use_bias=False)

    def __init__(self, output_stride):
        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6 , 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError()



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

        if args.model == DEEPLAB_MOBILENET:
            pass

        if self.args.use_aspp:
            self.aspp = ASPP(args.output_stride)

        #self.decoder = Decoder(num_classes)