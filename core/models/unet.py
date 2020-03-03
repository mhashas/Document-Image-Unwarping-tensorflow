import tensorflow as tf
from constants import *

class UNetDownBlock(tf.keras.layers.Layer):
    """This class is used to instantiate a UNet Downsampling block"""

    def __init__(self, out_channels, down_type=MAXPOOL, outermost=False, innermost=False, dropout_rate=0.2, bias=True, init_type=KAIMING_INIT):
        """
        Initializes a UNet Downsampling block

        Args:
            out_channels (int): number of output channe;s
            down_type (string): method used for downsampling, either maxpool or strided convolutions
            outermost (bool): true if it's the outermost block
            innermost (bool): true if it's the innermost block
            dropout_rate (float):
            bias (bool): if the convolutions use bias
            init_type (str): type of kernel initializer
        """
        super(UNetDownBlock, self).__init__()

        self.innermost = innermost
        self.outermost = outermost
        self.use_maxpool = down_type == MAXPOOL
        strides = 1 if self.use_maxpool else 2
        kernel_size = 3 if self.use_maxpool else 4

        self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=bias, kernel_initializer=init_type)
        self.relu = tf.keras.layers.LeakyReLU(0.2)
        self.norm = tf.keras.layers.BatchNormalization()
        self.maxpool = tf.keras.layers.MaxPool2D()
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)


    def call(self, x, training=False):
        """
        Computes a forward pass through the block

        Args:
            x (tf.Tensor): input
            training (bool): true if we are in training mode

        Returns:
            x (tf.Tensor): result of the forward pass
        """

        if self.outermost:
            x = self.conv(x)
            x = self.norm(x, training=training)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x, training=training)
            x = self.conv(x)
        else:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x, training=training)
            x = self.conv(x)
            x = self.norm(x, training=training)

        return x

class UNetUpBlock(tf.keras.layers.Layer):
    """This class is used to instantiate a UNet Upsampling block"""

    def __init__(self, out_channels, outermost=False, innermost=False, dropout=0.2, kernel_size=4, bias=True, init_type=KAIMING_INIT):
        """
        Initializes a UNet Upsampling block

        Args:
            out_channels (int): number of output channe;s
            outermost (bool): true if it's the outermost block
            innermost (bool): true if it's the innermost block
            dropout_rate (float):
            bias (bool): if the convolutions use bias
            init_type (str): type of kernel initializer
        """
        super(UNetUpBlock, self).__init__()

        self.innermost = innermost
        self.outermost = outermost

        self.conv = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=kernel_size, strides=2, padding='same', use_bias=bias, kernel_initializer=init_type)
        self.relu = tf.keras.layers.ReLU()
        self.norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout)

    def call(self, x, training=False):
        """
        Computes a forward pass through the block

        Args:
            x (tf.Tensor): input
            training (bool): true if we are in training mode

        Returns:
            x (tf.Tensor): result of the forward pass
        """

        if self.outermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x, training=training)
            x = self.conv(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x, training=training)
            x = self.conv(x)
            x = self.norm(x, training=training)
        else:
            x = self.relu(x)
            if self.dropout: x= self.dropout(x, training=training)
            x = self.conv(x)
            x = self.norm(x, training=training)

        return x

class UNet(tf.keras.Model):
    """Create a Unet-based Fully Convolutional Network
          X -------------------identity----------------------
          |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, args, num_classes):
        """
        Initializes UNet

        Args:
            args (argparse.ArgumentParser): command line arguments
            num_classes (int): specifies the output channels of the last convolution
        """
        super(UNet, self).__init__()

        self.refine_network = args.refine_network
        self.num_downs = args.num_downs
        self.ngf = args.ngf

        self.encoder = self.build_encoder(self.num_downs, self.ngf, down_type=args.down_type, init_type=args.init_type)
        self.decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, init_type=args.init_type)

        if self.refine_network:
            self.refine_encoder = self.build_encoder(self.num_downs, self.ngf, down_type=args.down_type, init_type=args.init_type)
            self.refine_decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, init_type=args.init_type)

    def build_encoder(self, num_downs, ngf, down_type=STRIDECONV, init_type=KAIMING_INIT):
        """
        Constructs a UNet downsampling encoder, consisting of $num_downs UNetDownBlocks

        Args:
            num_downs (int): the number of downsaplings in UNet. For example, # if |num_downs| == 7, image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int): the number of filters in the last conv layer
            down_type (str): if we should use strided convolution or maxpool for reducing the feature map
            init_type (str): type of kernel initializer

        Returns:
            tf.keras.Sequential: $num_downs UnetDownBlocks
        """

        layers = [UNetDownBlock(out_channels=ngf, down_type=down_type, outermost=True, init_type=init_type),
                  UNetDownBlock(out_channels=ngf * 2, down_type=down_type, init_type=init_type),
                  UNetDownBlock(out_channels=ngf * 4, down_type=down_type, init_type=init_type),
                  UNetDownBlock(out_channels=ngf * 8, down_type=down_type, init_type=init_type)]

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UNetDownBlock(out_channels=ngf * 8, down_type=down_type, init_type=init_type))

        layers.append(UNetDownBlock(out_channels=ngf * 8, down_type=down_type, innermost=True, init_type=init_type))

        return tf.keras.Sequential(layers)

    def build_decoder(self, num_downs, num_classes, ngf, init_type=KAIMING_INIT):
        """
        Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

        Args:
            num_downs (int): the number of upsamplings in UNet. For example, # if |num_downs| == 7, featurex of size 1x1 will become of size 128x128
            num_classes (int): classes to categorize
            ngf (int): the number of filters in the first upconv layer
            init_type (str): type of kernel initializer

        Returns:
            nn.Sequential: $num_downs UnetUpBlocks
        """

        layers = [UNetUpBlock(out_channels=ngf * 8, innermost=True, init_type=init_type)]

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UNetUpBlock(out_channels=ngf * 8, init_type=init_type))

        layers.append(UNetUpBlock(out_channels=ngf * 4, init_type=init_type))
        layers.append(UNetUpBlock(out_channels=ngf * 2, init_type=init_type))
        layers.append(UNetUpBlock(out_channels=ngf, init_type=init_type))
        layers.append(UNetUpBlock(out_channels=num_classes, outermost=True, init_type=init_type))

        return tf.keras.Sequential(layers)

    def encoder_forward(self, x, training=False, use_refine_network=False):
        """
        Computes a forward pass through the encoder network

        Args:
            x (tf.Tensor): input tensor
            training (bool): useful for layers such batch norm or dropout
            use_refine_network (bool): determines which encoder we use

        Returns:
            tf.Tensor: output tensor
        """

        skip_connections = []
        model = self.refine_encoder if use_refine_network else self.encoder

        for i, down in enumerate(model.layers):
            x = down(x, training=training)
            if down.use_maxpool:
                x = down.maxpool(x)

            if not down.innermost:
                skip_connections.append(x)

        return x, skip_connections

    def decoder_forward(self, x, skip_connections, training=False, use_refine_network=False):
        """
        Computes a forward pass through the encoder network

        Args:
            x (tf.Tensor): input tensor
            skip_connections (list): list of tf.Tensor of encoder skip connections
            training (bool): useful for layers such batch norm or dropout
            use_refine_network (bool): determines which encoder we use

        Returns:
           tf.Tensor: output tensor
        """

        model = self.refine_decoder if use_refine_network else self.decoder

        for i, up in enumerate(model.layers):
            if not up.innermost:
                skip = skip_connections[-i]
                out = tf.concat([out, skip], 3)
                out = up(out, training=training)
            else:
                out = up(x, training=training)

        return out

    def call(self, x, training=False):
        """
        Computes a forward pass through the  netowrk

        Args:
           x (tf.Tensor): input tensor
           training (bool): useful for layers such batch norm or dropout

        Returns:
           tf.Tensor: output tensor
        """

        output, skip_connections = self.encoder_forward(x, training=training)
        output = self.decoder_forward(output, skip_connections, training=training)

        if self.refine_network:
            refined_output = tf.concat((x, output), axis=3)
            refined_output, refined_skip_connections = self.encoder_forward(refined_output, training=training, use_refine_network=True)
            refined_output = self.decoder_forward(refined_output, refined_skip_connections, training=training, use_refine_network=True)
            return output, refined_output

        return output