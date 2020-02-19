import tensorflow as tf

from constants import *

class BasicBlock(tf.keras.layers.Layer):
    """Creates a ResNet basic block"""

    expansion = 1

    def __init__(self, out_channels, stride=1, dilation=1, downsample=None, init_type=KAIMING_INIT):
        """
        Initializes the BasicBlock

        Args:
            out_channels (int): final convolution output channels
            stride (int): convolution stride
            dilation (int): convolution dilation rate
            downsample (tf.Sequential): downsampling module
        """
        super(BasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(out_channels, strides=stride, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init_type)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(out_channels, strides=stride, kernel_size=3, padding="same", use_bias=False, kernel_initializer=init_type)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        """
        Computes a forward pass through the  netowrk

        Args:
           x (tf.Tensor): input tensor
           training (bool): useful for layers such batch norm or dropout

        Returns:
           tf.Tensor: output tensor
        """

        residual = x

        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)

        if self.downsample is not None:
            residual = self.downsample(residual, training=training)

        x = x + residual
        x = self.relu(x)

        return x

class Bottleneck(tf.keras.layers.Layer):
    """Creates a Resnet Bottleneck"""

    expansion = 4

    def __init__(self, out_channels, stride=1, dilation=1, downsample=None, init_type=KAIMING_INIT):
        """
        Initializes the Bottleneck

        Args:
            out_channels (int): final convolution output channels
            stride (int): convolution stride
            dilation (int): convolution dilation rate
            downsample (tf.keras.Sequential): downsampling module
        """
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False, kernel_initializer=init_type)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=stride, dilation_rate=dilation, padding="same", use_bias=False, kernel_initializer=init_type)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(out_channels * 4, kernel_size=1, use_bias=False, kernel_initializer=init_type)
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def call(self, x, training=False):
        """
        Computes a forward pass through the  netowrk

        Args:
           x (tf.Tensor): input tensor
           training (bool): useful for layers such batch norm or dropout

        Returns:
           tf.Tensor: output tensor
        """

        residual = x

        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x, training=training)

        if self.downsample is not None:
            residual = self.downsample(residual, training=training)

        x = x + residual
        x = self.relu(x)

        return x

class ResNet(tf.keras.models.Model):
    """Create a ResNet model"""

    def __init__(self, block, layers, output_stride, init_type=KAIMING_INIT):
        """
        Initializes the Resnet model

        Args:
            block (tf.keras.layer.Layer): Resnet block
            layers (list): number of layers
            output_stride (int): determines the dilation rates of the convolutions
        """
        super (ResNet, self).__init__()

        self.in_channels = 64
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError()

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False, kernel_initializer=init_type)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], init_type=init_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], init_type=init_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], init_type=init_type)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], init_type=init_type)

    def _make_layer(self, block, out_channels, blocks, stride=1, dilation=1, init_type=KAIMING_INIT):
        """
        Creates a ResNet layer

        Args:
            block (tf.keras.layer.Layer): ResNet block
            out_channels (int): number of output channels
            blocks (list): blocks per layer
            stride (int): convolution stride
            dilation (int): convolution dilation rate
            init_type (str): type of kernel initializer

        Returns:
            tf.keras.Sequential: ResNet layer
        """

        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = tf.keras.Sequential([tf.keras.layers.Conv2D(out_channels * block.expansion, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=init_type),
                                              tf.keras.layers.BatchNormalization()])

        layers = []
        layers.append(block(out_channels, stride, dilation, downsample=downsample, init_type=init_type))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(out_channels, dilation=dilation, init_type=init_type))

        return tf.keras.Sequential(layers)

    def _make_MG_unit(self, block, out_channels, blocks, stride=1, dilation=1, init_type=KAIMING_INIT):
        """
        Creates the last ResNet Layer

        Args:
            block (tf.keras.layer.Layer): ResNet block
            out_channels (int): number of output channels
            blocks (list): blocks per layer
            stride (int): convolution stride
            dilation (int): convolution dilation rate
            init_type (str): type of kernel initializer

        Returns:
            tf.keras.Sequential: ResNet layer
        """

        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = tf.keras.Sequential([tf.keras.layers.Conv2D(out_channels * block.expansion, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=init_type),
                                              tf.keras.layers.BatchNormalization()])

        layers = []
        layers.append(block(out_channels, stride, dilation=blocks[0]*dilation, downsample=downsample))
        self.in_channels = out_channels * block.expansion

        for i in range(1, len(blocks)):
            layers.append(block(out_channels, dilation=blocks[i]*dilation))

        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        """
        Computes a forward pass through the  netowrk

        Args:
           x (tf.Tensor): inut tensor
           training (bool): useful for layers such batch norm or dropout

        Returns:
           tf.Tensor: output tensor
           tf.Tensor: low_level features tensor
        """

        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, training=training)
        low_level_feat = x
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        return x, low_level_feat

def ResNet50(output_stride, init_type=KAIMING_INIT):
    """
    Creates ResNet 50

    Args:
        output_stride (int): determines the dilation rates of the convolutions
        init_type (str): type of kernel initializer

    Returns:
        tf.keras.Model: ResNet 50
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, init_type=init_type)

    return model

def ResNet101(output_stride, init_type=KAIMING_INIT):
    """
    Creates ResNet 101

    Args:
        output_stride (int): determines the dilation rates of the convolutions
        init_type (str): type of kernel initializer

    Returns:
        tf.keras.Model: ResNet 101
    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, init_type=init_type)

    return model

