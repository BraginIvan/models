import tensorflow as tf
from nets.efficientnet import utils
import numpy as np



class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

    Attributes:
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params):
        """Initializes a MBConv block.

        Args:
          block_args: BlockArgs, arguments to create a Block.
          global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._batch_norm = global_params.batch_norm
        self._data_format = global_params.data_format
        if self._data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._relu_fn = global_params.relu_fn or tf.nn.swish

        self.endpoints = None

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=utils.conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False,
                # name="expand_conv"
            )
            self._bn0 = self._batch_norm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        self._depthwise_conv = utils.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=utils.conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False,
            # name="depthwise_conv"
        )

        self._bn1 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)


        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=utils.conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False,
            # name="project_conv"
        )
        self._bn2 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)



    def call(self, inputs, drop_connect_rate=None):
        """Implementation of call().

        Args:
          inputs: the inputs tensor.
          drop_connect_rate: float, between 0 to 1, drop connect rate.

        Returns:
          A output tensor.
        """
        tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            x = self._relu_fn(self._bn0(self._expand_conv(inputs)))
        else:
            x = inputs
        tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

        x = self._relu_fn(self._bn1(self._depthwise_conv(x)))
        tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))



        self.endpoints = {'expansion_output': x}

        x = self._bn2(self._project_conv(x))
        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = utils.drop_connect(x, drop_connect_rate=drop_connect_rate)
                x = tf.add(x, inputs)
        tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
        return x




class MBConvBlockWithoutDepthwise(MBConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                kernel_initializer=utils.conv_kernel_initializer,
                padding='same',
                use_bias=False)
            self._bn0 = self._batch_norm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=self._block_args.strides,
            kernel_initializer=utils.conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def call(self, inputs, training=None, drop_connect_rate=None):
        """Implementation of call().

        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          drop_connect_rate: float, between 0 to 1, drop connect rate.

        Returns:
          A output tensor.
        """
        tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            x = self._relu_fn(self._bn0(self._expand_conv(inputs)))
        else:
            x = inputs
        tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

        self.endpoints = {'expansion_output': x}

        x = self._bn1(self._project_conv(x))
        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x_drop = utils.drop_connect(x, drop_connect_rate=drop_connect_rate)
                    x = tf.add(x_drop, inputs)
                else:
                    x = tf.add(x, inputs)
        tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
        return x