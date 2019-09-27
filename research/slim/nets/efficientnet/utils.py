import tensorflow as tf
from nets.efficientnet import efficientnet_model
import re
import numpy as np

from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function

class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return efficientnet_model.BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])],
            conv_type=int(options['c']) if 'c' in options else 0)

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % block.conv_type,
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.

        Args:
          string_list: a list of strings, each string is a notation of block.

        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.

        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    pass


@add_arg_scope
def drop_connect(inputs, drop_connect_rate, is_training=False):
    """Apply drop connect."""
    print('drop_connect is_training: '  + str(is_training))

    if not is_training:
        return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    # random_tensor = tf.add(random_tensor, tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype), name="drop_connect_rate_add")
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.

    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


@add_arg_scope
def get_batch_norm(is_training=None):
    class TpuBatchNormalization(tf.layers.BatchNormalization):
        # class TpuBatchNormalization(tf.layers.BatchNormalization):
        """Cross replica batch normalization."""

        def __init__(self, fused=None, **kwargs):
            # if fused in (True, None):
            #     raise ValueError('TpuBatchNormalization does not support fused=True.')
            super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

        def _cross_replica_average(self, t, num_shards_per_group):
            """Calculates the average value of input tensor across TPU replicas."""
            num_shards = tpu_function.get_tpu_context().number_of_shards
            group_assignment = None
            if num_shards_per_group > 1:
                if num_shards % num_shards_per_group != 0:
                    raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
                                     % (num_shards, num_shards_per_group))
                num_groups = num_shards // num_shards_per_group
                group_assignment = [[
                    x for x in range(num_shards) if x // num_shards_per_group == y
                ] for y in range(num_groups)]
            return tpu_ops.cross_replica_sum(t, group_assignment) / tf.cast(
                num_shards_per_group, t.dtype)

        def _moments(self, inputs, reduction_axes, keep_dims):
            """Compute the mean and variance: it overrides the original _moments."""
            shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
                inputs, reduction_axes, keep_dims=keep_dims)

            num_shards = tpu_function.get_tpu_context().number_of_shards or 1
            if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
                num_shards_per_group = 1
            else:
                num_shards_per_group = max(8, num_shards // 8)
            tf.logging.info('TpuBatchNormalization with num_shards_per_group %s',
                            num_shards_per_group)
            if num_shards_per_group > 1:
                # Compute variance using: Var[X]= E[X^2] - E[X]^2.
                shard_square_of_mean = tf.math.square(shard_mean)
                shard_mean_of_square = shard_variance + shard_square_of_mean
                group_mean = self._cross_replica_average(
                    shard_mean, num_shards_per_group)
                group_mean_of_square = self._cross_replica_average(
                    shard_mean_of_square, num_shards_per_group)
                group_variance = group_mean_of_square - tf.math.square(group_mean)
                return (group_mean, group_variance)
            else:
                return (shard_mean, shard_variance)

        def call(self, inputs, training=is_training):
            return super(TpuBatchNormalization, self).call(inputs, training=training)

    return TpuBatchNormalization

    # print('get_batch_norm is_training: '  + str(is_training))
    # class BatchNormalization(tf.layers.BatchNormalization):
    #
    #     def __init__(self, **kwargs):
    #         super(BatchNormalization, self).__init__(fused=False, **kwargs)
    #
    #     def call(self, inputs, training=is_training):
    #         return super(BatchNormalization, self).call(inputs, training=training)
    # return BatchNormalization
