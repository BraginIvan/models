# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nets.efficientnet.conv_block import MBConvBlock, MBConvBlockWithoutDepthwise

from nets.efficientnet import utils

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'relu_fn',
    'batch_norm', 'use_se',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type',
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    tf.logging.info('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))





class Model(tf.keras.Model):
    """A class implements tf.keras.Model for MNAS-like model.

      Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self, blocks_args=None, global_params=None):
        """Initializes an `Model` instance.

        Args:
          blocks_args: A list of BlockArgs to construct block modules.
          global_params: GlobalParams, a set of global parameters.

        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(Model, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._batch_norm = global_params.batch_norm

        self.endpoints = None

        self._build()

    def _get_conv_block(self, conv_type):
        conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
        return conv_block_map[conv_type]

    def _build(self):
        """Builds a model."""
        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))

            # The first block needs to take care of stride and filter size increase.
            conv_block = self._get_conv_block(block_args.conv_type)
            self._blocks.append(conv_block(block_args, self._global_params))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in xrange(block_args.num_repeat - 1):
                self._blocks.append(conv_block(block_args, self._global_params))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem part.
        self._conv_stem = tf.layers.Conv2D(
            filters=round_filters(32, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=utils.conv_kernel_initializer,
            padding='same',
            data_format=self._global_params.data_format,
            use_bias=False)
        self._bn0 = self._batch_norm(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        # Head part.
        self._conv_head = tf.layers.Conv2D(
            filters=round_filters(1280, self._global_params),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=utils.conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = self._batch_norm(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self._global_params.data_format)
        if self._global_params.num_classes:
            self._fc = tf.layers.Dense(
                self._global_params.num_classes,
                kernel_initializer=utils.dense_kernel_initializer)
        else:
            self._fc = None

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=None, features_only=None):
        # training = True
        """Implementation of call().

        Args:
          inputs: input tensors.
          training: boolean, whether the model is constructed for training.
          features_only: build the base feature network only.

        Returns:
          output tensors.
        """
        self.endpoints = {}
        # Calls Stem layers
        with tf.variable_scope('stem'):
            outputs = self._relu_fn(
                self._bn0(self._conv_stem(inputs)))
        tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
        self.endpoints['stem'] = outputs

        # Calls blocks.
        reduction_idx = 0
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if ((idx == len(self._blocks) - 1) or
                    self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            with tf.variable_scope('blocks_%s' % idx):
                drop_rate = self._global_params.drop_connect_rate
                if drop_rate:
                    drop_rate *= float(idx) / len(self._blocks)
                    tf.logging.info('block_%s drop_connect_rate: %s' % (idx, drop_rate))
                outputs = block.call(
                    outputs, drop_connect_rate=drop_rate)
                self.endpoints['block_%s' % idx] = outputs
                if is_reduction:
                    self.endpoints['reduction_%s' % reduction_idx] = outputs
                if block.endpoints:
                    for k, v in six.iteritems(block.endpoints):
                        self.endpoints['block_%s/%s' % (idx, k)] = v
                        if is_reduction:
                            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        self.endpoints['features'] = outputs


        return outputs
