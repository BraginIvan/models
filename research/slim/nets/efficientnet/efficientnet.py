from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets.efficientnet import efficientnet_model
from nets.efficientnet import utils

slim = tf.contrib.slim


def efficientnet_edgetpu_params(model_name):
    """Get efficientnet-edgetpu params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-edgetpu-S': (1.0, 1.0, 224, 0.2),
        'efficientnet-edgetpu-M': (1.0, 1.1, 240, 0.2),
        'efficientnet-edgetpu-L': (1.2, 1.4, 300, 0.3),
    }
    return params_dict[model_name]


def efficientnet_edgetpu(width_coefficient=None,
                         depth_coefficient=None,
                         dropout_rate=0.2,
                         drop_connect_rate=0.2):
    """Creates an efficientnet-edgetpu model."""
    blocks_args = [
        'r1_k3_s11_e4_i24_o24_c1_noskip',
        'r2_k3_s22_e8_i24_o32_c1',
        'r4_k3_s22_e8_i32_o48_c1',
        'r5_k5_s22_e8_i48_o96',
        'r4_k5_s11_e8_i96_o144',
        'r2_k5_s22_e8_i144_o192',
        # 'r2_k5_s22_e8_i144_o192_noskip',

    ]
    global_params = efficientnet_model.GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        data_format='channels_last',
        num_classes=1001,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.relu6,
        batch_norm=utils.get_batch_norm(),
        use_se=False)
    decoder = utils.BlockDecoder()
    return decoder.decode(blocks_args), global_params


def get_model_params(model_name):
    """Get the block args and global params for a given model."""
    width_coefficient, depth_coefficient, _, dropout_rate = (
        efficientnet_edgetpu_params(model_name))
    blocks_args, global_params = efficientnet_edgetpu(width_coefficient,
                                                      depth_coefficient,
                                                      dropout_rate)

    tf.logging.info('global_params= %s', global_params)
    tf.logging.info('blocks_args= %s', blocks_args)
    return blocks_args, global_params


def efficient_net_base(inputs, model_name, scope=None):
    """A helper function to create a base model and return global_pool.

    Args:
      inputs: input images tensor.
      model_name: string, the model name of a pre-defined MnasNet.

    Returns:
      features: global pool features.
      endpoints: the endpoints for each layer.

    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(inputs, tf.Tensor)
    blocks_args, global_params = get_model_params(model_name)

    with tf.variable_scope(scope):
        model = efficientnet_model.Model(blocks_args, global_params)
        features = model(inputs, features_only=True)

    features = tf.identity(features, 'global_pool')
    return features, model.endpoints

from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope, arg_scoped_arguments

def efficient_net_arg_scope(
        is_training=True,
        weight_decay=0.00004,
        stddev=0.09,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=0.001,
        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
        normalizer_fn=slim.batch_norm):
    """Defines the default MobilenetV1 arg scope.

    Args:
      is_training: Whether or not we're training the model. If this is set to
        None, the parameter is not added to the batch_norm arg_scope.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
      batch_norm_updates_collections: Collection for the update ops for
        batch norm.
      normalizer_fn: Normalization function to apply after convolution.

    Returns:
      An `arg_scope` to use for the mobilenet v1 model.
    """

    batch_norm_params = {
        # 'center': True,
        # 'scale': True,
        # 'decay': batch_norm_decay,
        # 'epsilon': batch_norm_epsilon,
        # 'updates_collections': batch_norm_updates_collections,
    }
    if is_training is not None:
        batch_norm_params['is_training'] = is_training

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with slim.arg_scope([slim.conv2d],
                        weights_initializer=weights_init, normalizer_fn=normalizer_fn):
        with slim.arg_scope([utils.get_batch_norm, utils.drop_connect], **batch_norm_params):

            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer) as sc:
                return sc
