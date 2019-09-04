from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from nets.efficientnet import efficientnet_model
from nets.efficientnet import utils

# import utils

slim = tf.contrib.slim

MEAN_RGB = [127.0, 127.0, 127.0]
STDDEV_RGB = [128.0, 128.0, 128.0]


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
        relu_fn=tf.nn.relu,
        batch_norm=tf.layers.BatchNormalization,
        use_se=False)
    decoder = utils.BlockDecoder()
    return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet-edgetpu'):
        width_coefficient, depth_coefficient, _, dropout_rate = (
            efficientnet_edgetpu_params(model_name))
        blocks_args, global_params = efficientnet_edgetpu(width_coefficient,
                                                          depth_coefficient,
                                                          dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    tf.logging.info('global_params= %s', global_params)
    tf.logging.info('blocks_args= %s', blocks_args)
    return blocks_args, global_params


def efficient_net_base(inputs, model_name, training, scope = None, override_params=None):
    # final_endpoint = 'Conv2d_13_pointwise',
    #                       min_depth=8,
    #                       conv_defs=None,
    #                       output_stride=None,
    #                       use_explicit_padding=False,
    #                       scope=None
    """A helper function to create a base model and return global_pool.

    Args:
      inputs: input images tensor.
      model_name: string, the model name of a pre-defined MnasNet.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        mnasnet_model.GlobalParams.

    Returns:
      features: global pool features.
      endpoints: the endpoints for each layer.

    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(inputs, tf.Tensor)
    blocks_args, global_params = get_model_params(model_name, override_params)

    with tf.variable_scope(scope):
        model = efficientnet_model.Model(blocks_args, global_params)
        features = model(inputs, training=training, features_only=True)

    features = tf.identity(features, 'global_pool')
    return features, model.endpoints


def efficient_net(inputs,
                  model_name,
                  is_training=True,
                  override_params=None,
                  model_dir=None,
                  fine_tuning=False):
    # num_classes = 1000,
    #     dropout_keep_prob = 0.999,
    #     min_depth = 8,
    #     depth_multiplier = 1.0,
    #     conv_defs = None,
    #     prediction_fn = tf.contrib.layers.softmax,
    #     spatial_squeeze = True,
    #     reuse = None,
    #     scope = 'MobilenetV1',
    #     global_pool = False

    """A helper functiion to creates a model and returns predicted logits.

    Args:
      inputs: input images tensor.
      model_name: string, the predefined model name.
      is_training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        efficientnet_model.GlobalParams.
      model_dir: string, optional model dir for saving configs.
      fine_tuning: boolean, whether the model is used for finetuning.

    Returns:
      logits: the logits tensor of classes.
      endpoints: the endpoints for each layer.

    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(inputs, tf.Tensor)
    if not is_training or fine_tuning:
        if not override_params:
            override_params = {}
        override_params['batch_norm'] = tf.layers.BatchNormalization
    blocks_args, global_params = get_model_params(model_name, override_params)
    if not is_training or fine_tuning:
        global_params = global_params._replace(batch_norm=tf.layers.BatchNormalization)

    if model_dir:
        param_file = os.path.join(model_dir, 'model_params.txt')
        if not tf.gfile.Exists(param_file):
            if not tf.gfile.Exists(model_dir):
                tf.gfile.MakeDirs(model_dir)
            with tf.gfile.GFile(param_file, 'w') as f:
                tf.logging.info('writing to %s' % param_file)
                f.write('model_name= %s\n\n' % model_name)
                f.write('global_params= %s\n\n' % str(global_params))
                f.write('blocks_args= %s\n\n' % str(blocks_args))

    with tf.variable_scope(model_name):
        model = efficientnet_model.Model(blocks_args, global_params)
        logits = model(inputs, training=is_training)

    logits = tf.identity(logits, 'logits')
    return logits, model.endpoints


efficient_net.default_image_size = 224

# efficient_net_arg_scope = ??

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
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'updates_collections': batch_norm_updates_collections,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

  with slim.arg_scope([slim.conv2d],
                      weights_initializer=weights_init, normalizer_fn=normalizer_fn):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer) as sc:
          return sc
