# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.
Another difference is that 'v2' ResNets do not include an activation function in
the main pathway. Also see [2; Fig. 4e].

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from changed_scripts import resnet_utils

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=True, 
              include_root_block=True,
              spatial_squeeze=False, #TODO True -> False
              reuse=None,
              scope=None):
  """Generator for v2 (preactivation) ResNet models.

  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.


  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
        if spatial_squeeze:
          logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = slim.softmax(logits, scope='predictions')
        
        return net, end_points # TODO logits-> net
        
    
resnet_v2.default_image_size = 224


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
resnet_v2_50.default_image_size = resnet_v2.default_image_size


def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
resnet_v2_101.default_image_size = resnet_v2.default_image_size


def resnet_v2_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
resnet_v2_152.default_image_size = resnet_v2.default_image_size


def resnet_v2_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
resnet_v2_200.default_image_size = resnet_v2.default_image_size


#######################################################################################################
############################### Intermediate Fully connected layers ###################################
#######################################################################################################

cutting_points = ['conv1', 'pool1', 
                  'block1_unit1_conv1', 'block1_unit1_conv2', 'block1_unit1_conv3', 'block1_unit1',
                  'block1_unit2_conv1', 'block1_unit2_conv2', 'block1_unit2_conv3', 'block1_unit2', 
                  'block1_unit3_conv1', 'block1_unit3_conv2', 'block1_unit3_conv3', 'block1_unit3',
                  'block2_unit1_conv1', 'block2_unit1_conv2', 'block2_unit1_conv3', 'block2_unit1',
                  'block2_unit2_conv1', 'block2_unit2_conv2', 'block2_unit2_conv3', 'block2_unit2', 
                  'block2_unit3_conv1', 'block2_unit3_conv2', 'block2_unit3_conv3', 'block2_unit3',
                  'block2_unit4_conv1', 'block2_unit4_conv2', 'block2_unit4_conv3', 'block2_unit4',
                  'block3_unit1_conv1', 'block3_unit1_conv2', 'block3_unit1_conv3', 'block3_unit1',
                  'block3_unit2_conv1', 'block3_unit2_conv2', 'block3_unit2_conv3', 'block3_unit2', 
                  'block3_unit3_conv1', 'block3_unit3_conv2', 'block3_unit3_conv3', 'block3_unit3',
                  'block3_unit4_conv1', 'block3_unit4_conv2', 'block3_unit4_conv3', 'block3_unit4',                  
                  'block3_unit5_conv1', 'block3_unit5_conv2', 'block3_unit5_conv3', 'block3_unit5',
                  'block3_unit6_conv1', 'block3_unit6_conv2', 'block3_unit6_conv3', 'block3_unit6',
                  'block4_unit1_conv1', 'block4_unit1_conv2', 'block4_unit1_conv3', 'block4_unit1',
                  'block4_unit2_conv1', 'block4_unit2_conv2', 'block4_unit2_conv3', 'block4_unit2', 
                  'block4_unit3_conv1', 'block4_unit3_conv2', 'block4_unit3_conv3', 'block4_unit3',
                  'postnorm',
                  'pool5', 
                  'my_fc_2', 'my_fc_3', 'my_fc_4', 'my_fc_5',
                  'concat',
                  'maxout',
                  'my_fc_1'
                  ]


def bottleneck_intermediate(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None, fc_after=None, cutting_points=None, name=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  cut_point = cutting_points.index(fc_after) # Where to cut the network off
  
  
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    if cut_point >= cutting_points.index( ('%s_conv1' % (name))):
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                            scope='conv1')
    else:
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            shortcut)
        
        
        
    if cut_point >= cutting_points.index(('%s_conv2' % (name))):    
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
    else:
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            residual)    
        
    if cut_point >= cutting_points.index(('%s_conv3' % (name))):     
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')
    else:
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            residual)    
    
    if cut_point >= cutting_points.index(name):  
        output = shortcut + residual
    else:
        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            residual)    

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)



def resnet_v2_intermediate(inputs,
              blocks,
              fc_after = None,
              dropout_rate= 0.5,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=True, 
              include_root_block=True,
              spatial_squeeze=False, #TODO True -> False
              reuse=None,
              scope=None):
  """Generator for v2 (preactivation) ResNet models.

  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.


  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  cut_point = cutting_points.index(fc_after) # Where to cut the network off
  
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          
          
          if cut_point >= cutting_points.index( 'conv1'):          
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None, normalizer_fn=None):
                net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          else:
              net = tf.nn.dropout(net, dropout_rate) 
              net= slim.flatten(net) #TODO
              
              net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
              end_points = slim.utils.convert_collection_to_dict(end_points_collection)
              return net, end_points
                
                
          if cut_point >= cutting_points.index( 'pool1'):      
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            
          else:
              net = tf.nn.dropout(net, dropout_rate) 
              net= slim.flatten(net)
              
              net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
              end_points = slim.utils.convert_collection_to_dict(end_points_collection)
              return net , end_points
          
          
        if cut_point >= cutting_points.index( 'block1_unit1_conv1'):  
            net = resnet_utils.stack_blocks_dense_intermediate(net=net, blocks=blocks, output_stride=output_stride, fc_after=fc_after, cutting_points=cutting_points)

        else:
            net = tf.nn.dropout(net, dropout_rate) 
            net= slim.flatten(net)
            
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points
        
        
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        
        if cut_point >= cutting_points.index('postnorm'):            
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

        else:
            net = tf.nn.dropout(net, dropout_rate) 
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
              
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points
            
        
        if global_pool:
          # Global average pooling.
          if cut_point > cutting_points.index( 'pool5'):
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          else:
              net = tf.nn.dropout(net, dropout_rate) 
              net= slim.flatten(net)
              
              net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
 
              end_points = slim.utils.convert_collection_to_dict(end_points_collection)
              return net, end_points
          
          
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        
        return net, end_points # TODO logits-> net


def resnet_v2_50_intermediate(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50',
                 fc_after = None, 
                 dropout_rate=0.5):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  cut_point = cutting_points.index(fc_after) # Where to cut the network off
  
  blocks = []
  
  if cut_point >= cutting_points.index( 'block1_unit1_conv1'):
      blocks.append(resnet_utils.Block(
          'block1', bottleneck_intermediate, [(256, 64, 1)] * 2 + [(256, 64, 2)]))
  if cut_point >= cutting_points.index( 'block2_unit1_conv1'):
      blocks.append(resnet_utils.Block(
          'block2', bottleneck_intermediate, [(512, 128, 1)] * 3 + [(512, 128, 2)]))
  if cut_point >= cutting_points.index( 'block3_unit1_conv1'):
      blocks.append(resnet_utils.Block(
          'block3', bottleneck_intermediate, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]))
  if cut_point >= cutting_points.index( 'block4_unit1_conv1'):
      blocks.append(resnet_utils.Block(
          'block4', bottleneck_intermediate, [(2048, 512, 1)] * 3))
      
 
  return resnet_v2_intermediate(inputs, blocks, num_classes=num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope, fc_after=fc_after, dropout_rate = 0.5)

