#Visualization of Kernels: https://arxiv.org/pdf/1311.2901.pdf

# Zeiler
#https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf

# Conv1 Kernels
# https://gist.github.com/kukuruza/03731dc494603ceab0c5

# Deeper Filters
#https://gist.github.com/awjuliani/acde9d491658265c3fbf6a74b91518e3#file-deep-layer-visualization-ipynb

# https://github.com/grishasergei/conviz

# https://github.com/tensorflow/tensorflow/issues/908

#https://gist.github.com/kukuruza/bb640cebefcc550f357c

# Deep Dream
#   
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb  #generate images that maximize the sum of activations of particular channel of a 
                                                                                                                #particular convolutional layer of the neural network
# Different Visualizations
# http://cs231n.github.io/understanding-cnn/
# t-SNE:    http://lvdmaaten.github.io/tsne/
#           http://www.sciencedirect.com/science/article/pii/S1046202314003211
#           http://cs.stanford.edu/people/karpathy/cnnembed/

# https://files.daylen.com/visualizing-residual-networks.pdf

from __future__ import print_function
from io import BytesIO
from functools import partial
import PIL.Image
from google.protobuf import text_format
import optparse
#from IPython.display import clear_output, Image, display, HTML

import math
import numpy as np
import time
import os
import matplotlib 
import matplotlib.pyplot as plt
import argparse
from random import randint

import tensorflow as tf
import resNetClassifier
from tensorflow.python.framework import graph_util
from changed_scripts   import dataset_utils
# Main slim library
slim = tf.contrib.slim

import my_functions

################################################################################
################# Visualization of first convolutional layer ###################
################################################################################


# Source: https://gist.github.com/kukuruza/03731dc494603ceab0c5
def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(math.sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x


#
# ... and somewhere inside "def train():" after calling "inference()"
#
"""
with tf.variable_scope('conv1'):
  tf.get_variable_scope().reuse_variables()
  weights = tf.get_variable('weights')
  grid = put_kernels_on_grid (weights)
tf.image.summary('conv1/kernels', grid, max_outputs=1)
"""

################################################################################
#################################  (Deep Dream) ################################
################################################################################
# Source : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb





################################################################################

# Helper functions for TF Graph visualization 
def rename_nodes(graph_def, rename_func):
    '''Re-names the nodes in graph_def using the rename func
    Args:
        graph_def:      GraphDef containing the nodes of the graph
        rename_func:    Function on how to rename the nodes of the graph
    Return:
        graph_def with new names
    '''
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def

# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
#tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))


################################################################################
################# Naive feature visualization (Deep Dream) #####################
################################################################################



# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0
   
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0, filename='images/naive_foo.png'):
    """
    Given a channel (combination of channels) t_obj, a image img0 is rendered to maximize the channels activations
    
    Args:
        t_obj: channel or combination of channels
        img0: original image
        iter_n: number of iterations
        step: step size
        filename: where to save the generated images
        
    Returns:
        Saves the generated image to filename
    """
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    
    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
        print(score, end = ' ')
    
    fig = plt.figure()
    plt.imshow(visstd(img))
    #plt.show()
    plt.savefig(filename)
    plt.close(fig)
    


    
################################################################################
################# Multiscale image generation (Deep Dream) #####################
################################################################################

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_multiscale(t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, filename='images/multiscale_foo.png'):
    """
    Given a channel (combination of channels) t_obj, a image img0 is rendered to maximize the channels activations
    
    Args:
        t_obj: channel or combination of channels
        img0: original image
        iter_n: number of iterations
        step: step size
        filename: where to save the generated images
        
    Returns:
        Saves the generated image to filename
    """
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
            print('.', end = ' ')
            
    
    fig = plt.figure()
    plt.imshow(visstd(img))
    plt.savefig(filename)
    #plt.show() 
    plt.close(fig)



################################################################################
########### Laplacian Pyramid Gradient Normalization (Deep Dream) ##############
################################################################################



def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]


def render_lapnorm(t_obj, img0=img_noise, visfunc=visstd,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4,filename='images/laplacian_foo.png'):
    """
    Given a channel (combination of channels) t_obj, a image img0 is rendered to maximize the channels activations
    
    Args:
        t_obj: channel or combination of channels
        img0: original image
        iter_n: number of iterations
        step: step size
        filename: where to save the generated images
        
    Returns:
        Saves the generated image to filename
    """
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    
    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g*step
            print('.', end = ' ')
    fig = plt.figure()
    plt.imshow(visstd(img))
    plt.savefig(filename)
    #plt.show()
    plt.close(fig)

    

################################################################################
#####################  Deep Dream to find Prototype  ###########################
################################################################################    
    
def render_deepdream(t_obj, img0=img_noise,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4, filename="optimal/foo.png"):
    """
    Given a channel (combination of channels) t_obj, a image img0 is rendered to maximize the channels activations
    
    Args:
        t_obj: channel or combination of channels
        img0: original image
        iter_n: number of iterations
        step: step size
        octave_n:
        octave_scale:
        filename: where to save the generated images
        
    Returns:
        Saves the generated image to filename
    """
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = ' ')
        #clear_output()
    fig = plt.figure()
    plt.imshow(visstd(img))
    plt.savefig(filename)
    #plt.show()
    plt.close(fig)
    
     

###########################################################

def adjust_most_accurate(most_activating_images):
    """
    Given the images that activate a channel the most, the deepdream algorithm is used to excite this level even further
    
    Args:
        most_activating_images: dicitonary containing the images that excite the given channels the most
        
    Returns:
        Saves the generated image 
    """
    
    
    for k,v in most_activating_images.items():

        
        acc_before = str.split(v, ",")[0]
        acc_before = str.split(acc_before, ".")[1]
        image_name = str.split(v, "'")[1]
        
        print("Processing image %s with an previous probability of %s." %(image_name, acc_before))
        
        channel = k
        filepath = ("mydata/train/%s" % (image_name))
        filename = ("optimal_stimuli/class_%s_before_acc_%s.jpg" % (k, acc_before))
    
        #print(filepath)
        #print(filename)
        for i in range(0, 28):
            filename = ("optimal_stimuli/class_%s_before_acc_%s_%s.jpg" % (k, acc_before, i))
            render_deepdream(T(layer)[i,channel] ,my_functions.get_img(filepath), octave_n = 10, iter_n = 50, filename=filename)  


    
if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option(
        '-a', '--adjust_most_accurate',
        help = "Do you want to adjust the most accurate images (adjust) or visualize channels (visualize)?",
        default = "adjust")
    parser.add_option(
        '-m', '--method',
        help="Method used. One of 'naive', 'multiscale', 'laplacian'", 
        default='laplacian')
    parser.add_option(
        '-l', '--layer',
        help="Which layer should be visualized (number from 0 to 52)", 
        default=0)
    parser.add_option(
        '-c', '--channel',
        help="Which channel should be visualized (1:number of channels). To visualize x random channels, type '-x', to visualize all, type '0'", 
        default=1)    
    parser.add_option(
        '-n', '--name',
        help="Filename of generate image", 
        default=1)
    
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession(graph=graph)
    
        
    model_fn =  'mydata/resnet_finetuned_plantclef2015_2/frozen_model.pb' #'/home/lolek/Downloads/tensorflow_inception_graph.pb' #

    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())       
                
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, input_map={'batch':t_preprocessed}) # resnet_v2_50/Pad

    #layers = [op.name for op in graph.get_operations() if 'import/' in op.name]
    #print(layers)
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    #print(layers)
    
    opts, _ = parser.parse_args()
    
    filename = opts.name
    
    if opts.adjust_most_accurate == "visualize":

            
        # Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
        # to have non-zero gradients for features with negative initial activations.
        layer =  'resnet_v2_50/block4/unit_3/bottleneck_v2/conv3/convolution'#'resnet_v2_50/block4/unit_3/bottleneck_v2/conv3/convolution' #'mixed4d_3x3_bottleneck_pre_relu'#
        channel = int(opts.channel) # picking some feature channel to visualize
        
            
        k = np.float32([1,4,6,4,1])
        k = np.outer(k, k)
        k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
        
        #render_deepdream(T('my_fc_1/BiasAdd')[:,80], my_functions.get_img("mydata/train/2.jpg"), octave_n = 10, iter_n = 100)
        
        
        all_methods = {'naive':render_naive, 'multiscale':render_multiscale, 'laplacian':render_lapnorm}
        used_method = all_methods.get(opts.method, "laplacian")
        
        all_layers = ['resnet_v2_50/conv1/convolution',                                     #0
                      'resnet_v2_50/block1/unit_1/bottleneck_v2/shortcut/convolution',      #1
                      'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/convolution',         #2
                      'resnet_v2_50/block1/unit_1/bottleneck_v2/conv2/convolution',         #3
                      'resnet_v2_50/block1/unit_1/bottleneck_v2/conv3/convolution',         #4
                      'resnet_v2_50/block1/unit_2/bottleneck_v2/conv1/convolution',         #5
                      'resnet_v2_50/block1/unit_2/bottleneck_v2/conv2/convolution',         #6
                      'resnet_v2_50/block1/unit_2/bottleneck_v2/conv3/convolution',         #7
                      'resnet_v2_50/block1/unit_3/bottleneck_v2/conv1/convolution',         #8
                      'resnet_v2_50/block1/unit_3/bottleneck_v2/conv2/convolution',     #9
                      'resnet_v2_50/block1/unit_3/bottleneck_v2/conv3/convolution',     #10
                      'resnet_v2_50/block2/unit_1/bottleneck_v2/shortcut/convolution',  #11
                      'resnet_v2_50/block2/unit_1/bottleneck_v2/conv1/convolution',     #12
                      'resnet_v2_50/block2/unit_1/bottleneck_v2/conv2/convolution',     #13
                      'resnet_v2_50/block2/unit_1/bottleneck_v2/conv3/convolution',     #14
                      'resnet_v2_50/block2/unit_2/bottleneck_v2/conv1/convolution',     #15
                      'resnet_v2_50/block2/unit_2/bottleneck_v2/conv2/convolution',     #16
                      'resnet_v2_50/block2/unit_2/bottleneck_v2/conv3/convolution',     #17
                      'resnet_v2_50/block2/unit_3/bottleneck_v2/conv1/convolution',     #18
                      'resnet_v2_50/block2/unit_3/bottleneck_v2/conv2/convolution',     #19
                      'resnet_v2_50/block2/unit_3/bottleneck_v2/conv3/convolution',     #20
                      'resnet_v2_50/block2/unit_4/bottleneck_v2/conv1/convolution',     #21
                      'resnet_v2_50/block2/unit_4/bottleneck_v2/conv2/convolution',     #22 
                      'resnet_v2_50/block2/unit_4/bottleneck_v2/conv3/convolution',     #23
                      'resnet_v2_50/block3/unit_1/bottleneck_v2/shortcut/convolution',  #24
                      'resnet_v2_50/block3/unit_1/bottleneck_v2/conv1/convolution',     #25
                      'resnet_v2_50/block3/unit_1/bottleneck_v2/conv2/convolution',     #26
                      'resnet_v2_50/block3/unit_1/bottleneck_v2/conv3/convolution',     #27
                      'resnet_v2_50/block3/unit_2/bottleneck_v2/conv1/convolution',     #28
                      'resnet_v2_50/block3/unit_2/bottleneck_v2/conv2/convolution',     #29
                      'resnet_v2_50/block3/unit_2/bottleneck_v2/conv3/convolution',     #30
                      'resnet_v2_50/block3/unit_3/bottleneck_v2/conv1/convolution',     #31
                      'resnet_v2_50/block3/unit_3/bottleneck_v2/conv2/convolution',     #32
                      'resnet_v2_50/block3/unit_3/bottleneck_v2/conv3/convolution',     #33
                      'resnet_v2_50/block3/unit_4/bottleneck_v2/conv1/convolution',     #34
                      'resnet_v2_50/block3/unit_4/bottleneck_v2/conv2/convolution',     #35
                      'resnet_v2_50/block3/unit_4/bottleneck_v2/conv3/convolution',     #36
                      'resnet_v2_50/block3/unit_5/bottleneck_v2/conv1/convolution',     #37
                      'resnet_v2_50/block3/unit_5/bottleneck_v2/conv2/convolution',     #38
                      'resnet_v2_50/block3/unit_5/bottleneck_v2/conv3/convolution',     #39
                      'resnet_v2_50/block3/unit_6/bottleneck_v2/conv1/convolution',     #40
                      'resnet_v2_50/block3/unit_6/bottleneck_v2/conv2/convolution',     #41
                      'resnet_v2_50/block3/unit_6/bottleneck_v2/conv3/convolution',     #42
                      'resnet_v2_50/block4/unit_1/bottleneck_v2/shortcut/convolution',  #43
                      'resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/convolution',     #44
                      'resnet_v2_50/block4/unit_1/bottleneck_v2/conv2/convolution',     #45
                      'resnet_v2_50/block4/unit_1/bottleneck_v2/conv3/convolution',     #46
                      'resnet_v2_50/block4/unit_2/bottleneck_v2/conv1/convolution',     #47
                      'resnet_v2_50/block4/unit_2/bottleneck_v2/conv2/convolution',     #48
                      'resnet_v2_50/block4/unit_2/bottleneck_v2/conv3/convolution',     #49
                      'resnet_v2_50/block4/unit_3/bottleneck_v2/conv1/convolution',     #50
                      'resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/convolution',     #51
                      'resnet_v2_50/block4/unit_3/bottleneck_v2/conv3/convolution']     #52
        
        for i,l in enumerate(all_layers):
            print(i, np.int(graph.get_tensor_by_name("import/"+l+':0').get_shape()[-1]) )
        
        
        for op in graph.get_operations():
            print (str(op.name))    
        #print(tf.shape(T('resnet_v2_50/block4/unit_3/bottleneck_v2/conv3/weights')[:,:,:,12]))

        
        
        
        try:
            used_layer = all_layers[int(opts.layer)]
        except:
            used_layer = all_layers[0]
            
        possible_channels = int(graph.get_tensor_by_name("import/"+layer+':0').get_shape()[-1]) 
        
        if channel == 0:
            print("You are going to show all %s channels. This may take a while" % (possible_channels))
            for channel_i in range(possible_channels):
                used_method(T(used_layer)[:,:,:,channel_i], filename = filename)
        elif channel > 0:
            print(used_layer, channel)
            used_method(T(used_layer)[:,:,:,channel-1], filename = filename)
        else:
            print("You are going to show %s of %s possible channels. This may take a while" % (possible_channels))
            no_channels = math.fabs(channel)
            for i in range(no_channels):
                channel_i = randint(0, possible_channels)
                used_method(T(used_layer)[:,:,:,channel_i], filename = filename)
   
    elif opts.adjust_most_accurate == "adjust":
        layer =  'my_fc_1/BiasAdd'
    
        most_acc= dataset_utils.read_label_file("mydata/labels", "most_accurate_images.txt")
        
        adjust_most_accurate(most_acc)
    