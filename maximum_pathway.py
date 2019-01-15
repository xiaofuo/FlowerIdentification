#/home/lolek/.local/lib/python3.5/site-packages/tensorflow/contrib/training/python/training/
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import my_functions

# Main slim library
slim = tf.contrib.slim


import resNetClassifier
import dataVisualisation
import my_resnet_preprocessing

import matplotlib.pyplot as plt
import scipy.misc
import sys
from changed_scripts import dataset_utils


from numpy import unravel_index

###################################################################################################################
###################################################################################################################
###################################################################################################################

dataset_dir ='mydata/PlantClefTraining2015'  
split ='train_set1' 
#layer='resnet_v2_50/block1/unit_3/bottleneck_v2' 
checkpoint_path="mydata/resnet_finetuned_plantclef2015_2/model.ckpt-150000"



all_outputs = ['resnet_v2_50/conv1', 'resnet_v2_50/block1/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block1/unit_1/bottleneck_v2', 'resnet_v2_50/block1/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block1/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block1/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block1/unit_2/bottleneck_v2', 'resnet_v2_50/block1/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block1/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block1/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block1/unit_3/bottleneck_v2', 'resnet_v2_50/block1', 'resnet_v2_50/block2/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block2/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_1/bottleneck_v2', 'resnet_v2_50/block2/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_2/bottleneck_v2', 'resnet_v2_50/block2/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_3/bottleneck_v2', 'resnet_v2_50/block2/unit_4/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_4/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_4/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_4/bottleneck_v2', 'resnet_v2_50/block2', 'resnet_v2_50/block3/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block3/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_1/bottleneck_v2', 'resnet_v2_50/block3/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_2/bottleneck_v2', 'resnet_v2_50/block3/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_3/bottleneck_v2', 'resnet_v2_50/block3/unit_4/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_4/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_4/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_4/bottleneck_v2', 'resnet_v2_50/block3/unit_5/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_5/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_5/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_5/bottleneck_v2', 'resnet_v2_50/block3/unit_6/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_6/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_6/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_6/bottleneck_v2', 'resnet_v2_50/block3', 'resnet_v2_50/block4/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block4/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block4/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block4/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block4/unit_1/bottleneck_v2', 'resnet_v2_50/block4/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block4/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block4/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block4/unit_2/bottleneck_v2', 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block4/unit_3/bottleneck_v2', 'resnet_v2_50/block4', 'net', 'my_fc_2/BiasAdd', 'my_fc_3/BiasAdd', 'my_fc_4/BiasAdd', 'my_fc_5/BiasAdd', 'concat', 'max_out', 'logits']

all_convs = ['resnet_v2_50/conv1', 'resnet_v2_50/block1/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block1/unit_1/bottleneck_v2', 'resnet_v2_50/block1/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block1/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block1/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block1/unit_2/bottleneck_v2', 'resnet_v2_50/block1/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block1/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block1/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block1/unit_3/bottleneck_v2', 'resnet_v2_50/block1', 'resnet_v2_50/block2/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block2/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_1/bottleneck_v2', 'resnet_v2_50/block2/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_2/bottleneck_v2', 'resnet_v2_50/block2/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_3/bottleneck_v2', 'resnet_v2_50/block2/unit_4/bottleneck_v2/conv1', 'resnet_v2_50/block2/unit_4/bottleneck_v2/conv2', 'resnet_v2_50/block2/unit_4/bottleneck_v2/conv3', 'resnet_v2_50/block2/unit_4/bottleneck_v2', 'resnet_v2_50/block2', 'resnet_v2_50/block3/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block3/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_1/bottleneck_v2', 'resnet_v2_50/block3/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_2/bottleneck_v2', 'resnet_v2_50/block3/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_3/bottleneck_v2', 'resnet_v2_50/block3/unit_4/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_4/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_4/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_4/bottleneck_v2', 'resnet_v2_50/block3/unit_5/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_5/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_5/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_5/bottleneck_v2', 'resnet_v2_50/block3/unit_6/bottleneck_v2/conv1', 'resnet_v2_50/block3/unit_6/bottleneck_v2/conv2', 'resnet_v2_50/block3/unit_6/bottleneck_v2/conv3', 'resnet_v2_50/block3/unit_6/bottleneck_v2', 'resnet_v2_50/block3', 'resnet_v2_50/block4/unit_1/bottleneck_v2/shortcut', 'resnet_v2_50/block4/unit_1/bottleneck_v2/conv1', 'resnet_v2_50/block4/unit_1/bottleneck_v2/conv2', 'resnet_v2_50/block4/unit_1/bottleneck_v2/conv3', 'resnet_v2_50/block4/unit_1/bottleneck_v2', 'resnet_v2_50/block4/unit_2/bottleneck_v2/conv1', 'resnet_v2_50/block4/unit_2/bottleneck_v2/conv2', 'resnet_v2_50/block4/unit_2/bottleneck_v2/conv3', 'resnet_v2_50/block4/unit_2/bottleneck_v2', 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv1', 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv2', 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv3', 'resnet_v2_50/block4/unit_3/bottleneck_v2', 'resnet_v2_50/block4']

unit_convs = ['resnet_v2_50/conv1',
              'resnet_v2_50/block1/unit_1/bottleneck_v2', 
              'resnet_v2_50/block1/unit_2/bottleneck_v2', 
              'resnet_v2_50/block1/unit_3/bottleneck_v2',
              'resnet_v2_50/block2/unit_1/bottleneck_v2', 
              'resnet_v2_50/block2/unit_2/bottleneck_v2',
              'resnet_v2_50/block2/unit_3/bottleneck_v2', 
              'resnet_v2_50/block2/unit_4/bottleneck_v2', 
              'resnet_v2_50/block3/unit_1/bottleneck_v2', 
              'resnet_v2_50/block3/unit_2/bottleneck_v2', 
              'resnet_v2_50/block3/unit_3/bottleneck_v2', 
              'resnet_v2_50/block3/unit_4/bottleneck_v2', 
              'resnet_v2_50/block3/unit_5/bottleneck_v2',
              'resnet_v2_50/block3/unit_6/bottleneck_v2',
              'resnet_v2_50/block4/unit_1/bottleneck_v2', 
              'resnet_v2_50/block4/unit_2/bottleneck_v2', 
              'resnet_v2_50/block4/unit_3/bottleneck_v2']

def max_path_random_image():
    """
    Plots the Level of activation of each layer for a random image
    """


    with tf.Graph().as_default():
        dataset = dataVisualisation.get_split(split, dataset_dir)
                
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                                            shuffle=False,
                                                                            common_queue_capacity=8000,
                                                                            common_queue_min=0)
                
        image_raw, label = data_provider.get(['image', 'label'])
                                
                #imaget = my_resnet_preprocessing.preprocess_image(image_raw, 224, 224, is_training=False)
                #image,augmented_image1,augmented_image2, augmented_image3, augmented_image4,augmented_image5, labels = tf.train.batch([imaget,imaget,imaget,imaget,imaget,imaget, label],  batch_size=1,
                #                                num_threads=1,
                #                                capacity=12 * 1)
                
                
                
                # Preprocessing return original image, center_crop and 4 corner crops with adjusted color values
        image  = my_resnet_preprocessing.preprocess_for_eval_v0(image_raw, 224, 224, is_training=False) 
            
        image, labels = tf.train.batch([image,label], 
                                                batch_size=1,
                                                num_threads=1,
                                                capacity=2 * 1)
                
                
                
                
        logits = resNetClassifier.my_cnn(image, is_training = False, dropout_rate =1, layer='all') #TODO


                
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)                  
                    
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            
            layer_act, default_label = sess.run([logits, labels])
            
            outputs = {}
            
            for l_out in all_outputs:
                outputs[l_out] = layer_act[l_out][0][0][0]
            
            
            default_label[0] = default_label
            comp_no = 25
            i = 0    
            while i < comp_no:                
                        
                layer_act, label = sess.run([logits, labels])
                
                if label == default_label:
                    i += 1
                    print("%s of %s images found" % (i, comp_no))
                    for l_out in all_outputs:
                        outputs[l_out] += layer_act[l_out][0][0][0] / comp_no 
                        
            
        for l_out in all_outputs: 
            data = outputs[l_out]
            plt.plot(np.arange(len(data)), data, 'ro')
            #plt.show()
            plt.savefig(("distribution/%s_%s.png" % (l_out, comp_no)))
            plt.close(fig)
            
 



def max_path(image_list):
    """
    Plots the level of activation of each layer for a image in the image_list
    
    Args:
        image_list: List of images to visualize
        
    """
    
    with tf.Graph().as_default():
        # tensorflow model implementation (Alexnet convolution)
        X = tf.placeholder(tf.float32, shape = [None, 224, 224, 3]) # placeholder for input images

        image = tf.reshape(X, shape=[224, 224, 3])
        image_pre = my_resnet_preprocessing.preprocess_image(image, 224, 224, False)
        image_pre = tf.reshape(image_pre, shape=[-1, 224, 224, 3])
        image_pre = tf.to_float(image_pre)

        # Test pretrained model
        logits = resNetClassifier.my_cnn(image_pre, is_training = False, dropout_rate =1, layer='all')
        
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)                  
                    
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            

            
            outputs = {}
            
            

            comp_no = len(image_list)
 
            for i in image_list:
                im = my_functions.get_img(os.path.join("./mydata/train/", i), (224,224,3)).astype(np.float32)
                im = np.expand_dims(im, axis = 0)                
                        
                layer_act = sess.run([logits],feed_dict={X: im})
                

                for l_out in all_outputs:
                    current_act = my_functions.get_data_of_array(layer_act[0][l_out]) / comp_no 
                    #print(current_act.shape)
                    outputs.setdefault(l_out, current_act)
                    
                        
            
        for index, l_out in enumerate(all_outputs): 
            data = outputs[l_out]
            for channel, d in enumerate(data):
                #print(d.shape)
                fig = plt.figure()
                plt.plot(np.arange(len(d)), d, 'ro')
                #plt.show()
                plt.savefig(("images/distribution/layer_%s_channel_%s_compostion_%s.png" % (index,channel, comp_no)))
                plt.close(fig)
            
        
def find_max_activation_per_filter(image_list, filename="max_activations.txt"):
    """
    Returns and saves the top-9 images which excide each channel the most and saves them into filename TODO Change the saving process!
    
    Args:
        image_list: Images to take into account
        filename: name of text file where information is saved
    Returns:
        top-9 images which excide each layer the most
    """
    max_act_per_layer = {}
    for key in unit_convs:
        max_act_per_layer[key] = {}
    

    
    with tf.Graph().as_default():
        # tensorflow model implementation (Alexnet convolution)
        X = tf.placeholder(tf.float32)#, shape = [None, 224, 224, 3]) # placeholder for input images

        image = tf.reshape(X, shape=[224, 224, 3])
        image_pre = my_resnet_preprocessing.preprocess_for_eval_v0(image, 224, 224, False)
        image_pre = tf.reshape(image_pre, shape=[-1, 224, 224, 3])
        image_pre = tf.to_float(image_pre)

        # Test pretrained model
        logits = resNetClassifier.my_cnn(image_pre, is_training = False, dropout_rate =1, layer='all')
        
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)                  
                    
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
            for ind,i in enumerate(image_list):
                sys.stdout.write('\r>> Passing image %d of %d through network' % (
                            ind+1, len(image_list)))
                sys.stdout.flush()
                
                im = my_functions.get_img(os.path.join("./mydata/train/", i), (224,224,3)).astype(np.float32)
                
                im = np.expand_dims(im, axis = 0)                
                        
                layer_act = sess.run([logits],feed_dict={X: im})
                
                for key, value in layer_act[0].items():
                    if key in unit_convs:
                        value = value[0]
                        
                        
                        # Get number of channels
                        if value.ndim == 3:
                            l = range(0, len(value[0][0]))
                        
                        for channel in l:                      
                            
                            
                            current_channel = max_act_per_layer[key].setdefault(channel, [[-sys.maxsize-1, -1, "x"]])
                            
                            s = np.sum(value[:,:,channel:channel+1])
                            #s = np.amax(value[:,:,channel:channel+1])
                            
                            #print(unravel_index(value[:,:,channel:channel+1].argmax(), value[:,:,channel:channel+1].shape))
                            ##############
                            
                            if len(current_channel) < 9:
                                for tuple_index, tuple_outputs in enumerate(max_act_per_layer[key][channel]):
                                    if tuple_outputs[0] < s:
                                        max_act_per_layer[key][channel].insert(tuple_index,[s, channel,i])
                                        
                                        if len(max_act_per_layer[key][channel]) > 9:
                                            max_act_per_layer[key][channel] = max_act_per_layer[key][channel][:9]

                                        break
                                else:
                                    max_act_per_layer[key][channel].append([s,channel, i])
                                
                                    
                                    
                            else:
                                for tuple_index, tuple_outputs in enumerate(max_act_per_layer[key][channel]):
                                    if tuple_outputs[0] < s:
                                        max_act_per_layer[key][channel].insert(tuple_index,[s, channel,i])
                                        
                                        if len(max_act_per_layer[key][channel]) > 9:
                                            max_act_per_layer[key][channel] = max_act_per_layer[key][channel][:9]

                                        break          
    
    dataset_utils.write_label_file(max_act_per_layer, "mydata", filename = filename)
    return max_act_per_layer
                            
def top_channel_per_layer(activation_dict, filename="max_activations_top1.txt"):
    """
    Returns the most activated channel in each layer, given the activations of each channel and saves them into filename
    
    Args:
        activation_dict: The dictionary of each channels activation
        filename: name of text file where information is saved
        
        
    """
    return_dict = {}
    for key, values in activation_dict.items():
        max_value = [-sys.maxsize-1,-1]
        for channel, value in values.items():
            if value[0][0] > max_value[0]:
                max_value = [value[0][0], channel]
        return_dict[key] = {max_value[1]:values[max_value[1]]}  
        
    dataset_utils.write_label_file(return_dict, "mydata",filename = filename)    
    return return_dict    



            
            
        
        
if __name__ == "__main__":
    # Testig with a few images only
    image_list = ["1.jpg", "2.jpg", "5.jpg", "13.jpg", "18.jpg", "19.jpg", "53.jpg","54.jpg","55.jpg","56.jpg","57.jpg"]
    label_list = [5810, 780, 112, 14841, 5483, 1235]
    
    activations = find_max_activation_per_filter(image_list)
    top = top_channel_per_layer(activations)
    
    print(activations['resnet_v2_50/conv1'])