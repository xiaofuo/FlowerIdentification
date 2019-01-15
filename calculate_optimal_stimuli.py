#/home/lolek/.local/lib/python3.5/site-packages/tensorflow/contrib/training/python/training/
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf

# Main slim library
slim = tf.contrib.slim


import resNetClassifier
import dataVisualisation
import my_resnet_preprocessing
import my_functions
import plantclef_download_and_convert_data
from changed_scripts import dataset_utils
import scipy.misc 

def find_most_accurate(label_directory = "mydata/PlantClefTraining2015", 
                       dataset_dir ="mydata/train/", 
                       model_directory ="mydata/resnet_finetuned_plantclef2015_2/model.ckpt-150000", 
                       dict_directory = "mydata/labels", 
                       dict_name = "most_accurate_images.txt" ):
    """
    Finds the images in Filepath, that have the highest output probability, while beeing true.
    
    Args:
        label_directory: Where to find the dictionary mapping from class_id to one-hot-labels
        dataset_dir: where to find the images
        model_directory: where to find your models checkpoints
        dict_directory: where to save the images with the highest activation
        dict_name: name of the txt-file of the dictionary containing images with the highest activation
    Returns:
        best_act: dictionary containing images with the highest activation
    
    """
    best_act = {}
    label_dict= dataset_utils.read_label_file(label_directory)
    label_dict = dict([int(v),k] for k,v in label_dict.items())

    
    with tf.Graph().as_default():
            
            X = tf.placeholder(tf.float32)
            
            image_pre = my_resnet_preprocessing.preprocess_image(X, 224, 224, False)
            image_pre = tf.reshape(image_pre, shape=[-1, 224, 224, 3])
            image_pre = tf.to_float(image_pre)
        
                
                
            logits = resNetClassifier.my_cnn(image_pre, is_training = False, dropout_rate=1.0, layer=None)
        

                
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                saver = tf.train.Saver()
                saver.restore(sess,model_directory )       
                
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                dataset_dir = "optimal_stimuli"
                for filename in os.listdir(dataset_dir):
                    if filename.endswith(".jpg"):  
                        filepath = ("%s/%s" % (dataset_dir, filename))
                        im = my_functions.get_img(filepath,).astype(np.float32)
                        
                        layer_act = sess.run([logits],feed_dict={X: im})
                        layer_act = my_functions.numpy_softmax(layer_act)
                        
                        
                        label,_,_,_,_,_ = plantclef_download_and_convert_data.get_class_name_from_xml("%s.xml" % (str.split(filepath, ".jpg")[0]))
                        label = label_dict[label]
                        

                                
                    
                        
                        if np.argmax(layer_act) == label:
                            best_act.setdefault(label, [-1, "test.jpg"])
                            
                            if best_act[label][0] < np.amax(layer_act):
                                best_act[label] = [np.amax(layer_act), filename]
                        
                    
                    
    dataset_utils.write_label_file(best_act, dict_directory, filename=dict_name)  
    return best_act


        
if __name__ == "__main__":
    most_acc = find_most_accurate()