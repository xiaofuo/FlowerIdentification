import math
import numpy as np
import time
import os

import tensorflow as tf

# Main slim library
slim = tf.contrib.slim

from changed_scripts import resnet_v2
from changed_scripts import dataset_utils


from resNetClassifier import load_batch
import dataVisualisation
import my_functions
import my_resnet_preprocessing
import changed_scripts.dataset_utils

from tensorflow.contrib.framework.python.ops import variables

##########################################################################
##########################################################################

flowers_data_dir = 'mydata/joinedDataset' 

num_families = 124
num_genus= 516
num_species = 1000
num_organs = 7

labels_dict =  dataset_utils.read_label_file("mydata/PlantClefTraining2015", "labels.txt")
class_id_to_family = my_functions.my_read_label_file("mydata/labels", "class_id_to_family.txt")
family_to_one_hot = my_functions.my_read_label_file("mydata/labels", "family_one_hot.txt")
class_id_to_genus = my_functions.my_read_label_file("mydata/labels", "class_id_to_genus.txt")
genus_to_one_hot = my_functions.my_read_label_file("mydata/labels", "genus_one_hot.txt")
class_id_to_species = my_functions.my_read_label_file("mydata/labels", "class_id_to_species.txt")
species_to_one_hot = my_functions.my_read_label_file("mydata/labels", "species_one_hot.txt")


##########################################################################
##########################################################################

def load_batch_intermediate(dataset, batch_size=28, height=224, width=224, is_training=False): 
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label_species, label_genus, label_family, label_organ  = data_provider.get(['image', 'label_species', 'label_genus', 'label_family', 'label_organ'])
       
    # Preprocess the image for display purposes.
    image_raw_disp = tf.expand_dims(image_raw, 0)
    image_raw_disp = tf.image.resize_images(image_raw_disp, [height, width])
    image_raw_disp = tf.squeeze(image_raw_disp)
    

    # Batch it up.
    if is_training:
        image= my_resnet_preprocessing.preprocess_for_train_version1(image_raw, height, width) #preprocess_image
        

        #image = tf.stack([image, image_flipped], 0)
        #label = tf.stack([label, label], 0)

        images, images_raw, label_species, labels_genus, labels_family, labels_organ = tf.train.batch(
          [image, image_raw_disp, label_species, label_genus, label_family, label_organ],
          batch_size=batch_size,
          num_threads=1,
          capacity=5000)
    
        return images, image_raw_disp, label_species, labels_genus, labels_family, labels_organ
    
    
    
    else:        
        image = my_resnet_preprocessing.preprocess_for_eval_v0(image_raw, height, width) #preprocess_image

        images, label_species, labels_genus, labels_family, labels_organ = tf.train.batch(
            [image, label_species, label_genus, label_family, label_organ],
            batch_size=batch_size,
            num_threads=1,
            capacity= 2*batch_size)
        
        return images, image_raw_disp, label_species, labels_genus, labels_family, labels_organ
    
       
   
    

  
  
    


#load_batch_intermediate(image_list = ["1.jpg", "2.jpg"], label_list=[10, 108])
##########################################################################
##########################################################################

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



def my_intermediate_cnn(images, is_training, dropout_rate =1,fc_after=None, num_classes=1000):
    """ Creates a neural network and calculates the logits of a given set of images
    
    Args: 
      images: batch of images of which logits need to be calculated
      is_training: boolean, indicates wether softmax should be calculated or not
      dropout_rate: keeping rate during dropout: should be one during evaluation
      
    Returns:
      logits: logits of images with softmax (if dropout != 0 and is_training=False), else without softmax
    """      
    
    cut_point = cutting_points.index(fc_after) # Where to cut the network off
    
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=0.0002)): 
        
        logits_res, end_points = resnet_v2.resnet_v2_50_intermediate(inputs=images, 
                                                                     num_classes = num_classes,
                                                                     is_training=is_training, 
                                                                     fc_after=fc_after) 
        
        end_points['net']=logits_res              
            
        
            
        
        if cut_point >= cutting_points.index( 'my_fc_2'):      
            logits_drop = tf.nn.dropout(logits_res, dropout_rate) 
            max_out_unit_1 = slim.fully_connected(logits_drop, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_2')
            end_points['my_fc_2/BiasAdd']=max_out_unit_1
        else:            
            net = tf.nn.dropout(logits_res, dropout_rate) 
            net= slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            return net, end_points
        
        if cut_point >= cutting_points.index( 'my_fc_3'):      
            max_out_unit_2 = slim.fully_connected(logits_drop, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_3')
            end_points['my_fc_3/BiasAdd']=max_out_unit_2
        else:            
            net = tf.nn.dropout(max_out_unit_1, dropout_rate) 
            net= slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            return net, end_points    
        
        if cut_point >= cutting_points.index( 'my_fc_4'):
            max_out_unit_3 = slim.fully_connected(logits_drop, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_4')
            end_points['my_fc_4/BiasAdd']=max_out_unit_3
        else:            
            net = tf.nn.dropout(max_out_unit_2, dropout_rate) 
            net= slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            return net, end_points        
        
        if cut_point >= cutting_points.index( 'my_fc_5'):
            max_out_unit_4 = slim.fully_connected(logits_drop, 512, activation_fn=None, normalizer_fn=None, scope='my_fc_5')
            end_points['my_fc_5/BiasAdd']=max_out_unit_4
        else:            
            net = tf.nn.dropout(max_out_unit_3, dropout_rate) 
            net= slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            return net, end_points   
        
        if cut_point >= cutting_points.index( 'concat'):
            max_out_joined = tf.concat([max_out_unit_1, max_out_unit_2, max_out_unit_3, max_out_unit_4], 1)
            end_points['concat']=max_out_joined
        else:            
            net = tf.nn.dropout(max_out_unit_4, dropout_rate) 
            net= slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            return net, end_points      
        
        if cut_point >= cutting_points.index( 'maxout'):
            max_out_act = my_functions.max_out(max_out_joined, num_units = 1, axis = 1)
            end_points['max_out']= max_out_act    
        else:           
            net = tf.nn.dropout(max_out_joined, dropout_rate) 
            net= slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            return net, end_points 
            
            
        if cut_point >= cutting_points.index( 'my_fc_1'):
            logits_squeeze = tf.squeeze(max_out_act, [1, 2], name='SpatialSqueeze')
            logits = slim.fully_connected(logits_squeeze, 1000, scope = "my_fc_1", activation_fn=None, normalizer_fn=None)
            
            #net = tf.nn.dropout(logits, dropout_rate) 
            net= slim.flatten(logits)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            
            end_points['logits']=logits
        else:           
            net = tf.nn.dropout(max_out_act, dropout_rate) 
            net= slim.flatten(net)
            net = slim.fully_connected(net, num_classes, activation_fn=None, normalizer_fn=None, scope='fc_intermediate')
            return net, end_points     

        return net, end_points
        

#######################################################################
#######################################################################


checkpoints_dir = 'mydata/resnet_finetuned_plantclef2015_6/'


checkpoint_dicts = {'conv1':'resnet_v2_50/conv1', 
                  'block1_unit1_conv1':'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1',  
                  'block1_unit1_conv2':'resnet_v2_50/block1/unit_1/bottleneck_v2/conv2',  
                  'block1_unit1_conv3':'resnet_v2_50/block1/unit_1/bottleneck_v2/conv3',  
                  'block1_unit1':'resnet_v2_50/block1/unit_1/bottleneck_v2/add',
                  'block1_unit2_conv1':'resnet_v2_50/block1/unit_2/bottleneck_v2/conv1', 
                  'block1_unit2_conv2':'resnet_v2_50/block1/unit_2/bottleneck_v2/conv2', 
                  'block1_unit2_conv3':'resnet_v2_50/block1/unit_2/bottleneck_v2/conv3', 
                  'block1_unit2':'resnet_v2_50/block1/unit_2/bottleneck_v2/add', 
                  'block1_unit3_conv1':'resnet_v2_50/block1/unit_3/bottleneck_v2/conv1', 
                  'block1_unit3_conv2':'resnet_v2_50/block1/unit_3/bottleneck_v2/conv2', 
                  'block1_unit3_conv3':'resnet_v2_50/block1/unit_3/bottleneck_v2/conv3', 
                  'block1_unit3':'resnet_v2_50/block1/unit_3/bottleneck_v2/add',
                  'block2_unit1_conv1':'resnet_v2_50/block2/unit_1/bottleneck_v2/conv1', 
                  'block2_unit1_conv2':'resnet_v2_50/block2/unit_1/bottleneck_v2/conv2', 
                  'block2_unit1_conv3':'resnet_v2_50/block2/unit_1/bottleneck_v2/conv3', 
                  'block2_unit1':'resnet_v2_50/block2/unit_1/bottleneck_v2/add',
                  'block2_unit2_conv1':'resnet_v2_50/block2/unit_2/bottleneck_v2/conv1', 
                  'block2_unit2_conv2':'resnet_v2_50/block2/unit_2/bottleneck_v2/conv2', 
                  'block2_unit2_conv3':'resnet_v2_50/block2/unit_2/bottleneck_v2/conv3', 
                  'block2_unit2':'resnet_v2_50/block2/unit_2/bottleneck_v2/add', 
                  'block2_unit3_conv1':'resnet_v2_50/block2/unit_3/bottleneck_v2/conv1', 
                  'block2_unit3_conv2':'resnet_v2_50/block2/unit_3/bottleneck_v2/conv2', 
                  'block2_unit3_conv3':'resnet_v2_50/block2/unit_3/bottleneck_v2/conv3', 
                  'block2_unit3':'resnet_v2_50/block2/unit_3/bottleneck_v2/add',
                  'block2_unit4_conv1':'resnet_v2_50/block2/unit_4/bottleneck_v2/conv1',  
                  'block2_unit4_conv2':'resnet_v2_50/block2/unit_4/bottleneck_v2/conv2',  
                  'block2_unit4_conv3':'resnet_v2_50/block2/unit_4/bottleneck_v2/conv3',  
                  'block2_unit4':'resnet_v2_50/block2/unit_4/bottleneck_v2/add',
                  'block3_unit1_conv1':'resnet_v2_50/block3/unit_1/bottleneck_v2/conv1', 
                  'block3_unit1_conv2':'resnet_v2_50/block3/unit_1/bottleneck_v2/conv2', 
                  'block3_unit1_conv3':'resnet_v2_50/block3/unit_1/bottleneck_v2/conv3', 
                  'block3_unit1':'resnet_v2_50/block3/unit_1/bottleneck_v2/add',
                  'block3_unit2_conv1':'resnet_v2_50/block3/unit_2/bottleneck_v2/conv1', 
                  'block3_unit2_conv2':'resnet_v2_50/block3/unit_2/bottleneck_v2/conv2', 
                  'block3_unit2_conv3':'resnet_v2_50/block3/unit_2/bottleneck_v2/conv3', 
                  'block3_unit2':'resnet_v2_50/block3/unit_2/bottleneck_v2/add',
                  'block3_unit3_conv1':'resnet_v2_50/block3/unit_3/bottleneck_v2/conv1', 
                  'block3_unit3_conv2':'resnet_v2_50/block3/unit_3/bottleneck_v2/conv2', 
                  'block3_unit3_conv3':'resnet_v2_50/block3/unit_3/bottleneck_v2/conv3', 
                  'block3_unit3':'resnet_v2_50/block3/unit_3/bottleneck_v2/add',
                  'block3_unit4_conv1':'resnet_v2_50/block3/unit_4/bottleneck_v2/conv1', 
                  'block3_unit4_conv2':'resnet_v2_50/block3/unit_4/bottleneck_v2/conv2', 
                  'block3_unit4_conv3':'resnet_v2_50/block3/unit_4/bottleneck_v2/conv3', 
                  'block3_unit4':'resnet_v2_50/block3/unit_3/bottleneck_v2/add',                 
                  'block3_unit5_conv1':'resnet_v2_50/block3/unit_5/bottleneck_v2/conv1', 
                  'block3_unit5_conv2':'resnet_v2_50/block3/unit_5/bottleneck_v2/conv2', 
                  'block3_unit5_conv3':'resnet_v2_50/block3/unit_5/bottleneck_v2/conv3', 
                  'block3_unit5':'resnet_v2_50/block3/unit_5/bottleneck_v2/add',
                  'block3_unit6_conv1':'resnet_v2_50/block3/unit_6/bottleneck_v2/conv1', 
                  'block3_unit6_conv2':'resnet_v2_50/block3/unit_6/bottleneck_v2/conv2', 
                  'block3_unit6_conv3':'resnet_v2_50/block3/unit_6/bottleneck_v2/conv3', 
                  'block3_unit6':'resnet_v2_50/block3/unit_6/bottleneck_v2/add',
                  'block4_unit1_conv1':'resnet_v2_50/block4/unit_1/bottleneck_v2/conv1',
                  'block4_unit1_conv2':'resnet_v2_50/block4/unit_1/bottleneck_v2/conv2', 
                  'block4_unit1_conv3':'resnet_v2_50/block4/unit_1/bottleneck_v2/conv3', 
                  'block4_unit1':'resnet_v2_50/block4/unit_1/bottleneck_v2/add',
                  'block4_unit2_conv1':'resnet_v2_50/block4/unit_2/bottleneck_v2/conv1', 
                  'block4_unit2_conv2':'resnet_v2_50/block4/unit_2/bottleneck_v2/conv2',  
                  'block4_unit2_conv3':'resnet_v2_50/block4/unit_2/bottleneck_v2/conv3', 
                  'block4_unit2':'resnet_v2_50/block4/unit_2/bottleneck_v2/add', 
                  'block4_unit3_conv1':'resnet_v2_50/block4/unit_3/bottleneck_v2/conv1', 
                  'block4_unit3_conv2':'resnet_v2_50/block4/unit_3/bottleneck_v2/conv2', 
                  'block4_unit3_conv3':'resnet_v2_50/block4/unit_3/bottleneck_v2/conv3', 
                  'block4_unit3':'resnet_v2_50/block4/unit_3/bottleneck_v2/add',
                  'postnorm':'resnt_v2_50/postnorm',
                  'pool5':'resnet_v2_50/pool5', 
                  'my_fc_2':'my_fc_2', 
                  'my_fc_3':'my_fc_3', 
                  'my_fc_4':'my_fc_4', 
                  'my_fc_5':'my_fc_5',
                  'my_fc_1':"my_fc_1"}

        
def get_init_fn(fc_after, checkpoints_dir = checkpoints_dir,checkpoint = 'model.ckpt-150000'):
    #Returns a function run by the chief worker to warm-start the training.
    checkpoint_exclude_scopes=["resnet_v2_50/logits", 'resnet_v2_50/fc_intermediate', 'fc_intermediate' ] 
    cut_point = cutting_points.index(fc_after)
    for ind_l, l in enumerate(cutting_points):
        if ind_l > cut_point:
            if l in checkpoint_dicts.keys():
                #print("!",l)
                checkpoint_exclude_scopes.append(checkpoint_dicts[l])
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    #print(exclusions)

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            #print(var.name)
            variables_to_restore.append(var)
    

    return slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, checkpoint), #Choose Model
      variables_to_restore)


def train(split, train_steps, train_dir, fc_after, level, checkpoints_dir = checkpoints_dir,checkpoint = 'model.ckpt-150000'):
    """ Trains the given neural network and saves the weights and summary information into a new checkpoint file in the train_dir
    
    Args: 
      split: Chooses split of flower dataset to train the network
      train_steps: Number of steps to train network
      train_dir: Directory in which checkpoints should be stored, and old checkpoints get loaded
    Returns:
      -
    """

    
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO) # showing INFO logs
        
        dataset = dataVisualisation.get_split(split_name = split, dataset_dir = flowers_data_dir, label_type="multiple")
        images, _, label_species, labels_genus, labels_family, labels_organ= load_batch_intermediate(dataset,height=224, width=224, is_training=True, batch_size=100)      
        

                    #print(family, genus, species)
                
        abstraction_levels = {"family":labels_family, "genus":labels_genus, "species":label_species, "organs":labels_organ}  
        levels_length = {"family":124, "genus":516, "species":1000, "organs":7}  
                    
        labels = tf.stack(abstraction_levels.get(level, label_species))
        
        one_hot_labels = slim.one_hot_encoding(labels, levels_length.get(level, 1000))
        

        
        
        

        # Forward pass with non-flipped images
        logits,_ = my_intermediate_cnn(images, is_training=True, fc_after=fc_after, num_classes = levels_length.get(level, 1000))
        #print(logits, one_hot_labels)
        #with tf.Session() as sess:
        #    print(sess.run(tf.shape(logits)))

        tf.losses.softmax_cross_entropy(one_hot_labels, logits)      
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/Total_Loss', total_loss)
        
        
        # Learning rate decay
        global_step = variables.get_or_create_global_step()
        boundaries = [tf.constant(100000, dtype= "int64"), tf.constant(200000, dtype= "int64"), tf.constant(300000, dtype= "int64")]
        values = [0.001, 0.0001, 0.00001, 0.000001]
        my_learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        
        #for v in tf.trainable_variables():
        #    print(v)
            
        #for v in slim.get_variables(scope="resnet_v2_50/fc_intermediate/"):
        #    print(v)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.MomentumOptimizer(learning_rate=my_learning_rate, momentum = 0.9) 
        train_op = slim.learning.create_train_op(total_loss=total_loss, optimizer=optimizer, variables_to_train=slim.get_variables(scope="fc_intermediate"))
        
        saver = tf.train.Saver(max_to_keep=1)
 
        
        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            log_every_n_steps=50,
            init_fn= get_init_fn(fc_after),
            number_of_steps=train_steps,
            global_step = global_step, 
            saver = saver)    
        
       


    
    print('Finished training. Last batch loss %f' % final_loss)
    
    
def eval(split, eval_steps, train_dir, fc_after, level): 
    """ Evaluates the given neural network, which is saved in train_dir
    
    Args: 
      split: Chooses split of flower dataset to train the network
      eval_steps: Number of steps to evaluate network
      train_dir: Directory in which checkpoints sare found
      fc_after: where to cut of the network
      level: which class do you want to eval on? (family, genus, species, organ)
      
    Returns:
      -
    """
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        
        dataset = dataVisualisation.get_split(split_name = split, dataset_dir = flowers_data_dir, label_type="multiple")
        images, _, label_species, labels_genus, labels_family, labels_organ= load_batch_intermediate(dataset,height=224, width=224, is_training=False, batch_size=100)      
        

                    #print(family, genus, species)
                
        abstraction_levels = {"family":labels_family, "genus":labels_genus, "species":label_species, "organs":labels_organ}  
        levels_length = {"family":124, "genus":516, "species":1000, "organs":7}  
                    
        labels = tf.stack(abstraction_levels.get(level, label_species))
        
        one_hot_labels = slim.one_hot_encoding(labels, levels_length.get(level, 1000)) 
        
        
        
        logits,_ = my_intermediate_cnn(images, is_training=True, fc_after=fc_after,num_classes = levels_length.get(level, 1000))
        logits = slim.softmax(logits)
        #logits = tf.cast(logits, tf.int64)
        predictions = tf.argmax(logits, 1)  
        predictions = tf.cast(predictions, tf.int64)
 
        
                     
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'eval/Recall@5': slim.metrics.streaming_sparse_recall_at_k(logits, labels, 5)
        })
        
        op = tf.summary.scalar('top5percentError', names_to_values['eval/Recall@5'], collections=[])  
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        op = tf.summary.scalar('top1percentError', names_to_values['eval/Accuracy'], collections=[])  
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


        print('Running evaluation Loop...')
        checkpoint_path = tf.train.latest_checkpoint(train_dir)
         
        metric_values = slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_path,
            logdir=train_dir,
            num_evals = eval_steps,  # TODO Adjust number of evaluations
            summary_op=tf.summary.merge_all(),
            eval_op=list(names_to_updates.values()),
            final_op=list(names_to_values.values()))
        
        print("Evaluation 'eval' started.")
        names_to_values = dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print('%s: %f' % (name, names_to_values[name]))
        
        return names_to_values['eval/Accuracy'], names_to_values['eval/Recall@5']

########################################################################################

def train_and_eval(): 
    """
    Trains fully fully_connected softmax classifiers for each class (family, genus, species, organ) after every possible convolution. 
    
    """
    level_set = ["family", "genus", "species", "organs"]
    done =['conv1', 'pool1', 
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
                  'block4_unit2_conv1', 'block4_unit2_conv2', 'block4_unit2_conv3', 'block4_unit2']
    for level in level_set:
        for l in cutting_points:
            if l not in done:
                if l != "conv1":
                    train_dir = ("layer_generalization/fc_%s_%s" % (l, level))
                    train('train_set1', 500, train_dir, fc_after=l, level=level)
                    train('train_set2', 1000, train_dir, fc_after=l, level=level)
                    
                    top1_1, top5_1 = eval('train_set1', 30, train_dir, fc_after=l, level=level)
                    top1_2, top5_2 = eval('train_set2', 30, train_dir, fc_after=l, level=level)
                    top1_3, top5_3 = eval('train_set3', 30, train_dir, fc_after=l, level=level)
                    
                    
                    with tf.gfile.Open("layer_generalization/Accuracy_2.txt", 'a') as f:
                            f.write('Level: %5s, Layer: %20s, Top 1 (Training): %4s,      Top 5 (Training): %4s,            Top 1 (Training): %4s,      Top 5 (Training): %4s,           Top 1 (Test): %4s,      Top 5 (Test): %4s\n' % (level, l, round(top1_1,2) , round(top5_1,2) , round(top1_2,2) , round( top5_2,2) , round(top1_3,2) , round( top5_3,2)))
                    done.append(l)
        done = []
        
def train_and_eval_slim(level_set, slim_cutting_points, start_step =1,, finish_step = 20, checkpoints_dir = checkpoints_dir,checkpoint = 'model.ckpt-150000',filename):
    """ Trains the fully_connected softmax classifiers for each given class (family, genus, species, organ) after each cutting point. 
    
    
    Args: 
      level_set: Classes to train on (family, genus, species, organ)
      slim_cutting_points: after which layer should the fc-classifer be trained?
      start_step: If you already started the training you can start at a later training step (start_step*500)
      finish_step: perform finish_step*500 training steps
      checkpoints_dir: checkpoints of model
      
    Returns:
      saves model weights into checkpoints_dir
      prints accuracy into filename text file
    """
    done = ['block1_unit3', 'block2_unit4']
    
    for level in level_set:        
        for l in slim_cutting_points:
            if l not in done:
                if l != "conv1":
                    if level =="organs" and l =="block3_unit6":
                        start_step = 5
                    else:
                        start_step = 1
                    
                    """
                    if checkpoint.endswith(".ckpt"):
                        train_dir = ("layer_generalization/fc_slim_untrained_%s_%s" % (l, level))
                        filename = "layer_generalization/Accuracy_slim_untrained.txt"
                    else:    
                        train_dir = ("layer_generalization/fc_slim_%s_%s_model2" % (l, level))
                        filename = "layer_generalization/Accuracy_slim_model2.txt"
                    """    
                        
                    for steps in range(start_step, finish_step,2):
                        train('train_set1', (steps*500), train_dir, fc_after=l, level=level,checkpoints_dir = checkpoints_dir,checkpoint=checkpoint)
                        train('train_set2', ((steps+1)*500), train_dir, fc_after=l, level=level,checkpoints_dir = checkpoints_dir,checkpoint=checkpoint)
                    
                    top1_1, top5_1 = eval('train_set1', 30, train_dir, fc_after=l, level=level)
                    top1_2, top5_2 = eval('train_set2', 30, train_dir, fc_after=l, level=level)
                    top1_3, top5_3 = eval('train_set3', 30, train_dir, fc_after=l, level=level)
                    
                    
                    with tf.gfile.Open(filename, 'a') as f:
                            f.write('Level: %5s, Layer: %20s, Top 1 (Training): %4s,      Top 5 (Training): %4s,            Top 1 (Training): %4s,      Top 5 (Training): %4s,           Top 1 (Test): %4s,      Top 5 (Test): %4s\n' % (level, l, round(top1_1,2) , round(top5_1,2) , round(top1_2,2) , round( top5_2,2) , round(top1_3,2) , round( top5_3,2)))
                    done.append(l)
        done = []

if __name__ == "__main__":
    #train_and_eval()
    level_set = ["organs"]
    slim_cutting_points = ['block1_unit3', 'block2_unit4', 'block3_unit6','block4_unit3']
    train_and_eval_slim(level_set, slim_cutting_points, checkpoints_dir='mydata/resnet_finetuned_plantclef2015_2/')   
    
    #level_set = ["family", "genus", "species", "organs"]
    #slim_cutting_points = ['block1_unit3', 'block2_unit4', 'block3_unit6','block4_unit3']
    #train_and_eval_slim(level_set, slim_cutting_points, checkpoints_dir='mydata/resnet_finetuned_plantclef2015_2/')
#    train_and_eval_slim(level_set, slim_cutting_points, checkpoints_dir = "mydata/resnet_checkpoints/" ,checkpoint = 'resnet_v2_50.ckpt')   
    