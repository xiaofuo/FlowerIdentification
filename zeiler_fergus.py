import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf

# Main slim library
slim = tf.contrib.slim

import tf_cnnvis.tf_cnnvis
import scipy.misc
import maximum_pathway

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def get_img(src, img_size=False):
    """
    Reades a image from source src and returns it as a numpy array
        
    Args:
        src: path of image to load
        img_size: if given: resizes the images to specified size
        
    Returns:
        Numpy array of image
    
    """
    img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img                
                
def visualize_cnnvis(image_list,output_dir, model_fn =  'mydata/resnet_finetuned_plantclef2015_3/frozen_model.pb'):  
    """
    Creates deconvolutions and feature map activities for the given network for every possible layer!
    
    Args:
        model_fn: Model to use.
        output_dir: Where to save the images
        image_list: images to use
        
    Returns:
        Saves the generated images
    """
    
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
   
    
    
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = t_input-imagenet_mean
    #tf.import_graph_def(graph_def, {'input':t_preprocessed})                
    tf.import_graph_def(graph_def, input_map={'resnet_v2_50/Pad':t_preprocessed})
    
    layers = [op.name for op in graph.get_operations()]
    print(layers)
    
    
    for i in image_list:
        im = get_img(os.path.join("./mydata/train/", i), (224,224,3)).astype(np.float32)
        im = np.expand_dims(im, axis = 0)
        


        layers = ["r", "p", "c"]
    
        tf_cnnvis.activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = layers, path_outdir =output_dir)
        tf_cnnvis.deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = "import/my_fc_1/BiasAdd", path_outdir =output_dir) #import/resnet_v2_50/pool5 #layers
        #tf_cnnvis.deepdream_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {X : im})
    print("DONE")
    
def deepdream_cnnvis(image_list,class_list,output_dir,model_fn =  'mydata/resnet_finetuned_plantclef2015_/frozen_model.pb'):  
     """
     Generates Deep dream like images for the given model. The images are changed acording to the classes given in class_list
    
    Args:
        model_fn: Model to use.
        output_dir: Where to save the images
        class_list: Which class should the network look for in the image?
        image_list: images given
        
    Returns:
        Saves the generated images
    """
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
   
    
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 0# 117.0
    t_preprocessed = t_input-imagenet_mean
    #tf.import_graph_def(graph_def, {'input':t_preprocessed})                
    tf.import_graph_def(graph_def, input_map={'resnet_v2_50/Pad':t_preprocessed}) # resnet_v2_50/Pad
    
    
    layers = [op.name for op in graph.get_operations()]
    
    for i, label in zip(image_list, class_list):
        im =get_img(os.path.join("./mydata/train", i), (224,224,3)).astype(np.float32) #TODO ./mydata/train/
        im = np.expand_dims(im, axis = 0)
                     
        tf_cnnvis.deepdream_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layer = "import/my_fc_1/BiasAdd", classes =[label], path_outdir =output_dir )

                
    print("DONE")    
    
def visualize_cnnvis_max(image_list, model_fn =  'mydata/resnet_finetuned_plantclef2015_3/frozen_model.pb', output_dir="./my_Outputs2"): 
     """
    Creates deconvolutions and feature map activities for the given network. Given the list of images, it searches for the layers which are activated most and creates 
    Visualizations for them.
    
    Args:
        model_fn: Model to use.
        image_list: list of images to use for analysis
        output_dir: where to save images
        
    Returns:
        Saves the generated images
    """
    activations = maximum_pathway.find_max_activation_per_filter(image_list)
    print("\nAll activations found and saved.")
    activations = maximum_pathway.top_channel_per_layer(activations)
    print("Found the top activations per layer.")
            
    
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
   
    
    
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = t_input-imagenet_mean
    #tf.import_graph_def(graph_def, {'input':t_preprocessed})                
    tf.import_graph_def(graph_def, input_map={'resnet_v2_50/Pad':t_preprocessed})
    
    for key in activations.keys():
        for channel_no, values in activations[key].items():
            for value in values:
                
    
                im = get_img(os.path.join("./mydata/train/", value[2]), (224,224,3)).astype(np.float32)
                im = np.expand_dims(im, axis = 0)
                
            

                layer = "import/%s" % (key)
                
                layers = [op.name for op in graph.get_operations() if (layer in op.name and "/bottleneck_v2/add" in op.name) or ("resnet_v2_50/conv1" == key and "resnet_v2_50/conv1/BiasAdd" in op.name )]
                
                layer = layers[0]
                
                tf_cnnvis.activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = layer, path_outdir =output_dir )
                tf_cnnvis.deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = layer, path_outdir = output_dir) #import/resnet_v2_50/pool5 #layers
            
           
    print("Visualization finished.")    
    
def test_vis(layer_to_img_dict, model_fn, output_dir ="./zeiler_fergus_vis" ):
     """
    Creates deconvolutions and feature map activities for the given network. For each layer in layer_to_img_dict it creates the outputs for all associated images.
    
    Args:
        model_fn: Model to use.
        layer_to_img_dict: dictionary mapping layer names to images
        output_dir: where to save images
        
    Returns:
        Saves the generated images
    """
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    
    
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = 2*(t_input -0.5)#-imagenet_mean
    tf.import_graph_def(graph_def, input_map={'resnet_v2_50/Pad':t_preprocessed})
   


    for layer_name, img_list in layer_to_img_dict.items():
        filename = layer_name.split("/")[2]

        layer = [v.name for v in sess.graph.get_operations() if layer_name == v.name]
        
        
        if len(layer) > 1:
            print("multiple choices, chose first")
            layer = layer[0]
        elif len(layer) == 0:
            print("ERROR: Layer %s not found" % layer_name)

        for img in img_list:
            im = get_img(os.path.join("./mydata/Max_act_V0/", img)).astype(np.float32)
            im = np.expand_dims(im, axis = 0)
                                

    
            tf_cnnvis.activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = layer, path_outdir = ("%s/%s" % (output_dir, filename)))
            tf_cnnvis.deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {t_input : im}, layers = layer, path_outdir = ("%s/%s" % (output_dir,filename)))
        
       
if __name__ == "__main__":
    layer_to_img_dict = {}
    layer_to_img_dict["import/resnet_v2_50/block1/unit_3/bottleneck_v2/conv3/BiasAdd"] = ["8865.jpg", "23872.jpg","63355.jpg","63788.jpg","66492.jpg","82865.jpg","83448.jpg","91643.jpg","105104.jpg"] # Channel 14
    layer_to_img_dict["import/resnet_v2_50/block2/unit_4/bottleneck_v2/conv3/BiasAdd"] = ["8351.jpg","9529.jpg","11022.jpg","15234.jpg","26659.jpg","53459.jpg","78761.jpg","104909.jpg","107183.jpg"] # Channel 47
    layer_to_img_dict["import/resnet_v2_50/block3/unit_6/bottleneck_v2/conv3/BiasAdd"] = ["7547.jpg","11460.jpg","16568.jpg","59547.jpg","59975.jpg","71193.jpg","85500.jpg","100135.jpg","104909.jpg"] # Channel 21
    layer_to_img_dict["import/resnet_v2_50/block4/unit_3/bottleneck_v2/conv3/BiasAdd"] = ["12788.jpg","28532.jpg","53267.jpg","65274.jpg","72290.jpg","77829.jpg","78324.jpg","97783.jpg","107257.jpg"] # Channel 9
    
    layer_to_img_dict["import/resnet_v2_50/block1/unit_3/bottleneck_v2/conv3/BiasAdd"] = ["26387.jpg","54771.jpg","57442.jpg","75240.jpg","79990.jpg","85326.jpg","100426.jpg","103896.jpg","104035.jpg"] # Channel 25
    layer_to_img_dict["import/resnet_v2_50/block2/unit_4/bottleneck_v2/conv3/BiasAdd"] = ["20854.jpg","23434.jpg","36099.jpg","61539.jpg","62266.jpg","74237.jpg","91352.jpg","93751.jpg","106742.jpg"] # Channel 0
    layer_to_img_dict["import/resnet_v2_50/block3/unit_6/bottleneck_v2/conv3/BiasAdd"] = ["2565.jpg","23052.jpg","27269.jpg","28552.jpg","66636.jpg","74242.jpg","98121.jpg","110383.jpg","110494.jpg"] # Channel 14
    layer_to_img_dict["import/resnet_v2_50/block4/unit_3/bottleneck_v2/conv3/BiasAdd"] = ["7634.jpg","13518.jpg","34140.jpg","42953.jpg","43117.jpg","52336.jpg","70250.jpg","77294.jpg","82007.jpg"] # Channel 12
    
    layer_to_img_dict["import/resnet_v2_50/block1/unit_3/bottleneck_v2/conv3/BiasAdd"] = ["10454.jpg","27974.jpg","36815.jpg","41241.jpg","43790.jpg","49546.jpg","69444.jpg","98172.jpg","105966.jpg"] # Channel 71
    layer_to_img_dict["import/resnet_v2_50/block2/unit_4/bottleneck_v2/conv3/BiasAdd"] = ["13748.jpg","20916.jpg","28921.jpg","31870.jpg","46491.jpg","57442.jpg","66754.jpg","79996.jpg","88590.jpg"] # Channel 79
    layer_to_img_dict["import/resnet_v2_50/block3/unit_6/bottleneck_v2/conv3/BiasAdd"] = ["13398.jpg","18367.jpg","30654.jpg","39878.jpg","54493.jpg","76251.jpg","93817.jpg","101750.jpg","112766.jpg"] # Channel 597    
    layer_to_img_dict["import/resnet_v2_50/block4/unit_3/bottleneck_v2/conv3/BiasAdd"] = ["28467.jpg","31870.jpg","50851.jpg","55093.jpg","59001.jpg","69758.jpg","82149.jpg","85326.jpg","94965.jpg"] # Channel 35   
    test_vis(layer_to_img_dict, model_fn =  'mydata/resnet_finetuned_plantclef2015_2/frozen_model.pb')
