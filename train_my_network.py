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
import changed_scripts.dataset_utils as dataset_utils
import my_resnet_preprocessing
import kernel_visualization
import my_functions


import layerwise_generalization

##############################################################################################
##############################################################################################




# Network directories. In paper refered to as networks 1 to 8 instead of 0 to 7

log_dir_network0 = 'mydata/resnet_finetuned_plantclef2015_test_3'   #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_test_3
log_dir_network1 = 'mydata/resnet_finetuned_plantclef2015_1'        #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_1
log_dir_network2 = 'mydata/resnet_finetuned_plantclef2015_2'        #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_2
log_dir_network3 = 'mydata/resnet_finetuned_plantclef2015_3'        #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_3
log_dir_network4 = 'mydata/resnet_finetuned_plantclef2015_4'        #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_4

log_dir_network5 = 'mydata/resnet_finetuned_plantclef2015_5'        #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_5
log_dir_network6 = 'mydata/resnet_finetuned_plantclef2015_6'        #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_6
log_dir_network7 = 'mydata/resnet_finetuned_plantclef2015_7'        #tensorboard --logdir=mydata/resnet_finetuned_plantclef2015_7





# Training the networks


def train_network1():
    """
    Training the first network on folds 1 and 2, evaluated on fold 3 (150.000 iterations, 75k on each set)
    """
    resNetClassifier.train('train_set2', 125000, log_dir_network1)
    for x in range(51,60,2):
        resNetClassifier.train('train_set1', 2500*x, log_dir_network1)
        resNetClassifier.train('train_set2', 2500*x+2500, log_dir_network1)
        resNetClassifier.eval('train_set3', log_dir_network1)


def train_network2():
    """
    Training the second network on folds 2 and 3, evaluated on fold 1 (150.000 iterations, 75k on each set)
    """
    for x in range(59,60,2):
        resNetClassifier.train('train_set2', 2500*x, log_dir_network2)
        resNetClassifier.train('train_set3', 2500*x+2500, log_dir_network2)
        resNetClassifier.eval('train_set1', log_dir_network2)    


def train_network3():    
    """
    Training the third network on folds 1 and 3, evaluated on fold 2 (150.000 iterations, 75k on each set)
    """
    for x in range(51,60,2):
        resNetClassifier.train('train_set3', 2500*x, log_dir_network4)
        resNetClassifier.train('train_set1', 2500*x+2500, log_dir_network4)
        resNetClassifier.eval('train_set2', log_dir_network4)        
        
def train_network4():    
    """
    Training the third network on folds 1 and 3, evaluated on fold 2 (150.000 iterations, 75k on each set)
    """
    for x in range(51,60,2):
        resNetClassifier.train('train_set1', 2500*x, log_dir_network3)
        resNetClassifier.train('train_set2', 2500*x+2500, log_dir_network3)
        resNetClassifier.eval('train_set3', log_dir_network3)                
        
  
def train_network5():    
    """
    Training the third network on folds 1 and 3, evaluated on fold 2 (150.000 iterations, 75k on each set)
    """
    for x in range(57,60,2):
        resNetClassifier.train('train_set1', 2500*x, log_dir_network5)
        resNetClassifier.train('train_set2', 2500*x+2500, log_dir_network5)
        #resNetClassifier.eval('train_set3', log_dir_network3)     
        
def train_network6():
    """
    Training the second network on folds 2 and 3, evaluated on fold 1 (150.000 iterations, 75k on each set)
    """
    for x in range(1,60,2):
        resNetClassifier.train('train_set2', 2500*x, log_dir_network6)
        resNetClassifier.train('train_set3', 2500*x+2500, log_dir_network6)
        #resNetClassifier.eval('train_set1', log_dir_network2)

def train_network7():    
    """
    Training the third network on folds 1 and 3, evaluated on fold 2 (150.000 iterations, 75k on each set)
    """
    for x in range(25,60,2):
        resNetClassifier.train('train_set3', 2500*x, log_dir_network7)
        resNetClassifier.train('train_set1', 2500*x+2500, log_dir_network7)
        #resNetClassifier.eval('train_set2', log_dir_network3)        
                  

# Voting of the three networks on one image.
# Voting is done by returning the class_id with the overall maximum "likelyhood"
# Saves the data into the given file (Default: 'predictions.txt')
#https://github.com/thomaspark-pkj/resnet-ensemble/blob/master/eval_image_classifier_ensemble.py
def final_evaluation(label_dir, dataset_dir, filename="predictions.txt", visualize_kernel = False):
    """
    Evaulates a CNN on the test-set and saves the predictions in the form  <ImageId;ClassId;Probability> into a txt-file (filename)
    
    Args:
        label_dir: Directory where labels dictionary can be found (mapping from one-hot encodings to class_id)
        dataset_dir: Directory where test dataset can be found
        filename: filename of txt-file to save predictions
        visualize_kernel: Do you want to visualize the first layer of convolutions?
        
    Returns:
        Saves the predictions into "filename"
    """
    number_images =8000
    
    # Choose the three networks to evaluate on
    checkpoint_paths = [ "mydata/resnet_finetuned_plantclef2015_5/model.ckpt-150000",  "mydata/resnet_finetuned_plantclef2015_6/model.ckpt-150000","mydata/resnet_finetuned_plantclef2015_7/model.ckpt-102500"]
    
    output_list = []
    labels_list = []
    
    for index in range(len(checkpoint_paths)):
        with tf.Graph().as_default() as graph:
            dataset = dataVisualisation.get_split('test_set', dataset_dir,label_type="one")
            
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                                           shuffle=False,
                                                                           common_queue_capacity=8000,
                                                                           common_queue_min=0)
            
            image_raw, label = data_provider.get(['image', 'label'])
                            

            
            # Preprocessing return original image, center_crop and 4 corner crops with adjusted color values
            image, augmented_image1, augmented_image2, augmented_image3, augmented_image4, augmented_image5 = my_resnet_preprocessing.preprocess_for_final_run2(image_raw, 224, 224) 
            
            image,augmented_image1,augmented_image2, augmented_image3, augmented_image4,augmented_image5, labels = tf.train.batch([image, augmented_image1, augmented_image2, augmented_image3, augmented_image4, augmented_image5, label], 
                                            batch_size=1,
                                            num_threads=1,
                                            capacity=2 * 1)
            
            
            
            
            logits1 = resNetClassifier.my_cnn(image, is_training = False, dropout_rate =1)
            logits2 = resNetClassifier.my_cnn(augmented_image1, is_training = False, dropout_rate =1)
            logits3 = resNetClassifier.my_cnn(augmented_image2, is_training = False, dropout_rate =1)
            logits4 = resNetClassifier.my_cnn(augmented_image3, is_training = False, dropout_rate =1)
            logits5 = resNetClassifier.my_cnn(augmented_image4, is_training = False, dropout_rate =1)            
            logits6 = resNetClassifier.my_cnn(augmented_image5, is_training = False, dropout_rate =1)

            total_output = np.empty([number_images * 1, dataset.num_classes])
            total_labels = np.empty([number_images * 1], dtype=np.int32)
            offset = 0
            
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_paths[index])
              
                
                if visualize_kernel:
                    tf.get_variable_scope().reuse_variables()
                    
                    #for v in tf.global_variables():
                    #    print(v.name)
                        
                    weights = tf.get_variable("resnet_v2_50/conv1/weights")
                    print(weights.get_shape()[0].value, weights.get_shape()[1].value, weights.get_shape()[2].value, weights.get_shape()[3].value)
                    
                    weights = tf.slice(weights,[0,0,0,1] , [weights.get_shape()[0].value, weights.get_shape()[1].value, weights.get_shape()[2].value, 2])
                    
                    grid = kernel_visualization.put_kernels_on_grid (weights)
                    
                    
                    sum1 = tf.summary.image('conv1/kernels', grid, max_outputs=1)
                    _, summary1, img = sess.run([merged, sum1, tf.squeeze(grid)])
                    visualize_writer.add_summary(summary1,2)
                    fig = plt.figure()
                    plt.imshow(img)
                    plt.savefig("images/kernelsOne_%s.png" % (index))
                    #plt.show() 
                    plt.close(fig)
                
                
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for i in range(number_images):
                    #print('step: %d/%d' % (i+1, number_images))
                    
                    logit1, logit2, logit3, logit4,logit5, logit6, media_id = sess.run([logits1, logits2, logits3, logits4, logits5, logits6, labels])
                    
                    media_id = media_id[0]

                    # Use Average for voting of logits
                    logits = tuple(i + j for i, j in zip(logit1[0], logit2[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit3[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit4[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit5[0]))
                    logits = tuple(i + j for i, j in zip(logits, logit6[0]))
                    logits = [x / 6 for x in logits] 
                    
                    
                    # Passing logits through softmax function to receive "probabilities"
                    logits = my_functions.numpy_softmax(logits)
                    
                    
                    total_output[offset:offset + 1] = logits
                    total_labels[offset:offset + 1] = media_id
                    offset += 1
                coord.request_stop()
                coord.join(threads)

            output_list.append(total_output)
            labels_list.append(total_labels)
            
        
    prediction_filename = filename
    #os.remove(prediction_filename)
            
    for i in range(number_images):
        image_id = labels_list[0][i]
        
        for p in range(1000):
            p1 = output_list[0][i][p]
            p2 = output_list[1][i][p]
            p3 = output_list[2][i][p]
            
            probability = np.amax([p1, p2, p3])
           
            
            class_id = dataset_utils.read_label_file(label_dir)[p]
            
            with tf.gfile.Open(prediction_filename, 'a') as f:
                f.write('%s;%s;%f\n' % (image_id, class_id, probability)) # <ImageId;ClassId;Probability>
                

def final_evaluation_generic(label_dir, dataset_dir, checkpoint_paths, preprocessing_methods, filename="predictions.txt"):
    """
    Evaulates a CNN on the test-set and saves the predictions in the form  <ImageId;ClassId;Probability> into a txt-file (filename)
    Can use multiple models, is not limited to 3
    
    Args:
        label_dir: Directory where labels dictionary can be found (mapping from one-hot encodings to class_id)
        dataset_dir: Directory where test dataset can be found
        checkpoint_paths: checkpoints of the used models
        preprocessing_methods: corresponding preprocessing methods for the models
        filename: filename of txt-file to save predictions
        
    Returns:
        Saves the predictions into "filename"
    """
    number_images =8000
        

    output_list = []
    labels_list = []
    
    for checkpoint, preprocessing_method in zip(checkpoint_paths, preprocessing_methods):
        with tf.Graph().as_default() as graph:
            dataset = dataVisualisation.get_split('test_set', dataset_dir,label_type="one")
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                                           shuffle=False,
                                                                           common_queue_capacity=8000,
                                                                           common_queue_min=0)
            
            image_raw, label = data_provider.get(['image', 'label'])
                            

            
            # Preprocessing return original image, center_crop and 4 corner crops with adjusted color values
            image  = preprocessing_method(image_raw, 224, 224) 
            
            images, labels = tf.train.batch([image,  label], 
                                            batch_size=1,
                                            num_threads=1,
                                            capacity=2 * 1)
            
            
            
            
            logits1 = resNetClassifier.my_cnn(images, is_training = False, dropout_rate =1)

            total_output = np.empty([number_images * 1, dataset.num_classes])
            total_labels = np.empty([number_images * 1], dtype=np.int32)
            offset = 0
            
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)
       

           
                
                
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for i in range(number_images):
                    #print('step: %d/%d' % (i+1, number_images))
                           
                    logit1, media_id = sess.run([logits1, labels])
                    
                    media_id = media_id[0]

                                       
                    # Passing logits through softmax function to receive "probabilities"
                    logits = my_functions.numpy_softmax(logit1)
                    
                    
                    total_output[offset:offset + 1] = logits
                    total_labels[offset:offset + 1] = media_id
                    offset += 1
                coord.request_stop()
                coord.join(threads)

            output_list.append(total_output)
            labels_list.append(total_labels)

            
    with tf.gfile.Open(filename, 'a') as f:
        for i in range(number_images):
            image_id = labels_list[0][i]
            
            for p in range(1000):
                predictions = []
                for index in range(len(output_list)):
                    predictions.append(output_list[index][i][p])
                
                probability = np.sum(predictions)/len(predictions) #np.amax(predictions)
                
                class_id = dataset_utils.read_label_file(label_dir)[p]
                f.write('%s;%s;%f\n' % (image_id, class_id, probability)) # <ImageId;ClassId;Probability>
                
################################################################################################    
################################################################################################    
################################################################################################    


if __name__ == "__main__":
    #train_network1()
    #train_network2()
    #train_network3()
    #train_network4()
    #train_network5()
    #train_network6()
    #train_network7()    
    

    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network0)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network0)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network0)
    
    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network1)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network1)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network1)

    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network2)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network2)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network2)

    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network3)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network3)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network3)

    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network4)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network4)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network4)
    
    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network5)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network5)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network5)    
    
    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network6)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network6)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network6)
    
    #top1, top5 = resNetClassifier.eval('train_set1', log_dir_network7)
    #top1, top5 = resNetClassifier.eval('train_set2', log_dir_network7)
    #top1, top5 = resNetClassifier.eval('train_set3', log_dir_network7)
    
    
    
    #final_evaluation(label_dir = 'mydata/PlantClefTraining2015' , dataset_dir='mydata/PlantCLEF2016Test/' , visualize_kernel=True)
    
    #checkpoint_paths = [ "mydata/resnet_finetuned_plantclef2015_2/model.ckpt-150000"]
    #preprocessing_methods = [my_resnet_preprocessing.preprocess_for_eval_v0]
    ##final_evaluation_generic(label_dir = 'mydata/PlantClefTraining2015' , dataset_dir='mydata/PlantCLEF2016Test/' , checkpoint_paths=checkpoint_paths, preprocessing_methods=preprocessing_methods, filename="mydata/RunFilesToolAndResults/RunFilesToolAndResults/OfficialRunfiles/image/prediction_model2.txt")
    
    #checkpoint_paths = ["mydata/resnet_finetuned_plantclef2015_test_3/model.ckpt-150000"]
    #preprocessing_methods = [my_resnet_preprocessing.preprocess_for_eval_v0]
    #final_evaluation_generic(label_dir = 'mydata/PlantClefTraining2015' , dataset_dir='mydata/PlantCLEF2016Test/' , checkpoint_paths=checkpoint_paths, preprocessing_methods=preprocessing_methods, filename="mydata/RunFilesToolAndResults/RunFilesToolAndResults/OfficialRunfiles/image/prediction_model0.txt")
    
    #checkpoint_paths = [ "mydata/resnet_finetuned_plantclef2015_2/model.ckpt-150000", "mydata/resnet_finetuned_plantclef2015_test_3/model.ckpt-150000"]
    #preprocessing_methods = [my_resnet_preprocessing.preprocess_for_eval_v0,my_resnet_preprocessing.preprocess_for_eval_v0]
    ##final_evaluation_generic(label_dir = 'mydata/PlantClefTraining2015' , dataset_dir='mydata/PlantCLEF2016Test/' , checkpoint_paths=checkpoint_paths, preprocessing_methods=preprocessing_methods, filename="mydata/RunFilesToolAndResults/RunFilesToolAndResults/OfficialRunfiles/image/prediction_model0_model2.txt")
    
