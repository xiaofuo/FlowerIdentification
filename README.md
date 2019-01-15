# FlowerIdentification
## Disclaimer
This was one of my first projects in Tensorflow and therefore neither is very elegent, nor very efficiently implemented. <\br>
The trained weights of the network are missing.

## Task
The task of the 2016 PlantCLEF challenge (https://www.imageclef.org/lifeclef/2016/plant) was to predict the *class* and *probability* of a flower displayed on an image. 

## Network
The code is based upon the research presented in </br>
Šulc, M., Mishkin, D. & Matas, J. (2016). Very deep residual networks with maxout for plant identifcation in the wild. Working notes of CLEF 2016 conference. (http://ceur-ws.org/Vol-1609/16090579.pdf)

A Deep Residual Neural Network to identify flowers is created. The network is based on a pre-trained model within the Tensorflow framework (https://github.com/tensorflow/models/tree/master/research/slim). 

To fine-tune the existing weights, images of the [PlantCLEF dataset](http://www.imageclef.org/lifeclef/2017/plant) were used.

## Description of Scripts
*   *plantclef\_download\_and\_convert\_data*
This script downloads the given dataset of the PlantCLEF challenge 2016 and converts it into tfRecords. It saves the *image, species\_label, genus\_label, family\_label, organ\_label* (see *changed\_scripts/dataset\_utils: image\_to\_tfexample()*).
*   *changed\_scripts/resnet\_utils and changed\_scripts/resnet\_v2*
Contains the architecture of the Residual Networks. Methods containing "intermediate" in their name are changed, so that they can be cut of at any level.
*   *resNetClassifier*
Example on how to train and evaluate a network. The network architecture is further edited (my\_cnn()) and initialized (get\_init\_fn()). The code contains the load\_batch() method, here the data is loaded and preprocessed (see my\_resnet\_preprocessing). Probably the method as to be adjusted, when it was created the tfRecords only contained one label \glqq label\grqq{*, not four! Just load the one you need, or all (see layerwise\_generalization's load\_batch method for more insights).
*   *train\_my\_network*
script calling resNetClassifier to train the models. Also contains the method *final\_evaluation()* and  *final\_evaluation\_generic()*. The first uses three networks to make a prediction, using the 6 time data augmentation. The latter uses n networks, but you have to specify which preprocessing method you want to use.
*   *my\_resnet\_preprocessing*
Here the preprocessing is done. One for training, one for evaluation and one for the final evaluation. You can call the first two by using *preprocess\_image()*. There are deprecated methods included, the most recent ones are *preprocess\_for\_train()* and *preprocess\_for\_eval()*. *Always* change the dtype of the image to float (tf.convert\_image\_dtype), int type and float type preprocessings for train and eval don't work! You can find more preprocessing methods in the Tensorflow documentation under "Image".
*   *zeiler\_fergus*
Contains code calling *tf\_cnnvis*, this github repo contains code to visualize individual layers of your network. 
*   *my\_functions*
Contains additional functions, e.g. the MaxOut function used in the network introduced by the CMP group in the PlantCLEF challenge 2016.
*   *DataVisualisation*
Contains the splitting method used to get the tfRecords you want to use. *Note*: the whole script besides the *show\_example()* method should be moved to *my\_functions* or another script. (It is still in this script, because I didn't want to mess my code up in the last days). At least rename it :D 
*   *calculate\_optimal\_stimuli*
Goes through the dataset and finds the image which has the highest softmax output for a class. Saves it into a text file.
*   *freezer*
Method used to freeze a graph. This means, save its weights and architecture together into a *.pb* file. This can then be used to be deployed e.g. on a smartphone. 
*   *crop\_acts*
*zeiler\_fergus* creates visualizations for all channels of one layer, this method now crops out the specified channels and saves them into a grid (9 images needed, change the code for more or less)
*   *grid\_plotter*
Plots 9 images into a grid, good for visualizations ;)
*   *kernel\_visualization*
Contains methods from the DeepDream tutorial of Tensorflow. The outputs seem reasonable but did not really give me insights into what the network learned. The outputs for all classes almost looked the same. I'm not sure whether I messed up something, it does not work as well with the ResNet or if the outputs actually look like this, when finetuned on the plantclef-challenge!
*   *layerwise\_generalization*
Similar to *ResNetClassifier*, but trains *ONLY ONE LAYER*: speciefied in train\_op: variable\_to\_train !\\
Used to train the fc-softmax-classifiers afther a cutoff of the network. The methods *train\_and\_eval()* as well as *train\_and\_eval\_slim()* are training and evaluating the new layers and saving the results into a text-file.
*   *maximum\_pathway*
Only the methods *fin\_max\_activation\_per\_filter()* and *top\_channel\_per\_layer()* are interesting. The find the 9 images which excite each channel of each layer the most and save them as a text file. The latter now searches for the top activated channels in each layer and saves it into a text-file. The txt-files are not really readable (see max\_act\_worker). Need to be changed.
*   *max\_act\_worker*
This script is only necessary because the max\_activation.txt is not readable (see maximum\_pathway). Change the file, maybe into a csv or something similar.

## Results
The best network resulted in a Top-1 accuracy of 64.48% and a top-5 accuracy of 82.73%. This network only used scaled, ramdom cuts of the images to train. Further preprocessing did not improve the results.
