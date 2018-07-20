# Overview--FCN-ZY3
FCN applying in zy3
Some detail can go to https://github.com/shekkizh/FCN.tensorflow.git

#FCN implementation
This file FCN.py uses pre-train model "VGG-19" and FCN-8s and you can refer to the paper https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

#What I do?
I use this model to train my own datasets in the field of remote sensing.

#Data description
I use the imagery from China's first high resolution satellite-ZY3. This imagey has four multispectral bands with spatial resolution of 5.8 meters 
and one panchromatic band with spatial resolution of 1 meters.

#Sampling design
I use a slide window with a certain stride to obtain training/validation/testing sample patchs.

#Copyright
TensorflowUtils.py and FCN.py is written by shekkizh (https://github.com/shekkizh/FCN.tensorflow.git)
input_data.py is written by myself and some ideas in it are learned from Tensorflow official examples, such as MNIST classification problems

#Acknowledgement
Be grateful to shekkizh (https://github.com/shekkizh/FCN.tensorflow.git) and Tensorflow offical

