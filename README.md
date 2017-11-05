# CNN features from AlexNet

This repo has the code of the Tensotflow implementation of the AlexNet CNN (Tensorflow > 1.2) and scripts to extract the features from C1, C5 and FC2 layers. The code is ready-to-go, including the trained weights of AlexNet and some images to test. 

 - The AlexNet weights were downloaded from: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
 - The TensorFlow implementation of Alexnet was firstly developed by: https://github.com/kratzert/finetune_alexnet_with_tensorflow

### Install

    - Install tf from pip
    - Download the Alexnet weights and locate them on the root of this project. Weights can be downloaded from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy.

### Todos

 - Get features from layers
 - Implement SVM + RBF