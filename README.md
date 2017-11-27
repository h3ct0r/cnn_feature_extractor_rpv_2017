# CNN features from AlexNet

This repo has the code of the Tensotflow implementation of the AlexNet CNN (Tensorflow > 1.2) and scripts to extract the features from C1, C5 and FC2 layers. The code is ready-to-go, including the trained weights of AlexNet and some images to test. 

 - The AlexNet weights were downloaded from: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
 - The TensorFlow implementation of Alexnet was firstly developed by: https://github.com/kratzert/finetune_alexnet_with_tensorflow

### Install

- Install pip dependences:
        `$ pip install tf tqdm `
- Install opencv2. Follow tutorial from http://milq.github.io/install-opencv-ubuntu-debian/.
- Download the Alexnet weights and locate them on the root of this project. Weights can be downloaded from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy.
        `$ cd cnn_feature_extractor_rpv_2017/ && wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy`

### How to run

- Put the images on a folder, every class separated by folder name: 000, 001, 002, 003, etc.
- Create a new config file in JSON format (examples in **config/** folder).
- Run by executing `python main.py -c config/*a_config_file.json*`

#### Run with the Iris Dataset

- `python main.py -c config/alexnet_iris.json`

### Results and plots

All the results and plots are defined on the config file, but generally they are located in the *plots/* and *results/* folders.

- *plots*: are PDF generated files with a normalized confusion matrix using a heatmap color map.
- *results*: are JSON files generated with all the relevant information about the experiment: confusion matrix, overall precision, average precision, etc.
- 
## Authors
- Héctor Azpúrua
- Patricia Almeida
- Willian Hofner