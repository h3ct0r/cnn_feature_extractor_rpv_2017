import os
import re
import cv2
import tqdm
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from alexnet import AlexNet
from caffe_classes import class_names
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold


class FeatureExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.train = {}
        self.test = {}

        self.imagenet_mean = None
        self.x_placeholder = None
        self.keep_prob = None
        self.model = None
        self.score = None
        self.softmax = None

        self.load_dataset_indexes()
        self.init_alexnet()
        pass

    @staticmethod
    def add_sample_to_dict(d, s_id, sample_path):
        sample_d = {
            'img': cv2.imread(sample_path),
            'path': sample_path
        }

        if s_id in d.keys():
            d[s_id].append(sample_d)
        else:
            d[s_id] = [sample_d]

    @staticmethod
    def add_fmap_to_dict(d, s_id, p1, p5, fc2):
        fmaps = {
            'p1': p1,
            'p5': p5,
            'fc2': fc2
        }

        if s_id in d.keys():
            d[s_id].append(fmaps)
        else:
            d[s_id] = [fmaps]

    def load_dataset_indexes(self):
        dataset_path = self.cfg['dataset']

        for root, dirs, files in os.walk(dataset_path):
            for name in files:
                if not name.endswith('.bmp'):
                    continue

                s = name.split('_')
                s_id = int(re.sub('[^0-9]', '', s[0]))
                full_img_path = os.path.join(root, name)

                if 'test' in root:
                    FeatureExtractor.add_sample_to_dict(self.test, s_id, full_img_path)
                else:
                    FeatureExtractor.add_sample_to_dict(self.train, s_id, full_img_path)

    def init_alexnet(self):
        self.imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
        self.x_placeholder = tf.placeholder(tf.float32, [1, 227, 227, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        # create model with default config ( == no skip_layer and 1000 units in the last layer)
        self.model = AlexNet(self.x_placeholder, self.keep_prob, 1000, [])

        # define activation of last layer as score
        self.score = self.model.fc8

        # create op to calculate softmax
        self.softmax = tf.nn.softmax(self.score)
        pass

    def process_samples(self, samples):
        sample_r = {}

        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Load the pretrained weights into the model
            self.model.load_initial_weights(sess)

            s_keys = samples.keys()
            for i in tqdm.trange(len(s_keys)):
                s_id = s_keys[i]
                v = samples[s_id]
                for e in v:
                    # Convert image to float32 and resize to (227x227)
                    img = cv2.resize(e['img'].astype(np.float32), (227, 227))

                    # Subtract the ImageNet mean
                    img -= self.imagenet_mean

                    # Reshape as needed to feed into model
                    img = img.reshape((1, 227, 227, 3))

                    pool1_tensor = sess.graph.get_tensor_by_name('pool1:0')
                    pool5_tensor = sess.graph.get_tensor_by_name('pool5:0')
                    fc2_tensor = sess.graph.get_tensor_by_name('fc7/fc7:0')

                    p1, p5, fc2 = sess.run([pool1_tensor, pool5_tensor, fc2_tensor],
                                           feed_dict={self.x_placeholder: img, self.keep_prob: 1})

                    FeatureExtractor.add_fmap_to_dict(sample_r,
                                                      s_id,
                                                      np.dstack(p1).flatten(),
                                                      np.dstack(p5).flatten(),
                                                      np.dstack(fc2).flatten())

                    # np.set_printoptions(threshold='nan')
                    # print("pool1 shape: {}".format(p1.shape))
                    # print("pool5 shape: {}".format(p5.shape))
                    # print("fc2 shape: {}".format(fc2.shape))

                    # Run the session and calculate the class probability
                    # probs = sess.run(self.softmax, feed_dict={self.x_placeholder: img, self.keep_prob: 1})
                    # Get the class name of the class with the highest probability
                    # class_name = class_names[np.argmax(probs)]
                    # prob = probs[0, np.argmax(probs)]
        return sample_r

    def train_test_svm(self, data, target, kfolds=5):

        # Creating folds (cv parameter)
        scores = cross_validation.cross_val_score(svm.SVC(), data, target, cv=kfolds)
        print 'SVM-RBF accuracy for 5-fold', scores.mean()

        print '\nSVM accuracy for each fold'
        # Create the folds (5, in this case). This function returns indices to split data in train test sets.
        print 'data:', data.shape
        kf = cross_validation.StratifiedKFold(target, n_folds=kfolds)

        scores_SVMRBF = 0
        scores_LinearSVM = 0

        fold = 1
        for train, test in kf:
            print "-------------------> Fold %d" % fold
            fold += 1

            # Using indices returned to separate the folds
            fold_train = [data[i] for i in train]
            fold_target = [target[i] for i in train]
            fold_train_test = [data[i] for i in test]
            fold_target_test = [target[i] for i in test]

            scores_SVMRBF = scores_SVMRBF + self.classifier_function(fold_train,
                                                                     fold_target,
                                                                     fold_train_test,
                                                                     fold_target_test,
                                                                     svm.SVC(),
                                                                     'SVM-RBF')

            scores_LinearSVM = scores_LinearSVM + self.classifier_function(fold_train,
                                                                           fold_target,
                                                                           fold_train_test,
                                                                           fold_target_test,
                                                                           svm.LinearSVC(),
                                                                           'Linear SVM')

        print '\nFinal accuracy'
        print 'SVM-RBF accuracy', scores_SVMRBF / float(kfolds)
        print 'Linear SVM accuracy', scores_LinearSVM / float(kfolds)

    '''
    	Trains a simple classifier (eg.: SVMs, DT, ...)
    '''
    def classifier_function(self, train, target, test, test_target, classifier, clf_name):
        # Start Simple Classifier

        classifier.fit(train, target)

        prediction = classifier.predict(test)
        accuracy = accuracy_score(test_target, prediction)

        print '%s Accuracy %.2f' % (clf_name, accuracy)
        return accuracy

    def export_features_by_class(self, fmap):
        target = []
        p1 = []
        p5 = []
        fc2 = []
        for k, v in fmap.items():
            for e in v:
                target.append(k)
                p1.append(e['p1'])
                p5.append(e['p5'])
                fc2.append(e['fc2'])

        return np.asarray(target), np.asarray(p1), np.asarray(p5), np.asarray(fc2)

    def start(self):
        print "[INFO]", "Processing train..."
        train_fmap = self.process_samples(self.train)
        train_target, train_p1, train_p5, train_fc2 = self.export_features_by_class(train_fmap)

        print "[INFO]", "Target data:"
        print '\ttarget:', train_target.shape
        print '\tp1:', train_p1.shape
        print '\tp5:', train_p5.shape
        print '\tfc2:', train_fc2.shape

        print "[INFO]", "Processing test..."
        test_fmap = self.process_samples(self.test)
        test_target, test_p1, test_p5, test_fc2 = self.export_features_by_class(test_fmap)

        print "[INFO]", "Test data:"
        print '\ttarget:', test_target.shape
        print '\tp1:', test_p1.shape
        print '\tp5:', test_p5.shape
        print '\tfc2:', test_fc2.shape

        full_target = np.concatenate((train_target, test_target), axis=0)
        full_p1 = np.concatenate((train_p1, test_p1), axis=0)
        full_p5 = np.concatenate((train_p5, test_p5), axis=0)
        full_fc2 = np.concatenate((train_fc2, test_fc2), axis=0)

        self.train_test_svm(full_p1, full_target)
        self.train_test_svm(full_p5, full_target)
        self.train_test_svm(full_fc2, full_target)
        pass
