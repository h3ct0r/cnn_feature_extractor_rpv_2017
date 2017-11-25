import os
import re
import cv2
import math
import tqdm
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from alexnet import AlexNet
from svn_classificator import SvnClassificator

class FeatureExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.dataset = {}

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

                FeatureExtractor.add_sample_to_dict(self.dataset, s_id, full_img_path)

        # clean missing numbers
        prev_num = 0
        old_keys = sorted(self.dataset.keys())
        for i in xrange(len(old_keys)):
            old_k = old_keys[i]
            self.dataset[prev_num] = self.dataset.pop(old_k)
            prev_num += 1

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

    @staticmethod
    def export_features_by_class(fmap, cutoff=0.3):
        if 0 < cutoff > 1:
            raise Exception("cutoff must be between 0 and 1")

        t_train = []
        p1_train = []
        p5_train = []
        fc2_train = []

        t_test = []
        p1_test = []
        p5_test = []
        fc2_test = []

        for k, v in fmap.items():
            total_e = len(v)
            cut_p = int(math.ceil(total_e * (1 - cutoff)))

            for i in xrange(len(v)):
                e = v[i]
                if i < cut_p:
                    t_train.append(k)
                    p1_train.append(e['p1'])
                    p5_train.append(e['p5'])
                    fc2_train.append(e['fc2'])
                else:
                    t_test.append(k)
                    p1_test.append(e['p1'])
                    p5_test.append(e['p5'])
                    fc2_test.append(e['fc2'])

        labels = [str(e) for e in sorted(list(set(t_train)))]

        return [np.asarray(t_train), np.asarray(p1_train), np.asarray(p5_train), np.asarray(fc2_train)], \
               [np.asarray(t_test), np.asarray(p1_test), np.asarray(p5_test), np.asarray(fc2_test)], \
               labels

    def start(self):
        print "[INFO]", "Processing data..."
        fmap = self.process_samples(self.dataset)
        train, test, labels = self.export_features_by_class(fmap)

        target_train, p1_train, p5_train, fc2_train = train
        target_test, p1_test, p5_test, fc2_test = test

        print "[INFO]", "Data shapes:"
        print '\tTarget:{}/{}'.format(target_train.shape[0], target_test.shape[0])
        print '\tC1:', p1_train.shape
        print '\tC5:', p5_train.shape
        print '\tFC2:', fc2_train.shape
        print '\tLabels:', labels

        svn_fn = SvnClassificator(self.cfg)
        # res_p1_1 = svn_fn.svm_simple(train_c1, train_target, test_c1, test_target, labels)
        # print res_p1_1

        # res_early = svn_fn.svm_early_fusion(target_train, p1_train, p5_train, fc2_train, target_test, p1_test, p5_test,
        #                                     fc2_test, labels)
        # print res_early
        res_late = svn_fn.svm_late_fusion(target_train, p1_train, p5_train, fc2_train, target_test, p1_test, p5_test,
                                            fc2_test, labels)
        print res_late
        pass