import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC

from itertools import groupby

class ClassificatorHelper(object):
    def __init__(self, cfg):
        self.cfg = cfg
    pass
    
    def svm_simple(self, train_data, train_target, test_data, test_target, labels, kfolds=5, debug_level=100):
        """
        Train a simple SVM with grid search
        :param train_data: 
        :param train_target: 
        :param test_data: 
        :param test_target: 
        :param labels: 
        :param kfolds: 
        :return: 
        """

        print "[INFO]", "SVM Simple execution ..."

        parameter_candidates = [
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 10, 100], 'kernel': ['rbf']}
        ]

        # perform a grid search with a kfolds cross validation
        search = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1,
                              verbose=debug_level)
        search.fit(train_data, train_target)

        # Capture and fit the best estimator from across the grid search
        best_svm = search.best_estimator_
        predictions = best_svm.predict(test_data)

        y_data = confusion_matrix(test_target, predictions)

        y_data_float = y_data.astype(float)
        y_data_float *= 1/y_data_float.max()

        class_report = classification_report(test_target, predictions, target_names=labels)

        accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

        return {
            'rbf': search.best_score_,
            'kernel': search.best_estimator_.kernel,
            'C': search.best_estimator_.C,
            'gamma': search.best_estimator_.gamma,
            'class_report': class_report,
            'y_data': y_data.tolist(),
            'y_data_normalized': y_data_float.tolist(),
            'labels': labels,
            'predictions': predictions.tolist(),
            'test_target': test_target.tolist(),
            'oa': accuracy[0],
            'aa': accuracy[1]
        }

    def svm_early_fusion(self, train_target, train_p1, train_p5, train_fc2, test_target, test_p1, test_p5, test_fc2,
                         labels, kfolds=5, debug_level=100):
        """
        Perform permutations of classificators and then join all of them
        :param train_target: 
        :param train_p1: 
        :param train_p5: 
        :param train_fc2: 
        :param test_target: 
        :param test_p1: 
        :param test_p5: 
        :param test_fc2: 
        :param labels: 
        :param kfolds: 
        :return: 
        """

        print "[INFO]", "SVM Early fusion ..."

        results = []
        fmaps_names = ['p1', 'p5', 'fc2']
        train_fmaps = [train_p1, train_p5, train_fc2]
        test_fmaps = [test_p1, test_p5, test_fc2]

        pairwise_comb = list(itertools.permutations(fmaps_names, 2))
        pairwise_comb.append(fmaps_names)

        parameter_candidates = [
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 10, 100], 'kernel': ['rbf']}
        ]

        for each in pairwise_comb:
            print(each)
            a_i = fmaps_names.index(each[0])
            b_i = fmaps_names.index(each[1])
            train_data = np.concatenate((train_fmaps[a_i], train_fmaps[b_i]), axis=1)
            test_data = np.concatenate((test_fmaps[a_i], test_fmaps[b_i]), axis=1)

            if len(each) > 2:
                train_data = np.concatenate((train_data, train_fmaps[2]), axis=1)
                test_data = np.concatenate((test_data, test_fmaps[2]), axis=1)

            # Fit the grid search
            search = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1,
                                  verbose=debug_level)
            search.fit(train_data, train_target)

            # Capture and fit the best estimator from across the grid search
            best_svm = search.best_estimator_
            predictions = best_svm.predict(test_data)

            y_data = confusion_matrix(test_target, predictions)

            y_data_float = y_data.astype(float)
            y_data_float *= 1 / y_data_float.max()

            class_report = classification_report(test_target, predictions, target_names=labels)

            accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

            results.append({
                'combination': each,
                'rbf': search.best_score_,
                'kernel': search.best_estimator_.kernel,
                'C': search.best_estimator_.C,
                'gamma': search.best_estimator_.gamma,
                'class_report': class_report,
                'y_data': y_data.tolist(),
                'y_data_normalized': y_data_float.tolist(),
                'labels': labels,
                'predictions': predictions.tolist(),
                'test_target': test_target.tolist(),
                'oa': accuracy[0],
                'aa': accuracy[1]
            })
        return results

    def svm_late_fusion(self, train_target, train_p1, train_p5, train_fc2, test_target, test_p1, test_p5,
                        test_fc2, labels, kfolds=5, debug_level=100):
        """
        Perform classification with all layers and then select the most voted result
        :param train_target: 
        :param train_p1: 
        :param train_p5: 
        :param train_fc2: 
        :param test_target: 
        :param test_p1: 
        :param test_p5: 
        :param test_fc2: 
        :param labels: 
        :param kfolds: 
        :return: 
        """

        print "[INFO]", "SVM Late fusion ..."

        parameter_candidates = [
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 10, 100], 'kernel': ['rbf']}
        ]

        # Fit the grid search
        search_p1 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1,
                                 verbose=debug_level)
        search_p1.fit(train_p1, train_target)
        predictions_p1 = search_p1.predict(test_p1)

        search_p5 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1,
                                 verbose=debug_level)
        search_p5.fit(train_p5, train_target)
        predictions_p5 = search_p5.predict(test_p5)

        search_fc2 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1,
                                  verbose=debug_level)
        search_fc2.fit(train_fc2, train_target)
        predictions_fc2 = search_fc2.predict(test_fc2)

        prediction_votes = zip(predictions_p1, predictions_p5, predictions_fc2)
        predictions = []
        for votes in prediction_votes:
            predictions.append(ClassificatorHelper.most_common_element_in_list(votes))

        y_data = confusion_matrix(test_target, predictions)

        y_data_float = y_data.astype(float)
        y_data_float *= 1 / y_data_float.max()

        class_report = classification_report(test_target, predictions, target_names=labels)

        accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

        return {
            'class_report': class_report,
            'y_data': y_data.tolist(),
            'y_data_normalized': y_data_float.tolist(),
            'labels': labels,
            'prediction_votes': prediction_votes,
            'predictions': predictions,
            'test_target': test_target.tolist(),
            'oa': accuracy[0],
            'aa': accuracy[1]
        }

    def random_forest_simple(self, train_data, train_target, test_data, test_target, labels, debug_level=100):

        print "[INFO]", "Random forest simple ..."

        clf = RandomForestClassifier(bootstrap=True, n_jobs=-1, random_state=0, verbose=debug_level)
        clf.fit(train_data, train_target)
        predictions = clf.predict(test_data)

        y_data = confusion_matrix(test_target, predictions)

        y_data_float = y_data.astype(float)
        y_data_float *= 1 / y_data_float.max()

        class_report = classification_report(test_target, predictions, target_names=labels)

        accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

        return {
            'class_report': class_report,
            'y_data': y_data.tolist(),
            'y_data_normalized': y_data_float.tolist(),
            'labels': labels,
            'predictions': predictions.tolist(),
            'test_target': test_target.tolist(),
            'oa': accuracy[0],
            'aa': accuracy[1]
        }

    def svm_linear_majority_voting(self, train_data, train_target, test_data, test_target, labels, sub_samples=5,
                                   debug_level=100):

        print "[INFO]", "SVM linear majority vote with subsampling ..."

        parameter_candidates = [
            {'C': [1, 10, 100], 'kernel': ['linear']}
        ]

        train_data, train_target = shuffle(train_data, train_target, random_state=0)
        train_data_split = np.array_split(train_data, sub_samples)
        train_target_split = np.array_split(train_target, sub_samples)

        estimators = []
        for i in xrange(sub_samples):
            search = GridSearchCV(estimator=svm.SVC(),
                                  param_grid=parameter_candidates,
                                  n_jobs=-1,
                                  verbose=debug_level)

            ltrain_data = train_data_split[i]
            ltarget = train_target_split[i]

            search.fit(ltrain_data, ltarget)
            estimators.append(search.best_estimator_)

        prediction_votes = zip(*[est.predict(test_data) for est in estimators])

        predictions = []
        for votes in prediction_votes:
            predictions.append(ClassificatorHelper.most_common_element_in_list(votes))

        y_data = confusion_matrix(test_target, predictions)

        y_data_float = y_data.astype(float)
        y_data_float *= 1 / y_data_float.max()

        class_report = classification_report(test_target, predictions, target_names=labels)

        accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

        return {
            'class_report': class_report,
            'y_data': y_data.tolist(),
            'y_data_normalized': y_data_float.tolist(),
            'labels': labels,
            'prediction_votes': prediction_votes,
            'predictions': list(predictions),
            'test_target': test_target.tolist(),
            'oa': accuracy[0],
            'aa': accuracy[1]
        }

    def svm_linear_bagging(self, train_data, train_target, test_data, test_target, labels, n_estimators=5,
                           debug_level=100):

        print "[INFO]", "SVM linear majority vote with subsampling ..."


        parameter_candidates = [
            {'C': [1, 10, 100], 'kernel': ['linear']}
        ]

        clf = OneVsRestClassifier(BaggingClassifier(GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates,
                                                                 n_jobs=-1,
                                                                 verbose=debug_level),
                                                    n_estimators=n_estimators))
        clf.fit(train_data, train_target)

        predictions = clf.predict(test_data)

        y_data = confusion_matrix(test_target, predictions)

        y_data_float = y_data.astype(float)
        y_data_float *= 1 / y_data_float.max()

        class_report = classification_report(test_target, predictions, target_names=labels)

        accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

        return {
            'class_report': class_report,
            'y_data': y_data.tolist(),
            'y_data_normalized': y_data_float.tolist(),
            'labels': labels,
            'predictions': predictions.tolist(),
            'test_target': test_target.tolist(),
            'oa': accuracy[0],
            'aa': accuracy[1]
        }

    def test_diversity_fc2(self, train_target, train_fc2, test_target, test_fc2, labels, kfolds=5, debug_level=0):
        res_svm_rbf = self.svm_simple(train_fc2, train_target, test_fc2, test_target, labels, kfolds=kfolds,
                                      debug_level=debug_level)
        #print "res_svm_rbf:", res_svm_rbf

        res_random_forest = self.random_forest_simple(train_fc2, train_target, test_fc2, test_target, labels,
                                                      debug_level=debug_level)
        #print "res_random_forest:", res_random_forest

        res_svn_linear = self.svm_linear_majority_voting(train_fc2, train_target, test_fc2, test_target, labels,
                                                         debug_level=debug_level)
        #print "res_svn_linear:", res_svn_linear

        res_svm_bagging = self.svm_linear_bagging(train_fc2, train_target, test_fc2, test_target, labels,
                                                  debug_level=debug_level)
        #print "res_svm_bagging:", res_svm_bagging

        return {
            'svm_rbf': res_svm_rbf,
            'random_forest': res_random_forest,
            'svm_linear': res_svn_linear,
            'svm_bagging': res_svm_bagging
        }

    @staticmethod
    def save_results(res, path_json, path_heatmap, labels):
        ClassificatorHelper.save_result_json(res, path_json, labels)
        ClassificatorHelper.plot_heatmap_from_result(res, path_heatmap, labels)

    @staticmethod
    def save_result_json(res, path, filename):
        with open(os.path.join(path, filename + ".json" ), 'w') as fp:
            json.dump(res, fp)

    @staticmethod
    def plot_heatmap_from_result(res, path, filename):
        return ClassificatorHelper.plot_heatmap(res["y_data_normalized"], res["labels"], path, filename)

    @staticmethod
    def plot_heatmap(y_data, y_labels, path, filename, show_plot=False):
        fig, ax = plt.subplots()
        # using the ax subplot object, we use the same
        # syntax as above, but it allows us a little
        # bit more advanced control
        # ax.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
        # ax.pcolor(data,cmap=plt.cm.Greens)
        # ax.pcolor(data,cmap=plt.cm.gnuplot)
        ax.set_xticks(np.arange(0, len(y_labels)))
        ax.set_yticks(np.arange(0, len(y_labels)))

        # cmap = plt.get_cmap('BlueRed2')
        plt.imshow(y_data, cmap=plt.cm.gnuplot, interpolation='nearest')
        # plt.imshow(data, cmap=plt.cm.gnuplot)
        # plt.clim(-0.05,0.25)
        plt.colorbar()

        # Here we put the x-axis tick labels
        # on the top of the plot.  The y-axis
        # command is redundant, but inocuous.
        ax.xaxis.tick_top()
        ax.yaxis.tick_left()
        # similar syntax as previous examples
        ax.set_xticklabels(y_labels, minor=False, fontsize=12, rotation=90)
        ax.set_yticklabels(y_labels, minor=False, fontsize=12)

        # Here we use a text command instead of the title
        # to avoid collision between the x-axis tick labels
        # and the normal title position
        # plt.text(0.5,1.08,'Main Plot Title',
        #         fontsize=25,
        #         horizontalalignment='center',
        #         transform=ax.transAxes
        #         )

        # standard axis elements
        # plt.ylabel('Y Axis Label',fontsize=10)
        # plt.xlabel('X Axis Label',fontsize=10)

        plt.savefig(os.path.join(path, filename+".pdf"), bbox_inches='tight')
        if show_plot:
            plt.show()

    @staticmethod
    def most_common_element_in_list(l):
        return max(groupby(sorted(l)), key=lambda (x, v): (len(list(v)), -l.index(x)))[0]

    @staticmethod
    def extract_accuracy_metrics(conf_matrix):
        """
        Overall Accuracy and Average Accuracy method extracted from:
        http://spatial-analyst.net/ILWIS/htm/ilwismen/confusion_matrix.htm
        :param conf_matrix: 
        :return: a tuple list with the overall and average accuracy, and their respective standard deviations
        """

        ACC = []
        correctly_classified = []
        for i in xrange(conf_matrix.shape[0]):
            tp = float(conf_matrix[i][i])
            ACC.append(tp / float(sum(conf_matrix[i])))
            correctly_classified.append(tp)

        return [(np.mean(ACC), np.std(ACC)),
                (sum(correctly_classified) / float(conf_matrix.sum()), np.std(correctly_classified))]

