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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from itertools import groupby


class ClassificatorHelper(object):
    def __init__(self, cfg):
        self.cfg = cfg
    pass

    def svm_simple(self, train_data, train_target, labels, kfolds=5, debug_level=100):
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
            {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 10], 'kernel': ['rbf']}
        ]

        # Create the folds. This function returns indices to split data in train test sets.
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True)

        results = []
        fold = 1
        for train_i, test_i in skf.split(train_data, train_target):

            #print "TRAIN:", train_i, "TEST:", test_i

            # Using indices returned to separate the folds
            fold_train = [train_data[i] for i in train_i]
            fold_target = [train_target[i] for i in train_i]

            fold_train_test = [train_data[i] for i in test_i]
            fold_target_test = [train_target[i] for i in test_i]

            # perform a grid search with a kfolds cross validation
            search = GridSearchCV(estimator=svm.SVC(decision_function_shape='ovr'), param_grid=parameter_candidates, n_jobs=-1, verbose=debug_level)
            search.fit(fold_train, fold_target)

            # Capture and fit the best estimator from across the grid search
            best_svm = search.best_estimator_
            predictions = best_svm.predict(fold_train_test)

            accuracy_s = accuracy_score(fold_target_test, predictions)
            print 'Fold:%s Accuracy %.2f' % (fold, accuracy_s)

            y_data = confusion_matrix(fold_target_test, predictions)

            y_data_float = y_data.astype(float)
            y_data_float *= 1 / y_data_float.max()

            class_report = classification_report(fold_target_test, predictions, target_names=labels)

            accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)
            #print 'Fold:{} Accuracy conf matrix:{}'.format(fold, accuracy)
            #print 'y_data:\n{}'.format(y_data)
            # print 'fold_target_test:{}, pred:{}'.format(fold_target_test, predictions)

            results.append({
                'rbf': search.best_score_,
                'kernel': search.best_estimator_.kernel,
                'C': search.best_estimator_.C,
                'gamma': search.best_estimator_.gamma,
                'class_report': class_report,
                'y_data': y_data.tolist(),
                'y_data_normalized': y_data_float.tolist(),
                'labels': labels,
                'predictions': predictions.tolist(),
                'test_target': list(fold_target_test),
                'aa': accuracy[0],
                'oa': accuracy[1]
            })

            fold += 1

        res = results[-1]

        aa = [e['aa'] for e in results]
        oa = [e['oa'] for e in results]

        res["aa_mean_std"] = (np.mean(aa), np.std(aa))
        res["oa_mean_std"] = (np.mean(oa), np.std(oa))

        print 'aa:', aa
        print 'oa:', oa
        print 'aa_mean_std:', res["aa_mean_std"]
        print 'oa_mean_std:', res["oa_mean_std"]

        return res

    def svm_early_fusion(self, train_target, train_p1, train_p5, train_fc2, labels, kfolds=5, debug_level=100):
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

        pairwise_comb = []
        #pairwise_comb = list(itertools.permutations(fmaps_names, 2))
        pairwise_comb.append(fmaps_names)

        parameter_candidates = [
            {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 10], 'kernel': ['rbf']}
        ]

        # Create the folds. This function returns indices to split data in train test sets.
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True)

        for each in pairwise_comb:
            print(each)
            a_i = fmaps_names.index(each[0])
            b_i = fmaps_names.index(each[1])
            train_data = np.concatenate((train_fmaps[a_i], train_fmaps[b_i]), axis=1)
            #test_data = np.concatenate((test_fmaps[a_i], test_fmaps[b_i]), axis=1)

            if len(each) > 2:
                train_data = np.concatenate((train_data, train_fmaps[2]), axis=1)
                #test_data = np.concatenate((test_data, test_fmaps[2]), axis=1)

            local_results = []
            fold = 1
            for train_i, test_i in skf.split(train_data, train_target):
                # print "TRAIN:", train_i, "TEST:", test_i

                # Using indices returned to separate the folds
                fold_train = [train_data[i] for i in train_i]
                fold_target = [train_target[i] for i in train_i]

                fold_train_test = [train_data[i] for i in test_i]
                fold_target_test = [train_target[i] for i in test_i]

                # perform a grid search with a kfolds cross validation
                search = GridSearchCV(estimator=svm.SVC(decision_function_shape='ovr'), param_grid=parameter_candidates,
                                      n_jobs=-1, verbose=debug_level)
                search.fit(fold_train, fold_target)

                # Capture and fit the best estimator from across the grid search
                best_svm = search.best_estimator_
                predictions = best_svm.predict(fold_train_test)

                accuracy_s = accuracy_score(fold_target_test, predictions)
                print 'Fold:%s Accuracy %.2f' % (fold, accuracy_s)

                y_data = confusion_matrix(fold_target_test, predictions)

                y_data_float = y_data.astype(float)
                y_data_float *= 1 / y_data_float.max()

                class_report = classification_report(fold_target_test, predictions, target_names=labels)

                accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

                local_results.append({
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
                    'test_target': list(fold_target_test),
                    'aa': accuracy[0],
                    'oa': accuracy[1]
                })

                fold += 1

            res = local_results[-1]

            aa = [e['aa'] for e in local_results]
            oa = [e['oa'] for e in local_results]

            res["aa_mean_std"] = (np.mean(aa), np.std(aa))
            res["oa_mean_std"] = (np.mean(oa), np.std(oa))

            print 'aa:', aa
            print 'oa:', oa
            print 'aa_mean_std:', res["aa_mean_std"]
            print 'oa_mean_std:', res["oa_mean_std"]

            results.append(res)

        return results

    def svm_late_fusion(self, train_target, train_p1, train_p5, train_fc2, labels, kfolds=5, debug_level=100):
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
            {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 10], 'kernel': ['rbf']}
        ]

        # Create the folds. This function returns indices to split data in train test sets.
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True)

        results = []
        fold = 1
        for train_i, test_i in skf.split(train_p1, train_target):
            search_p1 = GridSearchCV(estimator=svm.SVC(decision_function_shape='ovr'), param_grid=parameter_candidates,
                                  n_jobs=-1, verbose=debug_level)
            search_p1.fit(train_p1[train_i], train_target[train_i])
            predictions_p1 = search_p1.best_estimator_.predict(train_p1[test_i])

            search_p5 = GridSearchCV(estimator=svm.SVC(decision_function_shape='ovr'), param_grid=parameter_candidates,
                                     n_jobs=-1, verbose=debug_level)
            search_p5.fit(train_p5[train_i], train_target[train_i])
            predictions_p5 = search_p5.best_estimator_.predict(train_p5[test_i])

            search_fc2 = GridSearchCV(estimator=svm.SVC(decision_function_shape='ovr'), param_grid=parameter_candidates,
                                     n_jobs=-1, verbose=debug_level)
            search_fc2.fit(train_fc2[train_i], train_target[train_i])
            predictions_fc2 = search_fc2.best_estimator_.predict(train_fc2[test_i])

            prediction_votes = zip(predictions_p1, predictions_p5, predictions_fc2)
            predictions = []
            for votes in prediction_votes:
                predictions.append(ClassificatorHelper.most_common_element_in_list(votes))

            y_data = confusion_matrix(train_target[test_i], predictions)

            y_data_float = y_data.astype(float)
            y_data_float *= 1 / y_data_float.max()

            class_report = classification_report(train_target[test_i], predictions, target_names=labels)

            accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

            results.append({
                'class_report': class_report,
                'y_data': y_data.tolist(),
                'y_data_normalized': y_data_float.tolist(),
                'labels': labels,
                'prediction_votes': prediction_votes,
                'predictions': predictions,
                'test_target': list(train_target[test_i]),
                'oa': accuracy[0],
                'aa': accuracy[1]
            })

        res = results[-1]

        aa = [e['aa'] for e in results]
        oa = [e['oa'] for e in results]

        res["aa_mean_std"] = (np.mean(aa), np.std(aa))
        res["oa_mean_std"] = (np.mean(oa), np.std(oa))

        print 'aa:', aa
        print 'oa:', oa
        print 'aa_mean_std:', res["aa_mean_std"]
        print 'oa_mean_std:', res["oa_mean_std"]

        return res

    def random_forest_simple(self, train_data, train_target, labels, kfolds=5, debug_level=100):
        """
        Train a simple Random Forest
        :param train_data: 
        :param train_target: 
        :param test_data: 
        :param test_target: 
        :param labels: 
        :param kfolds: 
        :return: 
        """

        print "[INFO]", "Random forest simple ..."

        # Create the folds. This function returns indices to split data in train test sets.
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True)

        results = []
        fold = 1
        for train_i, test_i in skf.split(train_data, train_target):

            #print "TRAIN:", train_i, "TEST:", test_i

            # Using indices returned to separate the folds
            fold_train = [train_data[i] for i in train_i]
            fold_target = [train_target[i] for i in train_i]

            fold_train_test = [train_data[i] for i in test_i]
            fold_target_test = [train_target[i] for i in test_i]

            clf = RandomForestClassifier(bootstrap=True, n_jobs=-1, random_state=0, verbose=debug_level)
            clf.fit(fold_train, fold_target)
            predictions = clf.predict(fold_train_test)

            accuracy_s = accuracy_score(fold_target_test, predictions)
            print 'Fold:%s Accuracy %.2f' % (fold, accuracy_s)

            y_data = confusion_matrix(fold_target_test, predictions)

            y_data_float = y_data.astype(float)
            y_data_float *= 1 / y_data_float.max()

            class_report = classification_report(fold_target_test, predictions, target_names=labels)

            accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)
            #print 'Fold:{} Accuracy conf matrix:{}'.format(fold, accuracy)
            #print 'y_data:\n{}'.format(y_data)
            # print 'fold_target_test:{}, pred:{}'.format(fold_target_test, predictions)

            results.append({
                'class_report': class_report,
                'y_data': y_data.tolist(),
                'y_data_normalized': y_data_float.tolist(),
                'labels': labels,
                'predictions': predictions.tolist(),
                'test_target': list(fold_target_test),
                'aa': accuracy[0],
                'oa': accuracy[1]
            })

            fold += 1

        res = results[-1]

        aa = [e['aa'] for e in results]
        oa = [e['oa'] for e in results]

        res["aa_mean_std"] = (np.mean(aa), np.std(aa))
        res["oa_mean_std"] = (np.mean(oa), np.std(oa))

        print 'aa:', aa
        print 'oa:', oa
        print 'aa_mean_std:', res["aa_mean_std"]
        print 'oa_mean_std:', res["oa_mean_std"]

        return res

    def svm_linear_majority_voting(self, train_data, train_target, labels, sub_samples=5, kfolds=5,
                                   debug_level=100):

        """
        Train a SVM with a linear kernel performing majority voting on several sub samples of the original data
        :param train_data: 
        :param train_target: 
        :param labels: 
        :param sub_samples: 
        :param kfolds: 
        :param debug_level: 
        :return: 
        """

        print "[INFO]", "SVM linear majority vote with subsampling ..."

        parameter_candidates = [
            {'C': [1, 10, 100], 'kernel': ['linear']}
        ]

        # Create the folds. This function returns indices to split data in train test sets.
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True)

        results = []
        fold = 1
        for train_i, test_i in skf.split(train_data, train_target):
            # print "TRAIN:", train_i, "TEST:", test_i

            # Using indices returned to separate the folds
            fold_train = [train_data[i] for i in train_i]
            fold_target = [train_target[i] for i in train_i]

            fold_train_test = [train_data[i] for i in test_i]
            fold_target_test = [train_target[i] for i in test_i]

            fold_train, fold_target = shuffle(fold_train, fold_target)
            train_data_split = np.array_split(fold_train, sub_samples)
            train_target_split = np.array_split(fold_target, sub_samples)

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

            prediction_votes = zip(*[est.predict(fold_train_test) for est in estimators])

            predictions = []
            for votes in prediction_votes:
                predictions.append(ClassificatorHelper.most_common_element_in_list(votes))

            accuracy_s = accuracy_score(fold_target_test, predictions)
            print 'Fold:%s Accuracy %.2f' % (fold, accuracy_s)

            y_data = confusion_matrix(fold_target_test, predictions)

            y_data_float = y_data.astype(float)
            y_data_float *= 1 / y_data_float.max()

            class_report = classification_report(fold_target_test, predictions, target_names=labels)

            accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

            results.append({
                'class_report': class_report,
                'y_data': y_data.tolist(),
                'y_data_normalized': y_data_float.tolist(),
                'labels': labels,
                'prediction_votes': prediction_votes,
                'predictions': list(predictions),
                'test_target': list(fold_target_test),
                'oa': accuracy[0],
                'aa': accuracy[1]
            })

            fold +=1

        res = results[-1]

        aa = [e['aa'] for e in results]
        oa = [e['oa'] for e in results]

        res["aa_mean_std"] = (np.mean(aa), np.std(aa))
        res["oa_mean_std"] = (np.mean(oa), np.std(oa))

        print 'aa:', aa
        print 'oa:', oa
        print 'aa_mean_std:', res["aa_mean_std"]
        print 'oa_mean_std:', res["oa_mean_std"]

        return res

    def svm_linear_bagging(self, train_data, train_target, labels, n_estimators=5, kfolds=5,
                           debug_level=100):
        """
        Train a SVM with a linear kernel with Bagging
        :param train_data: 
        :param train_target: 
        :param labels: 
        :param n_estimators: 
        :param kfolds: 
        :param debug_level: 
        :return: 
        """

        print "[INFO]", "SVM linear majority vote with subsampling ..."


        parameter_candidates = [
            {'C': [1, 10, 100], 'kernel': ['linear']}
        ]

        # Create the folds. This function returns indices to split data in train test sets.
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True)

        results = []
        fold = 1
        for train_i, test_i in skf.split(train_data, train_target):
            # print "TRAIN:", train_i, "TEST:", test_i

            # Using indices returned to separate the folds
            fold_train = [train_data[i] for i in train_i]
            fold_target = [train_target[i] for i in train_i]

            fold_train_test = [train_data[i] for i in test_i]
            fold_target_test = [train_target[i] for i in test_i]

            clf = OneVsRestClassifier(BaggingClassifier(GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates,
                                                                     n_jobs=-1,
                                                                     verbose=debug_level),
                                                        n_estimators=n_estimators))
            clf.fit(train_data, train_target)

            predictions = clf.predict(fold_train_test)

            y_data = confusion_matrix(fold_target_test, predictions)

            y_data_float = y_data.astype(float)
            y_data_float *= 1 / y_data_float.max()

            class_report = classification_report(fold_target_test, predictions, target_names=labels)

            accuracy = ClassificatorHelper.extract_accuracy_metrics(y_data)

            results.append({
                'class_report': class_report,
                'y_data': y_data.tolist(),
                'y_data_normalized': y_data_float.tolist(),
                'labels': labels,
                'predictions': predictions.tolist(),
                'test_target': list(fold_target_test),
                'oa': accuracy[0],
                'aa': accuracy[1]
            })

            fold +=1

        res = results[-1]

        aa = [e['aa'] for e in results]
        oa = [e['oa'] for e in results]

        res["aa_mean_std"] = (np.mean(aa), np.std(aa))
        res["oa_mean_std"] = (np.mean(oa), np.std(oa))

        print 'aa:', aa
        print 'oa:', oa
        print 'aa_mean_std:', res["aa_mean_std"]
        print 'oa_mean_std:', res["oa_mean_std"]

        return res

    def test_diversity_fc2(self, train_target, train_fc2, labels, kfolds=5, debug_level=0):
        """
        Tests for diversity on the FC2 layer
        :param train_target: 
        :param train_fc2: 
        :param labels: 
        :param kfolds: 
        :param debug_level: 
        :return: 
        """
        res_svm_rbf = self.svm_simple(train_fc2, train_target, labels, kfolds=kfolds,
                                      debug_level=debug_level)

        res_random_forest = self.random_forest_simple(train_fc2, train_target, labels,
                                                      debug_level=debug_level)

        res_svn_linear = self.svm_linear_majority_voting(train_fc2, train_target, labels,
                                                         debug_level=debug_level)

        res_svm_bagging = self.svm_linear_bagging(train_fc2, train_target, labels,
                                                  debug_level=debug_level)

        return {
            'svm_rbf': res_svm_rbf,
            'random_forest': res_random_forest,
            'svm_linear': res_svn_linear,
            'svm_bagging': res_svm_bagging
        }

    @staticmethod
    def save_results(res, path_json, path_heatmap, labels):
        """
        Save the results (plot and json file) on their respective folders
        :param res: 
        :param path_json: 
        :param path_heatmap: 
        :param labels: 
        :return: 
        """
        ClassificatorHelper.save_result_json(res, path_json, labels)
        ClassificatorHelper.plot_heatmap_from_result(res, path_heatmap, labels)

    @staticmethod
    def save_result_json(res, path, filename):
        """
        Serialize a DICT in json format and save it to file
        :param res: 
        :param path: 
        :param filename: 
        :return: 
        """
        with open(os.path.join(path, filename + ".json" ), 'w') as fp:
            json.dump(res, fp)

    @staticmethod
    def plot_heatmap_from_result(res, path, filename):
        """
        Generate a heatmap and save it to file
        :param res: 
        :param path: 
        :param filename: 
        :return: 
        """
        return ClassificatorHelper.plot_heatmap(res["y_data_normalized"], res["labels"], path, filename)

    @staticmethod
    def plot_heatmap(y_data, y_labels, path, filename, show_plot=False):
        """
        Generate a heatmap from a confusion matrix
        :param y_data: 
        :param y_labels: 
        :param path: 
        :param filename: 
        :param show_plot: 
        :return: 
        """
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
        """
        Get the most vote element in a list
        :param l: 
        :return: 
        """
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

        # AA, OA
        return [np.mean(ACC), (sum(correctly_classified) / float(conf_matrix.sum()))]

