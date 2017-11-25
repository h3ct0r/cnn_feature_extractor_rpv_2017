import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from itertools import groupby


class SvnClassificator(object):
    def __init__(self, cfg):
        self.cfg = cfg
    pass

    def svm_simple(self, train_data, train_target, test_data, test_target, labels, kfolds=5):
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
        search = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1, verbose=100)
        search.fit(train_data, train_target)

        # Capture and fit the best estimator from across the grid search
        best_svm = search.best_estimator_
        predictions = best_svm.predict(test_data)

        y_data = confusion_matrix(test_target, predictions)

        y_data_float = y_data.astype(float)
        y_data_float *= 1/y_data_float.max()

        class_report = classification_report(test_target, predictions, target_names=labels)

        return {
            'rbf': search.best_score_,
            'kernel': search.best_estimator_.kernel,
            'C': search.best_estimator_.C,
            'gamma': search.best_estimator_.gamma,
            'class_report': class_report,
            'y_data': y_data,
            'y_data_normalized': y_data_float,
            'labels': labels,
            'predictions': predictions,
            'test_target': test_target
        }

    def svm_early_fusion(self, train_target, train_p1, train_p5, train_fc2, test_target, test_p1, test_p5, test_fc2,
                         labels, kfolds=5):
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

        ##{'C': [1, 10, 100], 'kernel': ['linear']},

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
            search = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1)
            search.fit(train_data, train_target)

            # Capture and fit the best estimator from across the grid search
            best_svm = search.best_estimator_
            predictions = best_svm.predict(test_data)

            y_data = confusion_matrix(test_target, predictions)

            y_data_float = y_data.astype(float)
            y_data_float *= 1 / y_data_float.max()

            class_report = classification_report(test_target, predictions, target_names=labels)

            results.append({
                'combination': each,
                'rbf': search.best_score_,
                'kernel': search.best_estimator_.kernel,
                'C': search.best_estimator_.C,
                'gamma': search.best_estimator_.gamma,
                'class_report': class_report,
                'y_data': y_data,
                'y_data_normalized': y_data_float,
                'labels': labels,
                'predictions': predictions,
                'test_target': test_target
            })
        return results

    def svm_late_fusion(self, train_target, train_p1, train_p5, train_fc2, test_target, test_p1, test_p5,
                        test_fc2, labels, kfolds=5):
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

        results = []

        parameter_candidates = [
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 10, 100], 'kernel': ['rbf']}
        ]

        # Fit the grid search
        search_p1 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1)
        search_p1.fit(train_p1, train_target)
        predictions_p1 = search_p1.predict(test_p1)

        search_p5 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1)
        search_p5.fit(train_p5, train_target)
        predictions_p5 = search_p5.predict(test_p5)

        search_fc2 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=kfolds, n_jobs=-1)
        search_fc2.fit(train_fc2, train_target)
        predictions_fc2 = search_fc2.predict(test_fc2)

        prediction_votes = zip(predictions_p1, predictions_p5, predictions_fc2)
        predictions = []
        for votes in prediction_votes:
            predictions.append(SvnClassificator.most_common_element_in_list(votes))

        y_data = confusion_matrix(test_target, predictions)

        y_data_float = y_data.astype(float)
        y_data_float *= 1 / y_data_float.max()

        class_report = classification_report(test_target, predictions, target_names=labels)

        results.append({
            'class_report': class_report,
            'y_data': y_data,
            'y_data_normalized': y_data_float,
            'labels': labels,
            'prediction_votes': prediction_votes,
            'predictions': predictions,
            'test_target': test_target
        })
        return results

    @staticmethod
    def plot_heatmap(y_labels, y_data):
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

        #plt.savefig(filename + '.pdf', bbox_inches='tight')
        plt.show()

    @staticmethod
    def most_common_element_in_list(l):
        return max(groupby(sorted(l)), key=lambda (x, v): (len(list(v)), -l.index(x)))[0]
