import matplotlib.pyplot as plt
import requests
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import learning_curve
import math


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def plot_tweet_multimedia_scatter():
    data = np.genfromtxt('datasets/2017-dataset.csv', names=True, delimiter=",")

    first_ngram = data["3"]
    second_ngram = data["4"]
    multimedia = data["1"].tolist()
    label = data["196"]

    max_multimedia = max(multimedia)

    times_tweeted = [0 for i in range(int(max_multimedia + 1))]
    times_not_tweeted = [0 for i in range(int(max_multimedia + 1))]
    multimedia_range = np.arange(max_multimedia + 1)

    for i in xrange(len(multimedia)):
        num_multimedia = int(multimedia[i])
        was_tweeted = label[i]
        if (was_tweeted == 1):
            times_tweeted[num_multimedia] += 1.0
        else:
            times_not_tweeted[num_multimedia] += 1.0

    times_tweeted = np.array(times_tweeted)
    times_not_tweeted = np.array(times_not_tweeted)

    total_page_appearances = times_tweeted + times_not_tweeted

    # tweeted_ratio = np.nan_to_num(times_tweeted / total_page_appearances)
    tweeted_ratio = times_tweeted/total_page_appearances

    x = []
    y = []
    for xx, yy in zip(multimedia_range, tweeted_ratio):
        if not np.isnan(yy):
            x.append(xx)
            y.append(yy)
    plt.title("Effect of Multimedia on Article Tweetability")
    # print tweeted_ratio
    plt.scatter(x, y)
    plt.xlabel("Number of Multimedia in Article")
    plt.ylabel("Probability of Article being Tweeted")
    plt.show()

def plot_tweet_headlinelength_bars():
    index = 2
    data = np.genfromtxt('datasets/2017-dataset.csv', names=True, delimiter=",")
    step = 10.0
    # print_pages = data["0"].tolist()
    values = data[str(index)].tolist()
    label = data["196"]
    max_value = int(math.ceil(max(values) / step)) * int(step) #max(values)
    print max_value
    times_tweeted = np.arange(max_value + 1, step=step)#[0 for i in range(int(max_value / 50.0 + 1))]
    times_not_tweeted = np.arange(max_value + 1, step=step)#[0 for i in range(int(max_value / 50.0 + 1))]
    value_range = np.arange(max_value + 1, step=step)

    for i in xrange(len(times_tweeted)):
        times_tweeted[i] = 0
        times_not_tweeted[i] = 0

    for i in xrange(len(values)):
        # value = int(values[i])
        value = int(math.ceil(values[i] / step)) * int(step)
        index = int(value / step)
        was_tweeted = label[i]
        if (was_tweeted == 1):
            # print value
            # print len(times_tweeted)
            times_tweeted[index] += 1.0
        else:
            times_not_tweeted[index] += 1.0

    fig, ax = plt.subplots()
    ax.set_title("Headline Length on Article Tweetability")

    rects1 = ax.scatter(value_range, times_tweeted, c="blue")
    rects2 = ax.scatter(value_range, times_not_tweeted, c="orange")

    ax.legend((rects1, rects2), ('Tweeted Articles', 'Not-Tweeted Articles'))
    ax.set_xlabel("Characters in Headline")
    ax.set_ylabel("Number of Articles")

    plt.show()

def plot_tweet_pubdate_bars():
    index = 58
    data = np.genfromtxt('datasets/2017-dataset.csv', names=True, delimiter=",")

    values = data[str(index)].tolist()
    label = data["196"]
    max_value = max(values)
    times_tweeted = [0 for i in range(int(max_value + 1))]
    times_not_tweeted = [0 for i in range(int(max_value + 1))]
    value_range = np.arange(max_value + 1)

    for i in xrange(len(times_tweeted)):
        times_tweeted[i] = 0
        times_not_tweeted[i] = 0

    for i in xrange(len(values)):
        value = int(values[i])
        was_tweeted = label[i]
        if (was_tweeted == 1):
            times_tweeted[index] += 1.0
        else:
            times_not_tweeted[index] += 1.0

    fig, ax = plt.subplots()
    ax.set_title("Publication Day of Week on Article Tweetability")
    width = 0.35
    rects1 = ax.bar(value_range, times_tweeted, width)
    rects2 = ax.bar(value_range+width, times_not_tweeted, width)
    ax.set_xticks(value_range + width / 2)
    ax.set_xticklabels(("Sun", "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat"))
    ax.legend((rects1[0], rects2[0]), ('Tweeted Articles', 'Not-Tweeted Articles'))
    ax.set_xlabel("Article Word Count")
    ax.set_ylabel("Number of Articles")

    plt.show()


def plot_tweet_wordcount_bars():
    index = 194
    data = np.genfromtxt('datasets/2017-dataset.csv', names=True, delimiter=",")
    step = 1000.0
    values = data[str(index)].tolist()
    label = data["196"]
    max_value = int(math.ceil(max(values) / step)) * int(step)
    print max_value
    times_tweeted = np.arange(max_value + 1, step=step)
    times_not_tweeted = np.arange(max_value + 1, step=step)
    value_range = np.arange(max_value + 1, step=step)

    for i in xrange(len(times_tweeted)):
        times_tweeted[i] = 0
        times_not_tweeted[i] = 0

    for i in xrange(len(values)):
        value = int(math.ceil(values[i] / step)) * int(step)
        index = int(value / step)
        was_tweeted = label[i]
        if (was_tweeted == 1):
            times_tweeted[index] += 1.0
        else:
            times_not_tweeted[index] += 1.0

    fig, ax = plt.subplots()
    ax.set_title("Article Word Count on Article Tweetability")

    rects1 = ax.scatter(value_range, times_tweeted, c="blue")
    rects2 = ax.scatter(value_range, times_not_tweeted, c="orange")

    ax.legend((rects1, rects2), ('Tweeted Articles', 'Not-Tweeted Articles'))
    ax.set_xlabel("Article Word Count")
    ax.set_ylabel("Number of Articles")

    plt.show()

def plot_attribute_vs_class(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    """
    has_first_ngram_and_tweeted:  342
    has_first_ngram_and_not_tweeted:  116
    no_first_ngram_and_tweeted:  1157
    no_first_ngram_and_not_tweeted:  1384
    answer is 1
    has_2_ngram_and_tweeted:  84
    has_2_ngram_and_not_tweeted:  116
    no_2_ngram_and_tweeted:  1414
    no_2_ngram_and_not_tweeted:  1375
    """

    plt.tight_layout()
    plt.ylabel('Was Tweeted')
    plt.xlabel('Has N-Gram')


def plot_tweet_distributions():
    data = {"2015": { "no": 91910, "yes": 3386 }, "2016": { "no": 73846, "yes": 2313 }, "2017": { "no": 39521, "yes": 1799 } }

    fig, ax = plt.subplots()
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    bars = []
    year_list = ['2015', '2016', '2017']
    for index in xrange(3):
        year = year_list[index]
        num_tweeted = data[year]["yes"]
        num_not_tweeted = data[year]["no"]
        print "num_tweeted: ", num_tweeted
        print "num_not_tweeted: ", num_not_tweeted

        bar = plt.bar([index*5, 50+index*5], [num_tweeted, num_not_tweeted], 5)
        bars.append(bar[0])

    plt.xlabel("Was Tweeted")
    plt.ylabel("Number of Articles")
    plt.title("Distribution of Articles Tweeted")
    plt.xticks([5, 55], ('True', 'False'))

    plt.legend(tuple(bars), ["2015", "2016", "2017 (through Sept.)"])
    plt.show()

def plot_feature_correlations(clf, year_title):
    # correlations of all features
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance Score")
    plt.title(year_title + ": Feature Importance as Reported by Random Forest")
    plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_) #, width, color='r', yerr=men_std)
    print clf.feature_importances_

    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# cnf_matrix = confusion_matrix(testing_labels, predicted)
# four_class_title = 'Four Class Confusion Matrix'
# two_class_title = 'Two Class Confusion Matrix'
# four_classes = ['Not Popular', 'Less Popular', 'Popular', 'Most Popular']
# two_classes = ['Not Popular', 'Popular']
# plot_confusion_matrix(cnf_matrix, classes=four_classes, title=four_class_title, normalize=True)
# plt.show()


# if __name__ == '__main__':
#
#     plot_tweet_headlinelength_bars()

# if __name__ == '__main__':
#     plot_tweet_multimedia_scatter()

# if __name__ == '__main__':
#
#     data = np.genfromtxt('datasets/2017-dataset.csv', names=True, delimiter=",")
#
#     first_ngram = data["3"]
#     second_ngram = data["4"]
#     print_page = data["0"]
#     label = data["196"]
#
#     cnf_matrix = confusion_matrix(second_ngram, label)
#
#     """
#     has_first_ngram_and_tweeted:  342
#     has_first_ngram_and_not_tweeted:  116
#     no_first_ngram_and_tweeted:  1157
#     no_first_ngram_and_not_tweeted:  1384
#     answer is 1
#     has_2_ngram_and_tweeted:  84
#     has_2_ngram_and_not_tweeted:  116
#     no_2_ngram_and_tweeted:  1414
#     no_2_ngram_and_not_tweeted:  1375
#     """
#
#     cnf_matrix = np.array([np.array([cnf_matrix[0][0], cnf_matrix[1][0]]), \
#                             np.array([cnf_matrix[0][1], cnf_matrix[1][1]])])
#
#     two_class_title = 'Two Class Confusion Matrix'
#
#     classes = ["False", "True"]
#     plot_attribute_vs_class(cnf_matrix, classes, normalize=False, title='Effect of 2nd Most Popular N-Gram on Article Tweetability', cmap=plt.cm.Blues)
#     plt.show()
