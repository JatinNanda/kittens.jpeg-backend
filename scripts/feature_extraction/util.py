import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
import sys
from sklearn.metrics import confusion_matrix
import itertools
import json
import csv
from read_archives import get_stops

def generate_correlations(instances, labels):
    correlations = [np.correlate(instances[:,i], labels)[0] for i in xrange(len(instances[0, :]))]
    sorted_correlations = [i[0] for i in sorted(enumerate(correlations), reverse=True, key=lambda x:x[1])]
    return sorted_correlations

def generate_train_test(instances, labels, split, selected_indices):
    training_features = []
    training_labels = []
    testing_features = []
    testing_labels = []
    TRAIN_TEST_SPLIT = split #0.7

    for i, row in enumerate(np.array(instances)):
        selected_features = [row[j] for j in selected_indices]
        # for the purpose of unlabelled testing
        if labels is not None:
            label = labels[i]
        else:
            label = None
        if random.uniform(0, 1) < TRAIN_TEST_SPLIT:
            training_features.append(selected_features)
            training_labels.append(label)
        else:
            testing_features.append(selected_features)
            testing_labels.append(label)

    return training_features, training_labels, testing_features, testing_labels

def train_test_same_year(instances, labels, split, num_top_corr):

    sorted_correlations = generate_correlations(instances, labels)
    selected_indices = sorted_correlations[:num_top_corr]
    training_features, training_labels, testing_features, testing_labels = generate_train_test(instances, labels, split, selected_indices)

    clf = RandomForestClassifier(n_estimators=10, max_depth=100)
    clf.fit(training_features, training_labels)

    # plt.plot(clf.feature_importances_)
    # plt.show()
    # plt.plot(correlations)
    # plt.show()
    # print training_features
    # print clf.score(testing_features, testing_labels)
    # plt.scatter(testing_labels, predicted)
    # plt.show()
    #
    # predicted = clf.predict(testing_features)

    print "ACCURACY: ", clf.score(testing_features, testing_labels)

# dataset_train and dataset_test should be tuples of (instances, labels)
def train_test_two_years(dataset_train, dataset_test, num_top_corr):
    sorted_correlations = generate_correlations(dataset_train[0], dataset_train[1])
    selected_indices = sorted_correlations[:num_top_corr]
    training_features, training_labels, _, _ = generate_train_test(dataset_train[0], dataset_train[1], 1, selected_indices)

    clf = RandomForestClassifier(n_estimators=10, max_depth=100)
    clf.fit(training_features, training_labels)

    _, _, testing_features, testing_labels = generate_train_test(dataset_test[0], dataset_test[1], 0, selected_indices)

    print "ACCURACY: ", clf.score(testing_features, testing_labels)

# trains and returns a model using (instances,labels) tuples throughout several years
def train_all_modern(all_train_datasets, num_top_corr):
    # aggregate training data and labels
    all_training_features = np.array(sum([list(train_set[0]) for train_set in all_train_datasets], []))
    all_training_labels = np.array(sum([list(train_set[1]) for train_set in all_train_datasets], []))

    sorted_correlations = generate_correlations(all_training_features, all_training_labels)
    selected_indices = sorted_correlations[:num_top_corr]
    all_training_features, all_training_labels, _, _ = generate_train_test(all_training_features, all_training_labels, 1, selected_indices)

    clf = RandomForestClassifier(n_estimators=10, max_depth=100)
    clf.fit(all_training_features, all_training_labels)
    return clf, selected_indices

# the fruit of our labor - this method trains the final classifier and deals with historic testing data
def train_modern_test_historical(all_train_datasets, test_instances, test_articles, num_top_corr, output_name_csv):

    trained_classifier, selected_indices = train_all_modern(all_train_datasets, num_top_corr)

    testing_features, _, _, _ = generate_train_test(test_instances, None, 1, selected_indices)

    stop_phrases = get_stops(test_articles[0]['pub_date'].split('-')[0])

    predicted = trained_classifier.predict(testing_features)

    with open("outputs/" + output_name_csv, "w") as indices:
        writer = csv.writer(indices)
        writer.writerow(["pub_date", "main_headline", "article_url", "classification"])
        for i in xrange(len(testing_features)):
            classification = predicted[i]
            main_headline = test_articles[i]["headline"]["main"].encode('utf-8')
            pub_date = test_articles[i]["pub_date"]
            article_url = test_articles[i]["web_url"]
            # manually convert classification output for headlines with stop phrases just in case
            if not any(phrase in main_headline for phrase in stop_phrases):
                writer.writerow([pub_date, main_headline, article_url, classification])
            else:
                writer.writerow([pub_date, main_headline, article_url, 0])
    print "SAVED OUTPUT TO: ", "outputs/" + output_name_csv
