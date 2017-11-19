import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
import sys
from sklearn.metrics import confusion_matrix
import itertools
import json
import csv
import matplotlib.pyplot as plt
from read_archives import get_stops

def generate_correlations(data):
    labels = [row[-1] for row in data]
    correlations = [np.correlate(data[str(i)], labels)[0] for i in xrange(len(data[0])-1)]
    sorted_correlations = [i[0] for i in sorted(enumerate(correlations), reverse=True, key=lambda x:x[1])]
    return sorted_correlations

def generate_train_test(data, split, selected_indices):
    training_features = []
    training_labels = []
    testing_features = []
    testing_labels = []
    TRAIN_TEST_SPLIT = split #0.7

    for i, row in enumerate(data):
        selected_features = [row[j] for j in selected_indices]
        label = row[-1]
        # print label
        if random.uniform(0, 1) < TRAIN_TEST_SPLIT:
            training_features.append(selected_features)
            training_labels.append(label)
        else:
            testing_features.append(selected_features)
            testing_labels.append(label)

    return training_features, training_labels, testing_features, testing_labels

def train_test_same_year(dataset_csv, split, num_top_corr):
    data = np.genfromtxt(dataset_csv, delimiter=',', names=True)

    sorted_correlations = generate_correlations(data)
    selected_indices = sorted_correlations[:num_top_corr]#[:50]#[:4]
    training_features, training_labels, testing_features, testing_labels = generate_train_test(data, split, selected_indices)

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

def train_test_two_years(train_data_csv, test_data_csv, num_top_corr):
    data = np.genfromtxt(train_data_csv, delimiter=',', names=True)

    sorted_correlations = generate_correlations(data)
    selected_indices = sorted_correlations[:num_top_corr]#[:50]#[:4]
    training_features, training_labels, _, _ = generate_train_test(data, 1, selected_indices)

    clf = RandomForestClassifier(n_estimators=10, max_depth=100)
    clf.fit(training_features, training_labels)

    data = np.genfromtxt(test_data_csv, delimiter=',', names=True)
    _, _, testing_features, testing_labels = generate_train_test(data, 0, selected_indices)

    print "ACCURACY: ", clf.score(testing_features, testing_labels)

def train_modern_test_historical(train_data_csv, test_data_csv, test_articles, num_top_corr, output_name_csv):
    data = np.genfromtxt(train_data_csv, delimiter=',', names=True)

    sorted_correlations = generate_correlations(data)
    selected_indices = sorted_correlations[:num_top_corr]#[:50]#[:4]
    training_features, training_labels, _, _ = generate_train_test(data, 1, selected_indices)

    clf = RandomForestClassifier(n_estimators=10, max_depth=100)
    clf.fit(training_features, training_labels)

    testing_features = np.genfromtxt(test_data_csv, delimiter=',', names=True)
    # print len(testing_features), len(testing_features[0])

    stop_phrases = get_stops(test_data_csv.split("-")[0])

    with open(output_name_csv, "w") as indices:
        writer = csv.writer(indices)
        writer.writerow(["main_headline", "classification"])
        for i, row in enumerate(testing_features):
            selected_features = [row[j] for j in selected_indices]
            predicted = clf.predict(np.array([np.array(list(selected_features))]))#[:-1])]))
            classification = predicted[0]
            main_headline = test_articles[i]["headline"]["main"].encode('utf-8')
            if not any(phrase in main_headline for phrase in stop_phrases):
                writer.writerow([main_headline, classification])
            else:
                writer.writerow([main_headline, classification - 1])
    print "SAVED OUTPUT TO: ", output_name_csv





# # testing_features = np.genfromtxt('1861-dataset.csv', delimiter=',', names=True)
# # testing_features = np.genfromtxt('2015-dataset.csv', delimiter=',', names=True)
# # for i, row in enumerate(testing_features):
# #     predicted = clf.predict(np.array([np.array(list(row)[:-1])]))
# #     if predicted[0] == 4:
# #         print i
#
#
# data = np.genfromtxt('magic_dataset_2016.csv', delimiter=',', names=True)
#
# # correlations = [np.correlate(data[str(i)], data['196'])[0] for i in xrange(197)]
# #
# # sorted_correlations = [i[0] for i in sorted(enumerate(correlations), reverse=True, key=lambda x:x[1])]
#
# # training_features = []
# # training_labels = []
# testing_features = []
# testing_labels = []
# # TRAIN_TEST_SPLIT = 0#1 #0.7
#
# selected_indices = sorted_correlations[:50]#[:4]
#
# for i, row in enumerate(data):
#     selected_features = [row[j] for j in selected_indices]
#     label = row[-1]
#     # if random.uniform(0, 1) < TRAIN_TEST_SPLIT:
#     #     training_features.append(selected_features)
#     #     training_labels.append(label)
#     # else:
#     testing_features.append(selected_features)
#     testing_labels.append(label)



# sys.exit(0)
# predicted = clf.predict(testing_features)
#
# # plt.plot(clf.feature_importances_)
# # plt.show()
# # print training_features
# print clf.score(testing_features, testing_labels)
# plt.scatter(testing_labels, predicted)
# plt.show()
