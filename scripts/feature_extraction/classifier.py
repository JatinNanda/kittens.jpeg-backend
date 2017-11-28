import requests
import sys
import random
import json
import sklearn
from sklearn.datasets import make_regression
from archive_top_ngrams import get_ngrams_from_article_json
from sklearn import svm
from sklearn.metrics import mean_squared_error
from get_features import get_all_instances
from util import train_test_same_year
from util import train_test_two_years
from util import train_modern_test_historical

'''
THERE'S 4 WAYS TO RUN THE CLASSIFIER:
1) python classifier.py (year)   # This will train/test on the same year
2) python classifier.py (train_year) (test_year)   # This will train/test on different years
3) python classifier.py historic (test_year)   # This will train on the 'years_in_db' array and test on the historic year, outputting a csv
4) python classifier.py historic all # This will train on the 'years_in_db' array and test on the historic years range 1851-1899, outputting multiple csvs
'''

rest_endpoint = 'http://ec2-54-167-62-52.compute-1.amazonaws.com/get_dataset'
rest_endpoint_historical = 'http://ec2-54-167-62-52.compute-1.amazonaws.com/get_historic_articles'

# specify how large the test/training data will be
dataset_size = 1000
percent_yes = 0.25
percent_no = 0.75

years_in_db = ['2017', '2016', '2015']

def classify_same_year(year):
    instances, labels = grab_instances_for_year(year)

    print "Classifying..."
    train_test_same_year(instances, labels, 0.7, -1)

def classify_different_years(train_year, test_year):
    dataset_train = grab_instances_for_year(train_year)
    dataset_test = grab_instances_for_year(test_year)

    print "Classifying..."
    train_test_two_years(dataset_train, dataset_test, -1)

def classify_historic_data(output_years):
    all_train_data = []
    for year in years_in_db:
        all_train_data.append(grab_instances_for_year(year))

    for year in output_years:
        test_instances, articles = grab_articles_from_history(str(year))
        print len(test_instances)
        train_modern_test_historical(all_train_data, test_instances, articles, -1, str(year) + '-headlines.csv')



# Highest level method to query database and generate dataset for a year (used for fetching data post twitter creation)
def grab_instances_for_year(year):
    print "Getting data from db for " +  year + "..."
    dataset_params = {'year' : year, 'num_yes' : int(dataset_size * percent_yes), 'num_no' : int(dataset_size * percent_no)}
    dataset = requests.get(rest_endpoint, params = dataset_params).json()

    yes_data = dataset[0]['yes']
    no_data = dataset[1]['no']
    train_dataset = yes_data + no_data
    random.shuffle(train_dataset)

    print "Extracting features from data..."
    get_ngrams_from_article_json(train_dataset)
    instances, labels = get_all_instances(train_dataset)
    return instances, labels

# used for getting historic articles for testing the classifier
def grab_articles_from_history(year):
    print "*** TESTING ON HISTORY ***"
    print "Getting data from db for " +  year + "..."
    dataset_params = {'year' : year, 'num_articles' : dataset_size}
    dataset = requests.get(rest_endpoint_historical, params = dataset_params).json()

    articles = dataset['articles']

    print "Extracting features from data..."
    get_ngrams_from_article_json(articles)
    instances, _ = get_all_instances(articles)
    return instances, articles

if __name__ == '__main__':
    train_year = None
    test_year = None
    if len(sys.argv) == 1:
        print "Please supply a train year and/or test year!"
    elif len(sys.argv) == 2:
        train_year = sys.argv[1]
        classify_same_year(train_year)
    elif len(sys.argv) > 2:
        train_year = sys.argv[1]
        test_year = sys.argv[2]
        # signal to do historic classification on next param year
        if train_year == 'historic':
            if test_year == 'all':
                classify_historic_data(range(1851, 1899))
            else:
                classify_historic_data([test_year])
        else:
            classify_different_years(train_year, test_year)



