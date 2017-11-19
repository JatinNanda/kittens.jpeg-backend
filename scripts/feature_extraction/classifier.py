import pymongo
from get_features import get_all_instances
import sklearn
# from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from archive_top_ngrams import get_ngrams_from_article_json
from sklearn import svm
from sklearn.metrics import mean_squared_error
import json
from return_of_richards_dabbling import train_test_same_year
from return_of_richards_dabbling import train_test_two_years
from return_of_richards_dabbling import train_modern_test_historical
from turn_regression_into_classification import regression_to_classification


#to query it, db.tweets.find(),
#db.joined.find()
#documents = db.joined.find({"created_at" : {"$regex" : '2015'}})

year = '2015'
num_classes = 10
test_year = "1900"
train_data_csv = 'magic_dataset_' + year + '.csv'
test_raw_archive = "raw-archives/" + test_year + "_8.json"#"raw-archives/1861_8.json"
test_data_csv = test_year + "-dataset.csv"
test_data_popularity_output = test_year + "_popularity.csv"

# db = pymongo.MongoClient().db # this is the database
# documents = db.joined.find({"created_at" : {"$regex" : year}})
# feature_input = [item for item in documents]
# get_ngrams_from_article_json(feature_input)
# instances, labels = get_all_instances(feature_input)

regression_to_classification(year, num_classes)
print "same class accuracy"
train_test_same_year(train_data_csv, 0.7, -1)

with open(test_raw_archive, "r") as infile:
    documents = json.load(infile)
get_ngrams_from_article_json(documents['response']['docs'])
instances, labels = get_all_instances(documents['response']['docs'])

train_modern_test_historical(train_data_csv, test_data_csv, documents['response']['docs'], -1, test_data_popularity_output)

""""
#data goes back to 1980
#1000 tweets each year from 2017, 2016, 2015, 2014 all joined
"""
