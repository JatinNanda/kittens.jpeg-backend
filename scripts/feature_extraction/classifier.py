import requests
import random
import json
import sklearn
from sklearn.datasets import make_regression
from archive_top_ngrams import get_ngrams_from_article_json
from sklearn import svm
from sklearn.metrics import mean_squared_error
from get_features import get_all_instances
from return_of_richards_dabbling import train_test_same_year
from return_of_richards_dabbling import train_test_two_years
from return_of_richards_dabbling import train_modern_test_historical
from turn_regression_into_classification import regression_to_classification

size_train = 10
size_test = 1000
train_year = '2015'
test_year = '2017'

# pull data from database
print "Getting data from db..."
dataset_params = {'year' : train_year, 'num_yes' : 10, 'num_no' : 20}
dataset = requests.get('http://ec2-54-167-62-52.compute-1.amazonaws.com/get_dataset', params = dataset_params).json()

# merge train and test data and randomly sample
yes_data = dataset[0]['yes']
no_data = dataset[1]['no']
train_dataset = yes_data + no_data
train_dataset = random.sample(train_dataset, size_train)

print "Extracting features from data..."
get_ngrams_from_article_json(train_dataset)
instances, labels = get_all_instances(train_dataset)

#test_year = "1900"
#test_raw_archive = "raw-archives/" + test_year + "_8.json"#"raw-archives/1861_8.json"
#test_data_csv = test_year + "-dataset.csv"
#test_data_popularity_output = test_year + "_popularity.csv"

# documents = db.joined.find({"created_at" : {"$regex" : year}})
# feature_input = [item for item in documents]

#regression_to_classification(year, num_classes)
print "same class accuracy"
#train_test_same_year(train_data_csv, 0.7, -1)

#with open(test_raw_archive, "r") as infile:
#    documents = json.load(infile)
#get_ngrams_from_article_json(documents['response']['docs'])
#instances, labels = get_all_instances(documents['response']['docs'])

#train_modern_test_historical(train_data_csv, test_data_csv, documents['response']['docs'], -1, test_data_popularity_output)

