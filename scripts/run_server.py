from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
from flask_pymongo import ObjectId

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'db'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/db'

mongo = PyMongo(app)

@app.route('/add_article', methods=['POST'])
def add_article():
  articles = mongo.db.nyt
  paper = request.json
  result = articles.insert_many(paper)
  return jsonify({'inserted' : 'true'})

@app.route('/add_tweet', methods=['POST'])
def add_tweet():
  tweets = mongo.db.tweets
  tweet = request.json
  result = tweets.insert_many(tweet)
  return jsonify({'inserted' : 'true'})

@app.route('/get_all_articles', methods=['GET'])
def get_all_articles():
    articles = mongo.db.nyt
    result = convert_object(articles.find().limit(100))
    return jsonify({'result' : result})

@app.route('/get_all_tweets', methods=['GET'])
def get_all_tweets():
    tweets = mongo.db.tweets
    result = convert_object(tweets.find().limit(100))
    return jsonify({'result' : result})

@app.route('/get_last_article', methods=['GET'])
def get_last_article():
    articles = mongo.db.nyt
    result = convert_object(articles.find().sort('pub_date', -1).limit(1))
    return jsonify({'result' : result})

@app.route('/get_historic_articles', methods=['GET'])
def get_historic_articles():
    year = request.args.get('year')
    num_articles = request.args.get('num_articles')
    if year is None:
        year = '1980'
    if num_articles is None:
        num_articles = 1000
    articles = mongo.db.nyt
    data = convert_object(articles.find({'pub_date': {'$regex' : year}}).limit(int(num_articles)))
    return jsonify({'articles' : data})

@app.route('/get_dataset_test', methods=['GET'])
def get_dataset_test():
    year = request.args.get('year')
    num_articles = request.args.get('num_articles')
    if year is None:
        year = '1980'
    if num_articles is None:
        num_articles = 1000
    articles = mongo.db.dataset
    data = convert_object(articles.find({'pub_date': {'$regex' : year}}).limit(int(num_articles)))
    return jsonify({'articles' : data})

@app.route('/get_dataset_train', methods=['GET'])
def get_dataset_train():
    year = request.args.get('year')
    num_yes = request.args.get('num_yes')
    num_no = request.args.get('num_no')
    if year is None:
        year = '2017'
    if num_yes is None:
        num_yes = 1000
    if num_no is None:
        num_no = 2000
    articles = mongo.db.dataset
    yes_data = convert_object(articles.find({'pub_date': {'$regex' : year}, 'was_tweeted' : 1}).limit(int(num_yes)))
    no_data = convert_object(articles.find({'pub_date': {'$regex' : year}, 'was_tweeted' : 0}).limit(int(num_no)))
    return jsonify({'yes' : yes_data}, {'no' : no_data})

@app.route('/get_dataset_counts', methods=['GET'])
def get_dataset_train_all():
    articles = mongo.db.dataset
    years = ['2015', '2016', '2017']
    ret = {}
    for year in years:
        yes_data = articles.find({'pub_date': {'$regex' : year}, 'was_tweeted' : 1}).count()
        no_data = articles.find({'pub_date': {'$regex' : year}, 'was_tweeted' : 0}).count()
        ret[year] = {'yes' : yes_data, 'no' : no_data}
    return jsonify(ret)

def convert_object(objects):
    result = []
    for obj in objects:
        obj['_id'] = str(obj['_id'])
        result.append(obj)
    return result

if __name__ == '__main__':
    app.run(debug=True, port = 80, host = '0.0.0.0')
