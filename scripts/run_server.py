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

def convert_object(objects):
    result = []
    for obj in objects:
        obj['_id'] = str(obj['_id'])
        result.append(obj)
    return result

if __name__ == '__main__':
    app.run(debug=True, port = 80, host = '0.0.0.0')
