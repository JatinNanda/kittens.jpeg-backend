# encoding: utf-8

import tweepy
import requests
import json
import os

import logging

base_url = "http://ec2-52-14-183-157.us-east-2.compute.amazonaws.com/"

def loadKeys(key_file):
    # Load keys here and replace the empty strings in the return statement with those keys
    with open(key_file) as json_data:
        data = json.load(json_data)
    return data["api_key"],data["api_secret"],data["token"],data["token_secret"]

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method

	KEY_FILE = 'api_keys/twitter_keys.json'
	log_folder = "logs-twitter/"

	if not os.path.exists(log_folder):
		os.makedirs(log_folder)

	start_month = 4
	start_year = 2007
	end_month = 9
	end_year = 2017

        log_file = str(start_month) + str(start_year) + "-" + str(end_month) + str(end_year) + ".log"

	api_key, api_secret, token, token_secret = loadKeys(KEY_FILE)

	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(api_key, api_secret)
	auth.set_access_token(token, token_secret)
	api = tweepy.API(auth)

        logging.basicConfig(filename=(log_folder + log_file), level=logging.INFO)
        logging.info("Getting started.")

	#initialize a list to hold all the tweepy Tweets
	alltweets = []
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	#save most recent tweets
	alltweets.extend(new_tweets)
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	overall_count = len(new_tweets)
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		logging.info("getting tweets before %s" % (oldest))
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		#save most recent tweets
		alltweets.extend(new_tweets)
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		overall_count += len(new_tweets)
		if overall_count >= 600 or len(new_tweets) < 200:
		        payload = json.loads(json.dumps([tweet._json for tweet in alltweets]))
			response = requests.post(base_url + "add_tweet", json = payload)
			alltweets = []
			if len(new_tweets) < 200:
				break
        logging.info("Done grabbing tweets.")

if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("nytimes")
