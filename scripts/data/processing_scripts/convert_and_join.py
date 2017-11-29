import pymongo
import pprint
import requests
import sys
import time

pp = pprint.PrettyPrinter()
db = pymongo.MongoClient().db

def convert_urls_in_tweets(created_year, num_tweets):
    res = db.tweets.find({'created_at': {'$regex': created_year}}).limit(num_tweets)
    newly_converted = 0
    already_converted = 0
    for tweet in res:
        # make sure there is a URL in the tweet
        if not len(tweet['entities']['urls']) > 0:
            continue
        if 'article_url' in tweet.keys():
            already_converted += 1
            print "Found Fully Converted: " + str(already_converted) + " at " + str(tweet['_id'])
            continue
        # get the full form of the url
        url = tweet['entities']['urls'][0]['expanded_url']
        # determine if the url is already in the correct format
        if url[-4:] == 'html':
            already_converted += 1
            extended = url
            print "Found Partially Converted: " + str(already_converted) + " at " + str(tweet['_id'])
        else:
            try:
                extended = requests.get(url).url.split('?')[0]
            except:
                continue
            time.sleep(0.01)
            newly_converted += 1
            print "In total, converted " + str(newly_converted) + " tweets"

        # update the entry in the db with the correct expanded url in either case (this is not costly)
        db.tweets.update({
            '_id' : tweet['_id']
            }, {
                '$set': {
                    'article_url': extended
                    }
                }, upsert=False, multi=False)

def join_articles_with_converted_tweets(pub_year, num_articles):
    # loop through all the articles
    res = db.nyt.find({'pub_date': {'$regex': pub_year}}).limit(num_articles)
    i = 0
    for article in res:
        i+= 1
        # if the article has a link associated with it, we proceed
        if not article['web_url']:
            print "No URL in this article!"
            continue

        # split for malformed article urls just in case
        search_url = article['web_url'].split('?')[0]

        # find the tweet that matches this url
        tweet = db.tweets.find({'article_url': search_url})

        # join these two entities and post this to the 'dataset' collection

        # we have no matching tweet for this article so we add this as a '0' instance.
        if tweet.count() == 0:
            print "Added negative instance!"
            article['was_tweeted'] = 0
            db.dataset.insert_one(article)
        # we have a matching tweet so we add this as a '1' instance and join twitter data
        else:
            article['was_tweeted'] = 1
            joined_item = dict(article.items() + tweet.next().items())
            joined_item.pop("_id", None)
            res = db.dataset.insert_one(joined_item).inserted_id
            print ("Added positive instance!" +  str(res))
        print "Labelled article #" + str(i)

if __name__ == "__main__":
    # perform URL conversions and joins on the given years, with num_tweets converted per year and num_articles joined
    join_years = ['2015', '2016', '2017']
    num_tweets = 3000
    num_articles = 100000

    for year in join_years:
        #convert_urls_in_tweets(year, num_tweets)
        join_articles_with_converted_tweets(year, num_articles)

