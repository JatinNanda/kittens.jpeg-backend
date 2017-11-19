import urllib2
import requests
import json
import datetime
import time
import sys, os
import logging
from urllib2 import HTTPError
from ConfigParser import SafeConfigParser

base_url = "ec2-54-237-153-115.compute-1.amazonaws.com"

def loadKeys(key_file):
    # Load keys here and replace the empty strings in the return statement with those keys
    with open(key_file) as json_data:
        data = json.load(json_data)
    return data["api_key"]

# helper function to iterate through dates
def daterange( start_month, start_year, end_month, end_year ):
    daterange = []
    for year in range(start_year, end_year + 1):
        month_range_start = start_month if (start_year == year) else 1
        month_range_end = end_month if (end_year == year) else 12
        for month in range(month_range_start, month_range_end + 1):
            daterange.append((month, year))
    return daterange

# helper function to get json into a form I can work with
def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

# get the articles from the NYTimes Article API
def getArticles(month, year, api_key):
    # LOOP THROUGH THE 101 PAGES NYTIMES ALLOWS FOR THAT DATE

    for n in range(5): # 5 tries
        try:
            request_string = "https://api.nytimes.com/svc/archive/v1/" + str(year) + "/" + str(month) + ".json?&api-key=" + api_key
            response = urllib2.urlopen(request_string)
            content = response.read()
            if content:
                articles = convert(json.loads(content))
                # if there are articles here
                if len(articles["response"]["docs"]) >= 1:
                    return articles["response"]["docs"]
                else:
                    return
            time.sleep(3) # wait so we don't overwhelm the API
            break
        except HTTPError as e:
            print e
            logging.error("HTTPError on month: %s year: %s (err no. %s: %s) Here's the URL of the call: %s", month, year, e.code, e.reason, request_string)
            if e.code == 403:
                print "Script hit a snag and got an HTTPError 403. Check your log file for more info."
                return
            if e.code == 429:
                print "Waiting. You've probably reached an API limit."
                time.sleep(30) # wait 30 seconds and try again
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print (exc_type, fname, exc_tb.tb_lineno)
            logging.error("Error on month: %s year: %s", month, year, file_number, sys.exc_info()[0])
            continue

def main():

    KEY_FILE = 'api_keys/nyt_keys.json'
    log_folder = "logs-nyt/"

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    start_month = 2
    start_year = 1868
    end_month = 9
    end_year = 2017

    api_key = loadKeys(KEY_FILE)

    log_file = str(start_month) + str(start_year) + "-" + str(end_month) + str(end_year) + ".log"

    if os.path.isfile(log_folder + log_file):
        os.remove(log_folder + log_file)

    logging.basicConfig(filename=(log_folder + log_file), level=logging.INFO)

    logging.info("Getting started.")
    try:
        # LOOP THROUGH THE SPECIFIED DATES
        for month, year in daterange( start_month, start_year, end_month, end_year ):
            logging.info("Working on month %s year %s.", str(month), str(year))
            payload = getArticles(month, year, api_key)
            requests.post(base_url + "add_article", json = payload)
    except:
        logging.error("Unexpected error: %s", str(sys.exc_info()[0]))
        return None
    finally:
        logging.info("Finished.")

if __name__ == '__main__' :
    main()
