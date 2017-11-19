import urllib2
import json
import datetime
import time
import sys, os
import logging
from urllib2 import HTTPError
from ConfigParser import SafeConfigParser


# helper function to iterate through dates
def daterange( start_month, start_year, end_month, end_year ):
    daterange = []
    for year in range(start_year, end_year + 1):
        month_range_start = start_month if (start_year == year) else 1
        month_range_end = end_month if (end_year == year) else 12
        for month in range(month_range_start, month_range_end + 1):
            daterange.append((month, year))
    print daterange
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

# helpful function to figure out what to name individual JSON files
def getJsonFileName(month, year, json_file_path):
    filename = "".join([str(year), "_", str(month)])
    json_file_name = ".".join([filename,'json'])
    json_file_name = "".join([json_file_path,json_file_name])
    return json_file_name

# get the articles from the NYTimes Article API
def getArticles(month, year, api_key, json_file_path):
    for n in range(5): # 5 tries
        try:
            print "\nretrieve month", month, "year", year
            # https://api.nytimes.com/svc/archive/v1/2016/1.json?month=9&year=2017&api-key=264d0364650940e3ab0d6690c2cfd065
            request_string = "https://api.nytimes.com/svc/archive/v1/" + str(year) + "/" + str(month) + ".json?&api-key=" + api_key
            print request_string
            response = urllib2.urlopen(request_string)
            content = response.read()
            if content:
                articles = convert(json.loads(content))
                # if there are articles here
                if len(articles["response"]["docs"]) >= 1:
                    json_file_name = getJsonFileName(month, year, json_file_path)
                    json_file = open(json_file_name, 'w')
                    # json_file.write(content)
                    json.dump([articles], json_file, indent=4)
                    json_file.close()
                # if no more articles, go to next date
                else:
                    return
            print "finished month ", month, "year ", year
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

# Main function where stuff gets done

def main():

    json_file_path = "archives-json/"
    log_folder = "logs/"

    if not os.path.exists(json_file_path):
        os.makedirs(json_file_path)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    #  TODO: ADJUST ME TO CHANGE RANGE
    if len(sys.argv) < 6:
        print 'Usage: python getTimesArticles.py <api_key> <start_month> <start_year> <end_month> <end_year>'
        sys.exit(0)
    start_month = int(sys.argv[2]) #9 
    start_year = int(sys.argv[3]) #1851
    end_month = int(sys.argv[4]) #9
    end_year = int(sys.argv[5]) #2017
    api_key = sys.argv[1] #"264d0364650940e3ab0d6690c2cfd065"

    log_file = str(start_month) + str(start_year) + "-" + str(end_month) + str(end_year) + ".log"

    if os.path.isfile(log_folder + log_file):
        os.remove(log_folder + log_file)

    logging.basicConfig(filename=(log_folder + log_file), level=logging.INFO)

    logging.info("Getting started.")
    try:
        # LOOP THROUGH THE SPECIFIED DATES
        for month, year in daterange( start_month, start_year, end_month, end_year ):
            logging.info("Working on month %s year %s.", str(month), str(year))
            getArticles(month, year, api_key, json_file_path)
    except:
        logging.error("Unexpected error: %s", str(sys.exc_info()[0]))
    finally:
        logging.info("Finished.")

if __name__ == '__main__' :
    main()
